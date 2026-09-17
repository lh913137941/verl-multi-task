"""Lease authorization advances only from matching lifecycle evidence."""

import pytest

from multi_task_scheduler.orchestration.contracts import (
    OperationContext,
    OperationResult,
    ReleaseKind,
    ReplicaKey,
    ServiceAction,
)
from multi_task_scheduler.orchestration.operation_journal import (
    OperationStatus,
    Phase,
)
from multi_task_scheduler.orchestration.receipts import (
    EvidenceHeader,
    GPURelease,
    ReleaseEvidence,
    ServiceEvidence,
)
from multi_task_scheduler.scheduler.lease import (
    IllegalLeaseTransitionError,
    LeaseState,
    LeaseStateMachine,
    MissingEvidenceError,
)
from multi_task_scheduler.scheduler.ledger import LeaseRecord


def _machine(*, state=LeaseState.PLANNED, release_digest=None):
    machine = LeaseStateMachine()
    machine.register(
        LeaseRecord(
            lease_id="l1",
            donor_session="donor",
            borrower_session="borrower",
            state=state.value,
            last_release_digest=release_digest,
        )
    )
    return machine


def _ctx(session, operation_id, *, lease_epoch=0):
    return OperationContext(
        protocol_version=1,
        gs_epoch="gs-1",
        task_id=f"task-{session}",
        task_session=session,
        operation_id=operation_id,
        lease_id="l1",
        lease_epoch=lease_epoch,
        command_seq=1,
    )


def _header(context, key, digest, revision=1):
    return EvidenceHeader(
        ctx=context,
        key=key,
        phase_revision=revision,
        digest=digest,
    )


def _service_result(session, operation_id, action, *, lease_epoch=0):
    context = _ctx(session, operation_id, lease_epoch=lease_epoch)
    key = ReplicaKey(task_session=session, replica_id="r1", runtime_epoch=0)
    service = ServiceEvidence(
        header=_header(context, key, f"service-{operation_id}", 4),
        action=action,
        version=7 if action is ServiceAction.ADD else None,
        ce_revision=2,
        lb_revision=3,
        route_epoch=4,
        capacity_revision=5,
        manager_revision=6,
        ce_commit_digest=f"ce-{operation_id}",
        lb_commit_digest=f"lb-{operation_id}",
        prerequisite_digest=f"prerequisite-{operation_id}",
    )
    return OperationResult(
        ctx=context,
        target=key,
        status=OperationStatus.SUCCEEDED,
        phase=Phase.DONE,
        phase_revision=6,
        service=service,
    )


def _release_result(session, operation_id, release_kind, *, lease_epoch=0):
    result = _service_result(
        session,
        operation_id,
        ServiceAction.REMOVE,
        lease_epoch=lease_epoch,
    )
    release = ReleaseEvidence(
        header=_header(result.ctx, result.target, f"release-{operation_id}", 7),
        release_kind=release_kind,
        permit_digest=result.service.header.digest,
        inventory_digest=f"inventory-{operation_id}",
        per_gpu=(
            GPURelease(
                gpu_uuid="u0",
                free_hbm_bytes=1024,
                residual_hbm_bytes=0,
                owned_processes=(),
                unknown_processes=(),
                device_work_complete=True,
                meets_release_budget=True,
            ),
        ),
        all_backends_confirmed=True,
        observation_interval_ms=100,
    )
    return OperationResult(
        ctx=result.ctx,
        target=result.target,
        status=result.status,
        phase=result.phase,
        phase_revision=7,
        service=result.service,
        release=release,
    )


def test_full_happy_path_requires_release_and_service_evidence():
    sm = _machine()
    sm.advance("l1", LeaseState.DONOR_DRAINING)
    donor_release = _release_result(
        "donor", "op-donate", ReleaseKind.DONOR_SLEEP_RELEASED
    )
    sm.advance(
        "l1",
        LeaseState.DONOR_RELEASED,
        supporting_result=donor_release,
    )
    assert sm.get("l1").last_release_digest == donor_release.release.header.digest

    sm.advance("l1", LeaseState.BORROWER_PREPARING)
    sm.advance(
        "l1",
        LeaseState.BORROWER_ACTIVE,
        supporting_result=_service_result("borrower", "op-add", ServiceAction.ADD),
    )
    sm.advance("l1", LeaseState.RECALLING)
    borrower_release = _release_result(
        "borrower", "op-remove", ReleaseKind.BORROWER_RUNTIME_DESTROYED
    )
    sm.advance(
        "l1",
        LeaseState.BORROWER_RELEASED,
        supporting_result=borrower_release,
    )
    assert sm.get("l1").last_release_digest == borrower_release.release.header.digest

    sm.advance("l1", LeaseState.DONOR_RESTORING)
    sm.advance(
        "l1",
        LeaseState.CLOSED,
        supporting_result=_service_result("donor", "op-restore", ServiceAction.ADD),
    )
    assert sm.get("l1").state == LeaseState.CLOSED.value
    assert sm.get("l1").operation_ids == (
        "op-donate",
        "op-add",
        "op-remove",
        "op-restore",
    )


def test_succeeded_status_without_release_evidence_cannot_handoff_gpu():
    sm = _machine()
    sm.advance("l1", LeaseState.DONOR_DRAINING)
    result = _service_result("donor", "op-donate", ServiceAction.REMOVE)
    with pytest.raises(MissingEvidenceError, match="ReleaseEvidence"):
        sm.advance("l1", LeaseState.DONOR_RELEASED, supporting_result=result)


def test_release_must_reference_matching_service_removal_proof():
    sm = _machine()
    sm.advance("l1", LeaseState.DONOR_DRAINING)
    result = _release_result("donor", "op-donate", ReleaseKind.DONOR_SLEEP_RELEASED)
    bad_release = ReleaseEvidence(
        header=result.release.header,
        release_kind=result.release.release_kind,
        permit_digest="different-service",
        inventory_digest=result.release.inventory_digest,
        per_gpu=result.release.per_gpu,
        all_backends_confirmed=True,
        observation_interval_ms=result.release.observation_interval_ms,
    )
    result = OperationResult(
        ctx=result.ctx,
        target=result.target,
        status=result.status,
        phase=result.phase,
        phase_revision=result.phase_revision,
        service=result.service,
        release=bad_release,
    )
    with pytest.raises(MissingEvidenceError, match="service-removal"):
        sm.advance("l1", LeaseState.DONOR_RELEASED, supporting_result=result)


def test_release_kind_and_participant_must_match_lease_edge():
    sm = _machine()
    sm.advance("l1", LeaseState.DONOR_DRAINING)
    wrong_kind = _release_result(
        "donor", "op-donate", ReleaseKind.BORROWER_RUNTIME_DESTROYED
    )
    with pytest.raises(MissingEvidenceError, match="release kind"):
        sm.advance("l1", LeaseState.DONOR_RELEASED, supporting_result=wrong_kind)

    wrong_session = _release_result(
        "borrower", "op-donate-2", ReleaseKind.DONOR_SLEEP_RELEASED
    )
    with pytest.raises(MissingEvidenceError, match="wrong lease participant"):
        sm.advance("l1", LeaseState.DONOR_RELEASED, supporting_result=wrong_session)


def test_stale_lease_epoch_cannot_advance_release_edge():
    sm = _machine()
    sm.advance("l1", LeaseState.DONOR_DRAINING)
    stale = _release_result(
        "donor",
        "op-donate",
        ReleaseKind.DONOR_SLEEP_RELEASED,
        lease_epoch=1,
    )
    with pytest.raises(MissingEvidenceError, match="lease identity/epoch"):
        sm.advance("l1", LeaseState.DONOR_RELEASED, supporting_result=stale)


def test_post_release_authorization_requires_stored_release_digest():
    sm = _machine(state=LeaseState.DONOR_RELEASED)
    with pytest.raises(MissingEvidenceError, match="confirmed release digest"):
        sm.advance("l1", LeaseState.BORROWER_PREPARING)

    sm = _machine(state=LeaseState.DONOR_RELEASED, release_digest="release-d1")
    sm.advance("l1", LeaseState.BORROWER_PREPARING)
    assert sm.get("l1").state == LeaseState.BORROWER_PREPARING.value


def test_service_activation_requires_add_evidence_not_status_only():
    sm = _machine(
        state=LeaseState.BORROWER_PREPARING,
        release_digest="release-d1",
    )
    remove_result = _service_result("borrower", "op-add", ServiceAction.REMOVE)
    with pytest.raises(MissingEvidenceError, match="ServiceEvidence\(ADD\)"):
        sm.advance(
            "l1",
            LeaseState.BORROWER_ACTIVE,
            supporting_result=remove_result,
        )


def test_illegal_skip_and_reconciling_terminal_behavior():
    sm = _machine()
    with pytest.raises(IllegalLeaseTransitionError):
        sm.advance("l1", LeaseState.DONOR_RELEASED)

    sm.advance("l1", LeaseState.DONOR_DRAINING)
    sm.advance("l1", LeaseState.RECONCILING)
    with pytest.raises(IllegalLeaseTransitionError):
        sm.advance("l1", LeaseState.DONOR_RELEASED)
