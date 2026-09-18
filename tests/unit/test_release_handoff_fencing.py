"""Release evidence must prove the exact GPU set before GS handoff."""

import pytest

from multi_task_scheduler.orchestration.contracts import (
    NodePlacement,
    OperationContext,
    OperationResult,
    PlacementSpec,
    ReplicaKey,
    ServiceAction,
)
from multi_task_scheduler.orchestration.operation_journal import OperationStatus, Phase
from multi_task_scheduler.orchestration.receipts import EvidenceHeader, ReleaseEvidence, ServiceEvidence
from multi_task_scheduler.scheduler.lease import LeaseState, LeaseStateMachine, MissingEvidenceError
from multi_task_scheduler.scheduler.ledger import LeaseRecord


def _placement() -> PlacementSpec:
    return PlacementSpec(
        node=NodePlacement("n1", ("u0", "u1"), (0, 1), (0, 1), (0, 1)),
        tp=2,
        dp=1,
        pp=1,
        model_signature="sig",
        placement_digest="placement-u0-u1",
    )


def _release_result(released_gpu_uuids: tuple[str, ...]) -> OperationResult:
    ctx = OperationContext(1, "gs-1", "task-donor", "donor", "op-donate", "l1", 0, 1)
    key = ReplicaKey("donor", "r1", 0)
    service_header = EvidenceHeader(ctx, key, 4, "service-remove")
    service = ServiceEvidence(
        header=service_header,
        action=ServiceAction.REMOVE,
        prerequisite_digest="exit-1",
        service_digest="service-state-1",
    )
    release = ReleaseEvidence(
        header=EvidenceHeader(ctx, key, 6, "release-1"),
        release_kind="DONOR_SLEEP_RELEASED",
        permit_digest=service_header.digest,
        inventory_digest="inventory-1",
        released_gpu_uuids=released_gpu_uuids,
    )
    return OperationResult(
        ctx=ctx,
        target=key,
        status=OperationStatus.SUCCEEDED,
        phase=Phase.DONE,
        phase_revision=6,
        service=service,
        release=release,
    )


def _machine() -> LeaseStateMachine:
    machine = LeaseStateMachine()
    machine.register(
        LeaseRecord(
            lease_id="l1",
            donor_session="donor",
            borrower_session="borrower",
            state=LeaseState.DONOR_DRAINING.value,
            placement=_placement(),
        )
    )
    return machine


def test_release_evidence_rejects_duplicate_gpu_uuids():
    with pytest.raises(ValueError, match="duplicates"):
        _release_result(("u0", "u0"))


def test_gs_rejects_release_that_does_not_cover_entire_lease_placement():
    machine = _machine()
    incomplete = _release_result(("u0",))
    with pytest.raises(MissingEvidenceError, match="exactly cover every GPU"):
        machine.advance("l1", LeaseState.DONOR_RELEASED, supporting_result=incomplete)


def test_gs_accepts_release_only_when_all_lease_gpus_are_proven():
    machine = _machine()
    complete = _release_result(("u0", "u1"))
    lease = machine.advance("l1", LeaseState.DONOR_RELEASED, supporting_result=complete)
    assert lease.state == LeaseState.DONOR_RELEASED.value
    assert lease.last_release_digest == complete.release.header.digest
