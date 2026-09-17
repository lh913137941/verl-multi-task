"""Release evidence must prove the exact GPU set before GS handoff."""

import pytest

from multi_task_scheduler.orchestration.contracts import (
    NodePlacement,
    OperationContext,
    OperationResult,
    PlacementSpec,
    ReleaseKind,
    ReplicaKey,
    ServiceAction,
)
from multi_task_scheduler.orchestration.operation_journal import OperationStatus, Phase
from multi_task_scheduler.orchestration.receipts import (
    EvidenceHeader,
    GPURelease,
    ReleaseEvidence,
    ServiceEvidence,
)
from multi_task_scheduler.scheduler.lease import (
    LeaseState,
    LeaseStateMachine,
    MissingEvidenceError,
)
from multi_task_scheduler.scheduler.ledger import LeaseRecord


def _placement() -> PlacementSpec:
    return PlacementSpec(
        node=NodePlacement(
            node_id="n1",
            gpu_uuids=("u0", "u1"),
            physical_gpu_ids=(0, 1),
            global_ranks=(0, 1),
            local_ranks=(0, 1),
        ),
        tp=2,
        dp=1,
        pp=1,
        model_signature="sig",
        placement_digest="placement-u0-u1",
    )


def _gpu(gpu_uuid: str) -> GPURelease:
    return GPURelease(
        gpu_uuid=gpu_uuid,
        free_hbm_bytes=1024,
        residual_hbm_bytes=0,
        owned_processes=(),
        unknown_processes=(),
        device_work_complete=True,
        meets_release_budget=True,
    )


def _release_result(per_gpu: tuple[GPURelease, ...]) -> OperationResult:
    ctx = OperationContext(
        protocol_version=1,
        gs_epoch="gs-1",
        task_id="task-donor",
        task_session="donor",
        operation_id="op-donate",
        lease_id="l1",
        lease_epoch=0,
        command_seq=1,
    )
    key = ReplicaKey(task_session="donor", replica_id="r1", runtime_epoch=0)
    service_header = EvidenceHeader(
        ctx=ctx,
        key=key,
        phase_revision=4,
        digest="service-remove",
    )
    service = ServiceEvidence(
        header=service_header,
        action=ServiceAction.REMOVE,
        version=None,
        ce_revision=1,
        lb_revision=2,
        route_epoch=3,
        capacity_revision=4,
        manager_revision=5,
        ce_commit_digest="ce-remove",
        lb_commit_digest="lb-remove",
        prerequisite_digest="exit-1",
    )
    release = ReleaseEvidence(
        header=EvidenceHeader(
            ctx=ctx,
            key=key,
            phase_revision=6,
            digest="release-1",
        ),
        release_kind=ReleaseKind.DONOR_SLEEP_RELEASED,
        permit_digest=service_header.digest,
        inventory_digest="inventory-1",
        per_gpu=per_gpu,
        all_backends_confirmed=True,
        observation_interval_ms=100,
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


def test_release_evidence_rejects_duplicate_gpu_proofs():
    with pytest.raises(ValueError, match="one proof per GPU"):
        _release_result((_gpu("u0"), _gpu("u0")))


def test_gs_rejects_release_that_does_not_cover_entire_lease_placement():
    machine = _machine()
    incomplete = _release_result((_gpu("u0"),))

    with pytest.raises(MissingEvidenceError, match="exactly cover every GPU"):
        machine.advance(
            "l1",
            LeaseState.DONOR_RELEASED,
            supporting_result=incomplete,
        )


def test_gs_accepts_release_only_when_all_lease_gpus_are_proven():
    machine = _machine()
    complete = _release_result((_gpu("u0"), _gpu("u1")))

    lease = machine.advance(
        "l1",
        LeaseState.DONOR_RELEASED,
        supporting_result=complete,
    )

    assert lease.state == LeaseState.DONOR_RELEASED.value
    assert lease.last_release_digest == complete.release.header.digest
