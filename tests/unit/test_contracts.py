"""Current cross-component contract validation."""

import pytest

from multi_task_scheduler.orchestration.contracts import (
    LeaseAuthorization,
    NodePlacement,
    OperationCommand,
    OperationContext,
    OperationResult,
    PlacementSpec,
    PublishedWeightSnapshot,
    RecallMode,
    ReplicaKey,
)
from multi_task_scheduler.orchestration.operation_journal import (
    OperationKind,
    OperationStatus,
    Phase,
)
from multi_task_scheduler.orchestration.replica_record import ReplicaState


def _ctx(**overrides):
    values = dict(
        protocol_version=1,
        gs_epoch="gs-1",
        task_id="task-a",
        task_session="s1",
        operation_id="op-1",
        lease_id="lease-1",
        lease_epoch=0,
        command_seq=0,
    )
    values.update(overrides)
    return OperationContext(**values)


def _placement():
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
        model_signature="sig-1",
        placement_digest="placement-1",
    )


def _authorization(kind, ctx, placement_digest="placement-1"):
    return LeaseAuthorization(
        lease_id=ctx.lease_id,
        gs_epoch=ctx.gs_epoch,
        donor_session="donor",
        borrower_session=ctx.task_session,
        placement_digest=placement_digest,
        lease_epoch=ctx.lease_epoch,
        purpose=kind,
        prior_release_digest=(
            "release-previous" if kind in {OperationKind.ADD, OperationKind.RESTORE} else None
        ),
        authorization_seq=1,
    )


def test_operation_context_identity_includes_all_fences():
    assert _ctx().identity == _ctx().identity
    assert _ctx().identity != _ctx(lease_epoch=1).identity
    assert _ctx().identity != _ctx(command_seq=1).identity
    assert _ctx().identity != _ctx(expected_revision=1).identity


def test_operation_context_requires_integer_protocol_and_string_gs_epoch():
    with pytest.raises(ValueError, match="protocol_version"):
        OperationContext("p1", "gs-1", "t", "s", "op", "l", 0, 0)
    with pytest.raises(ValueError, match="gs_epoch"):
        OperationContext(1, "", "t", "s", "op", "l", 0, 0)


def test_single_node_placement_requires_first_release_topology():
    spec = _placement()
    assert spec.node.gpu_uuids == ("u0", "u1")
    assert spec.tp == 2 and spec.dp == 1 and spec.pp == 1
    with pytest.raises(ValueError, match="same length"):
        NodePlacement("n1", ("u0",), (0, 1), (0,), (0,))
    with pytest.raises(ValueError, match="dp=1"):
        PlacementSpec(
            node=NodePlacement("n1", ("u0",), (0,), (0,), (0,)),
            tp=1,
            dp=2,
            pp=1,
            model_signature="sig",
            placement_digest="p",
        )


def test_add_command_requires_nested_identity_authorization_and_matching_placement():
    ctx = _ctx()
    target = ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0)
    command = OperationCommand(
        ctx=ctx,
        kind=OperationKind.ADD,
        target=target,
        authorization=_authorization(OperationKind.ADD, ctx),
        payload_digest="payload-1",
        remaining_budget_ms=1000,
        placement=_placement(),
    )
    assert command.ctx is ctx
    assert command.target is target
    assert command.kind is OperationKind.ADD

    with pytest.raises(ValueError, match="placement must match authorization"):
        OperationCommand(
            ctx=ctx,
            kind=OperationKind.ADD,
            target=target,
            authorization=_authorization(OperationKind.ADD, ctx, "different"),
            payload_digest="payload-1",
            remaining_budget_ms=1000,
            placement=_placement(),
        )


def test_force_verified_is_remove_only():
    ctx = _ctx()
    target = ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0)
    with pytest.raises(ValueError, match="FORCE_VERIFIED"):
        OperationCommand(
            ctx=ctx,
            kind=OperationKind.ADD,
            target=target,
            authorization=_authorization(OperationKind.ADD, ctx),
            payload_digest="payload-1",
            remaining_budget_ms=1000,
            placement=_placement(),
            recall_mode=RecallMode.FORCE_VERIFIED,
        )


def test_operation_result_keeps_phase_and_status_independent():
    result = OperationResult(
        ctx=_ctx(),
        target=ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0),
        status=OperationStatus.RUNNING,
        phase=Phase.CREATE,
        phase_revision=2,
        replica_state=ReplicaState.PREPARING,
    )
    assert result.phase is Phase.CREATE
    assert result.status is OperationStatus.RUNNING

    with pytest.raises(ValueError, match="DONE"):
        OperationResult(
            ctx=_ctx(),
            target=ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0),
            status=OperationStatus.RUNNING,
            phase=Phase.DONE,
            phase_revision=3,
        )


def test_published_weight_snapshot_requires_real_content_identity():
    snapshot = PublishedWeightSnapshot(
        snapshot_id="snapshot-7",
        manifest_digest="manifest-7",
        model_signature="sig-1",
        version=7,
        byte_size=1024,
        sender=object(),
    )
    assert snapshot.version == 7
    with pytest.raises(ValueError, match="snapshot_id"):
        PublishedWeightSnapshot(
            snapshot_id="",
            manifest_digest="manifest-7",
            model_signature="sig-1",
            version=7,
            byte_size=1024,
            sender=object(),
        )
