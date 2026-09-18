"""Current cross-component contract validation."""

import pytest

from multi_task_scheduler.orchestration.contracts import (
    CapabilityProof,
    CapacityRecord,
    LeaseAuthorization,
    NodePlacement,
    OperationCommand,
    OperationContext,
    OperationResult,
    PlacementSpec,
    PublishedWeightSnapshot,
    RecallMode,
    ReplicaKey,
    RouteEntry,
    RouteState,
    RuntimeReady,
    SyncHealth,
    TaskSnapshot,
)
from multi_task_scheduler.orchestration.operation_journal import (
    OperationKind,
    OperationStatus,
    Phase,
)
from multi_task_scheduler.orchestration.production_window import ProductionWindow, WindowState
from multi_task_scheduler.orchestration.replica_record import ReplicaKind, ReplicaRecord, ReplicaState


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


def _placement(*, task_gpu="u0"):
    return PlacementSpec(
        node=NodePlacement(
            node_id="n1",
            gpu_uuids=(task_gpu, "u1") if task_gpu == "u0" else (task_gpu,),
            physical_gpu_ids=(0, 1) if task_gpu == "u0" else (0,),
            global_ranks=(0, 1) if task_gpu == "u0" else (0,),
            local_ranks=(0, 1) if task_gpu == "u0" else (0,),
        ),
        tp=2 if task_gpu == "u0" else 1,
        dp=1,
        pp=1,
        model_signature="sig-1",
        placement_digest="placement-1" if task_gpu == "u0" else f"placement-{task_gpu}",
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


def _window(task_session="s1"):
    return ProductionWindow(
        task_session=task_session,
        epoch=3,
        revision=9,
        state=WindowState.CLOSED_BACKPRESSURE,
        eligible_pending=0,
        held_samples=0,
    )


def _record(task_session="s1", replica_id="r1"):
    placement = _placement() if task_session == "s1" else _placement(task_gpu="other-u0")
    return ReplicaRecord(
        key=ReplicaKey(task_session=task_session, replica_id=replica_id, runtime_epoch=0),
        kind=ReplicaKind.NATIVE,
        state=ReplicaState.ACTIVE,
        revision=1,
        placement=placement,
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


def test_add_and_restore_must_not_carry_recall_mode():
    ctx = _ctx()
    target = ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0)
    with pytest.raises(ValueError, match="must not carry recall_mode"):
        OperationCommand(
            ctx=ctx,
            kind=OperationKind.ADD,
            target=target,
            authorization=_authorization(OperationKind.ADD, ctx),
            payload_digest="payload-1",
            remaining_budget_ms=1000,
            placement=_placement(),
            recall_mode=RecallMode.NATURAL,
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
    with pytest.raises(ValueError, match="DONE"):
        OperationResult(
            ctx=_ctx(),
            target=result.target,
            status=OperationStatus.RUNNING,
            phase=Phase.DONE,
            phase_revision=3,
        )


def test_route_entry_and_capacity_keep_owner_fences():
    key = ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0)
    route = RouteEntry(key, object(), RouteState.DRAINING, 2, 3, 7, "op-1")
    assert route.state is RouteState.DRAINING
    capacity = CapacityRecord(frozenset({key}), 4, 6, "op-4")
    assert capacity.max_concurrent_samples == 6


def test_task_snapshot_includes_published_version_and_rejects_cross_session_facts():
    key = ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0)
    capacity = CapacityRecord(frozenset({key}), 2, 4, "op-2")
    snapshot = TaskSnapshot(
        task_session="s1",
        production=_window(),
        replica_records=(_record(),),
        ce_revision=3,
        published_version=11,
        lb_revision=4,
        capacity=capacity,
        sync_health=SyncHealth.HEALTHY,
        current_operation_id=None,
        consistency="STABLE",
    )
    assert snapshot.published_version == 11
    with pytest.raises(ValueError, match="production window"):
        TaskSnapshot(
            task_session="s1",
            production=_window("s2"),
            replica_records=(_record(),),
            ce_revision=3,
            published_version=11,
            lb_revision=4,
            capacity=capacity,
            sync_health=SyncHealth.HEALTHY,
            current_operation_id=None,
            consistency="UNKNOWN",
        )


def test_published_weight_snapshot_drops_observability_byte_size():
    snapshot = PublishedWeightSnapshot(
        snapshot_id="snapshot-7",
        manifest_digest="manifest-7",
        model_signature="sig-1",
        version=7,
        sender=object(),
    )
    assert snapshot.version == 7
    assert not hasattr(snapshot, "byte_size")


def test_runtime_ready_and_capability_proof_keep_verified_shapes():
    key = ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0)
    assert RuntimeReady(key, "RECEIVER_READY", None).loaded_version is None
    with pytest.raises(ValueError, match="requires loaded_version"):
        RuntimeReady(key, "SERVING_READY", None)
    proof = CapabilityProof(
        name="target_abort_resume",
        backend_version="vllm-x",
        model_signature="sig-1",
        placement_digest="placement-1",
        validation_id="validation-1",
    )
    assert proof.name == "target_abort_resume"
