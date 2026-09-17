"""Current cross-component contract validation."""

import pytest

from multi_task_scheduler.orchestration.contracts import (
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
    SyncHealth,
    SyncSnapshot,
    TaskSnapshot,
)
from multi_task_scheduler.orchestration.operation_journal import (
    OperationKind,
    OperationStatus,
    Phase,
)
from multi_task_scheduler.orchestration.production_window import (
    ProductionWindow,
    WindowState,
)
from multi_task_scheduler.orchestration.replica_record import (
    ReplicaKind,
    ReplicaRecord,
    ReplicaState,
)


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
        active_samples=1,
        output_queue_size=0,
        max_queue_size=8,
        producer_exhausted=False,
        policy_refresh_inflight=False,
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
    assert command.ctx is ctx
    assert command.target is target
    assert command.kind is OperationKind.ADD
    assert command.recall_mode is None

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

    restore_ctx = _ctx(operation_id="restore-1")
    restore = OperationCommand(
        ctx=restore_ctx,
        kind=OperationKind.RESTORE,
        target=target,
        authorization=_authorization(OperationKind.RESTORE, restore_ctx),
        payload_digest="restore-payload",
        remaining_budget_ms=1000,
    )
    assert restore.recall_mode is None
    with pytest.raises(ValueError, match="must not carry recall_mode"):
        OperationCommand(
            ctx=restore_ctx,
            kind=OperationKind.RESTORE,
            target=target,
            authorization=_authorization(OperationKind.RESTORE, restore_ctx),
            payload_digest="restore-payload",
            remaining_budget_ms=1000,
            recall_mode=RecallMode.NATURAL,
        )


def test_force_verified_is_remove_only():
    ctx = _ctx()
    target = ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0)
    with pytest.raises(ValueError, match="recall_mode"):
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

    with pytest.raises(ValueError, match="UNKNOWN"):
        OperationResult(
            ctx=_ctx(),
            target=ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0),
            status=OperationStatus.UNKNOWN,
            phase=Phase.CREATE,
            phase_revision=3,
        )

    assert OperationResult(
        ctx=_ctx(),
        target=ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0),
        status=OperationStatus.UNKNOWN,
        phase=Phase.RECONCILE,
        phase_revision=4,
    ).phase is Phase.RECONCILE


def test_route_entry_is_typed_and_keeps_route_fences():
    key = ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0)
    route = RouteEntry(
        key=key,
        head_server=object(),
        state=RouteState.DRAINING,
        replica_route_epoch=2,
        sync_epoch=3,
        serving_version=7,
        commit_operation_id="op-1",
    )
    assert route.state is RouteState.DRAINING
    assert route.replica_route_epoch == 2
    with pytest.raises(ValueError, match="commit_operation_id"):
        RouteEntry(key, object(), RouteState.ROUTABLE, 1, 0, 7, "")


def test_capacity_record_derives_total_capacity_from_committed_active_set():
    first = ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0)
    second = ReplicaKey(task_session="s1", replica_id="r2", runtime_epoch=0)
    capacity = CapacityRecord(
        active_ids=frozenset({first, second}),
        revision=4,
        per_replica_limit=6,
        last_commit_operation_id="op-4",
    )
    assert capacity.max_concurrent_samples == 12
    with pytest.raises(ValueError, match="per_replica_limit"):
        CapacityRecord(frozenset(), 0, 0, "op-1")


def test_sync_snapshot_keeps_health_separate_from_version_and_ce_revision():
    snapshot = SyncSnapshot(
        version=11,
        ce_revision=8,
        health=SyncHealth.BLOCKED,
        owner_operation_id="op-sync",
    )
    assert snapshot.health is SyncHealth.BLOCKED
    assert snapshot.version == 11
    with pytest.raises(ValueError, match="owner_operation_id"):
        SyncSnapshot(11, 8, SyncHealth.HEALTHY, "")


def test_task_snapshot_rejects_cross_session_owner_facts():
    key = ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0)
    capacity = CapacityRecord(
        active_ids=frozenset({key}),
        revision=2,
        per_replica_limit=4,
        last_commit_operation_id="op-2",
    )
    snapshot = TaskSnapshot(
        task_session="s1",
        production=_window(),
        replica_records=(_record(),),
        ce_revision=3,
        lb_revision=4,
        capacity=capacity,
        sync_health=SyncHealth.HEALTHY,
        current_operation_id=None,
        consistency="STABLE",
    )
    assert snapshot.task_session == "s1"

    with pytest.raises(ValueError, match="production window"):
        TaskSnapshot(
            task_session="s1",
            production=_window("s2"),
            replica_records=(_record(),),
            ce_revision=3,
            lb_revision=4,
            capacity=capacity,
            sync_health=SyncHealth.HEALTHY,
            current_operation_id=None,
            consistency="UNKNOWN",
        )

    foreign = ReplicaKey(task_session="s2", replica_id="r2", runtime_epoch=0)
    with pytest.raises(ValueError, match="capacity active_ids"):
        TaskSnapshot(
            task_session="s1",
            production=_window(),
            replica_records=(_record(),),
            ce_revision=3,
            lb_revision=4,
            capacity=CapacityRecord(frozenset({foreign}), 3, 4, "op-3"),
            sync_health=SyncHealth.HEALTHY,
            current_operation_id=None,
            consistency="IN_PROGRESS",
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
