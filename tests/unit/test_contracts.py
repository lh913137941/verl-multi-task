"""Cross-component contract value semantics and evidence validation."""

import pytest

from multi_task_scheduler.orchestration.contracts import (
    NodePlacement,
    OperationCommand,
    OperationContext,
    OperationResult,
    PlacementSpec,
    PreparedReplica,
    PublishedWeightSnapshot,
    RecallMode,
    ReleaseEvidence,
    ReleaseKind,
    TransferKind,
    TransferReceipt,
)
from multi_task_scheduler.orchestration.operation_journal import (
    OperationKind,
    OperationStatus,
    Phase,
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


def _placement():
    node = NodePlacement(
        node_id="n1",
        gpu_uuids=("u0", "u1"),
        physical_gpu_ids=(0, 1),
        global_ranks=(0, 1),
        local_ranks=(0, 1),
    )
    return PlacementSpec(
        node=node,
        tp=2,
        model_signature="sig-1",
        placement_digest="placement-1",
    )


def _snapshot(version=7, digest="manifest-7"):
    return PublishedWeightSnapshot(
        snapshot_id=f"snapshot-{version}",
        manifest_digest=digest,
        model_signature="sig-1",
        version=version,
        byte_size=1024,
        sender=object(),
    )


def test_operation_context_identity_includes_command_fence():
    assert _ctx().identity == _ctx().identity
    assert _ctx().identity != _ctx(lease_epoch=1).identity
    assert _ctx().identity != _ctx(command_seq=1).identity
    assert _ctx().identity != _ctx(expected_revision=1).identity


def test_operation_context_uses_int_protocol_and_string_gs_epoch():
    with pytest.raises(ValueError, match="protocol_version"):
        OperationContext("p1", "gs-1", "t", "s", "op", "l", 0, 0)
    with pytest.raises(ValueError, match="gs_epoch"):
        OperationContext(1, "", "t", "s", "op", "l", 0, 0)


def test_single_node_placement_world_size_and_uuid_order():
    spec = _placement()
    assert spec.world_size == 2
    assert spec.gpu_uuids == ("u0", "u1")
    assert spec.node.node_id == "n1"
    assert spec.placement_digest == "placement-1"
    assert spec.node_blocks == (spec.node,)  # compatibility read-only view


def test_placement_rejects_mismatched_arrays_or_unsupported_dp_pp():
    with pytest.raises(ValueError, match="same length"):
        NodePlacement("n1", ("u0",), (0, 1), (0,), (0,))
    with pytest.raises(ValueError, match="dp=1"):
        PlacementSpec(
            node=NodePlacement("n1", ("u0",), (0,), (0,), (0,)),
            model_signature="sig",
            placement_digest="p",
            dp=2,
        )


def test_prepared_replica_with_receivers_preserves_identity():
    prepared = PreparedReplica(
        replica_id="r1", runtime_epoch=3, operation_id="op-1", placement_digest="d1"
    )
    updated = prepared.with_receivers(
        receiver_ids=["w0", "w1"], receivers=[object(), object()], head_server=object()
    )
    assert (updated.replica_id, updated.runtime_epoch, updated.operation_id) == ("r1", 3, "op-1")
    assert updated.receiver_ids == ("w0", "w1")
    assert len(updated.receiver_descriptors) == 2


def test_transfer_receipt_requires_all_receivers_complete():
    partial = TransferReceipt(
        transfer_id="t1",
        kind=TransferKind.BOOTSTRAP,
        target_version=5,
        manifest_digest="m1",
        expected_receiver_ids=("w0", "w1"),
        receiver_states={"w0": "complete"},
    )
    assert not partial.all_receivers_complete
    complete = TransferReceipt(
        transfer_id="t1",
        kind=TransferKind.BOOTSTRAP,
        target_version=5,
        manifest_digest="m1",
        expected_receiver_ids=("w0", "w1"),
        receiver_states={"w0": "complete", "w1": "complete"},
    )
    assert complete.all_receivers_complete


def test_release_evidence_kind_distinguishes_sleep_from_destroy_and_requires_gpu_evidence():
    donor = ReleaseEvidence(
        replica_id="r1", runtime_epoch=1, lease_epoch=0,
        lb_excluded=True, ce_excluded=True, no_inflight_transfer=True,
        process_cleared_or_slept=True, per_gpu_hbm_free={"u0": 10},
        release_kind=ReleaseKind.DONOR_SLEEP_RELEASED,
    )
    borrower = ReleaseEvidence(
        replica_id="r1", runtime_epoch=1, lease_epoch=0,
        lb_excluded=True, ce_excluded=True, no_inflight_transfer=True,
        process_cleared_or_slept=True, per_gpu_hbm_free={"u0": 10},
        release_kind=ReleaseKind.BORROWER_RUNTIME_DESTROYED,
    )
    assert donor.complete and borrower.complete
    assert donor.release_kind is ReleaseKind.DONOR_SLEEP_RELEASED
    assert borrower.release_kind is ReleaseKind.BORROWER_RUNTIME_DESTROYED


def test_command_uses_operation_kind_and_force_only_on_remove():
    command = OperationCommand(
        protocol_version=1, gs_epoch="gs-1",
        target_task_id="task-b", target_task_session="s2",
        operation_id="op-2", payload_digest="digest", kind=OperationKind.ADD,
        lease_id="lease-1", lease_epoch=0, command_seq=1, replica_id="r1",
        placement=_placement(),
    )
    assert command.kind is OperationKind.ADD
    assert command.context.command_seq == 1
    with pytest.raises(ValueError, match="FORCE_VERIFIED"):
        OperationCommand(
            protocol_version=1, gs_epoch="gs-1",
            target_task_id="task-b", target_task_session="s2",
            operation_id="op-3", payload_digest="digest", kind=OperationKind.ADD,
            lease_id="lease-1", lease_epoch=0, command_seq=2, replica_id="r1",
            recall_mode=RecallMode.FORCE_VERIFIED,
        )


def test_operation_result_uses_typed_phase_and_status():
    result = OperationResult(
        identity_fields=_ctx(), phase=Phase.CREATE, phase_revision=2,
        state=OperationStatus.RUNNING, actual_replica_state="PREPARING",
    )
    assert result.phase is Phase.CREATE
    assert result.status is OperationStatus.RUNNING
    assert result.replica_state == "PREPARING"
    with pytest.raises(ValueError, match="DONE"):
        OperationResult(
            identity_fields=_ctx(), phase=Phase.DONE, phase_revision=3,
            state=OperationStatus.RUNNING, actual_replica_state="ACTIVE",
        )


def test_published_weight_snapshot_requires_real_content_identity():
    snap = _snapshot()
    assert snap.serving_version == 7
    with pytest.raises(ValueError, match="manifest_digest"):
        PublishedWeightSnapshot(
            snapshot_id="snapshot-7", manifest_digest="", model_signature="sig-1",
            version=7, byte_size=1024, sender=object(),
        )
