"""Cross-component contract value semantics and evidence validation."""

import pytest

from multi_task_scheduler.orchestration.contracts import (
    Command,
    GpuPlacement,
    NodeBlock,
    OperationContext,
    OperationResult,
    PlacementSpec,
    PreparedReplica,
    PublishedWeightSnapshot,
    ReleaseKind,
    ReleaseReceipt,
    TransferKind,
    TransferReceipt,
)
from multi_task_scheduler.orchestration.operation_journal import OperationType


def _ctx(**overrides):
    values = dict(
        protocol_version="p1",
        gs_epoch=1,
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
    block = NodeBlock(
        node_id="n1",
        gpus=(
            GpuPlacement(gpu_uuid="u0", physical_id=0, global_rank=0, local_rank=0),
            GpuPlacement(gpu_uuid="u1", physical_id=1, global_rank=1, local_rank=1),
        ),
    )
    return PlacementSpec(node_blocks=(block,), tp=2, model_signature="sig-1")


def _snapshot(version=7, digest="manifest-7"):
    return PublishedWeightSnapshot(
        snapshot_id=f"snapshot-{version}",
        manifest_digest=digest,
        model_signature="sig-1",
        version=version,
        byte_size=1024,
        sender=object(),
    )


def test_operation_context_identity_is_stable():
    assert _ctx().identity == _ctx().identity
    assert _ctx().identity != _ctx(lease_epoch=1).identity


def test_placement_spec_world_size_and_uuid_order():
    spec = _placement()
    assert spec.world_size == 2
    assert spec.gpu_uuids == ("u0", "u1")


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


def test_release_receipt_kind_distinguishes_sleep_from_destroy_and_requires_per_gpu_evidence():
    donor = ReleaseReceipt(
        replica_id="r1", runtime_epoch=1, lease_epoch=0,
        lb_excluded=True, ce_excluded=True, no_inflight_transfer=True,
        process_cleared_or_slept=True, per_gpu_hbm_free={"u0": 10},
        release_kind=ReleaseKind.DONOR_SLEEP,
    )
    borrower = ReleaseReceipt(
        replica_id="r1", runtime_epoch=1, lease_epoch=0,
        lb_excluded=True, ce_excluded=True, no_inflight_transfer=True,
        process_cleared_or_slept=True, per_gpu_hbm_free={"u0": 10},
        release_kind=ReleaseKind.BORROWER_RELEASE,
    )
    assert donor.complete and borrower.complete
    assert donor.release_kind is ReleaseKind.DONOR_SLEEP
    assert borrower.release_kind is ReleaseKind.BORROWER_RELEASE


def test_command_and_result_are_frozen_and_serializable():
    command = Command(
        protocol_version="p1", gs_epoch=1, target_task_id="task-b", target_task_session="s2",
        operation_id="op-2", payload_digest="digest", kind=OperationType.ADD,
        lease_id="lease-1", lease_epoch=0, command_seq=1, replica_id="r1", placement=_placement(),
    )
    assert command.kind is OperationType.ADD
    assert command.placement.world_size == 2
    result = OperationResult(
        identity_fields=_ctx(), phase="CREATE", phase_revision=2,
        state="RUNNING", actual_replica_state="PREPARING",
    )
    assert result.state == "RUNNING"


def test_published_weight_snapshot_requires_real_content_identity():
    snap = _snapshot()
    assert snap.serving_version == 7
    with pytest.raises(ValueError, match="manifest_digest"):
        PublishedWeightSnapshot(
            snapshot_id="snapshot-7", manifest_digest="", model_signature="sig-1",
            version=7, byte_size=1024, sender=object(),
        )
