"""ScaleTransaction ordering against fake adapters for the simplified design."""

import asyncio

import pytest

from multi_task_scheduler.orchestration.contracts import (
    OperationContext,
    PreparedReplica,
    PublishedWeightSnapshot,
    RecallMode,
    ReleaseKind,
    ReleaseReceipt,
    TransferKind,
    TransferReceipt,
)
from multi_task_scheduler.orchestration.receipts import (
    AbortReceipt,
    ServingReadiness,
    WeightReadiness,
)
from multi_task_scheduler.orchestration.replica_sync_gate import ReplicaSyncGate
from multi_task_scheduler.orchestration.scale_transaction import (
    CheckpointProtocol,
    DrainTimeoutError,
    FenceRejectedError,
    MissingPublishedSnapshotError,
    PublishedSnapshotStore,
    ReplicaProtocol,
    RoutingProtocol,
    ScaleTransaction,
    ServiceNotDetachedError,
    ServingProtocol,
    TransferIncompleteError,
)


def ctx(operation_id="op-1", lease_epoch=0):
    return OperationContext(
        protocol_version="p1", gs_epoch=1, task_id="task-a", task_session="s1",
        operation_id=operation_id, lease_id="lease-1", lease_epoch=lease_epoch, command_seq=0,
    )


def prepared(replica_id="r1"):
    return PreparedReplica(
        replica_id=replica_id, runtime_epoch=1, operation_id="op-1",
        placement_digest="d1", receiver_ids=("w0", "w1"), head_server_descriptor="head",
    )


def snapshot(version=5, digest="m1"):
    return PublishedWeightSnapshot(
        snapshot_id=f"snap-{version}", manifest_digest=digest, model_signature="sig",
        version=version, byte_size=123, sender=object(),
    )


def transfer(complete=True, version=5, digest="m1"):
    states = {"w0": "complete", "w1": "complete"} if complete else {"w0": "complete"}
    return TransferReceipt(
        transfer_id="t1", kind=TransferKind.BOOTSTRAP, target_version=version,
        manifest_digest=digest, expected_receiver_ids=("w0", "w1"), receiver_states=states,
    )


class FakeCheckpoint(CheckpointProtocol):
    def __init__(self, transfer_receipt=None):
        self.transfer_receipt = transfer_receipt
        self.calls = []

    async def bootstrap_target(self, ctx, prepared, published):
        self.calls.append(("bootstrap", prepared.replica_id, published.snapshot_id))
        return self.transfer_receipt or transfer(version=published.version, digest=published.manifest_digest)

    async def add_effective_replica(self, ctx, prepared):
        self.calls.append(("add_effective", prepared.replica_id))
        return 5

    async def remove_effective_replica(self, ctx, replica_id):
        self.calls.append(("remove_effective", replica_id))
        return 6

    async def run_native_sync(self, ctx, new_version, digest):
        self.calls.append(("native_sync", new_version))
        return snapshot(new_version, digest)


class FakeRouting(RoutingProtocol):
    def __init__(self, drained=True, finish_remove=True):
        self.drained = drained
        self.finish_remove_result = finish_remove
        self.calls = []

    async def begin_drain(self, ctx, replica_id):
        from multi_task_scheduler.orchestration.receipts import DrainReceipt
        self.calls.append(("begin_drain", replica_id))
        return DrainReceipt(replica_id=replica_id, routing_epoch=3, operation_id=ctx.operation_id)

    async def wait_drained(self, ctx, replica_id, routing_epoch):
        self.calls.append(("wait_drained", replica_id, routing_epoch))
        return self.drained

    async def commit_routable(self, ctx, head_server, receipt):
        self.calls.append(("commit_routable", head_server, receipt.target_version))
        return 4

    async def finish_remove(self, ctx, replica_id):
        self.calls.append(("finish_remove", replica_id))
        return self.finish_remove_result

    async def query_routing_operation(self, ctx, operation_id):
        return "unknown"


class FakeReplica(ReplicaProtocol):
    def __init__(self):
        self.calls = []

    async def materialize_hidden(self, ctx, placement, model_config):
        self.calls.append(("materialize",))
        return prepared()

    def _release(self, ctx, replica_id, kind):
        return ReleaseReceipt(
            replica_id=replica_id, runtime_epoch=1, lease_epoch=ctx.lease_epoch,
            lb_excluded=True, ce_excluded=True, no_inflight_transfer=True,
            process_cleared_or_slept=True, per_gpu_hbm_free={"u0": 10}, release_kind=kind,
        )

    async def sleep_runtime(self, ctx, replica_id):
        self.calls.append(("sleep", replica_id))
        return self._release(ctx, replica_id, ReleaseKind.DONOR_SLEEP)

    async def destroy_runtime(self, ctx, replica_id, purpose):
        self.calls.append(("destroy", replica_id, purpose))
        return self._release(ctx, replica_id, ReleaseKind.BORROWER_RELEASE)

    async def inspect_runtime(self, ctx, replica_id):
        return {"replica_id": replica_id}


class FakeServing(ServingProtocol):
    def __init__(self):
        self.calls = []

    async def wake_weights(self, ctx, replica_id):
        self.calls.append(("wake_weights", replica_id))
        return WeightReadiness(replica_id=replica_id, operation_id=ctx.operation_id)

    async def wake_kv_and_validate(self, ctx, receipt):
        self.calls.append(("wake_kv", receipt.target_version))
        return ServingReadiness(replica_id="r1", operation_id=ctx.operation_id)

    async def abort_target(self, ctx, replica_id):
        self.calls.append(("abort", replica_id))
        return AbortReceipt(replica_id=replica_id, operation_id=ctx.operation_id,
                            request_ids=(), abort_confirmed=True)


class FakeSnapshots(PublishedSnapshotStore):
    def __init__(self, current=None):
        self.current = current
        self.published = []

    def current_snapshot(self):
        return self.current

    def publish(self, published):
        self.current = published
        self.published.append(published)


def build(**kwargs):
    gate = kwargs.pop("gate", ReplicaSyncGate())
    checkpoint = kwargs.pop("checkpoint", FakeCheckpoint())
    routing = kwargs.pop("routing", FakeRouting())
    replica = kwargs.pop("replica", FakeReplica())
    serving = kwargs.pop("serving", FakeServing())
    snapshots = kwargs.pop("snapshots", FakeSnapshots(snapshot()))
    return ScaleTransaction(gate, checkpoint, routing, replica, serving, snapshots, **kwargs)


def test_add_uses_exact_published_snapshot_then_joins_e_then_routes():
    async def scenario():
        checkpoint = FakeCheckpoint()
        routing = FakeRouting()
        tx = build(checkpoint=checkpoint, routing=routing)
        receipt = await tx.add_and_publish(ctx(), prepared())
        assert receipt.serving_version == 5
        assert checkpoint.calls == [("bootstrap", "r1", "snap-5"), ("add_effective", "r1")]
        assert routing.calls[-1][0] == "commit_routable"
        assert tx._gate.owner is None
    asyncio.run(scenario())


def test_add_refuses_when_no_real_published_snapshot_exists():
    async def scenario():
        tx = build(snapshots=FakeSnapshots(None))
        with pytest.raises(MissingPublishedSnapshotError):
            await tx.add_and_publish(ctx(), prepared())
        assert tx._gate.owner is None
    asyncio.run(scenario())


def test_add_rejects_mismatched_transfer_evidence_before_joining_e():
    async def scenario():
        checkpoint = FakeCheckpoint(transfer_receipt=transfer(version=4, digest="old"))
        tx = build(checkpoint=checkpoint)
        with pytest.raises(TransferIncompleteError, match="pinned snapshot"):
            await tx.add_and_publish(ctx(), prepared())
        assert not any(call[0] == "add_effective" for call in checkpoint.calls)
    asyncio.run(scenario())


def test_prepare_exit_natural_runs_outside_gate_and_returns_evidence():
    async def scenario():
        gate = ReplicaSyncGate()
        routing = FakeRouting()
        tx = build(gate=gate, routing=routing)
        proof = await tx.prepare_exit(ctx(), "r1")
        assert proof.safe_to_leave_service
        assert gate.owner is None
        assert [call[0] for call in routing.calls] == ["begin_drain", "wait_drained"]
    asyncio.run(scenario())


def test_force_verified_fails_before_any_routing_side_effect_until_supported():
    async def scenario():
        routing = FakeRouting()
        tx = build(routing=routing)
        with pytest.raises(NotImplementedError, match="continuation"):
            await tx.prepare_exit(ctx(), "r1", RecallMode.FORCE_VERIFIED)
        assert routing.calls == []
    asyncio.run(scenario())


def test_remove_commit_revalidates_then_excludes_e_then_r():
    async def scenario():
        checkpoint = FakeCheckpoint()
        routing = FakeRouting()
        tx = build(checkpoint=checkpoint, routing=routing)
        proof = await tx.prepare_exit(ctx(), "r1")
        service = await tx.remove_and_commit(ctx(), proof)
        assert service.service_detached
        assert checkpoint.calls[-1] == ("remove_effective", "r1")
        assert routing.calls[-1] == ("finish_remove", "r1")
        assert tx._gate.owner is None
    asyncio.run(scenario())


def test_finalize_release_requires_service_removal_evidence():
    async def scenario():
        tx = build()
        from multi_task_scheduler.orchestration.receipts import RemovedReceipt
        unsafe = RemovedReceipt(
            replica_id="r1", operation_id="op-1", ce_revision=1,
            lb_excluded=False, capacity_released=True,
        )
        with pytest.raises(ServiceNotDetachedError):
            await tx.finalize_release(ctx(), unsafe, native=False)
    asyncio.run(scenario())


def test_remove_wrapper_destroys_only_after_service_commit():
    async def scenario():
        replica = FakeReplica()
        tx = build(replica=replica)
        service, release = await tx.remove(ctx(), "r1")
        assert service.service_detached
        assert release.complete
        assert replica.calls[-1][0] == "destroy"
    asyncio.run(scenario())


def test_donate_wrapper_sleeps_native_after_service_commit():
    async def scenario():
        replica = FakeReplica()
        tx = build(replica=replica)
        service, release = await tx.donate(ctx(), "r1")
        assert service.service_detached
        assert release.release_kind is ReleaseKind.DONOR_SLEEP
        assert replica.calls[-1][0] == "sleep"
    asyncio.run(scenario())


def test_restore_rejects_unsatisfied_fence_before_waking_runtime():
    async def scenario():
        serving = FakeServing()
        tx = build(serving=serving)
        with pytest.raises(FenceRejectedError):
            await tx.restore_and_publish(ctx(), prepared(), fence_satisfied=False)
        assert serving.calls == []
    asyncio.run(scenario())


def test_native_sync_publishes_exact_returned_snapshot():
    async def scenario():
        snapshots = FakeSnapshots(snapshot(5, "m5"))
        tx = build(snapshots=snapshots)
        published = await tx.native_weight_sync(ctx("op-sync"), 6, "m6")
        assert published.version == 6
        assert snapshots.current_snapshot() is published
        assert snapshots.published == [published]
    asyncio.run(scenario())


def test_native_sync_requires_nonempty_manifest_digest():
    async def scenario():
        tx = build()
        with pytest.raises(ValueError, match="manifest"):
            await tx.native_weight_sync(ctx("op-sync"), 6, "")
    asyncio.run(scenario())
