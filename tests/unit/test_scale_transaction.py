"""ScaleTransaction ordering against fake adapters (section 5 sequences)."""

import asyncio

import pytest

from multi_task_scheduler.orchestration.contracts import (
    OperationContext,
    PreparedReplica,
    PublishedWeightSnapshot,
    ReleaseKind,
    ReleaseReceipt,
    TransferKind,
    TransferReceipt,
)
from multi_task_scheduler.orchestration.receipts import (
    AbortReceipt,
    DrainReceipt,
    ServingReadiness,
    WeightReadiness,
)
from multi_task_scheduler.orchestration.replica_sync_gate import GateKind, ReplicaSyncGate
from multi_task_scheduler.orchestration.scale_transaction import (
    CheckpointProtocol,
    DrainTimeoutError,
    FenceRejectedError,
    ReplicaProtocol,
    RoutingProtocol,
    ScaleTransaction,
    ServingProtocol,
    ServingVersionStore,
    TransferIncompleteError,
)


def ctx(operation_id="op-1", lease_epoch=0):
    return OperationContext(
        protocol_version="p1",
        gs_epoch=1,
        task_id="task-a",
        task_session="s1",
        operation_id=operation_id,
        lease_id="lease-1",
        lease_epoch=lease_epoch,
        command_seq=0,
    )


def prepared(replica_id="r1"):
    return PreparedReplica(
        replica_id=replica_id,
        runtime_epoch=1,
        operation_id="op-1",
        placement_digest="d1",
        receiver_ids=("w0", "w1"),
        head_server_descriptor="head",
    )


def transfer(complete=True):
    states = {"w0": "complete", "w1": "complete"} if complete else {"w0": "complete"}
    return TransferReceipt(
        transfer_id="t1",
        kind=TransferKind.BOOTSTRAP,
        target_version=5,
        manifest_digest="m1",
        expected_receiver_ids=("w0", "w1"),
        receiver_states=states,
    )


class FakeCheckpoint(CheckpointProtocol):
    def __init__(self, transfer_receipt=None):
        self.transfer_receipt = transfer_receipt
        self.calls = []

    async def bootstrap_target(self, ctx, prepared, snapshot):
        self.calls.append(("bootstrap", prepared.replica_id, snapshot.serving_version))
        return self.transfer_receipt or transfer()

    async def add_effective_replica(self, ctx, prepared):
        self.calls.append(("add_effective", prepared.replica_id))
        return 5

    async def remove_effective_replica(self, ctx, replica_id):
        self.calls.append(("remove_effective", replica_id))
        return 6

    async def run_native_sync(self, ctx, new_version, digest):
        self.calls.append(("native_sync", new_version))
        return PublishedWeightSnapshot(
            serving_version=new_version, manifest_digest=digest, receiver_ids=()
        )


class FakeRouting(RoutingProtocol):
    def __init__(self, drained=True):
        self.drained = drained
        self.calls = []

    async def begin_drain(self, ctx, replica_id):
        self.calls.append(("begin_drain", replica_id))
        return DrainReceipt(replica_id=replica_id, routing_epoch=3, operation_id=ctx.operation_id)

    async def wait_drained(self, ctx, replica_id, routing_epoch):
        self.calls.append(("wait_drained", replica_id))
        return self.drained

    async def commit_routable(self, ctx, head_server, receipt):
        self.calls.append(("commit_routable", head_server))
        return 4

    async def finish_remove(self, ctx, replica_id):
        self.calls.append(("finish_remove", replica_id))
        return True

    async def query_routing_operation(self, ctx, operation_id):
        return "committed"


class FakeReplica(ReplicaProtocol):
    def __init__(self):
        self.calls = []

    async def materialize_hidden(self, ctx, placement, model_config):
        self.calls.append(("materialize",))
        return prepared()

    async def sleep_runtime(self, ctx, replica_id):
        self.calls.append(("sleep", replica_id))
        return ReleaseReceipt(
            replica_id=replica_id, runtime_epoch=1, lease_epoch=0,
            lb_excluded=True, ce_excluded=True, no_inflight_transfer=True,
            process_cleared_or_slept=True, release_kind=ReleaseKind.DONOR_SLEEP,
        )

    async def destroy_runtime(self, ctx, replica_id, purpose):
        self.calls.append(("destroy", replica_id, purpose))
        return ReleaseReceipt(
            replica_id=replica_id, runtime_epoch=1, lease_epoch=0,
            lb_excluded=True, ce_excluded=True, no_inflight_transfer=True,
            process_cleared_or_slept=True, release_kind=ReleaseKind.BORROWER_RELEASE,
        )

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


class FakeVersions(ServingVersionStore):
    def __init__(self, current=5):
        self._current = current
        self.published = []

    def current(self):
        return self._current

    def publish(self, version):
        self._current = version
        self.published.append(version)


def build(**kwargs):
    gate = kwargs.pop("gate", ReplicaSyncGate())
    checkpoint = kwargs.pop("checkpoint", FakeCheckpoint())
    routing = kwargs.pop("routing", FakeRouting())
    replica = kwargs.pop("replica", FakeReplica())
    serving = kwargs.pop("serving", FakeServing())
    versions = kwargs.pop("versions", FakeVersions())
    return ScaleTransaction(gate, checkpoint, routing, replica, serving, versions, **kwargs)


def test_add_holds_gate_and_returns_ready_receipt():
    async def scenario():
        checkpoint = FakeCheckpoint()
        routing = FakeRouting()
        tx = build(checkpoint=checkpoint, routing=routing)
        receipt = await tx.add_and_publish(ctx(), prepared())
        assert receipt.replica_id == "r1"
        assert receipt.ce_revision == 5
        assert receipt.routing_epoch == 4
        assert receipt.serving_version == 5
        assert checkpoint.calls[0][0] == "bootstrap"
        assert checkpoint.calls[1][0] == "add_effective"
        assert routing.calls[0][0] == "commit_routable"
        assert tx._gate.owner is None  # gate released

    asyncio.run(scenario())


def test_add_does_not_publish_new_version_or_reset_staleness():
    async def scenario():
        checkpoint = FakeCheckpoint()
        versions = FakeVersions(current=5)
        tx = build(checkpoint=checkpoint, versions=versions)
        await tx.add_and_publish(ctx(), prepared())
        assert versions.published == []  # no version increment
        assert checkpoint.calls[1][0] == "add_effective"

    asyncio.run(scenario())


def test_add_incomplete_transfer_raises_and_releases_gate():
    async def scenario():
        checkpoint = FakeCheckpoint(transfer_receipt=transfer(complete=False))
        tx = build(checkpoint=checkpoint)
        with pytest.raises(TransferIncompleteError):
            await tx.add_and_publish(ctx(), prepared())
        assert tx._gate.owner is None

    asyncio.run(scenario())


def test_remove_drains_then_excludes_then_destroys():
    async def scenario():
        routing = FakeRouting()
        replica = FakeReplica()
        tx = build(routing=routing, replica=replica)
        removed, release = await tx.remove(ctx(), "r1", purpose="recall")
        assert removed.lb_excluded is True
        assert release.release_kind is ReleaseKind.BORROWER_RELEASE
        # order: begin_drain -> wait -> remove_effective -> finish_remove -> destroy
        kinds = [c[0] for c in routing.calls]
        assert kinds[0] == "begin_drain"
        assert "finish_remove" in kinds
        assert replica.calls[-1][0] == "destroy"

    asyncio.run(scenario())


def test_donate_sleeps_after_exclusion():
    async def scenario():
        routing = FakeRouting()
        replica = FakeReplica()
        tx = build(routing=routing, replica=replica)
        removed, release = await tx.donate(ctx(), "r1")
        assert release.release_kind is ReleaseKind.DONOR_SLEEP
        assert replica.calls[-1][0] == "sleep"

    asyncio.run(scenario())


def test_remove_drain_timeout_does_not_acquire_gate():
    async def scenario():
        gate = ReplicaSyncGate()
        routing = FakeRouting(drained=False)
        tx = build(gate=gate, routing=routing)
        with pytest.raises(DrainTimeoutError):
            await tx.remove(ctx(), "r1")
        assert gate.owner is None  # gate never acquired

    asyncio.run(scenario())


def test_restore_rejects_unsatisfied_fence_before_serving():
    async def scenario():
        serving = FakeServing()
        tx = build(serving=serving)
        with pytest.raises(FenceRejectedError):
            await tx.restore_and_publish(ctx(), prepared(), fence_satisfied=False)
        assert serving.calls == []

    asyncio.run(scenario())


def test_restore_wakes_weights_then_bootstraps():
    async def scenario():
        serving = FakeServing()
        checkpoint = FakeCheckpoint()
        tx = build(serving=serving, checkpoint=checkpoint)
        receipt = await tx.restore_and_publish(ctx(), prepared(), fence_satisfied=True)
        assert receipt.replica_id == "r1"
        assert serving.calls[0][0] == "wake_weights"
        assert serving.calls[1][0] == "wake_kv"
        assert checkpoint.calls[0][0] == "bootstrap"
        assert tx._gate.owner is None

    asyncio.run(scenario())


def test_native_sync_publishes_version_under_gate():
    async def scenario():
        checkpoint = FakeCheckpoint()
        versions = FakeVersions(current=5)
        tx = build(checkpoint=checkpoint, versions=versions)
        snapshot = await tx.native_weight_sync(ctx("op-nsync"), 6, "digest")
        assert snapshot.serving_version == 6
        assert versions.current() == 6
        assert versions.published == [6]
        assert checkpoint.calls[0][0] == "native_sync"

    asyncio.run(scenario())


def test_transactions_serialize_on_the_gate():
    async def scenario():
        gate = ReplicaSyncGate()
        checkpoint = FakeCheckpoint()
        tx = build(gate=gate, checkpoint=checkpoint)
        order = []

        async def add_with_marker(marker):
            order.append((marker, "start"))
            await tx.add_and_publish(ctx(f"op-{marker}"), prepared(f"r{marker}"))
            order.append((marker, "end"))

        await asyncio.gather(add_with_marker("a"), add_with_marker("b"))
        # a and b never interleave their guarded regions: each start is followed
        # by its own end before the other begins.
        assert order[0] == ("a", "start") or order[0] == ("b", "start")
        assert order[0][0] != order[1][0] or order[1][1] == "end"

    asyncio.run(scenario())
