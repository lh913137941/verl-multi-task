"""ScaleTransaction ordering for the current simplified evidence flow."""

import asyncio

import pytest

from multi_task_scheduler.orchestration.contracts import (
    OperationContext,
    PreparedReplica,
    PublishedWeightSnapshot,
    ReceiverRef,
    ReplicaKey,
    RuntimeReady,
    ServiceAction,
)
from multi_task_scheduler.orchestration.receipts import (
    CommitOwner,
    CommitReceipt,
    EvidenceHeader,
    ExitEvidence,
    ServiceEvidence,
    WeightEvidence,
)
from multi_task_scheduler.orchestration.replica_sync_gate import ReplicaSyncGate
from multi_task_scheduler.orchestration.scale_transaction import (
    CheckpointProtocol,
    EvidenceMismatchError,
    MissingPublishedSnapshotError,
    PublishedSnapshotStore,
    RollouterProtocol,
    RuntimeProtocol,
    ScaleTransaction,
)
from multi_task_scheduler.orchestration.contracts import RecallMode


def ctx(operation_id="op-1"):
    return OperationContext(
        protocol_version=1,
        gs_epoch="gs-1",
        task_id="task-a",
        task_session="s1",
        operation_id=operation_id,
        lease_id="lease-1",
        lease_epoch=0,
        command_seq=0,
    )


def key():
    return ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=1)


def header(context, digest, revision=1):
    return EvidenceHeader(ctx=context, key=key(), phase_revision=revision, digest=digest)


def prepared(context=None):
    context = context or ctx()
    return PreparedReplica(
        key=key(),
        ctx=context,
        placement_digest="placement-1",
        model_signature="sig",
        receivers=(
            ReceiverRef("w0", "n1", "u0", 0, object()),
            ReceiverRef("w1", "n1", "u1", 1, object()),
        ),
        head_server=object(),
        manager_revision=1,
    )


def snapshot(version=5, digest="m1"):
    return PublishedWeightSnapshot(
        snapshot_id=f"snap-{version}",
        manifest_digest=digest,
        model_signature="sig",
        version=version,
        byte_size=123,
        sender=object(),
    )


def weight(context=None, version=5, digest="m1", evidence_digest="weight-1"):
    context = context or ctx()
    return WeightEvidence(
        header=header(context, evidence_digest),
        transfer_id="t1",
        snapshot_id=f"snap-{version}",
        manifest_digest=digest,
        version=version,
        receiver_versions={"w0": version, "w1": version},
        device_complete=True,
        temporary_topology_clean=True,
    )


def exit_evidence(context=None, drain_id="drain-1", digest="exit-1"):
    context = context or ctx()
    return ExitEvidence(
        header=header(context, digest, revision=2),
        drain_id=drain_id,
        recall_mode=RecallMode.NATURAL,
        inflight=0,
        admitting=0,
        queued=0,
        running=0,
        pending_admissions=0,
        closed_admission=True,
        all_backends_confirmed=True,
        lb_revision=3,
        observed_age_ms=5,
        engine_digest="engine-1",
        continuations=(),
        unresolved_count=0,
    )


class FakeSnapshots(PublishedSnapshotStore):
    def __init__(self, current=None):
        self.current = current

    def current_snapshot(self):
        return self.current


class FakeCheckpoint(CheckpointProtocol):
    def __init__(self, *, weight_result=None):
        self.weight_result = weight_result
        self.calls = []

    async def bootstrap_target(self, context, prepared_replica, published):
        self.calls.append(("bootstrap", published.snapshot_id))
        return self.weight_result or weight(
            context, published.version, published.manifest_digest
        )

    async def add_effective(self, context, prepared_replica, installed):
        self.calls.append(("add_effective", installed.header.digest))
        return CommitReceipt(
            header=header(context, "ce-add", revision=3),
            owner=CommitOwner.CE,
            action=ServiceAction.ADD,
            revision=7,
            version=installed.version,
            route_epoch=None,
        )

    async def remove_effective(self, context, target, proof):
        self.calls.append(("remove_effective", proof.header.digest))
        return CommitReceipt(
            header=header(context, "ce-remove", revision=4),
            owner=CommitOwner.CE,
            action=ServiceAction.REMOVE,
            revision=8,
            version=None,
            route_epoch=None,
        )


class FakeRuntime(RuntimeProtocol):
    def __init__(self):
        self.calls = []

    async def wake_kv_and_validate(self, context, target, installed):
        self.calls.append(("ready", installed.version))
        return RuntimeReady(
            key=target,
            receiver_ready=True,
            serving_ready=True,
            loaded_version=installed.version,
            inventory_revision=2,
        )


class FakeRollouter(RollouterProtocol):
    def __init__(self, *, changed_drain=False):
        self.changed_drain = changed_drain
        self.calls = []

    async def revalidate_exit(self, context, proof):
        self.calls.append(("revalidate", proof.drain_id))
        return exit_evidence(
            context,
            drain_id="other-drain" if self.changed_drain else proof.drain_id,
            digest="exit-fresh",
        )

    async def commit_service(self, context, prepared_replica, installed, ce_commit):
        self.calls.append(("commit_service", ce_commit.header.digest))
        return ServiceEvidence(
            header=header(context, "service-add", revision=4),
            action=ServiceAction.ADD,
            version=installed.version,
            ce_revision=ce_commit.revision,
            lb_revision=9,
            route_epoch=10,
            capacity_revision=11,
            manager_revision=12,
            ce_commit_digest=ce_commit.header.digest,
            lb_commit_digest="lb-add",
            prerequisite_digest=installed.header.digest,
        )

    async def commit_removal(self, context, proof, ce_commit):
        self.calls.append(("commit_removal", ce_commit.header.digest))
        return ServiceEvidence(
            header=header(context, "service-remove", revision=5),
            action=ServiceAction.REMOVE,
            version=None,
            ce_revision=ce_commit.revision,
            lb_revision=13,
            route_epoch=14,
            capacity_revision=15,
            manager_revision=16,
            ce_commit_digest=ce_commit.header.digest,
            lb_commit_digest="lb-remove",
            prerequisite_digest=proof.header.digest,
        )


def build(*, snapshots=None, checkpoint=None, rollouter=None, runtime=None):
    return ScaleTransaction(
        ReplicaSyncGate(),
        checkpoint or FakeCheckpoint(),
        rollouter or FakeRollouter(),
        runtime or FakeRuntime(),
        snapshots or FakeSnapshots(snapshot()),
    )


def test_add_pins_snapshot_validates_runtime_then_commits_ce_and_service():
    async def scenario():
        checkpoint = FakeCheckpoint()
        runtime = FakeRuntime()
        rollouter = FakeRollouter()
        tx = build(checkpoint=checkpoint, runtime=runtime, rollouter=rollouter)
        service = await tx.bootstrap_and_publish(ctx(), prepared())
        assert service.action is ServiceAction.ADD
        assert service.version == 5
        assert checkpoint.calls == [("bootstrap", "snap-5"), ("add_effective", "weight-1")]
        assert runtime.calls == [("ready", 5)]
        assert rollouter.calls == [("commit_service", "ce-add")]
        assert tx._gate.owner is None

    asyncio.run(scenario())


def test_add_refuses_without_immutable_published_snapshot():
    async def scenario():
        tx = build(snapshots=FakeSnapshots(None))
        with pytest.raises(MissingPublishedSnapshotError):
            await tx.bootstrap_and_publish(ctx(), prepared())
        assert tx._gate.owner is None

    asyncio.run(scenario())


def test_add_rejects_mismatched_weight_evidence_before_ce_commit():
    async def scenario():
        checkpoint = FakeCheckpoint(weight_result=weight(version=4, digest="old"))
        tx = build(checkpoint=checkpoint)
        with pytest.raises(EvidenceMismatchError, match="pinned snapshot"):
            await tx.bootstrap_and_publish(ctx(), prepared())
        assert all(call[0] != "add_effective" for call in checkpoint.calls)

    asyncio.run(scenario())


def test_remove_revalidates_exit_then_commits_ce_and_service_under_gate():
    async def scenario():
        checkpoint = FakeCheckpoint()
        rollouter = FakeRollouter()
        tx = build(checkpoint=checkpoint, rollouter=rollouter)
        service = await tx.remove_and_commit(ctx(), exit_evidence())
        assert service.action is ServiceAction.REMOVE
        assert checkpoint.calls == [("remove_effective", "exit-fresh")]
        assert rollouter.calls == [
            ("revalidate", "drain-1"),
            ("commit_removal", "ce-remove"),
        ]
        assert tx._gate.owner is None

    asyncio.run(scenario())


def test_remove_rejects_revalidation_that_changes_drain_identity():
    async def scenario():
        tx = build(rollouter=FakeRollouter(changed_drain=True))
        with pytest.raises(EvidenceMismatchError, match="drain identity"):
            await tx.remove_and_commit(ctx(), exit_evidence())
        assert tx._gate.owner is None

    asyncio.run(scenario())


def test_restore_stays_explicitly_unimplemented_without_verified_native_wake():
    async def scenario():
        tx = build()
        with pytest.raises(NotImplementedError, match="RESTORE"):
            await tx.restore_and_publish(ctx("restore-op"), key())

    asyncio.run(scenario())
