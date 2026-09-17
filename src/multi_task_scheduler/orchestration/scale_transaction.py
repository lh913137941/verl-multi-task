"""Trainer-side transaction ordering for the simplified orchestration contract.

Only the public design flow is represented here: ADD publishes through
WeightEvidence/CommitReceipt/ServiceEvidence, REMOVE revalidates ExitEvidence
under G before service removal, and unverified RESTORE remains an explicit
failure rather than falling back to a legacy boolean fence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .contracts import (
    OperationContext,
    PreparedReplica,
    PublishedWeightSnapshot,
    ReplicaKey,
    RuntimeReady,
    ServiceAction,
)
from .receipts import CommitOwner, CommitReceipt, ExitEvidence, ServiceEvidence, WeightEvidence
from .replica_sync_gate import GateKind, GateLease, ReplicaSyncGate


class MissingPublishedSnapshotError(RuntimeError):
    """ADD/RESTORE cannot proceed without immutable current Vpub contents."""


class EvidenceMismatchError(RuntimeError):
    """A phase result does not match the operation/runtime/prerequisite identity."""


class CheckpointProtocol(ABC):
    @abstractmethod
    async def bootstrap_target(
        self,
        ctx: OperationContext,
        prepared: PreparedReplica,
        snapshot: PublishedWeightSnapshot,
    ) -> WeightEvidence:
        """Load the exact pinned snapshot into one hidden target."""

    @abstractmethod
    async def add_effective(
        self,
        ctx: OperationContext,
        prepared: PreparedReplica,
        weight: WeightEvidence,
    ) -> CommitReceipt:
        """Commit E membership and return owner=CE/action=ADD."""

    @abstractmethod
    async def remove_effective(
        self,
        ctx: OperationContext,
        key: ReplicaKey,
        exit: ExitEvidence,
    ) -> CommitReceipt:
        """Remove one member from E and return owner=CE/action=REMOVE."""


class RollouterProtocol(ABC):
    @abstractmethod
    async def revalidate_exit(
        self, ctx: OperationContext, proof: ExitEvidence
    ) -> ExitEvidence:
        """Revalidate drain identity/fences while the caller owns G."""

    @abstractmethod
    async def commit_service(
        self,
        ctx: OperationContext,
        prepared: PreparedReplica,
        weight: WeightEvidence,
        ce_commit: CommitReceipt,
    ) -> ServiceEvidence:
        """Commit R/C/M after CE ADD."""

    @abstractmethod
    async def commit_removal(
        self,
        ctx: OperationContext,
        proof: ExitEvidence,
        ce_commit: CommitReceipt,
    ) -> ServiceEvidence:
        """Commit R/C/M after CE REMOVE."""


class RuntimeProtocol(ABC):
    @abstractmethod
    async def wake_kv_and_validate(
        self,
        ctx: OperationContext,
        key: ReplicaKey,
        weight: WeightEvidence,
    ) -> RuntimeReady:
        """Restore serving resources after the exact snapshot is installed."""


class PublishedSnapshotStore(ABC):
    @abstractmethod
    def current_snapshot(self) -> PublishedWeightSnapshot | None:
        """Return the exact currently published immutable snapshot."""


class ScaleTransaction:
    def __init__(
        self,
        gate: ReplicaSyncGate,
        checkpoint: CheckpointProtocol,
        rollouter: RollouterProtocol,
        runtime: RuntimeProtocol,
        snapshots: PublishedSnapshotStore,
        *,
        gate_timeout: float | None = None,
    ) -> None:
        self._gate = gate
        self._checkpoint = checkpoint
        self._rollouter = rollouter
        self._runtime = runtime
        self._snapshots = snapshots
        self._gate_timeout = gate_timeout

    async def _acquire(self, ctx: OperationContext, kind: GateKind) -> GateLease:
        return await self._gate.acquire(ctx.operation_id, kind, timeout=self._gate_timeout)

    def _pin_snapshot(self) -> PublishedWeightSnapshot:
        snapshot = self._snapshots.current_snapshot()
        if snapshot is None:
            raise MissingPublishedSnapshotError("no immutable published snapshot is available")
        return snapshot

    @staticmethod
    def _require_header(ctx: OperationContext, key: ReplicaKey, evidence) -> None:
        if evidence.header.ctx != ctx or evidence.header.key != key:
            raise EvidenceMismatchError("evidence header does not match operation/runtime identity")

    async def bootstrap_and_publish(
        self, ctx: OperationContext, prepared: PreparedReplica
    ) -> ServiceEvidence:
        """ADD: pin Vpub -> load -> runtime validate -> E ADD -> R/C/M commit."""
        if prepared.ctx != ctx:
            raise EvidenceMismatchError("PreparedReplica belongs to a different operation")
        lease = await self._acquire(ctx, GateKind.ADD)
        try:
            snapshot = self._pin_snapshot()
            if snapshot.model_signature != prepared.model_signature:
                raise EvidenceMismatchError("published snapshot model signature mismatch")
            weight = await self._checkpoint.bootstrap_target(ctx, prepared, snapshot)
            self._require_header(ctx, prepared.key, weight)
            if (
                weight.snapshot_id != snapshot.snapshot_id
                or weight.manifest_digest != snapshot.manifest_digest
                or weight.version != snapshot.version
            ):
                raise EvidenceMismatchError("WeightEvidence does not match pinned snapshot")
            expected_receivers = {receiver.receiver_id for receiver in prepared.receivers}
            if set(weight.receiver_versions) != expected_receivers:
                raise EvidenceMismatchError("WeightEvidence does not cover every prepared receiver")

            ready = await self._runtime.wake_kv_and_validate(ctx, prepared.key, weight)
            if (
                ready.key != prepared.key
                or not ready.receiver_ready
                or not ready.serving_ready
                or ready.loaded_version != weight.version
            ):
                raise EvidenceMismatchError("runtime readiness does not match installed weights")

            ce_commit = await self._checkpoint.add_effective(ctx, prepared, weight)
            self._require_header(ctx, prepared.key, ce_commit)
            if (
                ce_commit.owner is not CommitOwner.CE
                or ce_commit.action is not ServiceAction.ADD
                or ce_commit.version != weight.version
            ):
                raise EvidenceMismatchError("invalid CE ADD commit receipt")

            service = await self._rollouter.commit_service(
                ctx, prepared, weight, ce_commit
            )
            self._require_header(ctx, prepared.key, service)
            if (
                service.action is not ServiceAction.ADD
                or service.version != weight.version
                or service.prerequisite_digest != weight.header.digest
            ):
                raise EvidenceMismatchError("invalid ADD ServiceEvidence")
            return service
        finally:
            await lease.release()

    async def remove_and_commit(
        self, ctx: OperationContext, proof: ExitEvidence
    ) -> ServiceEvidence:
        """REMOVE/DONATE: revalidate exit -> E REMOVE -> R/C/M removal under G."""
        self._require_header(ctx, proof.header.key, proof)
        lease = await self._acquire(ctx, GateKind.REMOVE)
        try:
            fresh = await self._rollouter.revalidate_exit(ctx, proof)
            self._require_header(ctx, proof.header.key, fresh)
            if fresh.drain_id != proof.drain_id:
                raise EvidenceMismatchError("revalidated exit changed drain identity")

            ce_commit = await self._checkpoint.remove_effective(
                ctx, fresh.header.key, fresh
            )
            self._require_header(ctx, fresh.header.key, ce_commit)
            if (
                ce_commit.owner is not CommitOwner.CE
                or ce_commit.action is not ServiceAction.REMOVE
            ):
                raise EvidenceMismatchError("invalid CE REMOVE commit receipt")

            service = await self._rollouter.commit_removal(ctx, fresh, ce_commit)
            self._require_header(ctx, fresh.header.key, service)
            if (
                service.action is not ServiceAction.REMOVE
                or service.version is not None
                or service.prerequisite_digest != fresh.header.digest
            ):
                raise EvidenceMismatchError("invalid REMOVE ServiceEvidence")
            return service
        finally:
            await lease.release()

    async def restore_and_publish(
        self, ctx: OperationContext, key: ReplicaKey
    ) -> ServiceEvidence:
        """RESTORE stays unavailable until the native runtime can expose receivers safely."""
        raise NotImplementedError(
            "RESTORE requires verified native wake + receiver projection before bootstrap"
        )
