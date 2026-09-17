"""Transaction ordering for the simplified orchestration contract.

Long-running prepare/drain work stays outside G. Membership and service commits
run under the task-local replica sync gate. Physical sleep/destroy happens only
after service removal evidence exists and after G has been released.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .contracts import (
    OperationContext,
    PlacementSpec,
    PreparedReplica,
    PublishedWeightSnapshot,
    RecallMode,
    ReleaseReceipt,
    TransferReceipt,
)
from .receipts import (
    AbortReceipt,
    DrainReceipt,
    ExitEvidence,
    ReadyReceipt,
    RemovedReceipt,
    RestoredReceipt,
    ServingReadiness,
    WeightReadiness,
)
from .replica_sync_gate import GateKind, GateLease, ReplicaSyncGate


class DrainTimeoutError(RuntimeError):
    """The target did not reach a safe exit point before its budget expired."""


class TransferIncompleteError(RuntimeError):
    """A bootstrap transfer finished before every receiver was complete."""


class FenceRejectedError(RuntimeError):
    """A RESTORE fence was not satisfied (borrower not released)."""


class MissingPublishedSnapshotError(RuntimeError):
    """ADD/RESTORE cannot proceed without immutable current Vpub contents."""


class ServiceNotDetachedError(RuntimeError):
    """Physical release was requested before service removal was committed."""


class CheckpointProtocol(ABC):
    @abstractmethod
    async def bootstrap_target(
        self, ctx: OperationContext, prepared: PreparedReplica, snapshot: PublishedWeightSnapshot
    ) -> TransferReceipt:
        """Load the exact pinned snapshot into one hidden target."""

    @abstractmethod
    async def add_effective_replica(self, ctx: OperationContext, prepared: PreparedReplica) -> int:
        """Commit E membership and return its monotonically increasing revision."""

    @abstractmethod
    async def remove_effective_replica(self, ctx: OperationContext, replica_id: str) -> int:
        """Remove one member from E and return its monotonic revision."""

    @abstractmethod
    async def run_native_sync(
        self, ctx: OperationContext, new_serving_version: int, manifest_digest: str
    ) -> PublishedWeightSnapshot:
        """Transfer native weights to E and return the newly published snapshot."""


class RoutingProtocol(ABC):
    @abstractmethod
    async def begin_drain(self, ctx: OperationContext, replica_id: str) -> DrainReceipt:
        """Close new routing and return the routing fence used for drain checks."""

    @abstractmethod
    async def wait_drained(self, ctx: OperationContext, replica_id: str, routing_epoch: int) -> bool:
        """Observe all old attempts as settled for the target."""

    @abstractmethod
    async def commit_routable(
        self, ctx: OperationContext, head_server: Any, receipt: TransferReceipt
    ) -> int:
        """Publish R only after weight and CE evidence have been checked."""

    @abstractmethod
    async def finish_remove(self, ctx: OperationContext, replica_id: str) -> bool:
        """Commit R=REMOVED only after attempts are settled and E is removed."""

    @abstractmethod
    async def query_routing_operation(self, ctx: OperationContext, operation_id: str) -> str:
        """Return the owner-side routing phase for reconciliation."""


class ReplicaProtocol(ABC):
    @abstractmethod
    async def materialize_hidden(
        self, ctx: OperationContext, placement: PlacementSpec, model_config: object
    ) -> PreparedReplica:
        """Create a hidden runtime; success never implies routability."""

    @abstractmethod
    async def sleep_runtime(self, ctx: OperationContext, replica_id: str) -> ReleaseReceipt:
        """Sleep a native runtime after complete service detachment."""

    @abstractmethod
    async def destroy_runtime(
        self, ctx: OperationContext, replica_id: str, purpose: str
    ) -> ReleaseReceipt:
        """Destroy a borrowed runtime after complete service detachment."""

    @abstractmethod
    async def inspect_runtime(self, ctx: OperationContext, replica_id: str) -> object:
        """Read-only runtime facts for reconciliation."""


class ServingProtocol(ABC):
    @abstractmethod
    async def wake_weights(self, ctx: OperationContext, replica_id: str) -> WeightReadiness:
        """Wake only the weight-receive path; generation remains closed."""

    @abstractmethod
    async def wake_kv_and_validate(
        self, ctx: OperationContext, receipt: TransferReceipt
    ) -> ServingReadiness:
        """Restore serving state after the exact pinned snapshot was loaded."""

    @abstractmethod
    async def abort_target(self, ctx: OperationContext, replica_id: str) -> AbortReceipt:
        """Force path primitive. A verified continuation protocol is still required."""


class PublishedSnapshotStore(ABC):
    """Trainer-side holder for the current immutable Vpub snapshot."""

    @abstractmethod
    def current_snapshot(self) -> PublishedWeightSnapshot | None:
        """Return the exact currently published snapshot, or None if unavailable."""

    @abstractmethod
    def publish(self, snapshot: PublishedWeightSnapshot) -> None:
        """Publish the exact snapshot after verified native synchronization."""


class ScaleTransaction:
    def __init__(
        self,
        gate: ReplicaSyncGate,
        checkpoint: CheckpointProtocol,
        routing: RoutingProtocol,
        replica: ReplicaProtocol,
        serving: ServingProtocol,
        snapshots: PublishedSnapshotStore | None = None,
        *,
        gate_timeout: float | None = None,
    ) -> None:
        self._gate = gate
        self._checkpoint = checkpoint
        self._routing = routing
        self._replica = replica
        self._serving = serving
        self._snapshots = snapshots
        self._gate_timeout = gate_timeout

    async def _acquire(self, ctx: OperationContext, kind: GateKind) -> GateLease:
        return await self._gate.acquire(ctx.operation_id, kind, timeout=self._gate_timeout)

    def _pin_snapshot(self) -> PublishedWeightSnapshot:
        snapshot = self._snapshots.current_snapshot() if self._snapshots is not None else None
        if snapshot is None:
            raise MissingPublishedSnapshotError("no immutable published snapshot is available")
        return snapshot

    # -- ADD -------------------------------------------------------------- #

    async def add_and_publish(self, ctx: OperationContext, prepared: PreparedReplica) -> ReadyReceipt:
        """Install current Vpub, join E, then open R while G is held."""
        lease = await self._acquire(ctx, GateKind.ADD)
        try:
            snapshot = self._pin_snapshot()
            transfer = await self._checkpoint.bootstrap_target(ctx, prepared, snapshot)
            if not transfer.all_receivers_complete:
                raise TransferIncompleteError(
                    f"ADD bootstrap incomplete for {prepared.replica_id}: {transfer.receiver_states}"
                )
            if transfer.target_version != snapshot.version or transfer.manifest_digest != snapshot.manifest_digest:
                raise TransferIncompleteError("ADD bootstrap receipt does not match the pinned snapshot")
            ce_revision = await self._checkpoint.add_effective_replica(ctx, prepared)
            routing_epoch = await self._routing.commit_routable(
                ctx, prepared.head_server_descriptor, transfer
            )
            return ReadyReceipt(
                replica_id=prepared.replica_id,
                operation_id=ctx.operation_id,
                routing_epoch=routing_epoch,
                ce_revision=ce_revision,
                serving_version=snapshot.version,
            )
        finally:
            await lease.release()

    # -- unified exit ----------------------------------------------------- #

    async def prepare_exit(
        self,
        ctx: OperationContext,
        replica_id: str,
        recall_mode: RecallMode = RecallMode.NATURAL,
    ) -> ExitEvidence:
        """Close admission and establish exit evidence without holding G."""
        recall_mode = RecallMode(recall_mode)
        if recall_mode is RecallMode.FORCE_VERIFIED:
            # A target abort alone is insufficient: the design also requires
            # verified continuation on a surviving target. Until that native
            # protocol exists, reject before changing routing state.
            raise NotImplementedError("FORCE_VERIFIED requires verified continuation support")

        drain = await self._routing.begin_drain(ctx, replica_id)
        drained = await self._routing.wait_drained(ctx, replica_id, drain.routing_epoch)
        if not drained:
            raise DrainTimeoutError(f"natural drain timeout: {replica_id}")
        return ExitEvidence(
            replica_id=replica_id,
            routing_epoch=drain.routing_epoch,
            operation_id=ctx.operation_id,
            recall_mode=RecallMode.NATURAL,
            attempts_drained=True,
            continuation_confirmed=True,
        )

    async def remove_and_commit(self, ctx: OperationContext, proof: ExitEvidence) -> RemovedReceipt:
        """Revalidate exit, remove E, then commit R/C/M while G is held."""
        if proof.operation_id != ctx.operation_id or not proof.safe_to_leave_service:
            raise DrainTimeoutError(f"stale or unsafe exit evidence for {proof.replica_id}")
        lease = await self._acquire(ctx, GateKind.REMOVE)
        try:
            if not await self._routing.wait_drained(ctx, proof.replica_id, proof.routing_epoch):
                raise DrainTimeoutError(f"exit evidence became stale: {proof.replica_id}")
            ce_revision = await self._checkpoint.remove_effective_replica(ctx, proof.replica_id)
            lb_excluded = await self._routing.finish_remove(ctx, proof.replica_id)
            if not lb_excluded:
                raise DrainTimeoutError(f"routing owner refused remove: {proof.replica_id}")
            return RemovedReceipt(
                replica_id=proof.replica_id,
                operation_id=ctx.operation_id,
                ce_revision=ce_revision,
                lb_excluded=True,
                capacity_released=True,
            )
        finally:
            await lease.release()

    async def finalize_release(
        self,
        ctx: OperationContext,
        service: RemovedReceipt,
        *,
        native: bool,
        purpose: str = "recall",
    ) -> ReleaseReceipt:
        """Physically release only after service detachment evidence exists."""
        if service.operation_id != ctx.operation_id or not service.service_detached:
            raise ServiceNotDetachedError("physical release requires matching service-removal evidence")
        if native:
            return await self._replica.sleep_runtime(ctx, service.replica_id)
        return await self._replica.destroy_runtime(ctx, service.replica_id, purpose)

    # Thin compatibility wrappers for callers that still use the older API.
    async def donate(self, ctx: OperationContext, replica_id: str) -> tuple[RemovedReceipt, ReleaseReceipt]:
        proof = await self.prepare_exit(ctx, replica_id, RecallMode.NATURAL)
        service = await self.remove_and_commit(ctx, proof)
        release = await self.finalize_release(ctx, service, native=True, purpose="donate")
        return service, release

    async def remove(
        self, ctx: OperationContext, replica_id: str, purpose: str = "recall"
    ) -> tuple[RemovedReceipt, ReleaseReceipt]:
        proof = await self.prepare_exit(ctx, replica_id, RecallMode.NATURAL)
        service = await self.remove_and_commit(ctx, proof)
        release = await self.finalize_release(ctx, service, native=False, purpose=purpose)
        return service, release

    # -- RESTORE ---------------------------------------------------------- #

    async def restore_and_publish(
        self,
        ctx: OperationContext,
        prepared: PreparedReplica,
        fence_satisfied: bool,
    ) -> RestoredReceipt:
        if not fence_satisfied:
            raise FenceRejectedError(
                f"RESTORE fence not satisfied for {prepared.replica_id} (borrower not released)"
            )
        await self._serving.wake_weights(ctx, prepared.replica_id)

        lease = await self._acquire(ctx, GateKind.RESTORE)
        try:
            snapshot = self._pin_snapshot()
            transfer = await self._checkpoint.bootstrap_target(ctx, prepared, snapshot)
            if not transfer.all_receivers_complete:
                raise TransferIncompleteError(f"RESTORE bootstrap incomplete for {prepared.replica_id}")
            if transfer.target_version != snapshot.version or transfer.manifest_digest != snapshot.manifest_digest:
                raise TransferIncompleteError("RESTORE bootstrap receipt does not match the pinned snapshot")
            await self._serving.wake_kv_and_validate(ctx, transfer)
            ce_revision = await self._checkpoint.add_effective_replica(ctx, prepared)
            routing_epoch = await self._routing.commit_routable(
                ctx, prepared.head_server_descriptor, transfer
            )
            return RestoredReceipt(
                replica_id=prepared.replica_id,
                operation_id=ctx.operation_id,
                routing_epoch=routing_epoch,
                ce_revision=ce_revision,
                serving_version=snapshot.version,
            )
        finally:
            await lease.release()

    # -- native parameter sync ------------------------------------------- #

    async def native_weight_sync(
        self, ctx: OperationContext, new_serving_version: int, manifest_digest: str
    ) -> PublishedWeightSnapshot:
        """Publish the exact snapshot returned by verified native synchronization."""
        if not manifest_digest:
            raise ValueError("native sync requires a nonempty manifest digest")
        lease = await self._acquire(ctx, GateKind.NATIVE_SYNC)
        try:
            snapshot = await self._checkpoint.run_native_sync(
                ctx, new_serving_version, manifest_digest
            )
            if snapshot.version != new_serving_version or snapshot.manifest_digest != manifest_digest:
                raise TransferIncompleteError("native sync returned a mismatched published snapshot")
            if self._snapshots is not None:
                self._snapshots.publish(snapshot)
            return snapshot
        finally:
            await lease.release()
