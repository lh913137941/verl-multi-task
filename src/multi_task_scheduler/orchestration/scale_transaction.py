"""Transaction ordering for the DONATE -> ADD -> REMOVE -> RESTORE loop.

``ScaleTransaction`` encodes the section 5 sequence diagrams and the section 6
gate rules: drain/observe outside the gate, then acquire the replica-sync gate
and perform the CE-membership and routing commits, then release the gate and
only afterwards run the physical sleep/destroy. It talks to four protocol
adapters, so the core stays free of Ray/verl/torch/vLLM and can be driven by
in-memory fakes in unit tests.

The four adapters are:
- ``CheckpointProtocol``  -> Checkpoint Engine Manager (effective replica set E)
- ``RoutingProtocol``     -> request load balancer (routable set, attempts)
- ``ReplicaProtocol``     -> LLMServerManager (materialize / sleep / destroy)
- ``ServingProtocol``     -> individual Server (wake weights / KV / abort)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .contracts import (
    OperationContext,
    PlacementSpec,
    PreparedReplica,
    PublishedWeightSnapshot,
    ReleaseReceipt,
    TransferReceipt,
)
from .receipts import (
    AbortReceipt,
    DrainReceipt,
    ReadyReceipt,
    RemovedReceipt,
    RestoredReceipt,
    ServingReadiness,
    WeightReadiness,
)
from .replica_sync_gate import GateFencedError, GateKind, GateLease, ReplicaSyncGate


class DrainTimeoutError(RuntimeError):
    """The target did not drain (I/A/Q/R zero) before its budget expired."""


class TransferIncompleteError(RuntimeError):
    """A bootstrap transfer finished before every receiver was complete."""


class FenceRejectedError(RuntimeError):
    """A RESTORE fence was not satisfied (borrower not released)."""


class CheckpointProtocol(ABC):
    """Checkpoint Engine membership adapter (section 4.1 CE Manager owner)."""

    @abstractmethod
    async def bootstrap_target(
        self, ctx: OperationContext, prepared: PreparedReplica, snapshot: PublishedWeightSnapshot
    ) -> TransferReceipt:
        """Build a verified sender -> target receiver group and load weights."""

    @abstractmethod
    async def add_effective_replica(
        self, ctx: OperationContext, prepared: PreparedReplica
    ) -> int:
        """Return the new effective-membership revision. Requires G."""

    @abstractmethod
    async def remove_effective_replica(
        self, ctx: OperationContext, replica_id: str
    ) -> int:
        """Return the new effective-membership revision. Requires G."""

    @abstractmethod
    async def run_native_sync(
        self, ctx: OperationContext, new_serving_version: int, manifest_digest: str
    ) -> PublishedWeightSnapshot:
        """Transfer weights to E, load + CUDA-sync, reset staleness once."""


class RoutingProtocol(ABC):
    """Request load balancer adapter (section 4.1 LB owner)."""

    @abstractmethod
    async def begin_drain(
        self, ctx: OperationContext, replica_id: str
    ) -> DrainReceipt:
        """Atomically advance routing_epoch and drop the target from routable."""

    @abstractmethod
    async def wait_drained(
        self, ctx: OperationContext, replica_id: str, routing_epoch: int
    ) -> bool:
        """True once in-flight/admitting/queued/running counts are all zero."""

    @abstractmethod
    async def commit_routable(
        self, ctx: OperationContext, head_server: Any, receipt: TransferReceipt
    ) -> int:
        """Return the routing epoch after publishing the target as routable."""

    @abstractmethod
    async def finish_remove(self, ctx: OperationContext, replica_id: str) -> bool:
        """True when the target's attempts are drained and it may be deleted."""

    @abstractmethod
    async def query_routing_operation(
        self, ctx: OperationContext, operation_id: str
    ) -> str:
        """Return the routing phase for reconciliation after a lost receipt."""


class ReplicaProtocol(ABC):
    """Runtime manager adapter (section 4.2 Manager owner)."""

    @abstractmethod
    async def materialize_hidden(
        self,
        ctx: OperationContext,
        placement: PlacementSpec,
        model_config: object,
    ) -> PreparedReplica:
        """Create a hidden runtime; on success only HIDDEN, never published."""

    @abstractmethod
    async def sleep_runtime(
        self, ctx: OperationContext, replica_id: str
    ) -> ReleaseReceipt:
        """Real sleep after routing drain + CE exclusion; keep native anchors."""

    @abstractmethod
    async def destroy_runtime(
        self, ctx: OperationContext, replica_id: str, purpose: str
    ) -> ReleaseReceipt:
        """Borrowed reclaim / failed cleanup; native only for explicit exit."""

    @abstractmethod
    async def inspect_runtime(self, ctx: OperationContext, replica_id: str) -> object:
        """Read-only node/GPU/process/engine status."""


class ServingProtocol(ABC):
    """Individual server adapter (section 4.2 Server owner)."""

    @abstractmethod
    async def wake_weights(
        self, ctx: OperationContext, replica_id: str
    ) -> WeightReadiness:
        """Restore weights only; generation stays disabled."""

    @abstractmethod
    async def wake_kv_and_validate(
        self, ctx: OperationContext, receipt: TransferReceipt
    ) -> ServingReadiness:
        """Restore KV, clear stale prefix/MM cache, validate engine; not routed."""

    @abstractmethod
    async def abort_target(
        self, ctx: OperationContext, replica_id: str
    ) -> AbortReceipt:
        """Force-reclaim a borrowed target; never a whole cluster rebalance."""


class ServingVersionStore(ABC):
    """Holder for the Trainer's published serving version (Vpub)."""

    @abstractmethod
    def current(self) -> int:
        """Return the currently published serving version."""

    @abstractmethod
    def publish(self, version: int) -> None:
        """Advance Vpub under the gate after a successful native sync."""


class ScaleTransaction:
    def __init__(
        self,
        gate: ReplicaSyncGate,
        checkpoint: CheckpointProtocol,
        routing: RoutingProtocol,
        replica: ReplicaProtocol,
        serving: ServingProtocol,
        versions: ServingVersionStore | None = None,
        *,
        gate_timeout: float | None = None,
    ) -> None:
        self._gate = gate
        self._checkpoint = checkpoint
        self._routing = routing
        self._replica = replica
        self._serving = serving
        self._versions = versions
        self._gate_timeout = gate_timeout

    async def _acquire(
        self, ctx: OperationContext, kind: GateKind
    ) -> GateLease:
        return await self._gate.acquire(
            ctx.operation_id, kind, timeout=self._gate_timeout
        )

    async def _pin_snapshot(
        self, ctx: OperationContext, receiver_ids: tuple[str, ...]
    ) -> PublishedWeightSnapshot:
        version = self._versions.current() if self._versions is not None else 0
        return PublishedWeightSnapshot(
            serving_version=version,
            manifest_digest="",  # filled by CE backend during native transfer
            receiver_ids=receiver_ids,
        )

    # -- ADD -------------------------------------------------------------- #

    async def add_and_publish(
        self, ctx: OperationContext, prepared: PreparedReplica
    ) -> ReadyReceipt:
        """Bootstrap a hidden target and publish it, holding G throughout.

        ADD does not advance the parameter version and does not reset staleness
        (section 5.3).
        """
        lease = await self._acquire(ctx, GateKind.ADD)
        try:
            snapshot = await self._pin_snapshot(ctx, prepared.receiver_ids)
            transfer = await self._checkpoint.bootstrap_target(ctx, prepared, snapshot)
            if not transfer.all_receivers_complete:
                raise TransferIncompleteError(
                    f"ADD bootstrap incomplete for {prepared.replica_id}: "
                    f"{transfer.receiver_states}"
                )
            ce_revision = await self._checkpoint.add_effective_replica(ctx, prepared)
            routing_epoch = await self._routing.commit_routable(
                ctx, prepared.head_server_descriptor, transfer
            )
            return ReadyReceipt(
                replica_id=prepared.replica_id,
                operation_id=ctx.operation_id,
                routing_epoch=routing_epoch,
                ce_revision=ce_revision,
                serving_version=snapshot.serving_version,
            )
        except GateFencedError:
            raise
        finally:
            await lease.release()

    # -- shared drain + CE exclusion -------------------------------------- #

    async def _remove_and_commit(
        self, ctx: OperationContext, replica_id: str, drain: DrainReceipt
    ) -> RemovedReceipt:
        lease = await self._acquire(ctx, GateKind.REMOVE)
        try:
            # Re-verify the drain under the gate; a new attempt invalidates it.
            if not await self._routing.wait_drained(
                ctx, replica_id, drain.routing_epoch
            ):
                raise DrainTimeoutError(f"target re-drained: {replica_id}")
            ce_revision = await self._checkpoint.remove_effective_replica(
                ctx, replica_id
            )
            lb_excluded = await self._routing.finish_remove(ctx, replica_id)
            return RemovedReceipt(
                replica_id=replica_id,
                operation_id=ctx.operation_id,
                ce_revision=ce_revision,
                lb_excluded=lb_excluded,
                capacity_released=True,
            )
        finally:
            await lease.release()

    async def donate(
        self, ctx: OperationContext, replica_id: str
    ) -> tuple[RemovedReceipt, ReleaseReceipt]:
        """DONATE: drain -> CE exclude -> real sleep (section 5.2)."""
        drain = await self._routing.begin_drain(ctx, replica_id)
        if not await self._routing.wait_drained(ctx, replica_id, drain.routing_epoch):
            raise DrainTimeoutError(f"donor drain timeout: {replica_id}")
        removed = await self._remove_and_commit(ctx, replica_id, drain)
        release = await self._replica.sleep_runtime(ctx, replica_id)
        return removed, release

    async def remove(
        self, ctx: OperationContext, replica_id: str, purpose: str = "recall"
    ) -> tuple[RemovedReceipt, ReleaseReceipt]:
        """REMOVE: drain -> CE exclude -> destroy (section 5.4)."""
        drain = await self._routing.begin_drain(ctx, replica_id)
        if not await self._routing.wait_drained(ctx, replica_id, drain.routing_epoch):
            raise DrainTimeoutError(f"borrower drain timeout: {replica_id}")
        removed = await self._remove_and_commit(ctx, replica_id, drain)
        release = await self._replica.destroy_runtime(ctx, replica_id, purpose)
        return removed, release

    # -- RESTORE ---------------------------------------------------------- #

    async def restore_and_publish(
        self,
        ctx: OperationContext,
        prepared: PreparedReplica,
        fence_satisfied: bool,
    ) -> RestoredReceipt:
        """RESTORE: wake weights -> (G) bootstrap -> validate -> publish (5.5)."""
        if not fence_satisfied:
            raise FenceRejectedError(
                f"RESTORE fence not satisfied for {prepared.replica_id} "
                f"(borrower not released)"
            )
        await self._serving.wake_weights(ctx, prepared.replica_id)

        lease = await self._acquire(ctx, GateKind.RESTORE)
        try:
            snapshot = await self._pin_snapshot(ctx, prepared.receiver_ids)
            transfer = await self._checkpoint.bootstrap_target(ctx, prepared, snapshot)
            if not transfer.all_receivers_complete:
                raise TransferIncompleteError(
                    f"RESTORE bootstrap incomplete for {prepared.replica_id}"
                )
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
                serving_version=snapshot.serving_version,
            )
        finally:
            await lease.release()

    # -- native parameter sync ------------------------------------------- #

    async def native_weight_sync(
        self, ctx: OperationContext, new_serving_version: int, manifest_digest: str
    ) -> PublishedWeightSnapshot:
        """Full native weight update under G: transfer, reset, publish (6)."""
        lease = await self._acquire(ctx, GateKind.NATIVE_SYNC)
        try:
            snapshot = await self._checkpoint.run_native_sync(
                ctx, new_serving_version, manifest_digest
            )
            if self._versions is not None:
                self._versions.publish(new_serving_version)
            return snapshot
        finally:
            await lease.release()
