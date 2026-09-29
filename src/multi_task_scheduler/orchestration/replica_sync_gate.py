"""Task-local exclusion and fencing for standalone replica membership.

``ReplicaSyncGate`` (the "G" gate of the fusion design, section 6) serializes
every write that mutates Checkpoint Engine membership together with native
weight synchronization. Hidden materialization and drain observation do *not*
acquire it. A cancelled or superseded owner is fenced so its stale lease can
no longer perform protected writes.

This is a task-internal asyncio lock, not a global GS lock, and must not be
shared across Ray concurrency groups.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class GateKind(str, Enum):
    """Which membership-mutating operation owns the gate."""

    ADD = "add"
    REMOVE = "remove"
    RESTORE = "restore"
    NATIVE_SYNC = "native_sync"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class GateOwner:
    operation_id: str
    kind: GateKind
    epoch: int
    acquired_at: float


class GateTimeoutError(asyncio.TimeoutError):
    """A caller could not acquire the gate before its deadline."""


class GateReentryError(RuntimeError):
    """The active operation tried to acquire the same gate a second time."""


class GateFencedError(RuntimeError):
    """A released or superseded lease attempted a protected write."""


class ReplicaSyncGate:
    """Fair task-local exclusion with owner metadata and fencing."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: GateOwner | None = None
        self._epoch = 0
        self._blocked_reason: str | None = None
        self._blocked_operation_id: str | None = None
        self.wait_seconds = 0.0
        self.max_owner_seconds = 0.0

    @property
    def owner(self) -> GateOwner | None:
        return self._owner

    @property
    def health(self) -> str:
        return "BLOCKED" if self._blocked_reason is not None else "HEALTHY"

    @property
    def blocked_reason(self) -> str | None:
        return self._blocked_reason

    @property
    def blocked_operation_id(self) -> str | None:
        return self._blocked_operation_id

    def block(self, owner: GateOwner, reason: str) -> None:
        """Latch uncertain side effects; releasing the lock does not prove recovery.

        There is deliberately no boolean reset. ``reconcile()`` is the only
        clearing path and it requires the operation that latched the gate to run
        its owner-level reconciliation while the gate remains exclusively held.
        """
        if not self._is_active(owner):
            raise GateFencedError("Only the current gate owner may block synchronization")
        if not isinstance(reason, str) or not reason:
            raise ValueError("A blocked gate requires a reason")
        self._blocked_reason = reason
        self._blocked_operation_id = owner.operation_id

    def _require_healthy(self) -> None:
        if self._blocked_reason is not None:
            raise GateFencedError(f"Replica synchronization is BLOCKED: {self._blocked_reason}")

    async def reconcile(
        self,
        operation_id: str,
        reconciliation: Callable[[], Any],
    ) -> bool:
        """Clear a BLOCKED latch only after owner-level reconciliation succeeds.

        The reconciliation callback runs while the same task-local lock is held,
        so no membership change or native sync can race the repair. Returning
        normally is the proof boundary; any exception leaves the latch intact.
        Only the operation that originally blocked the gate may reconcile it.
        """
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id must be a nonempty string")
        if not callable(reconciliation):
            raise TypeError("reconciliation must be callable")
        if self._blocked_reason is None:
            return False
        if self._blocked_operation_id != operation_id:
            raise GateFencedError(
                "Only the operation that blocked synchronization may reconcile it"
            )

        wait_started = time.monotonic()
        await self._lock.acquire()
        self.wait_seconds += time.monotonic() - wait_started
        try:
            # Another reconciler may have completed while this coroutine waited.
            if self._blocked_reason is None:
                return False
            if self._blocked_operation_id != operation_id:
                raise GateFencedError(
                    "Blocked synchronization belongs to another operation"
                )

            self._epoch += 1
            owner = GateOwner(
                operation_id=operation_id,
                kind=GateKind.RECOVERY,
                epoch=self._epoch,
                acquired_at=time.monotonic(),
            )
            self._owner = owner
            try:
                result = reconciliation()
                if inspect.isawaitable(result):
                    await result
                self._blocked_reason = None
                self._blocked_operation_id = None
                return True
            finally:
                held_seconds = time.monotonic() - owner.acquired_at
                self.max_owner_seconds = max(self.max_owner_seconds, held_seconds)
                self._owner = None
        finally:
            self._lock.release()

    async def acquire(
        self,
        operation_id: str,
        kind: GateKind,
        timeout: float | None = None,
    ) -> GateLease:
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id must be a nonempty string")
        kind = GateKind(kind)
        self._require_healthy()
        if self._owner is not None and self._owner.operation_id == operation_id:
            raise GateReentryError(
                f"operation {operation_id!r} already owns the replica sync gate"
            )

        wait_started = time.monotonic()
        try:
            if timeout is None:
                await self._lock.acquire()
            else:
                await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise GateTimeoutError(
                f"timed out waiting for replica sync gate: {operation_id}"
            ) from exc

        self.wait_seconds += time.monotonic() - wait_started
        # The previous owner may have failed while this coroutine was waiting.
        if self._blocked_reason is not None:
            self._lock.release()
            self._require_healthy()
        self._epoch += 1
        owner = GateOwner(
            operation_id=operation_id,
            kind=kind,
            epoch=self._epoch,
            acquired_at=time.monotonic(),
        )
        self._owner = owner
        return GateLease(self, owner)

    def _is_active(self, owner: GateOwner) -> bool:
        return self._owner == owner and self._lock.locked()

    async def _release(self, owner: GateOwner) -> bool:
        if not self._is_active(owner):
            return False
        held_seconds = time.monotonic() - owner.acquired_at
        self.max_owner_seconds = max(self.max_owner_seconds, held_seconds)
        self._owner = None
        self._lock.release()
        return True


class GateLease:
    def __init__(self, gate: ReplicaSyncGate, owner: GateOwner) -> None:
        self._gate = gate
        self.owner = owner

    @property
    def active(self) -> bool:
        return self._gate._is_active(self.owner)

    async def release(self) -> bool:
        return await self._gate._release(self.owner)

    async def __aenter__(self) -> GateLease:
        if not self.active:
            raise GateFencedError("cannot enter an inactive gate lease")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release()

    async def guard(self, call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if not self.active:
            raise GateFencedError(
                f"gate lease is no longer active: "
                f"operation={self.owner.operation_id}, epoch={self.owner.epoch}"
            )
        result = call(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        if not self.active or self._gate.health != "HEALTHY":
            raise GateFencedError("Gate ownership or health changed before the result was committed")
        return result

