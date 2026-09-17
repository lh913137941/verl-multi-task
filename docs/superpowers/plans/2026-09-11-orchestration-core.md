# 5.3 Orchestration Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pure-Python Slice A+B+C orchestration core for standalone replica synchronization, operation replay, replica lifecycle validation, and ADD/REMOVE/RESTORE transaction ordering.

**Architecture:** Add a dependency-light `multi_task_scheduler.orchestration` package that imports neither Ray nor verl. A task-local `ReplicaSyncGate` fences all Checkpoint Engine membership writes and native synchronization; `OperationJournal` and `ReplicaRecord` make control operations replayable and validate lifecycle transitions; `ScaleTransaction` coordinates injected runtime, Checkpoint Engine, load-balancer, and capacity primitives without owning any distributed runtime handles.

**Tech Stack:** Python 3.10+, standard-library `asyncio`, `dataclasses`, `enum`, `typing.Protocol`, pytest 8.3.5. Tests use `asyncio.run()` and deterministic fakes; they do not require pytest-asyncio, Ray, verl, CUDA, vLLM, or NCCL.

**Spec:** `docs/verl-codebase-blueprint-map.md`; `../multi_task_verl/multi_task_scheduler/references/17-verl-v0.9-dynamic-replica-scaling-timing-and-mutual-exclusion.md`; `../multi_task_verl/multi_task_scheduler/references/26-verl-multi-task-feature-development-plan.md`; `../multi_task_verl/multi_task_scheduler/references/27-verl-multi-task-feature-scope-discussion-notes.md`.

## Global Constraints

- Execution is blocked until the user explicitly authorizes Slice A+B+C implementation and the repository rule `New business features remain empty` in `AGENTS.md` is updated for that scope. Authoring and reviewing this plan does not grant that implementation authorization.
- Scope is `experimental Fully Async + pure STANDALONE + vLLM non-PD`; HYBRID, V1, PD, forced reclaim, policy selection, and GPU runtime work remain outside this plan.
- Do not modify any file under `D:/多RL任务/verl/verl/` and do not restore the old Impl/SPI integration.
- Do not modify existing P1 TaskRunner, Trainer, Rollouter, Manager, Load Balancer, Replica, HTTP Server, Checkpoint Engine, GroupScheduler, runtime-profile, or entry wiring in this slice.
- `src/multi_task_scheduler/orchestration/` must remain pure Python and must not import Ray, verl, torch, vLLM, or trigger actor discovery or global registration.
- GroupScheduler receives metadata only. Runtime handles, workers, Placement Groups, resource pools, server handles, and donor Checkpoint Engine workers never enter borrower placement contracts.
- The runtime manager owns runtime references; Checkpoint Engine membership and load-balancer routing are separate projections; Trainer remains the only writer of Checkpoint Engine membership.
- Use one `ReplicaSyncGate` for ADD, REMOVE, RESTORE, and native synchronization. Hidden materialization and drain observation do not acquire this gate.
- BOOTSTRAP is distinct from native synchronization: it targets one replica, uses a pinned serving version, does not advance the version, and does not reset rollout staleness.
- A known ADD failure rolls back Checkpoint Engine membership, capacity, and hidden routing preparation before releasing the gate. An ambiguous load-balancer commit timeout remains queryable and must not trigger a blind delete.
- Same `(operation_id, lease_epoch)` is an idempotent replay; an older epoch is rejected.
- Tests in this plan are mock-only unit coverage. Passing them must not be reported as Ray, native verl, GPU, vLLM, NCCL, or end-to-end training success.
- Keep `src/multi_task_scheduler/__init__.py` at `IMPLEMENTATION_STAGE = "PASSTHROUGH_INTEGRATION"`; that marker changes only when Slice D is connected to native subclasses.
- Use `apply_patch` for source and test edits. Do not commit or push unless the user separately authorizes those Git actions.

## File Map

| File | Responsibility |
| --- | --- |
| `src/multi_task_scheduler/orchestration/__init__.py` | Dependency-light package marker; exports nothing to keep top-level imports inert. |
| `src/multi_task_scheduler/orchestration/replica_sync_gate.py` | Fair task-local exclusion, owner metadata, monotonic metrics, lease fencing, and cancellation-safe release. |
| `src/multi_task_scheduler/orchestration/replica_record.py` | Replica lifecycle state and legal transition validation, independent of Ray runtime objects. |
| `src/multi_task_scheduler/orchestration/operation_journal.py` | Operation identity, epoch replay, transition validation, query, and validation freeze. |
| `src/multi_task_scheduler/orchestration/receipts.py` | Frozen, serializable data returned between orchestration steps. |
| `src/multi_task_scheduler/orchestration/scale_transaction.py` | Injected ADD/REMOVE/RESTORE/native-sync ordering and rollback policy. |
| `tests/unit/test_replica_sync_gate.py` | Gate serialization, reentry, cancellation release, fencing, and metrics. |
| `tests/unit/test_replica_record.py` | Happy lifecycle, rollback, restore, and illegal-transition coverage. |
| `tests/unit/test_operation_journal.py` | Replay, epoch fencing, operation transitions, query, and freeze coverage. |
| `tests/unit/test_receipts.py` | Receipt value semantics and immutability. |
| `tests/unit/test_scale_transaction.py` | Orchestration race, rollback, timeout, freeze, ownership, and projection-invariant cases. |

Run all commands from `D:/多RL任务/verl-multi-task`. If the local virtual environment does not exist, create it with the repository's documented `uv` workflow before running tests; do not install GPU dependencies for this plan.

---

### Task 1: Replica synchronization gate

**Files:**
- Create: `src/multi_task_scheduler/orchestration/__init__.py`
- Create: `src/multi_task_scheduler/orchestration/replica_sync_gate.py`
- Test: `tests/unit/test_replica_sync_gate.py`

**Interfaces:**
- Consumes: only Python standard-library modules.
- Produces: `GateKind`, `GateOwner`, `GateTimeoutError`, `GateReentryError`, `GateFencedError`, `ReplicaSyncGate.acquire(operation_id: str, kind: GateKind, timeout: float | None = None) -> GateLease`, and `GateLease.guard(call, *args, **kwargs)`.

- [ ] **Step 1: Create the package marker and write the failing gate tests**

Create `src/multi_task_scheduler/orchestration/__init__.py`:

```python
"""Pure-Python orchestration primitives with no Ray or verl imports."""
```

Create `tests/unit/test_replica_sync_gate.py`:

```python
import asyncio

import pytest

from multi_task_scheduler.orchestration.replica_sync_gate import (
    GateFencedError,
    GateKind,
    GateReentryError,
    GateTimeoutError,
    ReplicaSyncGate,
)


def test_gate_serializes_waiters_in_arrival_order():
    async def scenario():
        gate = ReplicaSyncGate()
        first = await gate.acquire("add-1", GateKind.ADD)
        second_task = asyncio.create_task(gate.acquire("sync-1", GateKind.NATIVE_SYNC))
        third_task = asyncio.create_task(gate.acquire("remove-1", GateKind.REMOVE))
        await asyncio.sleep(0)

        assert not second_task.done()
        assert not third_task.done()

        await first.release()
        second = await asyncio.wait_for(second_task, timeout=0.1)
        assert gate.owner.operation_id == "sync-1"
        await second.release()

        third = await asyncio.wait_for(third_task, timeout=0.1)
        assert gate.owner.operation_id == "remove-1"
        await third.release()
        assert gate.owner is None

    asyncio.run(scenario())


def test_same_operation_cannot_reenter_active_gate():
    async def scenario():
        gate = ReplicaSyncGate()
        lease = await gate.acquire("add-1", GateKind.ADD)
        with pytest.raises(GateReentryError):
            await gate.acquire("add-1", GateKind.ADD)
        await lease.release()

    asyncio.run(scenario())


def test_wait_timeout_does_not_release_current_owner():
    async def scenario():
        gate = ReplicaSyncGate()
        lease = await gate.acquire("sync-1", GateKind.NATIVE_SYNC)
        with pytest.raises(GateTimeoutError):
            await gate.acquire("add-1", GateKind.ADD, timeout=0.01)
        assert gate.owner.operation_id == "sync-1"
        await lease.release()

    asyncio.run(scenario())


def test_cancellation_releases_gate_and_fences_old_lease():
    async def scenario():
        gate = ReplicaSyncGate()
        started = asyncio.Event()
        captured = {}

        async def blocked_owner():
            lease = await gate.acquire("sync-1", GateKind.NATIVE_SYNC)
            captured["lease"] = lease
            async with lease:
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(blocked_owner())
        await started.wait()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=0.01)

        assert gate.owner is None
        next_lease = await gate.acquire("add-1", GateKind.ADD)

        async def forbidden_write():
            raise AssertionError("fenced writes must not execute")

        with pytest.raises(GateFencedError):
            await captured["lease"].guard(forbidden_write)
        await next_lease.release()

    asyncio.run(scenario())


def test_exception_context_releases_gate_and_records_metrics():
    async def scenario():
        gate = ReplicaSyncGate()
        with pytest.raises(RuntimeError, match="boom"):
            lease = await gate.acquire("restore-1", GateKind.RESTORE)
            async with lease:
                raise RuntimeError("boom")

        assert gate.owner is None
        assert gate.wait_seconds >= 0
        assert gate.max_owner_seconds >= 0

    asyncio.run(scenario())
```

- [ ] **Step 2: Run the gate tests and verify the import fails**

Run (PowerShell):

```powershell
$env:PYTHONPATH="$PWD/src"
$env:PYTHONDONTWRITEBYTECODE="1"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
python -m pytest -q -p no:cacheprovider tests/unit/test_replica_sync_gate.py
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'multi_task_scheduler.orchestration.replica_sync_gate'`.

- [ ] **Step 3: Implement the minimal gate and lease**

Create `src/multi_task_scheduler/orchestration/replica_sync_gate.py`:

```python
"""Task-local exclusion and fencing for standalone replica membership."""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class GateKind(str, Enum):
    ADD = "add"
    REMOVE = "remove"
    NATIVE_SYNC = "native_sync"
    RESTORE = "restore"


@dataclass(frozen=True)
class GateOwner:
    operation_id: str
    kind: GateKind
    epoch: int
    acquired_at: float


class GateTimeoutError(asyncio.TimeoutError):
    """Raised when a caller cannot acquire the gate before its deadline."""


class GateReentryError(RuntimeError):
    """Raised when the active operation tries to acquire the same gate again."""


class GateFencedError(RuntimeError):
    """Raised when a released or superseded lease attempts a protected write."""


class ReplicaSyncGate:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: GateOwner | None = None
        self._epoch = 0
        self.wait_seconds = 0.0
        self.max_owner_seconds = 0.0

    @property
    def owner(self) -> GateOwner | None:
        return self._owner

    async def acquire(
        self,
        operation_id: str,
        kind: GateKind,
        timeout: float | None = None,
    ) -> GateLease:
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id must be a nonempty string")
        kind = GateKind(kind)
        if self._owner is not None and self._owner.operation_id == operation_id:
            raise GateReentryError(f"operation {operation_id!r} already owns the replica sync gate")

        wait_started = time.monotonic()
        try:
            if timeout is None:
                await self._lock.acquire()
            else:
                await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise GateTimeoutError(f"timed out waiting for replica sync gate: {operation_id}") from exc

        self.wait_seconds += time.monotonic() - wait_started
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

    async def guard(self, call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if not self.active:
            raise GateFencedError(
                f"gate lease is no longer active: operation={self.owner.operation_id}, epoch={self.owner.epoch}"
            )
        result = call(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def __aenter__(self) -> GateLease:
        if not self.active:
            raise GateFencedError(f"cannot enter inactive gate lease for {self.owner.operation_id}")
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.release()
```

- [ ] **Step 4: Run the focused and full unit suites**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/unit/test_replica_sync_gate.py
python -m pytest -q -p no:cacheprovider tests/unit
```

Expected: the focused file passes 5 tests; the existing unit suite remains green. This proves only pure-Python gate behavior.

- [ ] **Step 5: Commit the independently testable gate**

Run only after separate Git authorization:

```powershell
git add src/multi_task_scheduler/orchestration/__init__.py src/multi_task_scheduler/orchestration/replica_sync_gate.py tests/unit/test_replica_sync_gate.py
git commit -m "feat: add replica synchronization gate"
```

---

### Task 2: Replica lifecycle record

**Files:**
- Create: `src/multi_task_scheduler/orchestration/replica_record.py`
- Test: `tests/unit/test_replica_record.py`

**Interfaces:**
- Consumes: `GateOwner` from Task 1 for optional observability only.
- Produces: `ReplicaState`, `IllegalReplicaTransitionError`, and mutable `ReplicaRecord.transition_to(new_state: ReplicaState) -> None`.

- [ ] **Step 1: Write the failing lifecycle tests**

Create `tests/unit/test_replica_record.py`:

```python
import pytest

from multi_task_scheduler.orchestration.replica_record import (
    IllegalReplicaTransitionError,
    ReplicaRecord,
    ReplicaState,
)


def test_add_lifecycle_reaches_routable_in_order():
    record = ReplicaRecord("replica-1", operation_id="add-1", lease_epoch=7)
    for state in (
        ReplicaState.HIDDEN,
        ReplicaState.BOOTSTRAP_PENDING,
        ReplicaState.BOOTSTRAPPING,
        ReplicaState.SYNCED,
        ReplicaState.CE_EFFECTIVE,
        ReplicaState.ROUTABLE,
    ):
        record.transition_to(state)
    assert record.state is ReplicaState.ROUTABLE


def test_hidden_cannot_skip_bootstrap_and_become_routable():
    record = ReplicaRecord("replica-1", operation_id="add-1", lease_epoch=7)
    record.transition_to(ReplicaState.HIDDEN)
    with pytest.raises(IllegalReplicaTransitionError):
        record.transition_to(ReplicaState.ROUTABLE)


@pytest.mark.parametrize(
    "start",
    [ReplicaState.BOOTSTRAPPING, ReplicaState.SYNCED, ReplicaState.CE_EFFECTIVE],
)
def test_known_add_failure_can_roll_back_to_hidden(start):
    record = ReplicaRecord("replica-1", state=start, operation_id="add-1", lease_epoch=7)
    record.transition_to(ReplicaState.HIDDEN)
    assert record.state is ReplicaState.HIDDEN


def test_remove_lifecycle_ends_in_dormant_or_destroyed():
    for final_state in (ReplicaState.DORMANT, ReplicaState.DESTROYED):
        record = ReplicaRecord("replica-1", state=ReplicaState.ROUTABLE)
        record.transition_to(ReplicaState.DRAINING)
        record.transition_to(ReplicaState.RETIRING)
        record.transition_to(final_state)
        assert record.state is final_state


def test_dormant_donor_can_reenter_bootstrap_path():
    record = ReplicaRecord("replica-1", state=ReplicaState.DORMANT)
    record.transition_to(ReplicaState.BOOTSTRAP_PENDING)
    assert record.state is ReplicaState.BOOTSTRAP_PENDING
```

- [ ] **Step 2: Run the lifecycle tests and verify the import fails**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/unit/test_replica_record.py
```

Expected: FAIL during collection because `replica_record.py` does not exist.

- [ ] **Step 3: Implement the replica state machine**

Create `src/multi_task_scheduler/orchestration/replica_record.py`:

```python
"""Replica lifecycle metadata independent of distributed runtime handles."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .replica_sync_gate import GateOwner


class ReplicaState(str, Enum):
    MATERIALIZING = "materializing"
    HIDDEN = "hidden"
    BOOTSTRAP_PENDING = "bootstrap_pending"
    BOOTSTRAPPING = "bootstrapping"
    SYNCED = "synced"
    CE_EFFECTIVE = "ce_effective"
    ROUTABLE = "routable"
    DRAINING = "draining"
    RETIRING = "retiring"
    DORMANT = "dormant"
    DESTROYED = "destroyed"
    FAILED = "failed"


class IllegalReplicaTransitionError(RuntimeError):
    """Raised when orchestration skips a required replica lifecycle state."""


_ALLOWED = {
    ReplicaState.MATERIALIZING: {ReplicaState.HIDDEN, ReplicaState.FAILED},
    ReplicaState.HIDDEN: {ReplicaState.BOOTSTRAP_PENDING, ReplicaState.FAILED},
    ReplicaState.BOOTSTRAP_PENDING: {
        ReplicaState.BOOTSTRAPPING,
        ReplicaState.HIDDEN,
        ReplicaState.DORMANT,
        ReplicaState.FAILED,
    },
    ReplicaState.BOOTSTRAPPING: {
        ReplicaState.SYNCED,
        ReplicaState.HIDDEN,
        ReplicaState.DORMANT,
        ReplicaState.FAILED,
    },
    ReplicaState.SYNCED: {
        ReplicaState.CE_EFFECTIVE,
        ReplicaState.HIDDEN,
        ReplicaState.DORMANT,
        ReplicaState.FAILED,
    },
    ReplicaState.CE_EFFECTIVE: {
        ReplicaState.ROUTABLE,
        ReplicaState.HIDDEN,
        ReplicaState.DORMANT,
        ReplicaState.FAILED,
    },
    ReplicaState.ROUTABLE: {ReplicaState.DRAINING, ReplicaState.FAILED},
    ReplicaState.DRAINING: {ReplicaState.RETIRING, ReplicaState.FAILED},
    ReplicaState.RETIRING: {
        ReplicaState.DORMANT,
        ReplicaState.DESTROYED,
        ReplicaState.FAILED,
    },
    ReplicaState.DORMANT: {ReplicaState.BOOTSTRAP_PENDING, ReplicaState.FAILED},
    ReplicaState.DESTROYED: set(),
    ReplicaState.FAILED: set(),
}


@dataclass
class ReplicaRecord:
    replica_id: str
    state: ReplicaState = ReplicaState.MATERIALIZING
    operation_id: str | None = None
    lease_epoch: int = 0
    routing_epoch: int | None = None
    last_synced_weight_version: int | None = None
    bootstrap_target_version: int | None = None
    replica_sync_gate_owner: GateOwner | None = None
    lifecycle_pins: set[str] = field(default_factory=set)

    def transition_to(self, new_state: ReplicaState) -> None:
        new_state = ReplicaState(new_state)
        if new_state is self.state:
            return
        if new_state not in _ALLOWED[self.state]:
            raise IllegalReplicaTransitionError(
                f"illegal replica transition for {self.replica_id}: {self.state.value} -> {new_state.value}"
            )
        self.state = new_state
```

- [ ] **Step 4: Run focused and accumulated tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/unit/test_replica_record.py
python -m pytest -q -p no:cacheprovider tests/unit/test_replica_sync_gate.py tests/unit/test_replica_record.py
```

Expected: all lifecycle and gate tests pass.

- [ ] **Step 5: Commit the lifecycle model**

Run only after separate Git authorization:

```powershell
git add src/multi_task_scheduler/orchestration/replica_record.py tests/unit/test_replica_record.py
git commit -m "feat: add replica lifecycle state machine"
```

---

### Task 3: Replayable operation journal

**Files:**
- Create: `src/multi_task_scheduler/orchestration/operation_journal.py`
- Test: `tests/unit/test_operation_journal.py`

**Interfaces:**
- Consumes: no earlier runtime object; records only serializable identifiers and scalar metadata.
- Produces: `OperationType`, `OperationState`, `OperationRecord`, `ExpiredLeaseError`, `OperationIdentityError`, `IllegalOperationTransitionError`, and `OperationJournal.begin/require/query/transition/set_validation_freeze`.

- [ ] **Step 1: Write the failing journal tests**

Create `tests/unit/test_operation_journal.py`:

```python
import pytest

from multi_task_scheduler.orchestration.operation_journal import (
    ExpiredLeaseError,
    IllegalOperationTransitionError,
    OperationIdentityError,
    OperationJournal,
    OperationState,
    OperationType,
)


def test_same_operation_and_epoch_replays_current_record():
    journal = OperationJournal()
    original = journal.begin("add-1", 4, "replica-1", OperationType.ADD)
    journal.transition("add-1", OperationState.PREPARED)
    replay = journal.begin("add-1", 4, "replica-1", OperationType.ADD)
    assert replay is original
    assert replay.state is OperationState.PREPARED


def test_same_identity_rejects_conflicting_metadata():
    journal = OperationJournal()
    journal.begin("add-1", 4, "replica-1", OperationType.ADD)
    with pytest.raises(OperationIdentityError):
        journal.begin("add-1", 4, "replica-2", OperationType.ADD)


def test_older_epoch_is_rejected_and_newer_epoch_replaces_record():
    journal = OperationJournal()
    journal.begin("add-1", 4, "replica-1", OperationType.ADD)
    with pytest.raises(ExpiredLeaseError):
        journal.begin("add-1", 3, "replica-1", OperationType.ADD)
    newer = journal.begin("add-1", 5, "replica-1", OperationType.ADD)
    assert newer.lease_epoch == 5
    assert newer.state is OperationState.ACCEPTED


def test_require_fences_an_old_caller():
    journal = OperationJournal()
    journal.begin("add-1", 5, "replica-1", OperationType.ADD)
    with pytest.raises(ExpiredLeaseError):
        journal.require("add-1", 4)


def test_add_remove_and_restore_transitions_are_distinct():
    journal = OperationJournal()
    journal.begin("add-1", 1, "r1", OperationType.ADD)
    for state in (
        OperationState.PREPARED,
        OperationState.BOOTSTRAPPING,
        OperationState.CE_EFFECTIVE,
        OperationState.ROUTABLE,
    ):
        journal.transition("add-1", state)

    journal.begin("remove-1", 1, "r1", OperationType.REMOVE)
    for state in (OperationState.DRAINING, OperationState.CE_REMOVED, OperationState.DORMANT):
        journal.transition("remove-1", state)

    journal.begin("restore-1", 2, "r1", OperationType.RESTORE)
    journal.transition("restore-1", OperationState.BOOTSTRAPPING)
    assert journal.query("add-1").state is OperationState.ROUTABLE
    assert journal.query("remove-1").state is OperationState.DORMANT
    assert journal.query("restore-1").state is OperationState.BOOTSTRAPPING


def test_illegal_transition_is_rejected():
    journal = OperationJournal()
    journal.begin("add-1", 1, "r1", OperationType.ADD)
    with pytest.raises(IllegalOperationTransitionError):
        journal.transition("add-1", OperationState.ROUTABLE)


def test_validation_freeze_is_explicit_journal_state():
    journal = OperationJournal()
    assert not journal.validation_frozen
    journal.set_validation_freeze(True)
    assert journal.validation_frozen
    journal.set_validation_freeze(False)
    assert not journal.validation_frozen
```

- [ ] **Step 2: Run the journal tests and verify the import fails**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/unit/test_operation_journal.py
```

Expected: FAIL during collection because `operation_journal.py` does not exist.

- [ ] **Step 3: Implement the journal and operation state machine**

Create `src/multi_task_scheduler/orchestration/operation_journal.py`:

```python
"""Replayable operation metadata for scale transactions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class OperationType(str, Enum):
    ADD = "add"
    REMOVE = "remove"
    RESTORE = "restore"


class OperationState(str, Enum):
    ACCEPTED = "accepted"
    PREPARED = "prepared"
    BOOTSTRAPPING = "bootstrapping"
    CE_EFFECTIVE = "ce_effective"
    ROUTABLE = "routable"
    DRAINING = "draining"
    CE_REMOVED = "ce_removed"
    DESTROYED = "destroyed"
    DORMANT = "dormant"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ExpiredLeaseError(RuntimeError):
    """Raised when a command does not match the journal's current lease epoch."""


class OperationIdentityError(RuntimeError):
    """Raised when an operation ID is replayed with conflicting immutable fields."""


class IllegalOperationTransitionError(RuntimeError):
    """Raised when an operation skips a required transaction state."""


_FAILURE_STATES = {OperationState.FAILED, OperationState.ROLLED_BACK}
_ALLOWED = {
    OperationState.ACCEPTED: {
        OperationState.PREPARED,
        OperationState.BOOTSTRAPPING,
        OperationState.DRAINING,
        *_FAILURE_STATES,
    },
    OperationState.PREPARED: {OperationState.BOOTSTRAPPING, *_FAILURE_STATES},
    OperationState.BOOTSTRAPPING: {OperationState.CE_EFFECTIVE, *_FAILURE_STATES},
    OperationState.CE_EFFECTIVE: {OperationState.ROUTABLE, *_FAILURE_STATES},
    OperationState.ROUTABLE: set(),
    OperationState.DRAINING: {OperationState.CE_REMOVED, OperationState.FAILED},
    OperationState.CE_REMOVED: {
        OperationState.DESTROYED,
        OperationState.DORMANT,
        OperationState.FAILED,
    },
    OperationState.DESTROYED: set(),
    OperationState.DORMANT: set(),
    OperationState.FAILED: set(),
    OperationState.ROLLED_BACK: set(),
}


@dataclass
class OperationRecord:
    operation_id: str
    lease_epoch: int
    replica_id: str
    operation_type: OperationType
    state: OperationState = OperationState.ACCEPTED
    detail: dict[str, object] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)


class OperationJournal:
    def __init__(self) -> None:
        self._records: dict[str, OperationRecord] = {}
        self._validation_frozen = False

    @property
    def validation_frozen(self) -> bool:
        return self._validation_frozen

    def set_validation_freeze(self, frozen: bool) -> None:
        self._validation_frozen = bool(frozen)

    def begin(
        self,
        operation_id: str,
        lease_epoch: int,
        replica_id: str,
        operation_type: OperationType,
    ) -> OperationRecord:
        if not operation_id or not replica_id:
            raise ValueError("operation_id and replica_id must be nonempty")
        if lease_epoch < 0:
            raise ValueError("lease_epoch must be nonnegative")
        operation_type = OperationType(operation_type)
        existing = self._records.get(operation_id)
        if existing is not None:
            if lease_epoch < existing.lease_epoch:
                raise ExpiredLeaseError(
                    f"operation {operation_id!r} epoch {lease_epoch} is older than {existing.lease_epoch}"
                )
            if lease_epoch == existing.lease_epoch:
                if existing.replica_id != replica_id or existing.operation_type is not operation_type:
                    raise OperationIdentityError(f"conflicting replay for operation {operation_id!r}")
                return existing

        record = OperationRecord(operation_id, lease_epoch, replica_id, operation_type)
        self._records[operation_id] = record
        return record

    def require(self, operation_id: str, lease_epoch: int) -> OperationRecord:
        record = self._records[operation_id]
        if lease_epoch != record.lease_epoch:
            raise ExpiredLeaseError(
                f"operation {operation_id!r} epoch {lease_epoch} does not match {record.lease_epoch}"
            )
        return record

    def query(self, operation_id: str) -> OperationRecord | None:
        return self._records.get(operation_id)

    def transition(
        self,
        operation_id: str,
        new_state: OperationState,
        detail: dict[str, object] | None = None,
    ) -> OperationRecord:
        record = self._records[operation_id]
        new_state = OperationState(new_state)
        if new_state is not record.state:
            if new_state not in _ALLOWED[record.state]:
                raise IllegalOperationTransitionError(
                    f"illegal operation transition for {operation_id}: "
                    f"{record.state.value} -> {new_state.value}"
                )
            record.state = new_state
        if detail:
            record.detail.update(detail)
        record.updated_at = time.monotonic()
        return record
```

- [ ] **Step 4: Run focused and accumulated tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/unit/test_operation_journal.py
python -m pytest -q -p no:cacheprovider tests/unit/test_replica_sync_gate.py tests/unit/test_replica_record.py tests/unit/test_operation_journal.py
```

Expected: all journal, lifecycle, and gate tests pass.

- [ ] **Step 5: Commit the journal**

Run only after separate Git authorization:

```powershell
git add src/multi_task_scheduler/orchestration/operation_journal.py tests/unit/test_operation_journal.py
git commit -m "feat: add replayable scale operation journal"
```

---

### Task 4: Transaction receipts

**Files:**
- Create: `src/multi_task_scheduler/orchestration/receipts.py`
- Test: `tests/unit/test_receipts.py`

**Interfaces:**
- Consumes: only standard-library dataclasses.
- Produces: frozen `PreparedReplica`, `BootstrapReceipt`, `RoutableReceipt`, `DrainingReceipt`, `CEEffectiveRemoveReceipt`, and `ActiveReceipt` values.

- [ ] **Step 1: Write the failing receipt tests**

Create `tests/unit/test_receipts.py`:

```python
from dataclasses import FrozenInstanceError

import pytest

from multi_task_scheduler.orchestration.receipts import (
    ActiveReceipt,
    BootstrapReceipt,
    CEEffectiveRemoveReceipt,
    DrainingReceipt,
    PreparedReplica,
    RoutableReceipt,
)


def test_receipts_are_value_objects_with_expected_fields():
    server_handle = object()
    prepared = PreparedReplica(
        "r1", "http://server", server_handle, {"worker_ids": [1, 2]}, "add-1", 8
    )
    assert prepared.replica_id == "r1"
    assert prepared.server_handle is server_handle
    assert BootstrapReceipt("r1", 9, "add-1", 8, (("worker-0", True),)).weight_version == 9
    assert RoutableReceipt("r1", 3, "add-1", 8).routing_epoch == 3
    assert DrainingReceipt("r1", 4, 2, "remove-1", 8).inflight == 2
    assert CEEffectiveRemoveReceipt("r1", "remove-1", 8).operation_id == "remove-1"
    assert ActiveReceipt("r1", 8).lease_epoch == 8


def test_receipts_are_frozen():
    receipt = RoutableReceipt("r1", 3, "add-1", 8)
    with pytest.raises(FrozenInstanceError):
        receipt.routing_epoch = 4
```

- [ ] **Step 2: Run the receipt tests and verify the import fails**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/unit/test_receipts.py
```

Expected: FAIL during collection because `receipts.py` does not exist.

- [ ] **Step 3: Implement frozen receipt values**

Create `src/multi_task_scheduler/orchestration/receipts.py`:

```python
"""Serializable receipts exchanged by pure orchestration steps."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreparedReplica:
    replica_id: str
    server_address: str
    server_handle: object
    sync_descriptor: object
    operation_id: str
    lease_epoch: int


@dataclass(frozen=True)
class BootstrapReceipt:
    replica_id: str
    weight_version: int
    operation_id: str
    lease_epoch: int
    receiver_statuses: tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class RoutableReceipt:
    replica_id: str
    routing_epoch: int
    operation_id: str
    lease_epoch: int


@dataclass(frozen=True)
class DrainingReceipt:
    replica_id: str
    routing_epoch: int
    inflight: int
    operation_id: str
    lease_epoch: int


@dataclass(frozen=True)
class CEEffectiveRemoveReceipt:
    replica_id: str
    operation_id: str
    lease_epoch: int


@dataclass(frozen=True)
class ActiveReceipt:
    replica_id: str
    lease_epoch: int
```

- [ ] **Step 4: Run focused and accumulated tests**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/unit/test_receipts.py
python -m pytest -q -p no:cacheprovider tests/unit/test_replica_sync_gate.py tests/unit/test_replica_record.py tests/unit/test_operation_journal.py tests/unit/test_receipts.py
```

Expected: all four pure-data/primitives test files pass.

- [ ] **Step 5: Commit the receipt values**

Run only after separate Git authorization:

```powershell
git add src/multi_task_scheduler/orchestration/receipts.py tests/unit/test_receipts.py
git commit -m "feat: add orchestration transaction receipts"
```

---

### Task 5: Scale transaction orchestration

**Files:**
- Create: `src/multi_task_scheduler/orchestration/scale_transaction.py`
- Test: `tests/unit/test_scale_transaction.py`

**Interfaces:**
- Consumes: `ReplicaSyncGate`, `OperationJournal`, `ReplicaRecord`, and receipt values from Tasks 1–4.
- Produces: `CommitTimeoutError`, structural protocols `RuntimeOps`/`CheckpointOps`/`LoadBalancerOps`/`CapacityOps`, and `ScaleTransaction.prepare_replica`, `bootstrap_and_publish`, `begin_drain`, `wait_inflight_zero`, `remove_effective_replica`, `finish_remove_and_destroy`, `wrap_native_sync`, and `restore_donor`.

- [ ] **Step 1: Write deterministic fakes and the first five failing transaction tests**

Create `tests/unit/test_scale_transaction.py` with the following complete test module. The first five tests cover all three required race orders plus known rollback and ambiguous LB timeout; the remaining tests cover gate boundaries, freeze, ownership, restore, and projection consistency.

```python
import asyncio

import pytest

from multi_task_scheduler.orchestration.operation_journal import (
    OperationJournal,
    OperationState,
)
from multi_task_scheduler.orchestration.receipts import (
    BootstrapReceipt,
    CEEffectiveRemoveReceipt,
    DrainingReceipt,
    PreparedReplica,
    RoutableReceipt,
)
from multi_task_scheduler.orchestration.replica_record import ReplicaState
from multi_task_scheduler.orchestration.replica_sync_gate import GateKind, ReplicaSyncGate
from multi_task_scheduler.orchestration.scale_transaction import CommitTimeoutError, ScaleTransaction


class RecordingRuntime:
    def __init__(self, events):
        self.events = events

    async def materialize_hidden(self, operation_id, lease_epoch, replica_id, node_id, gpu_ids):
        self.events.append(("materialize", replica_id, node_id, tuple(gpu_ids)))
        return PreparedReplica(
            replica_id,
            f"http://{replica_id}",
            object(),
            {"replica_id": replica_id},
            operation_id,
            lease_epoch,
        )

    async def health_check(self, replica_id):
        self.events.append(("health", replica_id))

    async def wait_backend_idle(self, replica_id):
        self.events.append(("backend_idle", replica_id))

    async def destroy_temporary(self, replica_id):
        self.events.append(("destroy", replica_id))

    async def mark_dormant(self, replica_id):
        self.events.append(("dormant", replica_id))

    async def wake(self, replica_id):
        self.events.append(("wake", replica_id))

    async def sleep(self, replica_id):
        self.events.append(("sleep", replica_id))


class RecordingCE:
    def __init__(self, events):
        self.events = events
        self.effective = set()
        self.sync_started = asyncio.Event()
        self.allow_sync = asyncio.Event()
        self.allow_sync.set()
        self.fail_at = None
        self.receipt_epoch_delta = 0

    async def current_serving_version(self):
        return 11

    async def pin_snapshot(self, version):
        self.events.append(("pin", version))
        return ("snapshot", version)

    async def unpin_snapshot(self, token):
        self.events.append(("unpin", token[1]))

    async def bootstrap_replica(self, sync_descriptor, version, operation_id, lease_epoch):
        self.events.append(("bootstrap", sync_descriptor["replica_id"], version))
        if self.fail_at == "bootstrap":
            raise RuntimeError("bootstrap failed")
        return BootstrapReceipt(
            sync_descriptor["replica_id"],
            version,
            operation_id,
            lease_epoch + self.receipt_epoch_delta,
            (("worker-0", True),),
        )

    async def add_effective(self, prepared, operation_id):
        self.events.append(("ce_add", prepared.replica_id, operation_id))
        if self.fail_at == "ce_add":
            raise RuntimeError("ce add failed")
        self.effective.add(prepared.replica_id)

    async def remove_if_operation_matches(self, replica_id, operation_id):
        self.events.append(("ce_rollback", replica_id, operation_id))
        self.effective.discard(replica_id)

    async def remove_effective(self, replica_id, operation_id, lease_epoch):
        self.events.append(("ce_remove", replica_id, operation_id))
        self.effective.remove(replica_id)
        return CEEffectiveRemoveReceipt(replica_id, operation_id, lease_epoch)

    async def update_weights(self):
        self.events.append(("sync_start", tuple(sorted(self.effective))))
        self.sync_started.set()
        await self.allow_sync.wait()
        self.events.append(("sync_end", tuple(sorted(self.effective))))
        return "synced"


class RecordingLB:
    def __init__(self, events):
        self.events = events
        self.routable = set()
        self.routing_epoch = 0
        self.commit_error = None

    async def begin_drain(self, operation_id, lease_epoch, replica_id):
        self.routing_epoch += 1
        self.events.append(("begin_drain", replica_id))
        return DrainingReceipt(replica_id, self.routing_epoch, 0, operation_id, lease_epoch)

    async def wait_inflight_zero(self, replica_id, routing_epoch):
        self.events.append(("drained", replica_id))

    async def commit_routable_idempotently(
        self, operation_id, lease_epoch, replica_id, server_address, server_handle
    ):
        self.events.append(("lb_commit", operation_id, replica_id, server_address))
        if self.commit_error is not None:
            raise self.commit_error
        self.routing_epoch += 1
        self.routable.add(replica_id)
        return RoutableReceipt(replica_id, self.routing_epoch, operation_id, lease_epoch)

    async def rollback_hidden_prepare_if_uncommitted(self, operation_id, lease_epoch):
        self.events.append(("lb_rollback", operation_id))

    async def finish_remove(self, replica_id, routing_epoch):
        self.events.append(("lb_remove", replica_id))
        self.routable.remove(replica_id)


class RecordingCapacity:
    def __init__(self, events):
        self.events = events
        self.active = set()
        self.fail_commit = False

    async def prepare_add(self, operation_id, replica_id):
        token = (operation_id, replica_id)
        self.events.append(("capacity_prepare", *token))
        return token

    async def commit_add(self, token):
        self.events.append(("capacity_commit", *token))
        if self.fail_commit:
            raise RuntimeError("capacity commit failed")
        self.active.add(token[1])

    async def rollback_add(self, token):
        self.events.append(("capacity_rollback", *token))
        self.active.discard(token[1])

    async def decrease_on_remove(self, replica_id):
        self.events.append(("capacity_remove", replica_id))
        self.active.remove(replica_id)


def make_stack():
    events = []
    gate = ReplicaSyncGate()
    journal = OperationJournal()
    runtime = RecordingRuntime(events)
    ce = RecordingCE(events)
    lb = RecordingLB(events)
    capacity = RecordingCapacity(events)
    transaction = ScaleTransaction(gate, journal, runtime, ce, lb, capacity)
    return transaction, gate, journal, runtime, ce, lb, capacity, events


async def prepare(transaction, operation_id="add-1", epoch=1, replica_id="r1"):
    return await transaction.prepare_replica(operation_id, epoch, replica_id, "node-1", (0, 1))


async def add_routable(transaction, operation_id="add-1", epoch=1, replica_id="r1"):
    await prepare(transaction, operation_id, epoch, replica_id)
    return await transaction.bootstrap_and_publish(operation_id, epoch)


def event_index(events, name):
    return next(index for index, event in enumerate(events) if event[0] == name)


def test_native_sync_holding_gate_makes_add_wait():
    async def scenario():
        tx, _, _, _, ce, _, _, events = make_stack()
        await prepare(tx)
        ce.allow_sync.clear()
        sync_task = asyncio.create_task(tx.wrap_native_sync("sync-1"))
        await ce.sync_started.wait()
        add_task = asyncio.create_task(tx.bootstrap_and_publish("add-1", 1))
        await asyncio.sleep(0)
        assert not add_task.done()
        assert not any(event[0] == "bootstrap" for event in events)
        ce.allow_sync.set()
        await sync_task
        await add_task
        assert event_index(events, "sync_end") < event_index(events, "bootstrap")

    asyncio.run(scenario())


def test_add_finishes_before_next_native_sync_sees_membership():
    async def scenario():
        tx, _, _, _, _, _, _, events = make_stack()
        await add_routable(tx)
        await tx.wrap_native_sync("sync-1")
        sync_start = next(event for event in events if event[0] == "sync_start")
        assert sync_start[1] == ("r1",)
        assert event_index(events, "ce_add") < event_index(events, "sync_start")

    asyncio.run(scenario())


def test_remove_drain_overlaps_sync_but_ce_remove_waits_for_gate():
    async def scenario():
        tx, _, _, _, ce, _, _, events = make_stack()
        await add_routable(tx)
        await tx.begin_drain("remove-1", 2, "r1")
        await tx.wait_inflight_zero("remove-1", 2)
        ce.allow_sync.clear()
        sync_task = asyncio.create_task(tx.wrap_native_sync("sync-1"))
        await ce.sync_started.wait()
        remove_task = asyncio.create_task(tx.remove_effective_replica("remove-1", 2))
        await asyncio.sleep(0)
        assert not remove_task.done()
        ce.allow_sync.set()
        await sync_task
        await remove_task
        assert event_index(events, "begin_drain") < event_index(events, "sync_start")
        assert event_index(events, "sync_end") < event_index(events, "ce_remove")

    asyncio.run(scenario())


def test_known_add_failure_rolls_back_all_projections_and_releases_gate():
    async def scenario():
        tx, gate, journal, _, _, lb, capacity, events = make_stack()
        await prepare(tx)
        lb.commit_error = RuntimeError("known uncommitted failure")
        with pytest.raises(RuntimeError, match="known uncommitted"):
            await tx.bootstrap_and_publish("add-1", 1)
        assert gate.owner is None
        assert not tx.ce.effective
        assert not lb.routable
        assert not capacity.active
        assert journal.query("add-1").state is OperationState.ROLLED_BACK
        assert any(event[0] == "ce_rollback" for event in events)
        assert any(event[0] == "lb_rollback" for event in events)

    asyncio.run(scenario())


def test_lb_commit_timeout_keeps_queryable_state_and_replay_does_not_blind_delete():
    async def scenario():
        tx, gate, journal, _, ce, lb, capacity, events = make_stack()
        await prepare(tx)
        lb.commit_error = CommitTimeoutError("outcome unknown")
        with pytest.raises(CommitTimeoutError):
            await tx.bootstrap_and_publish("add-1", 1)
        assert gate.owner is None
        assert ce.effective == {"r1"}
        assert capacity.active == {"r1"}
        assert journal.query("add-1").state is OperationState.CE_EFFECTIVE
        assert not any(event[0] in {"ce_rollback", "lb_rollback"} for event in events)

        lb.commit_error = None
        receipt = await tx.bootstrap_and_publish("add-1", 1)
        assert receipt.replica_id == "r1"
        assert journal.query("add-1").state is OperationState.ROUTABLE
        assert not any(event[0] == "ce_rollback" for event in events)

    asyncio.run(scenario())


def test_hidden_create_does_not_acquire_gate():
    async def scenario():
        tx, gate, _, _, _, _, _, _ = make_stack()
        lease = await gate.acquire("sync-1", GateKind.NATIVE_SYNC)
        prepared = await asyncio.wait_for(prepare(tx), timeout=0.1)
        assert prepared.replica_id == "r1"
        assert gate.owner.operation_id == "sync-1"
        await lease.release()

    asyncio.run(scenario())


def test_begin_drain_does_not_acquire_gate():
    async def scenario():
        tx, gate, _, _, _, _, _, _ = make_stack()
        await add_routable(tx)
        lease = await gate.acquire("sync-1", GateKind.NATIVE_SYNC)
        receipt = await asyncio.wait_for(tx.begin_drain("remove-1", 2, "r1"), timeout=0.1)
        assert receipt.inflight == 0
        assert gate.owner.operation_id == "sync-1"
        await lease.release()

    asyncio.run(scenario())


def test_validation_freeze_records_prepared_add_without_ce_or_lb_commit():
    async def scenario():
        tx, gate, journal, _, ce, lb, capacity, events = make_stack()
        await prepare(tx)
        journal.set_validation_freeze(True)
        assert await tx.bootstrap_and_publish("add-1", 1) is None
        assert gate.owner is None
        assert not ce.effective
        assert not lb.routable
        assert not capacity.active
        assert journal.query("add-1").state is OperationState.PREPARED
        assert not any(event[0] in {"bootstrap", "ce_add", "lb_commit"} for event in events)

    asyncio.run(scenario())


def test_validation_freeze_defers_ce_remove_after_drain():
    async def scenario():
        tx, gate, journal, _, ce, lb, _, _ = make_stack()
        await add_routable(tx)
        await tx.begin_drain("remove-1", 2, "r1")
        await tx.wait_inflight_zero("remove-1", 2)
        journal.set_validation_freeze(True)
        assert await tx.remove_effective_replica("remove-1", 2) is None
        assert gate.owner is None
        assert ce.effective == {"r1"}
        assert lb.routable == {"r1"}
        assert journal.query("remove-1").state is OperationState.DRAINING

    asyncio.run(scenario())


def test_stale_bootstrap_receipt_is_fenced_and_rolled_back():
    async def scenario():
        tx, gate, journal, _, ce, lb, capacity, _ = make_stack()
        await prepare(tx)
        ce.receipt_epoch_delta = -1
        with pytest.raises(ValueError, match="bootstrap receipt"):
            await tx.bootstrap_and_publish("add-1", 1)
        assert gate.owner is None
        assert not ce.effective
        assert not lb.routable
        assert not capacity.active
        assert journal.query("add-1").state is OperationState.ROLLED_BACK

    asyncio.run(scenario())


def test_finish_remove_uses_runtime_and_lb_without_calling_ce_again():
    async def scenario():
        tx, _, journal, _, ce, lb, _, events = make_stack()
        await add_routable(tx)
        await tx.begin_drain("remove-1", 2, "r1")
        await tx.wait_inflight_zero("remove-1", 2)
        await tx.remove_effective_replica("remove-1", 2)
        ce_event_count = len([event for event in events if event[0].startswith("ce_")])
        receipt = await tx.finish_remove_and_destroy("remove-1", 2, is_donor=False)
        assert receipt.replica_id == "r1"
        assert journal.query("remove-1").state is OperationState.DESTROYED
        assert "r1" not in lb.routable
        assert len([event for event in events if event[0].startswith("ce_")]) == ce_event_count
        assert any(event[0] == "destroy" for event in events)

    asyncio.run(scenario())


def test_donor_restore_reuses_runtime_and_never_materializes_or_destroys_it():
    async def scenario():
        tx, _, journal, _, _, lb, _, events = make_stack()
        await add_routable(tx)
        await tx.begin_drain("remove-1", 2, "r1")
        await tx.wait_inflight_zero("remove-1", 2)
        await tx.remove_effective_replica("remove-1", 2)
        await tx.finish_remove_and_destroy("remove-1", 2, is_donor=True)
        events.clear()

        receipt = await tx.restore_donor("restore-1", 3, "r1")
        assert receipt.replica_id == "r1"
        assert journal.query("restore-1").state is OperationState.ROUTABLE
        assert "r1" in lb.routable
        assert any(event[0] == "wake" for event in events)
        assert not any(event[0] in {"materialize", "destroy"} for event in events)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_point", ["bootstrap", "ce_add", "capacity", "lb"])
def test_known_add_failures_never_leave_ce_and_lb_projections_disagree(failure_point):
    async def scenario():
        tx, _, _, _, ce, lb, capacity, _ = make_stack()
        await prepare(tx)
        if failure_point in {"bootstrap", "ce_add"}:
            ce.fail_at = failure_point
        elif failure_point == "capacity":
            capacity.fail_commit = True
        else:
            lb.commit_error = RuntimeError("known LB failure")

        with pytest.raises(RuntimeError):
            await tx.bootstrap_and_publish("add-1", 1)
        assert ("r1" in ce.effective) is ("r1" in lb.routable)
        assert "r1" not in ce.effective

    asyncio.run(scenario())
```

- [ ] **Step 2: Run the transaction tests and verify the import fails**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/unit/test_scale_transaction.py
```

Expected: FAIL during collection because `scale_transaction.py` does not exist.

- [ ] **Step 3: Implement protocols, ADD, REMOVE, native sync, and donor restore**

Create `src/multi_task_scheduler/orchestration/scale_transaction.py`:

```python
"""Pure orchestration of injected standalone replica lifecycle primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .operation_journal import OperationJournal, OperationState, OperationType
from .receipts import (
    ActiveReceipt,
    BootstrapReceipt,
    CEEffectiveRemoveReceipt,
    DrainingReceipt,
    PreparedReplica,
    RoutableReceipt,
)
from .replica_record import ReplicaRecord, ReplicaState
from .replica_sync_gate import GateKind, ReplicaSyncGate


class CommitTimeoutError(TimeoutError):
    """The LB commit outcome is unknown and must be resolved by query or replay."""


class RuntimeOps(Protocol):
    async def materialize_hidden(
        self,
        operation_id: str,
        lease_epoch: int,
        replica_id: str,
        node_id: str,
        gpu_ids: Sequence[int],
    ) -> PreparedReplica: ...

    async def health_check(self, replica_id: str) -> None: ...

    async def wait_backend_idle(self, replica_id: str) -> None: ...

    async def destroy_temporary(self, replica_id: str) -> None: ...

    async def mark_dormant(self, replica_id: str) -> None: ...

    async def wake(self, replica_id: str) -> None: ...

    async def sleep(self, replica_id: str) -> None: ...


class CheckpointOps(Protocol):
    async def current_serving_version(self) -> int: ...

    async def pin_snapshot(self, version: int) -> object: ...

    async def unpin_snapshot(self, token: object) -> None: ...

    async def bootstrap_replica(
        self,
        sync_descriptor: object,
        version: int,
        operation_id: str,
        lease_epoch: int,
    ) -> BootstrapReceipt: ...

    async def add_effective(self, prepared: PreparedReplica, operation_id: str) -> None: ...

    async def remove_if_operation_matches(self, replica_id: str, operation_id: str) -> None: ...

    async def remove_effective(
        self, replica_id: str, operation_id: str, lease_epoch: int
    ) -> CEEffectiveRemoveReceipt: ...

    async def update_weights(self) -> object: ...


class LoadBalancerOps(Protocol):
    async def begin_drain(
        self, operation_id: str, lease_epoch: int, replica_id: str
    ) -> DrainingReceipt: ...

    async def wait_inflight_zero(self, replica_id: str, routing_epoch: int) -> None: ...

    async def commit_routable_idempotently(
        self,
        operation_id: str,
        lease_epoch: int,
        replica_id: str,
        server_address: str,
        server_handle: object,
    ) -> RoutableReceipt: ...

    async def rollback_hidden_prepare_if_uncommitted(
        self, operation_id: str, lease_epoch: int
    ) -> None: ...

    async def finish_remove(self, replica_id: str, routing_epoch: int) -> None: ...


class CapacityOps(Protocol):
    async def prepare_add(self, operation_id: str, replica_id: str) -> object: ...

    async def commit_add(self, token: object) -> None: ...

    async def rollback_add(self, token: object) -> None: ...

    async def decrease_on_remove(self, replica_id: str) -> None: ...


@dataclass
class _PreparedContext:
    prepared: PreparedReplica
    capacity_token: object


class ScaleTransaction:
    def __init__(
        self,
        gate: ReplicaSyncGate,
        journal: OperationJournal,
        runtime: RuntimeOps,
        ce: CheckpointOps,
        lb: LoadBalancerOps,
        capacity: CapacityOps,
    ) -> None:
        self.gate = gate
        self.journal = journal
        self.runtime = runtime
        self.ce = ce
        self.lb = lb
        self.capacity = capacity
        self.replicas: dict[str, ReplicaRecord] = {}
        self._prepared: dict[str, _PreparedContext] = {}
        self._prepared_by_replica: dict[str, PreparedReplica] = {}
        self._capacity_committed: set[str] = set()
        self._routable_receipts: dict[str, RoutableReceipt] = {}
        self._draining_receipts: dict[str, DrainingReceipt] = {}
        self._remove_receipts: dict[str, CEEffectiveRemoveReceipt] = {}
        self._drained_operations: set[str] = set()

    async def prepare_replica(
        self,
        operation_id: str,
        lease_epoch: int,
        replica_id: str,
        node_id: str,
        gpu_ids: Sequence[int],
    ) -> PreparedReplica:
        operation = self.journal.begin(operation_id, lease_epoch, replica_id, OperationType.ADD)
        if operation.state is OperationState.PREPARED:
            return self._prepared[operation_id].prepared
        if operation.state is not OperationState.ACCEPTED:
            raise RuntimeError(f"operation {operation_id} cannot prepare from {operation.state.value}")

        record = ReplicaRecord(replica_id, operation_id=operation_id, lease_epoch=lease_epoch)
        self.replicas[replica_id] = record
        token = await self.capacity.prepare_add(operation_id, replica_id)
        materialized = False
        try:
            prepared = await self.runtime.materialize_hidden(
                operation_id, lease_epoch, replica_id, node_id, gpu_ids
            )
            materialized = True
            if (
                prepared.replica_id != replica_id
                or prepared.operation_id != operation_id
                or prepared.lease_epoch != lease_epoch
            ):
                raise ValueError("prepared replica receipt does not match operation identity")
            await self.runtime.health_check(replica_id)
        except Exception:
            if materialized:
                await self.runtime.destroy_temporary(replica_id)
            record.transition_to(ReplicaState.FAILED)
            await self.capacity.rollback_add(token)
            self.journal.transition(operation_id, OperationState.FAILED)
            raise

        record.transition_to(ReplicaState.HIDDEN)
        self._prepared[operation_id] = _PreparedContext(prepared, token)
        self._prepared_by_replica[replica_id] = prepared
        self.journal.transition(operation_id, OperationState.PREPARED)
        return prepared

    async def bootstrap_and_publish(
        self, operation_id: str, lease_epoch: int
    ) -> RoutableReceipt | None:
        operation = self.journal.require(operation_id, lease_epoch)
        if operation.state is OperationState.ROUTABLE:
            return self._routable_receipts[operation_id]
        if self.journal.validation_frozen:
            return None
        if operation.state not in {OperationState.PREPARED, OperationState.CE_EFFECTIVE}:
            raise RuntimeError(f"operation {operation_id} cannot publish from {operation.state.value}")

        context = self._prepared[operation_id]
        record = self.replicas[operation.replica_id]
        lease = await self.gate.acquire(operation_id, GateKind.ADD)
        async with lease:
            if operation.state is OperationState.CE_EFFECTIVE:
                return await self._commit_routable(operation, context, record)

            snapshot_token = None
            try:
                record.transition_to(ReplicaState.BOOTSTRAP_PENDING)
                record.transition_to(ReplicaState.BOOTSTRAPPING)
                self.journal.transition(operation_id, OperationState.BOOTSTRAPPING)
                version = await self.ce.current_serving_version()
                record.bootstrap_target_version = version
                snapshot_token = await self.ce.pin_snapshot(version)
                receipt = await lease.guard(
                    self.ce.bootstrap_replica,
                    context.prepared.sync_descriptor,
                    version,
                    operation_id,
                    lease_epoch,
                )
                if (
                    receipt.replica_id != record.replica_id
                    or receipt.weight_version != version
                    or receipt.operation_id != operation_id
                    or receipt.lease_epoch != lease_epoch
                    or not receipt.receiver_statuses
                    or not all(complete for _, complete in receipt.receiver_statuses)
                ):
                    raise ValueError("bootstrap receipt does not match target operation and receivers")
                record.last_synced_weight_version = version
                record.transition_to(ReplicaState.SYNCED)
                await lease.guard(self.ce.add_effective, context.prepared, operation_id)
                record.transition_to(ReplicaState.CE_EFFECTIVE)
                self.journal.transition(operation_id, OperationState.CE_EFFECTIVE)
                return await self._commit_routable(operation, context, record)
            except CommitTimeoutError:
                raise
            except Exception:
                await self._rollback_add(operation_id, context, record)
                raise
            finally:
                if snapshot_token is not None:
                    await self.ce.unpin_snapshot(snapshot_token)

    async def _commit_routable(self, operation, context, record) -> RoutableReceipt:
        await self.runtime.health_check(operation.replica_id)
        if operation.operation_id not in self._capacity_committed:
            await self.capacity.commit_add(context.capacity_token)
            self._capacity_committed.add(operation.operation_id)
        receipt = await self.lb.commit_routable_idempotently(
            operation.operation_id,
            operation.lease_epoch,
            operation.replica_id,
            context.prepared.server_address,
            context.prepared.server_handle,
        )
        if (
            receipt.replica_id != operation.replica_id
            or receipt.operation_id != operation.operation_id
            or receipt.lease_epoch != operation.lease_epoch
        ):
            raise ValueError("routable receipt does not match operation identity")
        record.routing_epoch = receipt.routing_epoch
        record.transition_to(ReplicaState.ROUTABLE)
        self.journal.transition(operation.operation_id, OperationState.ROUTABLE)
        self._routable_receipts[operation.operation_id] = receipt
        return receipt

    async def _rollback_add(self, operation_id, context, record) -> None:
        operation = self.journal.query(operation_id)
        await self.lb.rollback_hidden_prepare_if_uncommitted(operation_id, operation.lease_epoch)
        await self.ce.remove_if_operation_matches(record.replica_id, operation_id)
        await self.capacity.rollback_add(context.capacity_token)
        self._capacity_committed.discard(operation_id)
        if record.state not in {ReplicaState.HIDDEN, ReplicaState.FAILED}:
            record.transition_to(ReplicaState.HIDDEN)
        self.journal.transition(operation_id, OperationState.ROLLED_BACK)

    async def begin_drain(
        self, operation_id: str, lease_epoch: int, replica_id: str
    ) -> DrainingReceipt:
        operation = self.journal.begin(operation_id, lease_epoch, replica_id, OperationType.REMOVE)
        if operation.state is OperationState.DRAINING:
            return self._draining_receipts[operation_id]
        if operation.state is not OperationState.ACCEPTED:
            raise RuntimeError(f"operation {operation_id} cannot drain from {operation.state.value}")
        record = self.replicas[replica_id]
        receipt = await self.lb.begin_drain(operation_id, lease_epoch, replica_id)
        if (
            receipt.replica_id != replica_id
            or receipt.operation_id != operation_id
            or receipt.lease_epoch != lease_epoch
        ):
            raise ValueError("draining receipt does not match operation identity")
        record.routing_epoch = receipt.routing_epoch
        record.operation_id = operation_id
        record.lease_epoch = lease_epoch
        record.transition_to(ReplicaState.DRAINING)
        self.journal.transition(operation_id, OperationState.DRAINING)
        self._draining_receipts[operation_id] = receipt
        return receipt

    async def wait_inflight_zero(self, operation_id: str, lease_epoch: int) -> None:
        operation = self.journal.require(operation_id, lease_epoch)
        if operation.state is not OperationState.DRAINING:
            raise RuntimeError(f"operation {operation_id} is not draining")
        draining = self._draining_receipts[operation_id]
        await self.lb.wait_inflight_zero(operation.replica_id, draining.routing_epoch)
        await self.runtime.wait_backend_idle(operation.replica_id)
        self._drained_operations.add(operation_id)

    async def remove_effective_replica(
        self, operation_id: str, lease_epoch: int
    ) -> CEEffectiveRemoveReceipt | None:
        operation = self.journal.require(operation_id, lease_epoch)
        if operation.state is OperationState.CE_REMOVED:
            return self._remove_receipts[operation_id]
        if self.journal.validation_frozen:
            return None
        if operation.state is not OperationState.DRAINING:
            raise RuntimeError(f"operation {operation_id} cannot remove from {operation.state.value}")
        if operation_id not in self._drained_operations:
            raise RuntimeError(f"operation {operation_id} has not completed natural drain")

        lease = await self.gate.acquire(operation_id, GateKind.REMOVE)
        async with lease:
            receipt = await lease.guard(
                self.ce.remove_effective,
                operation.replica_id,
                operation_id,
                lease_epoch,
            )
            if (
                receipt.replica_id != operation.replica_id
                or receipt.operation_id != operation_id
                or receipt.lease_epoch != lease_epoch
            ):
                raise ValueError("CE remove receipt does not match operation identity")
            self.replicas[operation.replica_id].transition_to(ReplicaState.RETIRING)
            self.journal.transition(operation_id, OperationState.CE_REMOVED)
            self._remove_receipts[operation_id] = receipt
            return receipt

    async def finish_remove_and_destroy(
        self, operation_id: str, lease_epoch: int, *, is_donor: bool
    ) -> ActiveReceipt | None:
        operation = self.journal.require(operation_id, lease_epoch)
        if self.journal.validation_frozen:
            return None
        if operation.state not in {OperationState.CE_REMOVED, OperationState.DESTROYED, OperationState.DORMANT}:
            raise RuntimeError(f"operation {operation_id} cannot finish from {operation.state.value}")
        if operation.state in {OperationState.DESTROYED, OperationState.DORMANT}:
            return ActiveReceipt(operation.replica_id, operation.lease_epoch)

        draining = self._draining_receipts[operation_id]
        await self.lb.finish_remove(operation.replica_id, draining.routing_epoch)
        await self.capacity.decrease_on_remove(operation.replica_id)
        record = self.replicas[operation.replica_id]
        if is_donor:
            await self.runtime.sleep(operation.replica_id)
            await self.runtime.mark_dormant(operation.replica_id)
            record.transition_to(ReplicaState.DORMANT)
            self.journal.transition(operation_id, OperationState.DORMANT)
        else:
            await self.runtime.destroy_temporary(operation.replica_id)
            record.transition_to(ReplicaState.DESTROYED)
            self.journal.transition(operation_id, OperationState.DESTROYED)
        return ActiveReceipt(operation.replica_id, operation.lease_epoch)

    async def wrap_native_sync(self, operation_id: str) -> object:
        lease = await self.gate.acquire(operation_id, GateKind.NATIVE_SYNC)
        async with lease:
            return await lease.guard(self.ce.update_weights)

    async def restore_donor(
        self, operation_id: str, lease_epoch: int, replica_id: str
    ) -> RoutableReceipt | None:
        operation = self.journal.begin(operation_id, lease_epoch, replica_id, OperationType.RESTORE)
        if operation.state is OperationState.ROUTABLE:
            return self._routable_receipts[operation_id]
        if self.journal.validation_frozen:
            return None
        record = self.replicas[replica_id]
        if operation.state is OperationState.CE_EFFECTIVE:
            prepared = self._prepared_by_replica[replica_id]
            retry_lease = await self.gate.acquire(operation_id, GateKind.RESTORE)
            async with retry_lease:
                await self.runtime.wake(replica_id)
                await self.runtime.health_check(replica_id)
                routable = await self.lb.commit_routable_idempotently(
                    operation_id,
                    lease_epoch,
                    replica_id,
                    prepared.server_address,
                    prepared.server_handle,
                )
                if (
                    routable.replica_id != replica_id
                    or routable.operation_id != operation_id
                    or routable.lease_epoch != lease_epoch
                ):
                    raise ValueError("restore routable receipt does not match operation identity")
                record.routing_epoch = routable.routing_epoch
                record.transition_to(ReplicaState.ROUTABLE)
                self.journal.transition(operation_id, OperationState.ROUTABLE)
                self._routable_receipts[operation_id] = routable
                return routable
        if operation.state is not OperationState.ACCEPTED or record.state is not ReplicaState.DORMANT:
            raise RuntimeError(f"donor {replica_id} is not dormant")

        capacity_token = await self.capacity.prepare_add(operation_id, replica_id)
        existing = self._prepared_by_replica[replica_id]
        prepared = PreparedReplica(
            replica_id,
            existing.server_address,
            existing.server_handle,
            existing.sync_descriptor,
            operation_id,
            lease_epoch,
        )
        self._prepared_by_replica[replica_id] = prepared
        lease = await self.gate.acquire(operation_id, GateKind.RESTORE)
        snapshot_token = None
        woke = False
        async with lease:
            try:
                record.operation_id = operation_id
                record.lease_epoch = lease_epoch
                record.transition_to(ReplicaState.BOOTSTRAP_PENDING)
                record.transition_to(ReplicaState.BOOTSTRAPPING)
                self.journal.transition(operation_id, OperationState.BOOTSTRAPPING)
                version = await self.ce.current_serving_version()
                record.bootstrap_target_version = version
                snapshot_token = await self.ce.pin_snapshot(version)
                receipt = await lease.guard(
                    self.ce.bootstrap_replica,
                    prepared.sync_descriptor,
                    version,
                    operation_id,
                    lease_epoch,
                )
                if (
                    receipt.replica_id != replica_id
                    or receipt.weight_version != version
                    or receipt.operation_id != operation_id
                    or receipt.lease_epoch != lease_epoch
                    or not receipt.receiver_statuses
                    or not all(complete for _, complete in receipt.receiver_statuses)
                ):
                    raise ValueError("restore bootstrap receipt does not match donor operation")
                record.last_synced_weight_version = version
                record.transition_to(ReplicaState.SYNCED)
                await lease.guard(self.ce.add_effective, prepared, operation_id)
                record.transition_to(ReplicaState.CE_EFFECTIVE)
                self.journal.transition(operation_id, OperationState.CE_EFFECTIVE)
                await self.capacity.commit_add(capacity_token)
                await self.runtime.wake(replica_id)
                woke = True
                await self.runtime.health_check(replica_id)
                routable = await self.lb.commit_routable_idempotently(
                    operation_id,
                    lease_epoch,
                    replica_id,
                    prepared.server_address,
                    prepared.server_handle,
                )
                if (
                    routable.replica_id != replica_id
                    or routable.operation_id != operation_id
                    or routable.lease_epoch != lease_epoch
                ):
                    raise ValueError("restore routable receipt does not match operation identity")
                record.routing_epoch = routable.routing_epoch
                record.transition_to(ReplicaState.ROUTABLE)
                self.journal.transition(operation_id, OperationState.ROUTABLE)
                self._routable_receipts[operation_id] = routable
                return routable
            except CommitTimeoutError:
                raise
            except Exception:
                await self.lb.rollback_hidden_prepare_if_uncommitted(operation_id, lease_epoch)
                await self.ce.remove_if_operation_matches(replica_id, operation_id)
                await self.capacity.rollback_add(capacity_token)
                if woke:
                    await self.runtime.sleep(replica_id)
                if record.state is not ReplicaState.DORMANT:
                    record.transition_to(ReplicaState.DORMANT)
                self.journal.transition(operation_id, OperationState.ROLLED_BACK)
                raise
            finally:
                if snapshot_token is not None:
                    await self.ce.unpin_snapshot(snapshot_token)
```

- [ ] **Step 4: Run the transaction tests and inspect the first behavioral failure**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/unit/test_scale_transaction.py -x
```

Expected: PASS. If a test fails, stop at the first failure and fix the implementation while preserving the exact interface and ordering in this task. Do not weaken an assertion to obtain green output.

- [ ] **Step 5: Run the full mock-only unit suite**

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/unit
```

Expected: all existing and new unit tests pass. Report new tests as mock-only orchestration coverage; do not report distributed or GPU validation.

- [ ] **Step 6: Check dependency and import boundaries**

Run:

```powershell
rg -n "^(from|import) (ray|verl|torch|vllm)" src/multi_task_scheduler/orchestration tests/unit/test_replica_sync_gate.py tests/unit/test_replica_record.py tests/unit/test_operation_journal.py tests/unit/test_receipts.py tests/unit/test_scale_transaction.py
python -c "import sys; import multi_task_scheduler.orchestration.scale_transaction; assert not any(name == 'ray' or name.startswith('verl') for name in sys.modules)"
```

Expected: `rg` returns no matches; the Python process exits 0 without importing Ray or verl.

- [ ] **Step 7: Commit the independently testable orchestration core**

Run only after separate Git authorization:

```powershell
git add src/multi_task_scheduler/orchestration/scale_transaction.py tests/unit/test_scale_transaction.py
git commit -m "feat: add standalone scale transaction orchestration"
```

---

## Final Verification and Review Gate

- [ ] Run the repository's complete unit suite with the same dependency-light environment:

```powershell
$env:PYTHONPATH="$PWD/src"
$env:PYTHONDONTWRITEBYTECODE="1"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
python -m pytest -q -p no:cacheprovider tests/unit
```

- [ ] Confirm the scope remains A+B+C only:

```powershell
git status --short
git diff --stat
git diff -- src/multi_task_scheduler/__init__.py src/multi_task_scheduler/integration src/multi_task_scheduler/checkpoint src/multi_task_scheduler/rollout src/multi_task_scheduler/scheduler
```

Expected: only the orchestration modules and their unit tests are implementation changes; the final diff command prints nothing. Existing untracked documentation remains untouched.

- [ ] Record the result by layer: pure-Python mock unit tests executed; CPU Ray, native verl imports, vLLM, NCCL, GPU memory release, cross-job GS leases, and end-to-end training not executed.

- [ ] Stop for review before Slice D. Slice D must separately decide how `ReplicaSyncGate` is owned by the Trainer, how TaskRunner commands remain reachable during `run()`, and how the injected protocols bind to native CE/LB/runtime methods.
