"""Replica lifecycle metadata independent of distributed runtime handles.

The public lifecycle is intentionally small. Fine-grained bootstrap, CE, route,
sleep and destroy steps belong in the operation journal and evidence records,
not in a second mutable lifecycle state machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ReplicaKind(str, Enum):
    NATIVE = "NATIVE"
    BORROWED = "BORROWED"


class ReplicaState(str, Enum):
    PREPARING = "PREPARING"
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    DETACHED = "DETACHED"
    DORMANT = "DORMANT"
    RESTORING = "RESTORING"
    DESTROYED = "DESTROYED"
    QUARANTINED = "QUARANTINED"


class IllegalReplicaTransitionError(RuntimeError):
    """Raised when orchestration skips a required evidence-backed state edge."""


_ALLOWED = {
    ReplicaState.PREPARING: {
        ReplicaState.ACTIVE,
        ReplicaState.DESTROYED,
        ReplicaState.QUARANTINED,
    },
    ReplicaState.ACTIVE: {ReplicaState.DRAINING},
    ReplicaState.DRAINING: {
        ReplicaState.DETACHED,
        ReplicaState.ACTIVE,
        ReplicaState.QUARANTINED,
    },
    ReplicaState.DETACHED: {
        ReplicaState.DORMANT,
        ReplicaState.DESTROYED,
        ReplicaState.QUARANTINED,
    },
    ReplicaState.DORMANT: {
        ReplicaState.RESTORING,
        ReplicaState.QUARANTINED,
    },
    ReplicaState.RESTORING: {
        ReplicaState.ACTIVE,
        ReplicaState.DORMANT,
        ReplicaState.QUARANTINED,
    },
    ReplicaState.DESTROYED: set(),
    ReplicaState.QUARANTINED: set(),
}


@dataclass
class ReplicaRecord:
    replica_id: str
    kind: ReplicaKind = ReplicaKind.BORROWED
    state: ReplicaState | None = None
    revision: int = 0
    operation_id: str | None = None
    lease_id: str | None = None
    lease_epoch: int = 0
    runtime_epoch: int = 0
    routing_epoch: int | None = None
    loaded_version: int | None = None
    last_error: object | None = None
    lifecycle_pins: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.kind = ReplicaKind(self.kind)
        if not self.replica_id:
            raise ValueError("replica_id must be nonempty")
        if self.revision < 0 or self.lease_epoch < 0 or self.runtime_epoch < 0:
            raise ValueError("revision and epochs must be nonnegative")
        if self.state is None:
            self.state = (
                ReplicaState.ACTIVE
                if self.kind is ReplicaKind.NATIVE
                else ReplicaState.PREPARING
            )
        else:
            self.state = ReplicaState(self.state)
        self._validate_kind_state(self.state)

    def _validate_kind_state(self, state: ReplicaState) -> None:
        if self.kind is ReplicaKind.BORROWED and state in {
            ReplicaState.DORMANT,
            ReplicaState.RESTORING,
        }:
            raise IllegalReplicaTransitionError(
                f"borrowed replica {self.replica_id} cannot enter {state.value}"
            )
        if self.kind is ReplicaKind.NATIVE and state is ReplicaState.DESTROYED:
            raise IllegalReplicaTransitionError(
                f"native replica {self.replica_id} must sleep, never destroy"
            )

    def transition_to(self, new_state: ReplicaState) -> None:
        new_state = ReplicaState(new_state)
        if new_state is self.state:
            return
        if new_state not in _ALLOWED[self.state]:
            raise IllegalReplicaTransitionError(
                f"illegal replica transition for {self.replica_id}: "
                f"{self.state.value} -> {new_state.value}"
            )
        self._validate_kind_state(new_state)
        self.state = new_state
        self.revision += 1
