"""Manager-owned replica lifecycle record from simplified design §5/§6.4.

The record stores lifecycle identity and revisions only. Fine-grained bootstrap,
CE, route, sleep and destroy work belongs to the operation journal/evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .contracts import OperationError, PlacementSpec, ReplicaKey


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
    key: ReplicaKey
    kind: ReplicaKind
    state: ReplicaState
    revision: int
    placement: PlacementSpec
    lease_id: str | None = None
    active_operation_id: str | None = None
    lease_epoch: int | None = None
    loaded_version: int | None = None
    last_error: OperationError | None = None

    def __post_init__(self) -> None:
        self.kind = ReplicaKind(self.kind)
        self.state = ReplicaState(self.state)
        if self.revision < 0:
            raise ValueError("revision must be nonnegative")
        if self.lease_epoch is not None and self.lease_epoch < 0:
            raise ValueError("lease_epoch must be nonnegative")
        if self.loaded_version is not None and self.loaded_version < 0:
            raise ValueError("loaded_version must be nonnegative")
        self._validate_kind_state(self.state)

    def _validate_kind_state(self, state: ReplicaState) -> None:
        if self.kind is ReplicaKind.BORROWED and state in {
            ReplicaState.DORMANT,
            ReplicaState.RESTORING,
        }:
            raise IllegalReplicaTransitionError(
                f"borrowed replica {self.key.replica_id} cannot enter {state.value}"
            )
        if self.kind is ReplicaKind.NATIVE and state is ReplicaState.DESTROYED:
            raise IllegalReplicaTransitionError(
                f"native replica {self.key.replica_id} must sleep, never destroy"
            )

    def transition_to(self, new_state: ReplicaState) -> None:
        new_state = ReplicaState(new_state)
        if new_state is self.state:
            return
        if new_state not in _ALLOWED[self.state]:
            raise IllegalReplicaTransitionError(
                f"illegal replica transition for {self.key.replica_id}: "
                f"{self.state.value} -> {new_state.value}"
            )
        self._validate_kind_state(new_state)
        self.state = new_state
        self.revision += 1
