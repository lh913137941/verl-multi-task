"""Replica lifecycle metadata independent of distributed runtime handles.

``ReplicaState`` follows the two lifecycles of the fusion design (section 3.4):
the borrowed path (``CREATING -> HIDDEN -> BOOTSTRAPPING -> CE_EFFECTIVE ->
ACTIVE -> DRAINING -> CE_REMOVED -> DESTROYING -> DESTROYED``) and the donor
sleep/wake path (``... -> CE_REMOVED -> SLEEPING -> DORMANT -> WAKING_WEIGHTS ->
BOOTSTRAPPING -> ...``). Rollback returns a borrowed target to ``HIDDEN`` and a
donor target to ``DORMANT``; corrupt transfer/cleanup ends in ``QUARANTINED``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .replica_sync_gate import GateOwner


class ReplicaKind(str, Enum):
    NATIVE = "NATIVE"
    BORROWED = "BORROWED"


class ReplicaState(str, Enum):
    CREATING = "CREATING"
    HIDDEN = "HIDDEN"
    FAILED_HIDDEN = "FAILED_HIDDEN"
    BOOTSTRAPPING = "BOOTSTRAPPING"
    CE_EFFECTIVE = "CE_EFFECTIVE"
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    CE_REMOVED = "CE_REMOVED"
    DESTROYING = "DESTROYING"
    DESTROYED = "DESTROYED"
    SLEEPING = "SLEEPING"
    DORMANT = "DORMANT"
    WAKING_WEIGHTS = "WAKING_WEIGHTS"
    QUARANTINED = "QUARANTINED"


class IllegalReplicaTransitionError(RuntimeError):
    """Raised when orchestration skips a required replica lifecycle state."""


# Union of the borrowed and donor machines, including rollback edges. The
# kind-specific terminal choice (DESTROYED vs DORMANT) is enforced by the
# transaction layer; this table only rejects genuinely illegal jumps.
_ALLOWED = {
    ReplicaState.CREATING: {ReplicaState.HIDDEN, ReplicaState.FAILED_HIDDEN},
    ReplicaState.HIDDEN: {
        ReplicaState.BOOTSTRAPPING,
        ReplicaState.DESTROYING,
        ReplicaState.QUARANTINED,
    },
    ReplicaState.FAILED_HIDDEN: {
        ReplicaState.DESTROYING,
        ReplicaState.QUARANTINED,
    },
    ReplicaState.BOOTSTRAPPING: {
        ReplicaState.CE_EFFECTIVE,
        ReplicaState.HIDDEN,
        ReplicaState.DORMANT,
        ReplicaState.QUARANTINED,
    },
    ReplicaState.CE_EFFECTIVE: {
        ReplicaState.ACTIVE,
        ReplicaState.HIDDEN,
        ReplicaState.DORMANT,
        ReplicaState.QUARANTINED,
    },
    ReplicaState.ACTIVE: {ReplicaState.DRAINING},
    ReplicaState.DRAINING: {ReplicaState.CE_REMOVED, ReplicaState.QUARANTINED},
    ReplicaState.CE_REMOVED: {
        ReplicaState.DESTROYING,
        ReplicaState.SLEEPING,
        ReplicaState.QUARANTINED,
    },
    ReplicaState.DESTROYING: {ReplicaState.DESTROYED, ReplicaState.QUARANTINED},
    ReplicaState.DESTROYED: set(),
    ReplicaState.SLEEPING: {ReplicaState.DORMANT, ReplicaState.QUARANTINED},
    ReplicaState.DORMANT: {ReplicaState.WAKING_WEIGHTS, ReplicaState.QUARANTINED},
    ReplicaState.WAKING_WEIGHTS: {ReplicaState.BOOTSTRAPPING, ReplicaState.QUARANTINED},
    ReplicaState.QUARANTINED: set(),
}


@dataclass
class ReplicaRecord:
    replica_id: str
    kind: ReplicaKind = ReplicaKind.BORROWED
    state: ReplicaState | None = None
    operation_id: str | None = None
    lease_epoch: int = 0
    runtime_epoch: int = 0
    routing_epoch: int | None = None
    last_synced_weight_version: int | None = None
    bootstrap_target_version: int | None = None
    replica_sync_gate_owner: GateOwner | None = None
    lifecycle_pins: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.kind = ReplicaKind(self.kind)
        if self.state is None:
            # A donor native replica is already serving; only a borrowed
            # target starts life in CREATING (section 3.4).
            self.state = (
                ReplicaState.ACTIVE
                if self.kind is ReplicaKind.NATIVE
                else ReplicaState.CREATING
            )
        else:
            self.state = ReplicaState(self.state)

    def transition_to(self, new_state: ReplicaState) -> None:
        new_state = ReplicaState(new_state)
        if new_state is self.state:
            return
        if new_state not in _ALLOWED[self.state]:
            raise IllegalReplicaTransitionError(
                f"illegal replica transition for {self.replica_id}: "
                f"{self.state.value} -> {new_state.value}"
            )
        self.state = new_state
