"""Replayable operation metadata for scale transactions.

Tracks each control operation's fine-grained phase (section 3.4) and exposes
its coarse result state (section 3.2 ``OperationResult.state``). The same
``(operation_id, lease_epoch)`` is an idempotent replay; an older epoch is
rejected; conflicting identity or payload digest is a CONFLICT. This provides
the "same ID same digest returns existing status, different digest returns
CONFLICT" contract without claiming network exactly-once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class OperationType(str, Enum):
    DONATE = "DONATE"
    ADD = "ADD"
    REMOVE = "REMOVE"
    RESTORE = "RESTORE"


class OperationPhase(str, Enum):
    """Fine-grained task-internal progress (section 3.4)."""

    ACCEPTED = "ACCEPTED"
    PREPARING = "PREPARING"
    DRAINING = "DRAINING"
    WAIT_GATE = "WAIT_GATE"
    APPLYING = "APPLYING"
    COMMITTED = "COMMITTED"
    ROLLED_BACK = "ROLLED_BACK"
    RECONCILING = "RECONCILING"
    QUARANTINED = "QUARANTINED"


class OperationState(str, Enum):
    """Coarse result state reported in ``OperationResult.state`` (section 3.2)."""

    ACCEPTED = "ACCEPTED"
    RUNNING = "RUNNING"
    COMMITTED = "COMMITTED"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"
    RECONCILING = "RECONCILING"
    QUARANTINED = "QUARANTINED"


class ExpiredLeaseError(RuntimeError):
    """A command does not match the journal's current lease epoch."""


class OperationIdentityError(RuntimeError):
    """An operation ID is replayed with conflicting immutable or digest fields."""


class IllegalOperationTransitionError(RuntimeError):
    """An operation skips a required transaction phase."""


_PHASE_ALLOWED = {
    OperationPhase.ACCEPTED: {OperationPhase.PREPARING, OperationPhase.DRAINING},
    OperationPhase.PREPARING: {
        OperationPhase.WAIT_GATE,
        OperationPhase.ROLLED_BACK,
        OperationPhase.QUARANTINED,
    },
    OperationPhase.DRAINING: {
        OperationPhase.WAIT_GATE,
        OperationPhase.ROLLED_BACK,
        OperationPhase.QUARANTINED,
    },
    OperationPhase.WAIT_GATE: {
        OperationPhase.APPLYING,
        OperationPhase.ROLLED_BACK,
        OperationPhase.QUARANTINED,
    },
    OperationPhase.APPLYING: {
        OperationPhase.COMMITTED,
        OperationPhase.ROLLED_BACK,
        OperationPhase.RECONCILING,
        OperationPhase.QUARANTINED,
    },
    OperationPhase.COMMITTED: set(),
    OperationPhase.ROLLED_BACK: set(),
    OperationPhase.RECONCILING: {
        OperationPhase.COMMITTED,
        OperationPhase.ROLLED_BACK,
        OperationPhase.QUARANTINED,
    },
    OperationPhase.QUARANTINED: set(),
}

_RUNNING_PHASES = {
    OperationPhase.PREPARING,
    OperationPhase.DRAINING,
    OperationPhase.WAIT_GATE,
    OperationPhase.APPLYING,
}

_PHASE_TO_STATE = {
    OperationPhase.ACCEPTED: OperationState.ACCEPTED,
    OperationPhase.COMMITTED: OperationState.COMMITTED,
    OperationPhase.ROLLED_BACK: OperationState.ROLLED_BACK,
    OperationPhase.RECONCILING: OperationState.RECONCILING,
    OperationPhase.QUARANTINED: OperationState.QUARANTINED,
}


def phase_to_state(phase: OperationPhase) -> OperationState:
    if phase in _PHASE_TO_STATE:
        return _PHASE_TO_STATE[phase]
    if phase in _RUNNING_PHASES:
        return OperationState.RUNNING
    return OperationState.ACCEPTED


@dataclass
class OperationRecord:
    operation_id: str
    lease_epoch: int
    replica_id: str
    operation_type: OperationType
    phase: OperationPhase = OperationPhase.ACCEPTED
    payload_digest: str | None = None
    command_seq: int = 0
    detail: dict[str, object] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)

    @property
    def state(self) -> OperationState:
        return phase_to_state(self.phase)


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
        *,
        payload_digest: str | None = None,
        command_seq: int = 0,
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
                    f"operation {operation_id!r} epoch {lease_epoch} is older than "
                    f"{existing.lease_epoch}"
                )
            if lease_epoch == existing.lease_epoch:
                if (
                    existing.replica_id != replica_id
                    or existing.operation_type is not operation_type
                    or existing.payload_digest != payload_digest
                    or existing.command_seq != command_seq
                ):
                    raise OperationIdentityError(
                        f"conflicting replay for operation {operation_id!r}"
                    )
                return existing
            raise OperationIdentityError(
                f"operation {operation_id!r} cannot be reused for a different lease epoch"
            )

        record = OperationRecord(
            operation_id=operation_id,
            lease_epoch=lease_epoch,
            replica_id=replica_id,
            operation_type=operation_type,
            payload_digest=payload_digest,
            command_seq=command_seq,
        )
        self._records[operation_id] = record
        return record

    def require(self, operation_id: str, lease_epoch: int) -> OperationRecord:
        record = self._records[operation_id]
        if lease_epoch != record.lease_epoch:
            raise ExpiredLeaseError(
                f"operation {operation_id!r} epoch {lease_epoch} does not match "
                f"{record.lease_epoch}"
            )
        return record

    def query(self, operation_id: str) -> OperationRecord | None:
        return self._records.get(operation_id)

    def transition(
        self,
        operation_id: str,
        new_phase: OperationPhase,
        detail: dict[str, object] | None = None,
    ) -> OperationRecord:
        record = self._records[operation_id]
        new_phase = OperationPhase(new_phase)
        if new_phase is not record.phase:
            if new_phase not in _PHASE_ALLOWED[record.phase]:
                raise IllegalOperationTransitionError(
                    f"illegal operation transition for {operation_id}: "
                    f"{record.phase.value} -> {new_phase.value}"
                )
            record.phase = new_phase
        if detail:
            record.detail.update(detail)
        record.updated_at = time.monotonic()
        return record
