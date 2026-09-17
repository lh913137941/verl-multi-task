"""Replayable operation metadata aligned with simplified design §6.1/§6.3.

``status`` and ``phase`` are deliberately independent facts.  In particular,
``Phase.DONE`` can describe either a known success or a known failure, so code
must never infer the terminal ``OperationStatus`` from the phase alone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class OperationKind(str, Enum):
    ADD = "ADD"
    DONATE = "DONATE"
    REMOVE = "REMOVE"
    RESTORE = "RESTORE"


class Phase(str, Enum):
    VALIDATE = "VALIDATE"
    CREATE = "CREATE"
    DRAIN = "DRAIN"
    ABORT_TARGET = "ABORT_TARGET"
    WAIT_CONTINUATION = "WAIT_CONTINUATION"
    WAIT_GATE = "WAIT_GATE"
    WAKE_WEIGHTS = "WAKE_WEIGHTS"
    LOAD_WEIGHTS = "LOAD_WEIGHTS"
    JOIN_CE = "JOIN_CE"
    COMMIT_SERVICE = "COMMIT_SERVICE"
    LEAVE_CE = "LEAVE_CE"
    COMMIT_REMOVAL = "COMMIT_REMOVAL"
    SLEEP = "SLEEP"
    DESTROY = "DESTROY"
    RECONCILE = "RECONCILE"
    DONE = "DONE"


class OperationStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


# Compatibility names for callers outside this repository.  Repository code
# uses the simplified-design names above.
OperationType = OperationKind
OperationPhase = Phase
OperationState = OperationStatus


class ExpiredLeaseError(RuntimeError):
    """A command does not match the journal's current lease epoch."""


class OperationIdentityError(RuntimeError):
    """An operation ID is replayed with conflicting immutable fields."""


class IllegalOperationTransitionError(RuntimeError):
    """An operation skips a required transaction phase/status edge."""


_PHASE_ALLOWED = {
    Phase.VALIDATE: {Phase.CREATE, Phase.DRAIN, Phase.WAIT_GATE, Phase.RECONCILE, Phase.DONE},
    Phase.CREATE: {Phase.WAIT_GATE, Phase.DESTROY, Phase.RECONCILE, Phase.DONE},
    Phase.DRAIN: {Phase.WAIT_GATE, Phase.ABORT_TARGET, Phase.RECONCILE, Phase.DONE},
    Phase.ABORT_TARGET: {Phase.WAIT_CONTINUATION, Phase.RECONCILE, Phase.DONE},
    Phase.WAIT_CONTINUATION: {Phase.WAIT_GATE, Phase.RECONCILE, Phase.DONE},
    Phase.WAIT_GATE: {
        Phase.WAKE_WEIGHTS,
        Phase.LOAD_WEIGHTS,
        Phase.LEAVE_CE,
        Phase.RECONCILE,
        Phase.DONE,
    },
    Phase.WAKE_WEIGHTS: {Phase.LOAD_WEIGHTS, Phase.RECONCILE, Phase.DONE},
    Phase.LOAD_WEIGHTS: {Phase.JOIN_CE, Phase.RECONCILE, Phase.DONE},
    Phase.JOIN_CE: {Phase.COMMIT_SERVICE, Phase.RECONCILE, Phase.DONE},
    Phase.COMMIT_SERVICE: {Phase.DONE, Phase.RECONCILE},
    Phase.LEAVE_CE: {Phase.COMMIT_REMOVAL, Phase.RECONCILE, Phase.DONE},
    Phase.COMMIT_REMOVAL: {Phase.SLEEP, Phase.DESTROY, Phase.DONE, Phase.RECONCILE},
    Phase.SLEEP: {Phase.DONE, Phase.RECONCILE},
    Phase.DESTROY: {Phase.DONE, Phase.RECONCILE},
    Phase.RECONCILE: {Phase.DONE},
    Phase.DONE: set(),
}

_TERMINAL_STATUSES = {
    OperationStatus.SUCCEEDED,
    OperationStatus.FAILED,
    OperationStatus.UNKNOWN,
}


def phase_to_state(
    phase: Phase,
    *,
    terminal_status: OperationStatus | None = None,
) -> OperationStatus:
    """Compatibility helper without collapsing independent status semantics.

    ``DONE`` is ambiguous by design, therefore its status must be supplied.
    ``RECONCILE`` represents an unresolved side effect and maps to UNKNOWN.
    """
    phase = Phase(phase)
    if phase is Phase.VALIDATE:
        return OperationStatus.ACCEPTED
    if phase is Phase.RECONCILE:
        return OperationStatus.UNKNOWN
    if phase is Phase.DONE:
        if terminal_status is None:
            raise ValueError("DONE phase requires an explicit terminal OperationStatus")
        terminal_status = OperationStatus(terminal_status)
        if terminal_status not in _TERMINAL_STATUSES:
            raise ValueError("DONE requires SUCCEEDED, FAILED, or UNKNOWN")
        return terminal_status
    return OperationStatus.RUNNING


@dataclass
class OperationRecord:
    operation_id: str
    lease_epoch: int
    replica_id: str
    operation_kind: OperationKind
    phase: Phase = Phase.VALIDATE
    status: OperationStatus = OperationStatus.ACCEPTED
    phase_revision: int = 0
    payload_digest: str | None = None
    command_seq: int = 0
    detail: dict[str, object] = field(default_factory=dict)
    phase_results: dict[Phase, object] = field(default_factory=dict)
    phase_timings_ms: dict[Phase, int] = field(default_factory=dict)
    error: object | None = None
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)

    @property
    def operation_type(self) -> OperationKind:
        """Compatibility alias for the superseded field name."""
        return self.operation_kind

    @property
    def state(self) -> OperationStatus:
        """Compatibility alias; status is stored, not derived from phase."""
        return self.status


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
        operation_kind: OperationKind,
        *,
        payload_digest: str | None = None,
        command_seq: int = 0,
    ) -> OperationRecord:
        if not operation_id or not replica_id:
            raise ValueError("operation_id and replica_id must be nonempty")
        if lease_epoch < 0 or command_seq < 0:
            raise ValueError("lease_epoch and command_seq must be nonnegative")
        operation_kind = OperationKind(operation_kind)

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
                    or existing.operation_kind is not operation_kind
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
            operation_kind=operation_kind,
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
        new_phase: Phase,
        *,
        status: OperationStatus | None = None,
        phase_result: object | None = None,
        elapsed_ms: int | None = None,
        error: object | None = None,
        detail: dict[str, object] | None = None,
    ) -> OperationRecord:
        record = self._records[operation_id]
        new_phase = Phase(new_phase)
        if new_phase is not record.phase and new_phase not in _PHASE_ALLOWED[record.phase]:
            raise IllegalOperationTransitionError(
                f"illegal operation transition for {operation_id}: "
                f"{record.phase.value} -> {new_phase.value}"
            )

        if status is None:
            if new_phase is Phase.DONE:
                raise IllegalOperationTransitionError(
                    "DONE requires an explicit SUCCEEDED/FAILED/UNKNOWN status"
                )
            new_status = phase_to_state(new_phase)
        else:
            new_status = OperationStatus(status)

        if new_phase is Phase.DONE:
            if new_status not in _TERMINAL_STATUSES:
                raise IllegalOperationTransitionError(
                    "DONE requires SUCCEEDED, FAILED, or UNKNOWN"
                )
        elif new_status in {OperationStatus.SUCCEEDED, OperationStatus.FAILED}:
            raise IllegalOperationTransitionError(
                f"terminal status {new_status.value} requires phase DONE"
            )
        elif new_status is OperationStatus.UNKNOWN and new_phase is not Phase.RECONCILE:
            raise IllegalOperationTransitionError(
                "UNKNOWN is only valid while reconciling or at DONE"
            )

        changed = new_phase is not record.phase or new_status is not record.status
        record.phase = new_phase
        record.status = new_status
        if phase_result is not None:
            record.phase_results[new_phase] = phase_result
            changed = True
        if elapsed_ms is not None:
            if elapsed_ms < 0:
                raise ValueError("elapsed_ms must be nonnegative")
            record.phase_timings_ms[new_phase] = elapsed_ms
            changed = True
        if error is not None:
            record.error = error
            changed = True
        if detail:
            record.detail.update(detail)
            changed = True
        if changed:
            record.phase_revision += 1
            record.updated_at = time.monotonic()
        return record
