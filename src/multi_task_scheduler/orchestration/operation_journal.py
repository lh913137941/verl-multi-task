"""Replayable lifecycle journal and its acceptance fences."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class OperationKind(str, Enum):
    ADD = "ADD"
    DONATE = "DONATE"
    REMOVE = "REMOVE"
    RESTORE = "RESTORE"


class OperationStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


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


class Outcome(str, Enum):
    KNOWN_NOT_APPLIED = "KNOWN_NOT_APPLIED"
    KNOWN_APPLIED = "KNOWN_APPLIED"
    UNKNOWN = "UNKNOWN"


class ExpiredLeaseError(RuntimeError):
    """A command does not match the journal's accepted lease epoch."""


class OperationIdentityError(RuntimeError):
    """A replay or newly accepted operation conflicts with journal fences."""


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


def _default_status_for_phase(phase: Phase) -> OperationStatus:
    if phase is Phase.VALIDATE:
        return OperationStatus.ACCEPTED
    if phase is Phase.RECONCILE:
        return OperationStatus.UNKNOWN
    if phase is Phase.DONE:
        raise IllegalOperationTransitionError(
            "DONE requires an explicit SUCCEEDED/FAILED/UNKNOWN status"
        )
    return OperationStatus.RUNNING


def _require_command_shape(command: object) -> None:
    """Validate the journal-facing command shape without importing contracts."""
    try:
        ctx = command.ctx
        target = command.target
        kind = command.kind
        payload_digest = command.payload_digest
        remaining_budget_ms = command.remaining_budget_ms
    except AttributeError as exc:
        raise TypeError("journal begin requires a complete OperationCommand") from exc
    if not ctx.operation_id or not target.replica_id:
        raise ValueError("operation_id and target replica_id must be nonempty")
    if not isinstance(payload_digest, str) or not payload_digest:
        raise ValueError("payload_digest must be a nonempty string")
    if remaining_budget_ms < 0:
        raise ValueError("remaining_budget_ms must be nonnegative")
    OperationKind(kind)


def _command_identity(command: object) -> tuple:
    """Immutable business identity; transport budget is not replay identity."""
    return (
        command.ctx,
        OperationKind(command.kind),
        command.target,
        command.authorization,
        command.placement,
        command.candidate,
        command.recall_mode,
        command.payload_digest,
    )


@dataclass
class OperationRecord:
    command: object
    status: OperationStatus = OperationStatus.ACCEPTED
    phase: Phase = Phase.VALIDATE
    phase_revision: int = 0
    phase_results: dict[Phase, object] = field(default_factory=dict)
    phase_timings_ms: dict[Phase, int] = field(default_factory=dict)
    error: object | None = None
    deadline_at: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def operation_id(self) -> str:
        return self.command.ctx.operation_id

    @property
    def lease_epoch(self) -> int:
        return self.command.ctx.lease_epoch

    @property
    def target(self):
        return self.command.target


class OperationJournal:
    def __init__(self, *, clock=None) -> None:
        self._records: dict[str, OperationRecord] = {}
        self._clock = time.monotonic if clock is None else clock

    def begin(self, command: object) -> OperationRecord:
        """Idempotently accept one command while enforcing task/sequence fences."""
        _require_command_shape(command)
        operation_id = command.ctx.operation_id
        existing = self._records.get(operation_id)
        if existing is not None:
            if command.ctx.lease_epoch < existing.lease_epoch:
                raise ExpiredLeaseError(
                    f"operation {operation_id!r} epoch {command.ctx.lease_epoch} is older than "
                    f"{existing.lease_epoch}"
                )
            if command.ctx.lease_epoch > existing.lease_epoch:
                raise OperationIdentityError(
                    f"operation {operation_id!r} cannot be reused for another lease epoch"
                )
            if _command_identity(existing.command) != _command_identity(command):
                raise OperationIdentityError(
                    f"conflicting replay for operation {operation_id!r}"
                )
            return existing

        active = next(
            (
                record
                for record in self._records.values()
                if record.command.ctx.task_session == command.ctx.task_session
                and record.phase is not Phase.DONE
            ),
            None,
        )
        if active is not None:
            raise OperationIdentityError(
                "another lifecycle operation is active for this task: "
                f"{active.operation_id!r}"
            )

        prior_sequences = [
            record.command.ctx.command_seq
            for record in self._records.values()
            if record.target == command.target
        ]
        if prior_sequences and command.ctx.command_seq <= max(prior_sequences):
            raise OperationIdentityError(
                f"stale command_seq {command.ctx.command_seq} for {command.target!r}; "
                f"last accepted sequence is {max(prior_sequences)}"
            )

        now = self._clock()
        record = OperationRecord(
            command=command,
            deadline_at=now + command.remaining_budget_ms / 1000.0,
            created_at=now,
            updated_at=now,
        )
        self._records[operation_id] = record
        return record

    def remaining_budget_ms(self, operation_id: str) -> int:
        """Return the first acceptance budget; retries never extend it."""
        return max(
            0,
            int((self._records[operation_id].deadline_at - self._clock()) * 1000),
        )

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
    ) -> OperationRecord:
        record = self._records[operation_id]
        new_phase = Phase(new_phase)
        if new_phase is not record.phase and new_phase not in _PHASE_ALLOWED[record.phase]:
            raise IllegalOperationTransitionError(
                f"illegal operation transition for {operation_id}: "
                f"{record.phase.value} -> {new_phase.value}"
            )

        new_status = (
            _default_status_for_phase(new_phase)
            if status is None
            else OperationStatus(status)
        )
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
        if changed:
            record.phase_revision += 1
            record.updated_at = self._clock()
        return record
