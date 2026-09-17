"""Global lease authorization state machine for simplified design §8-§14.

Authorization changes are GS decisions. Receipt-backed edges advance only from
a matching terminal operation status; ``ACCEPTED``/``RUNNING`` never transfer
GPU authority.
"""

from __future__ import annotations

from enum import Enum

from multi_task_scheduler.orchestration.operation_journal import OperationStatus

from .ledger import LeaseRecord


class LeaseState(str, Enum):
    PLANNED = "PLANNED"
    DONOR_DRAINING = "DONOR_DRAINING"
    DONOR_RELEASED = "DONOR_RELEASED"
    BORROWER_PREPARING = "BORROWER_PREPARING"
    BORROWER_ACTIVE = "BORROWER_ACTIVE"
    RECALLING = "RECALLING"
    BORROWER_RELEASED = "BORROWER_RELEASED"
    DONOR_RESTORING = "DONOR_RESTORING"
    CLOSED = "CLOSED"
    RECONCILING = "RECONCILING"
    QUARANTINED = "QUARANTINED"


class IllegalLeaseTransitionError(RuntimeError):
    """A lease skipped a required state."""


class MissingReceiptError(RuntimeError):
    """A receipt-backed lease transition lacked a matching successful result."""


_ALLOWED = {
    LeaseState.PLANNED: {LeaseState.DONOR_DRAINING},
    LeaseState.DONOR_DRAINING: {LeaseState.DONOR_RELEASED, LeaseState.RECONCILING},
    LeaseState.DONOR_RELEASED: {LeaseState.BORROWER_PREPARING},
    LeaseState.BORROWER_PREPARING: {LeaseState.BORROWER_ACTIVE, LeaseState.RECONCILING},
    LeaseState.BORROWER_ACTIVE: {LeaseState.RECALLING, LeaseState.RECONCILING},
    LeaseState.RECALLING: {LeaseState.BORROWER_RELEASED, LeaseState.RECONCILING},
    LeaseState.BORROWER_RELEASED: {LeaseState.DONOR_RESTORING},
    LeaseState.DONOR_RESTORING: {LeaseState.CLOSED, LeaseState.RECONCILING, LeaseState.QUARANTINED},
    LeaseState.CLOSED: set(),
    LeaseState.RECONCILING: set(),
    LeaseState.QUARANTINED: set(),
}

_RECEIPT_REQUIRED = {
    (LeaseState.DONOR_DRAINING, LeaseState.DONOR_RELEASED): OperationStatus.SUCCEEDED,
    (LeaseState.BORROWER_PREPARING, LeaseState.BORROWER_ACTIVE): OperationStatus.SUCCEEDED,
    (LeaseState.RECALLING, LeaseState.BORROWER_RELEASED): OperationStatus.SUCCEEDED,
    (LeaseState.DONOR_RESTORING, LeaseState.CLOSED): OperationStatus.SUCCEEDED,
}


class LeaseStateMachine:
    def __init__(self) -> None:
        self._leases: dict[str, LeaseRecord] = {}

    def register(self, lease: LeaseRecord) -> None:
        lease.state = LeaseState(lease.state).value
        self._leases[lease.lease_id] = lease

    def get(self, lease_id: str) -> LeaseRecord:
        return self._leases[lease_id]

    def advance(
        self,
        lease_id: str,
        new_state: LeaseState,
        *,
        supporting_result_state: OperationStatus | str | None = None,
        operation_id: str | None = None,
    ) -> LeaseRecord:
        lease = self._leases[lease_id]
        current = LeaseState(lease.state)
        new_state = LeaseState(new_state)

        if new_state not in _ALLOWED[current]:
            raise IllegalLeaseTransitionError(
                f"illegal lease transition for {lease_id}: "
                f"{current.value} -> {new_state.value}"
            )

        required = _RECEIPT_REQUIRED.get((current, new_state))
        if required is not None:
            try:
                supplied = (
                    OperationStatus(supporting_result_state)
                    if supporting_result_state is not None
                    else None
                )
            except ValueError:
                supplied = None
            if supplied is not required:
                supplied_value = (
                    supporting_result_state.value
                    if isinstance(supporting_result_state, OperationStatus)
                    else supporting_result_state
                )
                raise MissingReceiptError(
                    f"lease {lease_id} {current.value} -> {new_state.value} "
                    f"requires a {required.value} operation result, "
                    f"got {supplied_value!r}"
                )

        lease.state = new_state.value
        if operation_id is not None:
            lease.last_operation_id = operation_id
        return lease
