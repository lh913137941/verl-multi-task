"""Global lease state machine (section 5.6), pure Python.

Every authorization change is written to the ledger *before* a command is
issued, and each receipt-backed transition must be supported by a matching
operation result — a lease never advances on "RPC already sent" alone.

Receipt-backed transitions require the exact operation result state the design
names; the authorization steps (issue DONATE, authorize ADD, issue REMOVE,
authorize RESTORE) are GS decisions with no receipt.
"""

from __future__ import annotations

from enum import Enum

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
    """A receipt-backed lease transition lacked its supporting operation result."""


_ALLOWED = {
    LeaseState.PLANNED: {LeaseState.DONOR_DRAINING},
    LeaseState.DONOR_DRAINING: {LeaseState.DONOR_RELEASED, LeaseState.RECONCILING},
    LeaseState.DONOR_RELEASED: {LeaseState.BORROWER_PREPARING},
    LeaseState.BORROWER_PREPARING: {LeaseState.BORROWER_ACTIVE, LeaseState.RECONCILING},
    LeaseState.BORROWER_ACTIVE: {LeaseState.RECALLING, LeaseState.RECONCILING},
    LeaseState.RECALLING: {LeaseState.BORROWER_RELEASED},
    LeaseState.BORROWER_RELEASED: {LeaseState.DONOR_RESTORING},
    LeaseState.DONOR_RESTORING: {LeaseState.CLOSED, LeaseState.QUARANTINED},
    LeaseState.CLOSED: set(),
    LeaseState.RECONCILING: set(),
    LeaseState.QUARANTINED: set(),
}

# Receipt state (OperationResult.state) each receipt-backed edge requires.
_RECEIPT_REQUIRED = {
    (LeaseState.DONOR_DRAINING, LeaseState.DONOR_RELEASED): "COMMITTED",
    (LeaseState.BORROWER_PREPARING, LeaseState.BORROWER_ACTIVE): "COMMITTED",
    (LeaseState.RECALLING, LeaseState.BORROWER_RELEASED): "COMMITTED",
    (LeaseState.DONOR_RESTORING, LeaseState.CLOSED): "COMMITTED",
}


class LeaseStateMachine:
    """Advance a lease only along the section 5.6 edges."""

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
        supporting_result_state: str | None = None,
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
            if supporting_result_state != required:
                raise MissingReceiptError(
                    f"lease {lease_id} {current.value} -> {new_state.value} "
                    f"requires a {required} operation result, "
                    f"got {supporting_result_state!r}"
                )

        lease.state = new_state.value
        if operation_id is not None:
            lease.last_operation_id = operation_id
        return lease
