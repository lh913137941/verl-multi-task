"""Global lease state machine uses simplified OperationStatus receipts."""

import pytest

from multi_task_scheduler.orchestration.operation_journal import OperationStatus
from multi_task_scheduler.scheduler.lease import (
    IllegalLeaseTransitionError,
    LeaseState,
    LeaseStateMachine,
    MissingReceiptError,
)
from multi_task_scheduler.scheduler.ledger import LeaseRecord


def _machine():
    sm = LeaseStateMachine()
    sm.register(LeaseRecord(lease_id="l1", donor_session="s1"))
    return sm


def test_full_happy_path_requires_succeeded_receipts_on_receipt_edges():
    sm = _machine()
    sm.advance("l1", LeaseState.DONOR_DRAINING)
    sm.advance("l1", LeaseState.DONOR_RELEASED, supporting_result_state=OperationStatus.SUCCEEDED)
    sm.advance("l1", LeaseState.BORROWER_PREPARING)
    sm.advance("l1", LeaseState.BORROWER_ACTIVE, supporting_result_state="SUCCEEDED")
    sm.advance("l1", LeaseState.RECALLING)
    sm.advance("l1", LeaseState.BORROWER_RELEASED, supporting_result_state="SUCCEEDED")
    sm.advance("l1", LeaseState.DONOR_RESTORING)
    sm.advance("l1", LeaseState.CLOSED, supporting_result_state="SUCCEEDED")
    assert sm.get("l1").state == LeaseState.CLOSED.value


def test_receipt_edge_without_receipt_raises():
    sm = _machine()
    sm.advance("l1", LeaseState.DONOR_DRAINING)
    with pytest.raises(MissingReceiptError):
        sm.advance("l1", LeaseState.DONOR_RELEASED)


def test_old_committed_state_is_rejected():
    sm = _machine()
    sm.advance("l1", LeaseState.DONOR_DRAINING)
    with pytest.raises(MissingReceiptError, match="SUCCEEDED"):
        sm.advance("l1", LeaseState.DONOR_RELEASED, supporting_result_state="COMMITTED")


def test_failed_or_unknown_result_cannot_advance_authorization():
    for status in (OperationStatus.FAILED, OperationStatus.UNKNOWN):
        sm = _machine()
        sm.advance("l1", LeaseState.DONOR_DRAINING)
        with pytest.raises(MissingReceiptError):
            sm.advance("l1", LeaseState.DONOR_RELEASED, supporting_result_state=status)


def test_illegal_skip_raises():
    sm = _machine()
    with pytest.raises(IllegalLeaseTransitionError):
        sm.advance("l1", LeaseState.DONOR_RELEASED)


def test_reconciling_edges():
    sm = _machine()
    sm.advance("l1", LeaseState.DONOR_DRAINING)
    sm.advance("l1", LeaseState.RECONCILING)
    with pytest.raises(IllegalLeaseTransitionError):
        sm.advance("l1", LeaseState.DONOR_RELEASED)


def test_quarantine_after_restore_failure():
    sm = _machine()
    for state, receipt in [
        (LeaseState.DONOR_DRAINING, None),
        (LeaseState.DONOR_RELEASED, "SUCCEEDED"),
        (LeaseState.BORROWER_PREPARING, None),
        (LeaseState.BORROWER_ACTIVE, "SUCCEEDED"),
        (LeaseState.RECALLING, None),
        (LeaseState.BORROWER_RELEASED, "SUCCEEDED"),
        (LeaseState.DONOR_RESTORING, None),
    ]:
        sm.advance("l1", state, supporting_result_state=receipt)
    sm.advance("l1", LeaseState.QUARANTINED)
    assert sm.get("l1").state == LeaseState.QUARANTINED.value


def test_borrower_preparing_can_reconcile():
    sm = _machine()
    sm.advance("l1", LeaseState.DONOR_DRAINING)
    sm.advance("l1", LeaseState.DONOR_RELEASED, supporting_result_state="SUCCEEDED")
    sm.advance("l1", LeaseState.BORROWER_PREPARING)
    sm.advance("l1", LeaseState.RECONCILING)
    assert sm.get("l1").state == LeaseState.RECONCILING.value
