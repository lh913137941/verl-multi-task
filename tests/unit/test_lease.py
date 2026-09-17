"""Global lease state machine (section 5.6)."""

import pytest

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


def test_full_happy_path_requires_receipts_on_receipt_edges():
    sm = _machine()
    sm.advance("l1", LeaseState.DONOR_DRAINING)  # GS issues DONATE
    sm.advance("l1", LeaseState.DONOR_RELEASED, supporting_result_state="COMMITTED")
    sm.advance("l1", LeaseState.BORROWER_PREPARING)  # GS authorizes ADD
    sm.advance("l1", LeaseState.BORROWER_ACTIVE, supporting_result_state="COMMITTED")
    sm.advance("l1", LeaseState.RECALLING)  # GS issues REMOVE
    sm.advance("l1", LeaseState.BORROWER_RELEASED, supporting_result_state="COMMITTED")
    sm.advance("l1", LeaseState.DONOR_RESTORING)  # GS authorizes RESTORE
    sm.advance("l1", LeaseState.CLOSED, supporting_result_state="COMMITTED")
    assert sm.get("l1").state == LeaseState.CLOSED.value


def test_receipt_edge_without_receipt_raises():
    sm = _machine()
    sm.advance("l1", LeaseState.DONOR_DRAINING)
    with pytest.raises(MissingReceiptError):
        sm.advance("l1", LeaseState.DONOR_RELEASED)


def test_receipt_edge_with_wrong_state_raises():
    sm = _machine()
    sm.advance("l1", LeaseState.DONOR_DRAINING)
    with pytest.raises(MissingReceiptError):
        sm.advance("l1", LeaseState.DONOR_RELEASED, supporting_result_state="ROLLED_BACK")


def test_illegal_skip_raises():
    sm = _machine()
    with pytest.raises(IllegalLeaseTransitionError):
        sm.advance("l1", LeaseState.DONOR_RELEASED)


def test_reconciling_edges():
    sm = _machine()
    sm.advance("l1", LeaseState.DONOR_DRAINING)
    sm.advance("l1", LeaseState.RECONCILING)
    # RECONCILING is terminal in this first-pass skeleton.
    with pytest.raises(IllegalLeaseTransitionError):
        sm.advance("l1", LeaseState.DONOR_RELEASED)


def test_quarantine_after_restore_failure():
    sm = _machine()
    for state, receipt in [
        (LeaseState.DONOR_DRAINING, None),
        (LeaseState.DONOR_RELEASED, "COMMITTED"),
        (LeaseState.BORROWER_PREPARING, None),
        (LeaseState.BORROWER_ACTIVE, "COMMITTED"),
        (LeaseState.RECALLING, None),
        (LeaseState.BORROWER_RELEASED, "COMMITTED"),
        (LeaseState.DONOR_RESTORING, None),
    ]:
        sm.advance("l1", state, supporting_result_state=receipt)
    sm.advance("l1", LeaseState.QUARANTINED)
    assert sm.get("l1").state == LeaseState.QUARANTINED.value


def test_borrower_active_and_preparing_can_reconcile():
    sm = _machine()
    sm.advance("l1", LeaseState.DONOR_DRAINING)
    sm.advance("l1", LeaseState.DONOR_RELEASED, supporting_result_state="COMMITTED")
    sm.advance("l1", LeaseState.BORROWER_PREPARING)
    sm.advance("l1", LeaseState.RECONCILING)
    assert sm.get("l1").state == LeaseState.RECONCILING.value
