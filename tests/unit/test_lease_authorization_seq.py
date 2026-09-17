"""Lease authorization_seq fencing and same-operation replay behavior."""

import pytest

from multi_task_scheduler.scheduler.lease import LeaseStateMachine
from multi_task_scheduler.scheduler.ledger import LeaseRecord


def _machine():
    machine = LeaseStateMachine()
    machine.register(
        LeaseRecord(
            lease_id="l1",
            donor_session="donor",
            borrower_session="borrower",
        )
    )
    return machine


def test_authorization_seq_is_monotonic_across_distinct_operations():
    machine = _machine()
    machine.validate_authorization("l1", "op-1", 5)
    machine.record_authorization("l1", "op-1", 5)

    with pytest.raises(ValueError, match="stale authorization_seq"):
        machine.validate_authorization("l1", "op-old", 4)
    with pytest.raises(ValueError, match="stale authorization_seq"):
        machine.validate_authorization("l1", "op-equal", 5)

    machine.validate_authorization("l1", "op-2", 6)
    machine.record_authorization("l1", "op-2", 6)


def test_same_operation_replay_keeps_original_authorization_seq_after_newer_auth():
    machine = _machine()
    machine.record_authorization("l1", "op-1", 5)
    machine.record_authorization("l1", "op-2", 6)

    machine.validate_authorization("l1", "op-1", 5)
    with pytest.raises(ValueError, match="conflicting authorization_seq"):
        machine.validate_authorization("l1", "op-1", 7)
