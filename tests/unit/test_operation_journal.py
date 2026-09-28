import pytest
from multi_task_scheduler.orchestration.contracts import OperationCommand, OperationKind, OperationStatus, ReplicaKey
from multi_task_scheduler.orchestration.operation_journal import OperationIdentityError, OperationJournal

def command(op="op-1", replica="r0"):
    return OperationCommand(op, OperationKind.ADD, ReplicaKey("task-a", replica), "l1")

def test_replay_and_single_active_fence():
    j = OperationJournal(); first = j.begin(command())
    assert j.begin(command()) is first
    with pytest.raises(OperationIdentityError): j.begin(command("op-2", "r1"))
    j.finish("op-1", OperationStatus.SUCCEEDED, "done")
    assert j.begin(command("op-2", "r1")).operation_id == "op-2"

def test_conflicting_replay_and_terminal_rewrite_fail():
    j = OperationJournal(); j.begin(command())
    with pytest.raises(OperationIdentityError): j.begin(command("op-1", "r1"))
    j.finish("op-1", OperationStatus.FAILED, "failed")
    with pytest.raises(OperationIdentityError): j.finish("op-1", OperationStatus.UNKNOWN, "lost")


def test_unknown_outcome_keeps_task_fenced_until_reconciled():
    journal = OperationJournal()
    first = command()
    journal.begin(first)
    journal.finish(first.operation_id, OperationStatus.UNKNOWN, "owner response lost")

    assert journal.begin(first).status is OperationStatus.UNKNOWN
    with pytest.raises(OperationIdentityError, match="another lifecycle operation"):
        journal.begin(command("op-2", "r1"))


def test_unknown_operation_can_be_reopened_only_for_same_operation_reconciliation():
    journal = OperationJournal()
    op = command("op-reconcile")
    journal.begin(op)
    journal.mark_running(op.operation_id)
    journal.finish(op.operation_id, OperationStatus.UNKNOWN, "owner fact unavailable")

    reopened = journal.reopen_unknown(op.operation_id)
    assert reopened.status is OperationStatus.RUNNING
    assert reopened.result is None

    journal.finish(op.operation_id, OperationStatus.SUCCEEDED, "reconciled")
    assert journal.query(op.operation_id).status is OperationStatus.SUCCEEDED

    with pytest.raises(OperationIdentityError, match="only UNKNOWN"):
        journal.reopen_unknown(op.operation_id)
