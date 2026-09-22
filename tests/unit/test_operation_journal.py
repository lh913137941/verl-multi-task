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
