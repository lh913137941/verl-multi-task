import ast
import threading
import time
from pathlib import Path

import pytest

from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    OperationCommand,
    OperationEvidence,
    OperationKind,
    OperationRecord,
    OperationStatus,
    ReplicaKey,
    ReplicaKind,
    ReplicaState,
)
from multi_task_scheduler.orchestration.operation_journal import OperationJournal

SOURCE = Path(__file__).resolve().parents[2] / "src/multi_task_scheduler"
INTEGRATION = "integration/verl/experimental_fully_async"


def isolated(relative, name, parent, **scope):
    path = SOURCE / relative
    tree = ast.parse(path.read_text())
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == name
    )
    node.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            node,
        ],
        type_ignores=[],
    )
    env = {"Parent": parent, **scope}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env[name]


class RemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class FakeRay:
    class exceptions:
        class GetTimeoutError(Exception):
            pass

        class RayActorError(Exception):
            pass

    @staticmethod
    def get(value, timeout=None):
        return value


def taskrunner_class():
    class Parent:
        def __init__(self):
            self.components = {}

    return isolated(
        f"{INTEGRATION}/task_runner.py",
        "MultiTaskFullyAsyncTaskRunner",
        Parent,
        OperationJournal=OperationJournal,
        OperationCommand=OperationCommand,
        OperationRecord=OperationRecord,
        OperationStatus=OperationStatus,
        OperationKind=OperationKind,
        OperationEvidence=OperationEvidence,
        EvidenceType=EvidenceType,
        threading=threading,
        ray=FakeRay,
    )


def test_taskrunner_minimal_journal_surface_and_replay_does_not_relaunch():
    cls = taskrunner_class()
    runner = cls()
    runner.task_session = "task-a"
    runner._control_ready = True
    launched = []
    runner._launch_operation = launched.append

    command = OperationCommand(
        "op",
        OperationKind.ADD,
        ReplicaKey("task-a", "r0"),
        "l1",
    )
    assert runner.submit_operation(command).status is OperationStatus.ACCEPTED
    assert runner.submit_operation(command).status is OperationStatus.ACCEPTED
    assert launched == ["op"]
    assert runner.query_operation("missing").status is OperationStatus.UNKNOWN


def test_taskrunner_executes_add_and_commits_terminal_record():
    cls = taskrunner_class()
    runner = cls()
    runner.task_session = "task-a"
    runner._control_ready = True
    runner._launch_operation = lambda operation_id: None

    class Rollouter:
        prepare_replica = RemoteMethod(lambda target: None)

    class Trainer:
        bootstrap_and_publish = RemoteMethod(
            lambda operation: OperationEvidence(
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                1,
            )
        )

    runner.components = {"rollouter": Rollouter(), "trainer": Trainer()}
    command = OperationCommand(
        "op",
        OperationKind.ADD,
        ReplicaKey("task-a", "r0"),
        "l1",
    )
    runner.submit_operation(command)
    runner._execute_operation("op")

    record = runner.query_operation("op")
    assert record.status is OperationStatus.SUCCEEDED
    assert record.result == EvidenceType.SERVICE_COMMITTED.value


def test_manager_owns_state_and_kind_maps_without_replica_record():
    class Parent:
        def __init__(self, *args):
            pass

    allowed = {
        ReplicaState.CREATING: {
            ReplicaState.ACTIVE,
            ReplicaState.RELEASED,
            ReplicaState.QUARANTINED,
        },
        ReplicaState.ACTIVE: {ReplicaState.DRAINING},
        ReplicaState.DRAINING: {
            ReplicaState.ACTIVE,
            ReplicaState.DORMANT,
            ReplicaState.RELEASED,
            ReplicaState.QUARANTINED,
        },
        ReplicaState.DORMANT: {
            ReplicaState.ACTIVE,
            ReplicaState.QUARANTINED,
        },
        ReplicaState.RELEASED: set(),
        ReplicaState.QUARANTINED: set(),
    }
    cls = isolated(
        f"{INTEGRATION}/llm_server_manager.py",
        "MultiTaskLLMServerManager",
        Parent,
        MultiTaskvLLMReplica=object(),
        MultiTaskGlobalRequestLoadBalancer=object(),
        ReplicaKey=ReplicaKey,
        ReplicaKind=ReplicaKind,
        ReplicaState=ReplicaState,
        _ALLOWED=allowed,
    )
    manager = cls(object())
    key = ReplicaKey("task-a", "r0")
    manager.register_replica(key, ReplicaKind.BORROWED)
    manager.transition_replica(key, ReplicaState.ACTIVE)
    manager.transition_replica(key, ReplicaState.DRAINING)
    manager.transition_replica(key, ReplicaState.RELEASED)
    assert manager.replica_meta(key) == (
        ReplicaKind.BORROWED,
        ReplicaState.RELEASED,
    )
    assert not hasattr(manager, "_lifecycle")


def rollouter_class():
    class Parent:
        def __init__(self, *args, **kwargs):
            self.paused = True
            self.max_concurrent_samples = 8

    return isolated(
        f"{INTEGRATION}/rollouter.py",
        "MultiTaskFullyAsyncRollouter",
        Parent,
        ReplicaKey=ReplicaKey,
        ReplicaKind=ReplicaKind,
        ReplicaState=ReplicaState,
        OperationRecord=OperationRecord,
        OperationEvidence=OperationEvidence,
        EvidenceType=EvidenceType,
        ray=FakeRay,
        time=time,
    )


def test_rollouter_idle_detection_does_not_read_lb():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "r0")
    rollouter.llm_server_manager = type(
        "M",
        (),
        {
            "replica_state": {key: ReplicaState.ACTIVE},
            "replica_kind": {key: ReplicaKind.NATIVE},
        },
    )()
    assert rollouter.collect_idle_candidates() == ((key, ReplicaKind.NATIVE),)
    rollouter.paused = False
    assert rollouter.collect_idle_candidates() == ()


def test_rollouter_natural_exit_closes_admission_then_returns_exit_ready():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "r0")
    calls = []

    class LB:
        begin_drain = RemoteMethod(lambda target: calls.append(("begin", target)) or "s0")
        has_unsettled_requests = RemoteMethod(lambda server_id: False)
        finish_remove = RemoteMethod(lambda target: calls.append(("finish", target)))

    class Manager:
        global_load_balancer = LB()

        def __init__(self):
            self.replica_state = {key: ReplicaState.ACTIVE}
            self.replica_kind = {key: ReplicaKind.NATIVE}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def transition_replica(self, target, state):
            self.replica_state[target] = state
            calls.append(("state", state))

    manager = Manager()
    rollouter.llm_server_manager = manager
    evidence = rollouter.prepare_exit(key, operation_id="op")

    assert evidence.type is EvidenceType.EXIT_READY
    assert evidence.operation_id == "op"
    assert manager.replica_state[key] is ReplicaState.DRAINING
    assert calls == [
        ("state", ReplicaState.DRAINING),
        ("begin", key),
        ("finish", key),
    ]


def test_rollouter_force_fails_before_mutating_m_without_verified_backend():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "r0")

    class Manager:
        def __init__(self):
            self.replica_state = {key: ReplicaState.ACTIVE}
            self.replica_kind = {key: ReplicaKind.BORROWED}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def transition_replica(self, target, state):
            self.replica_state[target] = state

    manager = Manager()
    rollouter.llm_server_manager = manager
    with pytest.raises(NotImplementedError, match="targeted abort"):
        rollouter.prepare_exit(key, operation_id="op", force=True)
    assert manager.replica_state[key] is ReplicaState.ACTIVE
