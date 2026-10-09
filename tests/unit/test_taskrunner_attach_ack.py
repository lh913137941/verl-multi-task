"""Regression tests for TaskRunner -> GS registration acknowledgement under Ray delays.

Extract the production method only; these tests must work in CPU-only CI
without importing VERL, CUDA, NPU, or Ray.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/multi_task_scheduler/integration/verl/experimental_fully_async/task_runner.py"
)


class GetTimeoutError(Exception):
    pass


class Ref:
    def __init__(self, kind):
        self.kind = kind


class RemoteMethod:
    def __init__(self, kind, calls):
        self.kind = kind
        self.calls = calls

    def remote(self, *args):
        self.calls.append((self.kind, args))
        return Ref(self.kind)


class FakeGS:
    def __init__(self, calls):
        self.attach_task = RemoteMethod("attach", calls)
        self.get_task_runners = RemoteMethod("ledger", calls)
        self.detach_task = RemoteMethod("detach", calls)


class FakeRay:
    exceptions = SimpleNamespace(GetTimeoutError=GetTimeoutError)

    def __init__(self, plan, actor):
        self.plan = {key: list(values) for key, values in plan.items()}
        self.actor = actor
        self.get_calls = []

    def get_runtime_context(self):
        return SimpleNamespace(current_actor=self.actor)

    def get(self, ref, *, timeout):
        self.get_calls.append((ref.kind, timeout))
        answers = self.plan[ref.kind]
        result = answers.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def instantiate(plan):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    parent = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MultiTaskFullyAsyncTaskRunner"
    )
    method = next(
        node for node in parent.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_attach_task_with_reconciliation"
    )
    isolated = ast.Module(
        body=[ast.ClassDef(
            name="IsolatedRunner",
            bases=[],
            keywords=[],
            body=[method],
            decorator_list=[],
        )],
        type_ignores=[],
    )
    events = []
    actor = SimpleNamespace(_actor_id="real-actor-A")
    ray = FakeRay(plan, actor)
    logs = []

    logger = SimpleNamespace(
        warning=lambda *args: logs.append(("warning", args)),
        info=lambda *args: logs.append(("info", args)),
    )
    scope = {
        "ray": ray,
        "logger": logger,
        "CONTROL_RPC_TIMEOUT_S": 1.0,
    }
    exec(compile(ast.fix_missing_locations(isolated), str(SOURCE), "exec"), scope)
    runner = scope["IsolatedRunner"]()
    runner.task_session = "real-actor-A"
    runner.group_scheduler = FakeGS(events)
    runner._attached_to_gs = False
    return runner, ray, events, logs


def test_attach_acks_immediately_with_one_rpc():
    runner, ray, events, _ = instantiate({"attach": [None]})
    runner._attach_task_with_reconciliation()
    assert runner._attached_to_gs is True
    assert [event[0] for event in events] == ["attach"]
    assert [kind for kind, _ in ray.get_calls] == ["attach"]


def test_attach_waits_on_same_ref_after_delayed_ack():
    runner, ray, events, _ = instantiate({
        "attach": [GetTimeoutError(), GetTimeoutError(), None]
    })
    runner._attach_task_with_reconciliation()
    assert runner._attached_to_gs is True
    assert [event[0] for event in events] == ["attach"]
    assert [kind for kind, _ in ray.get_calls] == ["attach"] * 3


def test_attach_lost_ack_can_be_verified_by_actor_identity():
    runner, _, events, _ = instantiate({
        "attach": [GetTimeoutError()] * 3,
        "ledger": [{"real-actor-A": SimpleNamespace(_actor_id="real-actor-A")}],
    })
    runner._attach_task_with_reconciliation()
    assert runner._attached_to_gs is True
    assert [event[0] for event in events] == ["attach", "ledger"]


def test_attach_unconfirmed_is_compensated_and_fails_closed():
    runner, _, events, _ = instantiate({
        "attach": [GetTimeoutError()] * 3,
        "ledger": [{}],
        "detach": [None],
    })
    with pytest.raises(RuntimeError, match="compensating detach acknowledged"):
        runner._attach_task_with_reconciliation()
    assert runner._attached_to_gs is False
    assert [event[0] for event in events] == ["attach", "ledger", "detach"]


def test_attach_ledger_unavailable_compensates_without_claiming_success():
    runner, _, events, _ = instantiate({
        "attach": [GetTimeoutError()] * 3,
        "ledger": [GetTimeoutError()],
        "detach": [GetTimeoutError()],
    })
    with pytest.raises(RuntimeError, match="compensating detach also unconfirmed"):
        runner._attach_task_with_reconciliation()
    assert runner._attached_to_gs is False
    assert [event[0] for event in events] == ["attach", "ledger", "detach"]


def test_attach_conflicting_actor_identity_never_detaches_other_task():
    runner, _, events, _ = instantiate({
        "attach": [GetTimeoutError()] * 3,
        "ledger": [{"real-actor-A": SimpleNamespace(_actor_id="OTHER")}],
    })
    with pytest.raises(RuntimeError, match="different ActorHandle"):
        runner._attach_task_with_reconciliation()
    assert runner._attached_to_gs is False
    assert [event[0] for event in events] == ["attach", "ledger"]


def test_attach_remote_rejection_is_not_mistaken_for_timeout():
    runner, _, events, _ = instantiate({"attach": [ValueError("duplicate session")]})
    with pytest.raises(ValueError, match="duplicate session"):
        runner._attach_task_with_reconciliation()
    assert runner._attached_to_gs is False
    assert [event[0] for event in events] == ["attach"]
