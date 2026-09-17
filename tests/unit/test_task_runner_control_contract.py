"""Scoped TaskRunner control-contract checks with an inert parent.

This is pure-Python/AST coverage only. It does not import verl/Ray and does not
claim native runtime or GPU execution success.
"""

import ast
from pathlib import Path

from multi_task_scheduler.orchestration.contracts import OperationCommand, OperationResult
from multi_task_scheduler.orchestration.operation_journal import (
    OperationJournal,
    OperationKind,
    OperationStatus,
    Phase,
)


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/multi_task_scheduler/integration/verl/experimental_fully_async/task_runner.py"
)


def _task_runner_class():
    parsed = ast.parse(SOURCE.read_text())
    node = next(
        item
        for item in parsed.body
        if isinstance(item, ast.ClassDef)
        and item.name == "MultiTaskFullyAsyncTaskRunner"
    )
    node.bases = [ast.Name(id="TestParent", ctx=ast.Load())]
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

    class TestParent:
        def __init__(self):
            self.components = {}

    scope = {
        "TestParent": TestParent,
        "OperationJournal": OperationJournal,
        "OperationResult": OperationResult,
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)
    return scope["MultiTaskFullyAsyncTaskRunner"]


def _command():
    return OperationCommand(
        protocol_version=1,
        gs_epoch="gs-1",
        target_task_id="task-a",
        target_task_session="session-a",
        operation_id="op-1",
        payload_digest="digest-1",
        kind=OperationKind.ADD,
        lease_id="lease-1",
        lease_epoch=0,
        command_seq=3,
        replica_id="r1",
    )


def test_submit_operation_returns_typed_accepted_result_and_is_idempotent():
    runner = _task_runner_class()()
    first = runner.submit_operation(_command())
    second = runner.submit_operation(_command())

    assert isinstance(first, OperationResult)
    assert first.phase is Phase.VALIDATE
    assert first.status is OperationStatus.ACCEPTED
    assert first.phase_revision == 0
    assert second == first
    assert len(runner._ensure_journal()._records) == 1


def test_query_operation_prefers_public_session_first_signature_but_keeps_old_call():
    runner = _task_runner_class()()
    runner.submit_operation(_command())

    public = runner.query_operation("session-a", "op-1")
    legacy = runner.query_operation("op-1")
    assert public is legacy
    assert public.operation_id == "op-1"


def test_probe_task_accepts_session_and_does_not_claim_runtime_facts():
    runner = _task_runner_class()()
    probe = runner.probe_task("session-a")
    assert probe == {
        "task_session": "session-a",
        "trainer_attached": False,
        "rollouter_attached": False,
        "operation_count": 0,
    }
