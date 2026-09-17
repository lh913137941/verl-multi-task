"""TaskRunner command-sequence fencing for delayed lifecycle commands."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from multi_task_scheduler.orchestration.contracts import (
    LeaseAuthorization,
    NodePlacement,
    OperationCommand,
    OperationContext,
    OperationResult,
    PlacementSpec,
    QueryResult,
    ReplicaKey,
)
from multi_task_scheduler.orchestration.operation_journal import (
    OperationIdentityError,
    OperationJournal,
    OperationKind,
    OperationStatus,
    Outcome,
)
from multi_task_scheduler.orchestration.receipts import ReleaseEvidence, ServiceEvidence

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/multi_task_scheduler/integration/verl/experimental_fully_async/task_runner.py"
)


def _isolated_task_runner():
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
        "OperationCommand": OperationCommand,
        "OperationResult": OperationResult,
        "QueryResult": QueryResult,
        "OperationIdentityError": OperationIdentityError,
        "OperationStatus": OperationStatus,
        "Outcome": Outcome,
        "ServiceEvidence": ServiceEvidence,
        "ReleaseEvidence": ReleaseEvidence,
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)
    return scope["MultiTaskFullyAsyncTaskRunner"]


def _placement() -> PlacementSpec:
    return PlacementSpec(
        node=NodePlacement(
            node_id="n1",
            gpu_uuids=("u0",),
            physical_gpu_ids=(0,),
            global_ranks=(0,),
            local_ranks=(0,),
        ),
        tp=1,
        dp=1,
        pp=1,
        model_signature="sig",
        placement_digest="placement-u0",
    )


def _command(operation_id: str, command_seq: int) -> OperationCommand:
    ctx = OperationContext(
        protocol_version=1,
        gs_epoch="gs-1",
        task_id="task-a",
        task_session="s1",
        operation_id=operation_id,
        lease_id="l1",
        lease_epoch=1,
        command_seq=command_seq,
    )
    return OperationCommand(
        ctx=ctx,
        kind=OperationKind.ADD,
        target=ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0),
        authorization=LeaseAuthorization(
            lease_id="l1",
            gs_epoch="gs-1",
            donor_session="donor",
            borrower_session="s1",
            placement_digest="placement-u0",
            lease_epoch=1,
            purpose=OperationKind.ADD,
            prior_release_digest="release-0",
            authorization_seq=command_seq + 10,
        ),
        payload_digest=f"payload-{operation_id}",
        remaining_budget_ms=1000,
        placement=_placement(),
    )


def test_distinct_operations_require_strictly_increasing_command_seq():
    runner = _isolated_task_runner()()

    first_command = _command("op-1", 5)
    first = runner.submit_operation(first_command)
    assert first.status is OperationStatus.ACCEPTED
    assert runner.submit_operation(first_command) == first

    with pytest.raises(OperationIdentityError, match="stale command_seq"):
        runner.submit_operation(_command("op-old", 4))
    with pytest.raises(OperationIdentityError, match="stale command_seq"):
        runner.submit_operation(_command("op-equal", 5))

    newer = runner.submit_operation(_command("op-2", 6))
    assert newer.status is OperationStatus.ACCEPTED

    # A retry of the already accepted older operation remains a query/replay of
    # that same operation; it must not be rejected merely because op-2 exists.
    assert runner.submit_operation(first_command) == first
