"""Control-plane tests for GS lease-state/GPU-authorization coupling.

The GroupScheduler class body is executed without Ray decoration so these tests
exercise the real synchronous control logic only; they do not claim Ray/GPU
runtime validation.
"""

from __future__ import annotations

import ast
from pathlib import Path
import uuid

import pytest

from multi_task_scheduler.orchestration.contracts import NodePlacement, PlacementSpec
from multi_task_scheduler.scheduler.lease import LeaseState, LeaseStateMachine
from multi_task_scheduler.scheduler.ledger import (
    Ledger,
    LeaseRecord,
    ProtocolInstance,
    ResourceManifest,
)

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/multi_task_scheduler/scheduler/group_scheduler.py"
)


def _isolated_group_scheduler():
    parsed = ast.parse(SOURCE.read_text())
    node = next(
        item
        for item in parsed.body
        if isinstance(item, ast.ClassDef) and item.name == "GroupScheduler"
    )
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
    scope = {
        "uuid": uuid,
        "Ledger": Ledger,
        "LeaseRecord": LeaseRecord,
        "ProtocolInstance": ProtocolInstance,
        "ResourceManifest": ResourceManifest,
        "LeaseState": LeaseState,
        "LeaseStateMachine": LeaseStateMachine,
        "PROTOCOL_VERSION": 1,
        "RUNTIME_KIND": "verl-multi-task:test",
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)
    return scope["GroupScheduler"]


def _placement() -> PlacementSpec:
    return PlacementSpec(
        node=NodePlacement(
            node_id="n1",
            gpu_uuids=("u0", "u1"),
            physical_gpu_ids=(0, 1),
            global_ranks=(0, 1),
            local_ranks=(0, 1),
        ),
        tp=2,
        dp=1,
        pp=1,
        model_signature="sig",
        placement_digest="placement-u0-u1",
    )


def _scheduler_with_donor_resources():
    scheduler = _isolated_group_scheduler()()
    scheduler.ledger.register_task("task-donor", "donor")
    scheduler.ledger.register_resources(
        ResourceManifest(
            owner_task_session="donor",
            placement=_placement(),
            replica_id="native-r1",
        )
    )
    return scheduler


def _gpu_records(scheduler):
    return tuple(
        scheduler.ledger.gpus[(scheduler.ledger.protocol.gs_epoch, "n1", gpu_uuid)]
        for gpu_uuid in ("u0", "u1")
    )


def test_donor_release_to_borrower_preparing_commits_gpu_user_ledger():
    scheduler = _scheduler_with_donor_resources()
    lease = LeaseRecord(
        lease_id="l1",
        donor_session="donor",
        borrower_session="borrower",
        state=LeaseState.DONOR_RELEASED.value,
        last_release_digest="release-donor",
        placement=_placement(),
    )
    scheduler.open_lease(lease)

    result = scheduler.advance_lease("l1", LeaseState.BORROWER_PREPARING.value)

    assert result["state"] == LeaseState.BORROWER_PREPARING.value
    for gpu in _gpu_records(scheduler):
        assert gpu.current_user == "borrower"
        assert gpu.lease_id == "l1"
        assert gpu.state == "LENT"


def test_borrower_release_to_donor_restoring_clears_borrower_authorization():
    scheduler = _scheduler_with_donor_resources()
    lease = LeaseRecord(
        lease_id="l1",
        donor_session="donor",
        borrower_session="borrower",
        state=LeaseState.BORROWER_RELEASED.value,
        last_release_digest="release-borrower",
        placement=_placement(),
    )
    for gpu_uuid in ("u0", "u1"):
        scheduler.ledger.set_current_user(
            (scheduler.ledger.protocol.gs_epoch, "n1", gpu_uuid),
            "borrower",
            "l1",
            0,
        )
    scheduler.open_lease(lease)

    result = scheduler.advance_lease("l1", LeaseState.DONOR_RESTORING.value)

    assert result["state"] == LeaseState.DONOR_RESTORING.value
    for gpu in _gpu_records(scheduler):
        assert gpu.current_user is None
        assert gpu.lease_id is None
        assert gpu.state == "FREE"


def test_gpu_preflight_failure_does_not_advance_lease_state():
    scheduler = _scheduler_with_donor_resources()
    lease = LeaseRecord(
        lease_id="l1",
        donor_session="wrong-donor",
        borrower_session="borrower",
        state=LeaseState.DONOR_RELEASED.value,
        last_release_digest="release-donor",
        placement=_placement(),
    )
    scheduler.open_lease(lease)

    with pytest.raises(ValueError, match="does not match donor"):
        scheduler.advance_lease("l1", LeaseState.BORROWER_PREPARING.value)

    assert scheduler.lease_sm.get("l1").state == LeaseState.DONOR_RELEASED.value
    assert all(gpu.current_user is None for gpu in _gpu_records(scheduler))
