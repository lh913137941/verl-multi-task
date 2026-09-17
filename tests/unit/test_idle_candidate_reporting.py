"""LB owns IdleCandidateReport sequencing and transport retry identity."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from multi_task_scheduler.orchestration.contracts import IdleCandidateReport, ReplicaKey
from multi_task_scheduler.orchestration.production_window import (
    ProductionWindow,
    ReplicaView,
    ServerActivity,
    WindowState,
    select_idle_candidates,
)
from multi_task_scheduler.orchestration.receipts import Ack

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/multi_task_scheduler/rollout/load_balancer.py"
)


def _isolated_lb(ray_substitute=None):
    parsed = ast.parse(SOURCE.read_text())
    node = next(
        item
        for item in parsed.body
        if isinstance(item, ast.ClassDef)
        and item.name == "MultiTaskGlobalRequestLoadBalancer"
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
        def __init__(self, *args, **kwargs):
            pass

    scope = {
        "TestParent": TestParent,
        "DEFAULT_ROUTING_CACHE_SIZE": 128,
        "ProductionWindow": ProductionWindow,
        "IdleCandidateReport": IdleCandidateReport,
        "select_idle_candidates": select_idle_candidates,
        "Ack": Ack,
        "ray": ray_substitute or SimpleNamespace(get=lambda value, **kwargs: value),
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)
    return scope["MultiTaskGlobalRequestLoadBalancer"]


def _window(state=WindowState.CLOSED_BACKPRESSURE, revision=4):
    return ProductionWindow(
        task_session="s1",
        epoch=3,
        revision=revision,
        state=state,
        eligible_pending=0,
        held_samples=0,
        active_samples=1,
        output_queue_size=0,
        max_queue_size=8,
        producer_exhausted=state is WindowState.EXHAUSTED,
        policy_refresh_inflight=False,
    )


def _view():
    return ReplicaView(
        activity=ServerActivity(
            key=ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0),
            engine_seq=5,
            observed_age_ms=10,
            admitting=0,
            queued=0,
            running=0,
            pending_admissions=0,
            transfer_inflight=False,
            all_backends_observed=True,
        ),
        manager_revision=2,
        lb_revision=3,
        placement_digest="placement-r1",
        stable_idle_ms=500,
        gpu_count=1,
    )


def _build(lb, window):
    return lb.build_idle_candidate_report(
        window,
        [_view()],
        gs_epoch="gs-1",
        lb_session="lb-1",
        valid_for_ms=1000,
        observations_fresh=True,
        min_active_gpus=1,
        current_active_gpus=2,
        routable_count=2,
    )


def test_lb_assigns_monotonic_source_seq_and_window_revision():
    lb = _isolated_lb()({})

    first = _build(lb, _window(revision=4))
    second = _build(lb, _window(revision=5))

    assert first.source_seq == 0
    assert second.source_seq == 1
    assert first.production_revision == 4
    assert second.production_revision == 5
    assert first.candidates[0].source_seq == first.source_seq
    assert second.candidates[0].source_seq == second.source_seq
    assert first.candidates[0].engine_digest
    assert second.candidates[0].engine_digest
    assert lb.last_idle_report is second


def test_open_window_builds_empty_replacement_report():
    lb = _isolated_lb()({})
    report = _build(lb, _window(state=WindowState.OPEN))
    assert report.candidates == ()
    assert report.source_seq == 0


def test_transport_retry_resends_same_report_without_allocating_new_seq():
    ack = Ack(accepted=True, revision=0)
    remote = Mock(return_value=ack)
    scheduler = SimpleNamespace(
        report_idle_candidates=SimpleNamespace(remote=remote)
    )
    lb = _isolated_lb()({}, group_scheduler=scheduler)
    report = _build(lb, _window())

    first = lb.report_idle_candidates(report)
    second = lb.report_idle_candidates(report)

    assert first == ack and second == ack
    assert lb.last_idle_report is report
    assert report.source_seq == 0
    assert lb._idle_source_seq == 0
    assert remote.call_count == 2
    assert all(call.args == (report,) for call in remote.call_args_list)
