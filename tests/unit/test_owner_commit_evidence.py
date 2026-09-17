"""Regression tests for owner revisions and evidence-preserving routing commits.

The native VERL parents are replaced with inert test parents; these tests cover
only the orchestration overlay bodies and do not claim Ray/GPU integration.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "src/multi_task_scheduler"


def _isolated_class(relative, name, parent, **globals_for_test):
    path = SOURCE / relative
    parsed = ast.parse(path.read_text())
    node = next(item for item in parsed.body if isinstance(item, ast.ClassDef) and item.name == name)
    node.bases = [ast.Name(id="TestParent", ctx=ast.Load())]
    node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    scope = {"TestParent": parent, **globals_for_test}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[name]


def test_ce_membership_revision_is_monotonic_and_idempotent():
    class Parent:
        def __init__(self, *args, **kwargs):
            pass

    cls = _isolated_class(
        "checkpoint/checkpoint_engine_manager.py",
        "MultiTaskCheckpointEngineManager",
        Parent,
    )
    manager = cls()
    r1 = SimpleNamespace(replica_id="r1")
    r2 = SimpleNamespace(replica_id="r2")

    assert manager.add_effective_replica(None, r1) == 1
    assert manager.add_effective_replica(None, r1) == 1
    assert manager.add_effective_replica(None, r2) == 2
    assert manager.remove_effective_replica(None, "r1") == 3
    assert manager.remove_effective_replica(None, "missing") == 3
    assert manager.effective_revision == 3
    assert set(manager.effective_replicas) == {"r2"}


def test_lb_does_not_remove_replica_with_unsettled_attempts():
    class Parent:
        def __init__(self, *args, **kwargs):
            pass

    cls = _isolated_class(
        "rollout/load_balancer.py",
        "MultiTaskGlobalRequestLoadBalancer",
        Parent,
        DEFAULT_ROUTING_CACHE_SIZE=128,
    )
    lb = cls({"r1": object()})
    lb.begin_drain("r1")
    lb.attempts["r1"] = {"a1": SimpleNamespace(state="RUNNING")}
    assert lb.finish_remove("r1") is False
    assert "r1" in lb.draining_ids

    lb.attempts["r1"]["a1"] = SimpleNamespace(state="RELEASED")
    assert lb.finish_remove("r1") is True
    assert "r1" not in lb.draining_ids
    assert "r1" not in lb.routable_ids


def test_lb_new_route_requires_weight_member_and_lease_evidence():
    class Parent:
        def __init__(self, *args, **kwargs):
            pass

    cls = _isolated_class(
        "rollout/load_balancer.py",
        "MultiTaskGlobalRequestLoadBalancer",
        Parent,
        DEFAULT_ROUTING_CACHE_SIZE=128,
    )
    lb = cls({})
    with pytest.raises(ValueError, match="requires serving_version"):
        lb.commit_routable("borrowed-r1")
    epoch = lb.commit_routable(
        "borrowed-r1", serving_version=7, ce_revision=4, lease_valid=True
    )
    assert epoch == 1
    assert "borrowed-r1" in lb.routable_ids


def test_lb_can_cancel_drain_for_previously_committed_route_without_new_add_evidence():
    class Parent:
        def __init__(self, *args, **kwargs):
            pass

    cls = _isolated_class(
        "rollout/load_balancer.py",
        "MultiTaskGlobalRequestLoadBalancer",
        Parent,
        DEFAULT_ROUTING_CACHE_SIZE=128,
    )
    lb = cls({"native-r1": object()})
    assert lb.begin_drain("native-r1") == 1
    assert lb.commit_routable("native-r1") == 2
    assert "native-r1" in lb.routable_ids
    assert "native-r1" not in lb.draining_ids
