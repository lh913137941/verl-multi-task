"""Regression coverage for two Fully Async jobs sharing a Ray namespace.

Tests the real TaskRunner._create_trainer method with lightweight stubs:
the native VERL pool-name collision happens before any GPU work starts.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/multi_task_scheduler/integration/verl/experimental_fully_async/task_runner.py"
)


def _build_trainer_stubs():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    klass = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MultiTaskFullyAsyncTaskRunner"
    )
    method = next(
        item for item in klass.body
        if isinstance(item, ast.FunctionDef) and item.name == "_create_trainer"
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ClassDef(
                    name="IsolatedTaskRunner",
                    bases=[],
                    keywords=[],
                    body=[method],
                    decorator_list=[],
                )
            ],
            type_ignores=[],
        )
    )
    created = []
    roles = SimpleNamespace(Rollout="rollout")
    class DummyTrainer:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            self.init_workers = SimpleNamespace(remote=lambda: "workers-started")

        @classmethod
        def remote(cls, **kwargs):
            instance = cls(kwargs)
            created.append(instance)
            return instance

    def create_resource_pool_manager(config, *, roles):
        assert "rollout" not in roles
        return SimpleNamespace(
            resource_pool_spec={"trainer_pool": [2], "other_pool": [1]},
            mapping={"actor": "trainer_pool", "ref": "trainer_pool",
                     "reward": "other_pool"},
            resource_pool_dict={},
        )

    scope = {
        "Role": roles,
        "MultiTaskFullyAsyncTrainer": DummyTrainer,
        "create_resource_pool_manager": create_resource_pool_manager,
        "ray": SimpleNamespace(get=lambda value: value),
    }
    exec(compile(module, str(SOURCE), "exec"), scope)
    return scope["IsolatedTaskRunner"], created


def _start_trainer(klass, session):
    runner = klass()
    runner.task_session = session
    runner.components = {
        "tokenizer": object(),
        "role_worker_mapping": {
            "actor": object(), "ref": object(),
            "reward": object(), "rollout": object(),
        },
        "ray_worker_group_cls": object(),
    }
    runner._create_trainer(SimpleNamespace(trainer=SimpleNamespace(device="npu")))
    return runner.components["trainer"].kwargs["resource_pool_manager"]


def test_independent_sessions_have_no_trainer_pg_name_collision():
    klass, created = _build_trainer_stubs()
    donor = _start_trainer(klass, "donor-actor")
    borrower = _start_trainer(klass, "borrower-actor")

    assert len(created) == 2
    assert set(donor.resource_pool_spec).isdisjoint(borrower.resource_pool_spec)
    assert donor.resource_pool_spec["trainer_pool_mt_donor-actor_"] == [2]
    assert borrower.resource_pool_spec["trainer_pool_mt_borrower-actor_"] == [2]
    # This is the exact PG-name construction used by RayResourcePool in VERL.
    donor_pg = f"{next(iter(donor.resource_pool_spec))}verl_group_1:0"
    borrower_pg = f"{next(iter(borrower.resource_pool_spec))}verl_group_1:0"
    assert donor_pg != borrower_pg
    assert "trainer_poolverl_group_1:0" not in (donor_pg, borrower_pg)


def test_all_roles_keep_same_pool_ownership_and_gpu_counts():
    klass, _ = _build_trainer_stubs()
    for session in ("task-a", "task-b"):
        pools = _start_trainer(klass, session)
        assert pools.mapping == {
            "actor": f"trainer_pool_mt_{session}_",
            "ref": f"trainer_pool_mt_{session}_",
            "reward": f"other_pool_mt_{session}_",
        }
        assert sum(sum(x) for x in pools.resource_pool_spec.values()) == 3
        assert pools.resource_pool_dict == {}


def test_missing_session_fails_before_resource_pool_or_trainer_creation():
    klass, created = _build_trainer_stubs()
    runner = klass()
    runner.task_session = None
    runner.components = {
        "tokenizer": object(),
        "role_worker_mapping": {"actor": object()},
        "ray_worker_group_cls": object(),
    }
    with pytest.raises(RuntimeError, match="require task_session"):
        runner._create_trainer(SimpleNamespace(trainer=SimpleNamespace(device="npu")))
    assert not created
