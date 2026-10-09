"""Two-job RewardLoop Actor naming regression with native behavior substitutes.

These tests extract production AST methods and need no Ray/VERL/NPU installation.
"""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/multi_task_scheduler/integration/verl/experimental_fully_async/rollouter.py"
)


def _isolated_reward_manager():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_TaskScopedRewardLoopManager"
    )
    cls.bases = [ast.Name(id="NativeRewardLoopManager", ctx=ast.Load())]
    cls.decorator_list = []

    registry = []

    class RewardWorkerClass:
        def options(self, *, name, scheduling_strategy):
            assert scheduling_strategy.soft is True
            registry.append((name, scheduling_strategy.node_id))
            return self

        def remote(self, config, router):
            assert router == "native-reward-router"
            return object()

    class NativeRewardLoopManager:
        def __init__(self, config, rm_resource_pool=None):
            assert rm_resource_pool is None
            self.config = config
            self.reward_router_address = "native-reward-router"
            self.reward_loop_workers_class = RewardWorkerClass()
            self._init_reward_loop_workers()

        def compute_rm_score(self, data):
            return ("native-compute", data)

    fake_ray = SimpleNamespace(
        nodes=lambda: [
            {"NodeID": "node-0", "Alive": True, "Resources": {"CPU": 16}},
            {"NodeID": "node-1", "Alive": True, "Resources": {"CPU": 8}},
        ],
        util=SimpleNamespace(
            scheduling_strategies=SimpleNamespace(
                NodeAffinitySchedulingStrategy=lambda *, node_id, soft: SimpleNamespace(
                    node_id=node_id, soft=soft
                ),
            ),
        ),
    )
    env = {"ray": fake_ray, "NativeRewardLoopManager": NativeRewardLoopManager}
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), env)
    return env["_TaskScopedRewardLoopManager"], registry


def test_two_verl_jobs_create_distinct_reward_loop_actor_names():
    cls, registry = _isolated_reward_manager()
    config = SimpleNamespace(reward=SimpleNamespace(num_workers=3))
    donor = cls(config=config, task_session="donor-actor")
    borrower = cls(config=config, task_session="borrower-actor")
    assert len(donor.reward_loop_workers) == len(borrower.reward_loop_workers) == 3
    assert [name for name, _ in registry] == [
        "reward_loop_worker_0_mt_donor-actor",
        "reward_loop_worker_1_mt_donor-actor",
        "reward_loop_worker_2_mt_donor-actor",
        "reward_loop_worker_0_mt_borrower-actor",
        "reward_loop_worker_1_mt_borrower-actor",
        "reward_loop_worker_2_mt_borrower-actor",
    ]
    assert [node for _, node in registry] == [
        "node-0", "node-1", "node-0", "node-0", "node-1", "node-0",
    ]
    assert donor.compute_rm_score("batch") == ("native-compute", "batch")
    assert not {name for name, _ in registry} & {
        "reward_loop_worker_0", "reward_loop_worker_1", "reward_loop_worker_2"
    }


def test_reward_loop_rejects_absent_task_session_before_launching_actors():
    cls, registry = _isolated_reward_manager()
    config = SimpleNamespace(reward=SimpleNamespace(num_workers=2))
    for session in (None, "", 7):
        with pytest.raises(ValueError, match="task_session"):
            cls(config=config, task_session=session)
    assert registry == []


def test_rollouter_passes_session_to_reward_loop_worker_manager():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    rollouter = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MultiTaskFullyAsyncRollouter"
    )
    method = next(
        node for node in rollouter.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_create_reward_loop_manager"
    )
    class_node = ast.ClassDef(
        name="IsolatedRollouter",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    created = []

    def make_manager(*, config, rm_resource_pool, task_session):
        created.append((config, rm_resource_pool, task_session))
        return object()

    env = {
        "asyncio": asyncio,
        "_TaskScopedRewardLoopManager": make_manager,
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[class_node], type_ignores=[])),
                str(SOURCE), "exec"),
        env,
    )
    rollouter_obj = env["IsolatedRollouter"]()
    rollouter_obj.config = object()
    rollouter_obj.task_session = "task-session-123"
    asyncio.run(rollouter_obj._create_reward_loop_manager())
    assert created == [(rollouter_obj.config, None, "task-session-123")]
    assert rollouter_obj.reward_loop_manager is not None

    rollouter_obj.task_session = ""
    with pytest.raises(RuntimeError, match="task_session"):
        asyncio.run(rollouter_obj._create_reward_loop_manager())
    assert len(created) == 1
