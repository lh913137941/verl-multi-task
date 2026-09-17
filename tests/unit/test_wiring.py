"""Scoped wiring checks for the current orchestration surface only.

These tests execute selected class bodies with inert runtime substitutes. They
validate argument/owner wiring and explicit unsupported boundaries, not real
VERL, Ray GPU, CUDA or NCCL execution.
"""

import ast
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
    Phase,
)
from multi_task_scheduler.orchestration.production_window import (
    ProductionWindow,
    WindowState,
)
from multi_task_scheduler.orchestration.receipts import ReleaseEvidence, ServiceEvidence
from multi_task_scheduler.orchestration.replica_sync_gate import (
    GateFencedError,
    GateKind,
    ReplicaSyncGate,
)

SOURCE = Path(__file__).resolve().parents[2] / "src/multi_task_scheduler"
INTEGRATION = "integration/verl/experimental_fully_async"


def _isolated_class(relative, name, parent, **globals_for_test):
    path = SOURCE / relative
    parsed = ast.parse(path.read_text())
    node = next(item for item in parsed.body if isinstance(item, ast.ClassDef) and item.name == name)
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
    scope = {"TestParent": parent, **globals_for_test}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[name]


def _command():
    ctx = OperationContext(
        protocol_version=1,
        gs_epoch="gs-1",
        task_id="task-a",
        task_session="s1",
        operation_id="op-1",
        lease_id="l1",
        lease_epoch=0,
        command_seq=0,
    )
    target = ReplicaKey(task_session="s1", replica_id="r1", runtime_epoch=0)
    placement = PlacementSpec(
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
    return OperationCommand(
        ctx=ctx,
        kind=OperationKind.ADD,
        target=target,
        authorization=LeaseAuthorization(
            lease_id="l1",
            gs_epoch="gs-1",
            donor_session="donor",
            borrower_session="s1",
            placement_digest="placement-u0",
            lease_epoch=0,
            purpose=OperationKind.ADD,
            prior_release_digest="release-0",
            authorization_seq=1,
        ),
        payload_digest="payload-1",
        remaining_budget_ms=1000,
        placement=placement,
    )


def _window(revision=9):
    return ProductionWindow(
        task_session="s1",
        epoch=3,
        revision=revision,
        state=WindowState.EXHAUSTED,
        eligible_pending=0,
        held_samples=0,
        active_samples=1,
        output_queue_size=0,
        max_queue_size=8,
        producer_exhausted=True,
        policy_refresh_inflight=False,
    )


def test_task_runner_submit_and_query_use_only_current_contract():
    class Parent:
        def __init__(self):
            self.components = {}

    runner_class = _isolated_class(
        f"{INTEGRATION}/task_runner.py",
        "MultiTaskFullyAsyncTaskRunner",
        Parent,
        OperationJournal=OperationJournal,
        OperationCommand=OperationCommand,
        OperationResult=OperationResult,
        QueryResult=QueryResult,
        OperationIdentityError=OperationIdentityError,
        OperationStatus=OperationStatus,
        Outcome=Outcome,
        Phase=Phase,
        ServiceEvidence=ServiceEvidence,
        ReleaseEvidence=ReleaseEvidence,
    )
    runner = runner_class()
    command = _command()
    first = runner.submit_operation(command)
    second = runner.submit_operation(command)
    assert first == second
    assert first.status is OperationStatus.ACCEPTED
    query = runner.query_operation("s1", "op-1")
    assert query.found is True
    assert query.value == first
    assert not hasattr(runner, "begin_operation")


def test_task_runner_attaches_before_native_run_and_detaches_after():
    events = []
    task_handle = object()

    class Parent:
        def __init__(self):
            self.components = {}

        def run(self, config):
            events.append("native_run")
            return "ok"

    scheduler = SimpleNamespace(
        attach_task=SimpleNamespace(remote=Mock(side_effect=lambda *args: events.append("attach"))),
        detach_task=SimpleNamespace(remote=Mock(side_effect=lambda *args: events.append("detach"))),
    )
    runner_class = _isolated_class(
        f"{INTEGRATION}/task_runner.py",
        "MultiTaskFullyAsyncTaskRunner",
        Parent,
        ray=SimpleNamespace(
            get=lambda value, **kwargs: value,
            get_runtime_context=lambda: SimpleNamespace(
                get_actor_id=lambda: "task-a", current_actor=task_handle
            ),
        ),
        get_or_create_group_scheduler=lambda: scheduler,
        logger=logging.getLogger(__name__),
    )
    runner = runner_class()
    assert runner.run(object()) == "ok"
    assert events == ["attach", "native_run", "detach"]


def test_manager_forwards_native_load_balancer_configuration():
    replica_class = object()
    load_balancer_class = object()
    actor_class = SimpleNamespace(remote=Mock(return_value=object()))
    ray_substitute = SimpleNamespace(remote=Mock(return_value=actor_class))

    class Parent:
        def __init__(self, *args):
            assert self.rollout_replica_class is replica_class

    manager_class = _isolated_class(
        f"{INTEGRATION}/llm_server_manager.py",
        "MultiTaskLLMServerManager",
        Parent,
        MultiTaskvLLMReplica=replica_class,
        MultiTaskGlobalRequestLoadBalancer=load_balancer_class,
        DEFAULT_ROUTING_CACHE_SIZE=123,
        ray=ray_substitute,
    )
    manager = manager_class(object(), rollout_resource_pool=object(), group_scheduler="gs")
    manager.server_addresses = ["a", "b"]
    manager.server_handles = [object(), object()]
    manager.rollout_config = SimpleNamespace(full_determinism=True)
    asyncio.run(manager._init_global_load_balancer())
    actor_class.remote.assert_called_once_with(
        servers=dict(zip(manager.server_addresses, manager.server_handles, strict=True)),
        max_cache_size=123,
        full_determinism=True,
        group_scheduler="gs",
    )


def test_rollouter_owns_production_window_but_not_idle_reporting():
    class Parent:
        def __init__(self, *args, **kwargs):
            pass

    rollouter_class = _isolated_class(
        f"{INTEGRATION}/rollouter.py",
        "MultiTaskFullyAsyncRollouter",
        Parent,
        MultiTaskLLMServerManager=SimpleNamespace(),
        FullyAsyncAgentLoopManager=SimpleNamespace(),
        FullyAsyncLLMServerClient=object(),
        ProductionWindow=ProductionWindow,
    )
    rollouter = rollouter_class(object(), object(), None, "cuda")
    assert rollouter.production_window is None

    window = _window()
    rollouter.set_production_window(window)
    assert rollouter.production_window is window
    assert not hasattr(rollouter, "report_idle_candidates")
    assert not hasattr(rollouter, "begin_drain")

    with pytest.raises(ValueError, match="revision"):
        rollouter.set_production_window(_window(revision=8))


def test_rollouter_lifecycle_runtime_actions_stay_explicitly_unimplemented():
    class Parent:
        def __init__(self, *args, **kwargs):
            pass

    rollouter_class = _isolated_class(
        f"{INTEGRATION}/rollouter.py",
        "MultiTaskFullyAsyncRollouter",
        Parent,
        MultiTaskLLMServerManager=SimpleNamespace(),
        FullyAsyncAgentLoopManager=SimpleNamespace(),
        FullyAsyncLLMServerClient=object(),
        ProductionWindow=ProductionWindow,
    )
    rollouter = rollouter_class(object(), object(), None, "cuda")
    for method, args in (
        (rollouter.prepare_replica, (None, None, None)),
        (rollouter.prepare_exit, (None,)),
        (rollouter.revalidate_exit, (None, None)),
        (rollouter.commit_service, (None, None, None, None)),
        (rollouter.commit_removal, (None, None, None)),
        (rollouter.finalize_release, (None, None)),
    ):
        with pytest.raises(NotImplementedError):
            method(*args)


def test_trainer_native_sync_uses_gate_and_unknown_failure_blocks_followups():
    async def scenario(failure):
        class Parent:
            async def _fit_update_weights(self):
                assert self.replica_sync_gate.owner.kind is GateKind.NATIVE_SYNC
                if failure:
                    raise RuntimeError("native sync interrupted")
                return {"timing": 1}

        trainer_class = _isolated_class(
            f"{INTEGRATION}/trainer.py",
            "MultiTaskFullyAsyncTrainer",
            Parent,
            ReplicaSyncGate=ReplicaSyncGate,
            GateKind=GateKind,
        )
        trainer = trainer_class()
        trainer.local_trigger_step = 1
        trainer.current_param_version = 7
        if failure:
            with pytest.raises(RuntimeError):
                await trainer._fit_update_weights()
            assert trainer.replica_sync_gate.health == "BLOCKED"
            with pytest.raises(GateFencedError):
                await trainer.replica_sync_gate.acquire("later", GateKind.ADD)
        else:
            assert await trainer._fit_update_weights() == {"timing": 1}
            assert trainer.replica_sync_gate.health == "HEALTHY"
        assert trainer.replica_sync_gate.owner is None

    asyncio.run(scenario(False))
    asyncio.run(scenario(True))


def test_trainer_exposes_only_snapshot_based_current_entry_points():
    trainer_class = _isolated_class(
        f"{INTEGRATION}/trainer.py",
        "MultiTaskFullyAsyncTrainer",
        object,
        omega_conf_to_dataclass=Mock(),
        MultiTaskCheckpointEngineManager=Mock(),
        ReplicaSyncGate=ReplicaSyncGate,
    )
    trainer = trainer_class()
    assert trainer.current_snapshot() is None
    assert not hasattr(trainer, "publish_serving_version")
    with pytest.raises(NotImplementedError):
        asyncio.run(trainer.bootstrap_and_publish(None, None))
    with pytest.raises(NotImplementedError):
        asyncio.run(trainer.remove_and_commit(None, None))
    with pytest.raises(NotImplementedError):
        asyncio.run(trainer.restore_and_publish(None, None))
