"""Scoped wiring tests for the current orchestration surface only.

These tests execute selected class bodies with explicit runtime substitutes.
They do not import verl GPU runtimes or claim CUDA/NCCL validation.
"""

import ast
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from multi_task_scheduler.orchestration.contracts import (
    LeaseAuthorization,
    OperationCommand,
    OperationContext,
    OperationResult,
    PlacementSpec,
    NodePlacement,
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
    ReplicaObservation,
    ReplicaView,
    select_idle_candidates,
)
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
    authorization = LeaseAuthorization(
        lease_id="l1",
        gs_epoch="gs-1",
        donor_session="donor",
        borrower_session="s1",
        placement_digest="placement-u0",
        lease_epoch=0,
        purpose=OperationKind.ADD,
        prior_release_digest="release-0",
        authorization_seq=1,
    )
    return OperationCommand(
        ctx=ctx,
        kind=OperationKind.ADD,
        target=target,
        authorization=authorization,
        payload_digest="d1",
        remaining_budget_ms=1000,
        placement=placement,
    )


def test_task_runner_creation_preserves_role_arguments_and_initialization_order():
    events = []

    def rpc(name):
        return SimpleNamespace(remote=Mock(side_effect=lambda: events.append(name)))

    trainer = SimpleNamespace(init_workers=rpc("trainer.init_workers"))
    rollouter = SimpleNamespace(
        init_workers=rpc("rollouter.init_workers"),
        set_max_required_samples=rpc("rollouter.set_max_required_samples"),
    )
    trainer_class = SimpleNamespace(remote=Mock(return_value=trainer))
    rollouter_class = SimpleNamespace(remote=Mock(return_value=rollouter))
    role = SimpleNamespace(Rollout=object())
    resource_pool = object()
    pool_factory = Mock(return_value=resource_pool)
    runner_class = _isolated_class(
        f"{INTEGRATION}/task_runner.py",
        "MultiTaskFullyAsyncTaskRunner",
        object,
        ray=SimpleNamespace(get=lambda value: value),
        Role=role,
        MultiTaskFullyAsyncTrainer=trainer_class,
        MultiTaskFullyAsyncRollouter=rollouter_class,
        create_resource_pool_manager=pool_factory,
    )
    runner = runner_class()
    runner.group_scheduler = object()
    runner.components = {
        "tokenizer": object(),
        "processor": object(),
        "ray_worker_group_cls": object(),
        "role_worker_mapping": {"actor": object(), role.Rollout: object()},
    }
    config = SimpleNamespace(trainer=SimpleNamespace(device="cuda"))
    runner._create_trainer(config)
    runner._create_rollouter(config)
    trainer_class.remote.assert_called_once_with(
        config=config,
        tokenizer=runner.components["tokenizer"],
        role_worker_mapping={"actor": runner.components["role_worker_mapping"]["actor"]},
        resource_pool_manager=resource_pool,
        ray_worker_group_cls=runner.components["ray_worker_group_cls"],
        device_name="cuda",
    )
    rollouter_class.remote.assert_called_once_with(
        config=config,
        tokenizer=runner.components["tokenizer"],
        processor=runner.components["processor"],
        device_name="cuda",
        group_scheduler=runner.group_scheduler,
    )
    assert events == [
        "trainer.init_workers",
        "rollouter.init_workers",
        "rollouter.set_max_required_samples",
    ]


@pytest.mark.parametrize("native_failure", [False, True])
def test_runner_attaches_before_native_run_and_detaches_after(native_failure):
    events = []
    task_handle = object()
    config = object()

    class Parent:
        def __init__(self):
            self.components = {}

        def run(self, received):
            assert received is config
            events.append("native_run")
            if native_failure:
                raise RuntimeError("native failure")
            return "native result"

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
    if native_failure:
        with pytest.raises(RuntimeError, match="native failure"):
            runner.run(config)
    else:
        assert runner.run(config) == "native result"
    assert events == ["attach", "native_run", "detach"]


def test_task_runner_uses_only_current_submit_and_query_contract():
    class Parent:
        def __init__(self):
            pass

    runner_class = _isolated_class(
        f"{INTEGRATION}/task_runner.py",
        "MultiTaskFullyAsyncTaskRunner",
        Parent,
        OperationJournal=OperationJournal,
        OperationCommand=OperationCommand,
        OperationResult=OperationResult,
        QueryResult=QueryResult,
        OperationIdentityError=OperationIdentityError,
        Outcome=Outcome,
    )
    runner = runner_class()
    command = _command()
    first = runner.submit_operation(command)
    second = runner.submit_operation(command)
    assert first == second
    assert first.status is OperationStatus.ACCEPTED
    query = runner.query_operation("s1", "op-1")
    assert query.found is True
    assert query.value.status is OperationStatus.ACCEPTED
    assert query.outcome is Outcome.UNKNOWN
    assert not hasattr(runner, "begin_operation")


def test_manager_forwards_native_load_balancer_flags():
    replica_class = object()
    load_balancer_class = object()
    forwarded = []

    class Parent:
        def __init__(self, *args):
            assert self.rollout_replica_class is replica_class
            forwarded.append(args)

    actor_class = SimpleNamespace(remote=Mock(return_value=object()))
    ray_substitute = SimpleNamespace(remote=Mock(return_value=actor_class))
    manager_class = _isolated_class(
        f"{INTEGRATION}/llm_server_manager.py",
        "MultiTaskLLMServerManager",
        Parent,
        MultiTaskvLLMReplica=replica_class,
        MultiTaskGlobalRequestLoadBalancer=load_balancer_class,
        DEFAULT_ROUTING_CACHE_SIZE=123,
        ray=ray_substitute,
    )
    config, pool, scheduler = object(), object(), object()
    manager = manager_class(config, rollout_resource_pool=pool, group_scheduler=scheduler)
    assert forwarded == [(config, None, pool)]
    manager.server_addresses = ["server-a", "server-b"]
    manager.server_handles = [object(), object()]
    manager.rollout_config = SimpleNamespace(full_determinism=True)
    asyncio.run(manager._init_global_load_balancer())
    actor_class.remote.assert_called_once_with(
        servers=dict(zip(manager.server_addresses, manager.server_handles, strict=True)),
        max_cache_size=123,
        full_determinism=True,
        group_scheduler=scheduler,
    )


def test_load_balancer_constructor_preserves_native_arguments_and_gs_handle():
    constructor = Mock()

    class Parent:
        def __init__(self, *args, **kwargs):
            constructor(*args, **kwargs)

    balancer_class = _isolated_class(
        "rollout/load_balancer.py",
        "MultiTaskGlobalRequestLoadBalancer",
        Parent,
        DEFAULT_ROUTING_CACHE_SIZE=123,
    )
    servers, scheduler = {"server-a": object()}, object()
    balancer = balancer_class(
        servers, max_cache_size=5, full_determinism=True, group_scheduler=scheduler
    )
    constructor.assert_called_once_with(servers, max_cache_size=5, full_determinism=True)
    assert balancer.group_scheduler is scheduler


def test_replica_borrowed_cuda_setup_fails_and_native_delegates():
    class Parent:
        def __init__(self, *args, **kwargs):
            pass

        def _setup_env_cuda_visible_devices(self, *args, **kwargs):
            return "native-setup"

    ray_sub = SimpleNamespace(remote=Mock(side_effect=[object(), object()]))
    replica_class = _isolated_class(
        "rollout/replica.py",
        "MultiTaskvLLMReplica",
        Parent,
        ray=ray_sub,
        MultiTaskvLLMHttpServer=object(),
        MultiTaskCheckpointEngineWorker=object(),
        RayClassWithInitArgs=Mock(),
    )
    native = replica_class("model", replica_rank=0)
    assert native._setup_env_cuda_visible_devices() == "native-setup"
    borrowed = replica_class("model", replica_rank=1, replica_kind="borrowed")
    with pytest.raises(NotImplementedError):
        borrowed._setup_env_cuda_visible_devices()


def test_rollouter_passes_gs_and_preserves_native_agent_loop_arguments():
    constructor = Mock()

    class Parent:
        def __init__(self, *args, **kwargs):
            constructor(*args, **kwargs)

    manager = SimpleNamespace(get_client=Mock(return_value=object()))
    manager_factory = SimpleNamespace(create=AsyncMock(return_value=manager))
    agent_factory = SimpleNamespace(create=AsyncMock(return_value=object()))
    client_class = object()
    rollouter_class = _isolated_class(
        f"{INTEGRATION}/rollouter.py",
        "MultiTaskFullyAsyncRollouter",
        Parent,
        MultiTaskLLMServerManager=manager_factory,
        FullyAsyncAgentLoopManager=agent_factory,
        FullyAsyncLLMServerClient=client_class,
    )
    config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(mode="async")))
    scheduler = object()
    rollouter = rollouter_class(config, object(), object(), "cuda", group_scheduler=scheduler)
    constructor.assert_called_once()
    rollouter.config = config
    rollouter.use_rm = False
    rollouter.reward_loop_manager = SimpleNamespace(reward_loop_workers=[object()])
    rollouter.teacher_model_manager = SimpleNamespace(get_client=Mock(return_value=object()))
    rollouter.get_hybrid_worker_group = Mock(return_value=None)
    asyncio.run(rollouter._init_async_rollout_manager())
    manager_factory.create.assert_awaited_once_with(
        config=config, worker_group=None, group_scheduler=scheduler
    )
    agent_factory.create.assert_awaited_once()


def test_rollouter_exposes_only_current_lifecycle_entry_points():
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
        select_idle_candidates=select_idle_candidates,
    )
    rollouter = rollouter_class(object(), object(), None, "cuda")
    rollouter.set_production_window(
        ProductionWindow(
            production_epoch=3,
            source_seq=1,
            exhausted_this_round=True,
            eligible_pending=0,
            held=0,
        )
    )
    observation = ReplicaObservation(
        replica_id="r1",
        in_flight=0,
        admitting=0,
        queued=0,
        running=0,
        pending_admissions=0,
        production_epoch=3,
        all_backends_observed=True,
    )
    report = rollouter.report_idle_candidates(
        rollouter.production_window,
        [ReplicaView(observation)],
        observations_fresh=True,
        min_active_gpus=1,
        current_active_gpus=2,
        routable_count=2,
    )
    assert report.candidate_ids == ("r1",)
    with pytest.raises(NotImplementedError):
        rollouter.prepare_exit(None)
    assert not hasattr(rollouter, "begin_drain")


def test_trainer_gate_and_current_transaction_entry_points():
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
    assert isinstance(trainer.replica_sync_gate, ReplicaSyncGate)
    with pytest.raises(NotImplementedError):
        asyncio.run(trainer.bootstrap_and_publish(None, None))
    with pytest.raises(NotImplementedError):
        asyncio.run(trainer.remove_and_commit(None, None))
    with pytest.raises(NotImplementedError):
        asyncio.run(trainer.restore_and_publish(None, None))
    assert not hasattr(trainer, "publish_serving_version")


@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError])
def test_native_sync_runs_under_gate_and_uncertainty_blocks_followups(failure):
    async def scenario():
        events = []

        class Parent:
            async def _fit_update_weights(self):
                assert self.replica_sync_gate.owner.kind is GateKind.NATIVE_SYNC
                events.append("native_sync")
                if failure:
                    raise failure("native sync interrupted")
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
            with pytest.raises(failure):
                await trainer._fit_update_weights()
            assert trainer.replica_sync_gate.health == "BLOCKED"
            with pytest.raises(GateFencedError):
                await trainer.replica_sync_gate.acquire("add", GateKind.ADD)
        else:
            assert await trainer._fit_update_weights() == {"timing": 1}
            assert trainer.replica_sync_gate.health == "HEALTHY"
        assert trainer.replica_sync_gate.owner is None
        assert events == ["native_sync"]

    asyncio.run(scenario())
