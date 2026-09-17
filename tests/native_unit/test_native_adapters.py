"""Explicit real-parent tests; missing verl/vLLM dependencies fail collection.

These tests import actual parent classes and inspect inheritance/delegation.
They do not create GPU workers or prove full actor initialization/training.
"""

import inspect

import pytest

from verl.checkpoint_engine.base import CheckpointEngineManager, CheckpointEngineWorker
from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerManager, FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

from multi_task_scheduler.checkpoint.checkpoint_engine_manager import MultiTaskCheckpointEngineManager
from multi_task_scheduler.checkpoint.checkpoint_engine_worker import MultiTaskCheckpointEngineWorker
from multi_task_scheduler.integration.verl.experimental_fully_async.llm_server_manager import MultiTaskLLMServerManager
from multi_task_scheduler.integration.verl.experimental_fully_async.rollouter import MultiTaskFullyAsyncRollouter
from multi_task_scheduler.integration.verl.experimental_fully_async.task_runner import MultiTaskFullyAsyncTaskRunner
from multi_task_scheduler.integration.verl.experimental_fully_async.trainer import MultiTaskFullyAsyncTrainer
from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.rollout.http_server import MultiTaskvLLMHttpServer
from multi_task_scheduler.rollout.load_balancer import MultiTaskGlobalRequestLoadBalancer
from multi_task_scheduler.rollout.replica import MultiTaskvLLMReplica


pytestmark = pytest.mark.native


@pytest.mark.parametrize("extension,native,inherited", [
    (MultiTaskFullyAsyncTaskRunner, FullyAsyncTaskRunner, ("_initialize_components", "_run_training_loop")),
    (MultiTaskFullyAsyncTrainer, FullyAsyncTrainer, ("init_workers", "fit")),
    (MultiTaskFullyAsyncRollouter, FullyAsyncRollouter, ("init_workers", "fit")),
])
def test_real_actor_subclasses_preserve_native_methods_and_resource_options(extension, native, inherited):
    extended_class = unwrap_native_actor_class(extension)
    native_class = unwrap_native_actor_class(native)
    assert issubclass(extended_class, native_class)
    assert not inspect.isabstract(extended_class)
    assert extension._default_options == native._default_options
    for method in inherited:
        assert getattr(extended_class, method) is getattr(native_class, method)


@pytest.mark.parametrize("extension,native,inherited", [
    (MultiTaskLLMServerManager, FullyAsyncLLMServerManager, ("_initialize_llm_servers", "get_client", "get_replicas")),
    (MultiTaskGlobalRequestLoadBalancer, GlobalRequestLoadBalancer, ("acquire_server", "release_server", "add_servers")),
    (MultiTaskvLLMReplica, vLLMReplica, ("init_standalone", "launch_servers")),
    (MultiTaskvLLMHttpServer, vLLMHttpServer, ("__init__", "generate", "sleep")),
    (MultiTaskCheckpointEngineManager, CheckpointEngineManager, ("__init__", "update_weights")),
    (MultiTaskCheckpointEngineWorker, CheckpointEngineWorker, ("__init__", "update_weights")),
])
def test_real_ordinary_subclasses_inherit_original_business_methods(extension, native, inherited):
    assert issubclass(extension, native)
    assert not inspect.isabstract(extension)
    for method in inherited:
        assert getattr(extension, method) is getattr(native, method)


def test_real_native_task_runner_constructor_is_preserved_without_discovering_gs():
    runner = unwrap_native_actor_class(MultiTaskFullyAsyncTaskRunner)()
    assert runner.running is False
    assert runner.components == {}
    assert not runner.shutdown_event.is_set()
    assert runner.group_scheduler is None


def test_real_native_load_balancer_constructor_and_routing_are_preserved():
    server_reference = object()  # Native local routing test; not a server ActorHandle.
    scheduler_reference = object()
    balancer = MultiTaskGlobalRequestLoadBalancer(
        servers={"server-a": server_reference}, max_cache_size=4,
        full_determinism=True, group_scheduler=scheduler_reference,
    )
    assert balancer.group_scheduler is scheduler_reference
    assert balancer.acquire_server("sample-a") == ("server-a", server_reference)
    assert balancer.get_inflight_count("server-a") == 1
    balancer.release_server("server-a")
    assert balancer.get_inflight_count("server-a") == 0
