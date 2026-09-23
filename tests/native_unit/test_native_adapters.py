import asyncio
from types import SimpleNamespace

import pytest
import ray

from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncLLMServerManager,
    FullyAsyncRollouter,
)
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.workers.rollout.router import GlobalRequestLoadBalancer

from multi_task_scheduler.integration.verl.experimental_fully_async.llm_server_manager import (
    MultiTaskLLMServerManager,
)
from multi_task_scheduler.integration.verl.experimental_fully_async.message_queue import (
    MultiTaskMessageQueue,
)
from multi_task_scheduler.integration.verl.experimental_fully_async.rollouter import (
    MultiTaskFullyAsyncRollouter,
)
from multi_task_scheduler.integration.verl.experimental_fully_async.task_runner import (
    MultiTaskFullyAsyncTaskRunner,
)
from multi_task_scheduler.integration.verl.experimental_fully_async.trainer import (
    MultiTaskFullyAsyncTrainer,
)
from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.contracts import (
    AttemptState,
    EvidenceType,
    ReplicaKey,
)
from multi_task_scheduler.orchestration.exactly_once import DuplicateCompletionError
from multi_task_scheduler.rollout.load_balancer import MultiTaskGlobalRequestLoadBalancer

pytestmark = pytest.mark.native


def test_actor_subclasses_keep_native_business_methods():
    for extension, native, method in [
        (MultiTaskFullyAsyncTaskRunner, FullyAsyncTaskRunner, "_run_training_loop"),
        (MultiTaskFullyAsyncTrainer, FullyAsyncTrainer, "fit"),
        (MultiTaskFullyAsyncRollouter, FullyAsyncRollouter, "fit"),
    ]:
        extended_class = unwrap_native_actor_class(extension)
        native_class = unwrap_native_actor_class(native)
        assert issubclass(extended_class, native_class)
        assert getattr(extended_class, method) is getattr(native_class, method)


def test_manager_and_lb_extend_native_classes():
    assert issubclass(MultiTaskLLMServerManager, FullyAsyncLLMServerManager)
    assert issubclass(MultiTaskGlobalRequestLoadBalancer, GlobalRequestLoadBalancer)


def test_lb_adds_only_request_exit_metadata():
    server = object()
    lb = MultiTaskGlobalRequestLoadBalancer({"s": server}, full_determinism=True)
    assert lb.require_release_fields() == ["request_id"]
    assert lb.acquire_server("r") == ("s", server)
    assert lb.query_attempt("r") is AttemptState.ADMITTED
    lb.release_server("s", request_id="r")
    assert lb.query_attempt("r") is AttemptState.SETTLED


def test_late_release_does_not_erase_verified_continuation_terminal_state():
    server = object()
    lb = MultiTaskGlobalRequestLoadBalancer({"s": server}, full_determinism=True)
    lb.acquire_server("r")
    evidence = lb.confirm_continuation("r", "client-1", "prefix-1")
    assert evidence.type is EvidenceType.EXIT_READY
    assert lb.query_attempt("r") is AttemptState.TERMINATED
    lb.release_server("s", request_id="r")
    assert lb.query_attempt("r") is AttemptState.TERMINATED


def test_initial_native_route_and_drain_keep_exact_request_facts():
    server = object()
    key = ReplicaKey("task-a", "native-0")
    lb = MultiTaskGlobalRequestLoadBalancer(
        {"s": server},
        full_determinism=True,
        initial_routes={key: "s"},
    )
    assert lb.routes[key] == "s"
    lb.acquire_server("r")

    assert lb.begin_drain(key) == "s"
    assert "s" not in lb.get_all_servers()
    assert lb.requests_for_server("s") == ("r",)
    with pytest.raises(ValueError, match="requests remain admitted"):
        lb.finish_remove(key)

    lb.release_server("s", request_id="r")
    assert lb.query_attempt("r") is AttemptState.SETTLED
    lb.finish_remove(key)
    assert key not in lb.routes


def test_message_queue_deduplicates_completed_native_samples():
    queue_cls = unwrap_native_actor_class(MultiTaskMessageQueue)
    queue = queue_cls({}, max_queue_size=8, task_session="task-a")
    first_payload = ray.cloudpickle.dumps(SimpleNamespace(sample_id="s1", value=1))
    replay_payload = bytes(first_payload)

    first = asyncio.run(queue.put_sample_once(first_payload))
    replay = asyncio.run(queue.put_sample_once(replay_payload))
    assert replay is first
    assert asyncio.run(queue.get_queue_size()) == 1

    conflicting = ray.cloudpickle.dumps(SimpleNamespace(sample_id="s1", value=2))
    with pytest.raises(DuplicateCompletionError):
        asyncio.run(queue.put_sample_once(conflicting))
    assert asyncio.run(queue.get_queue_size()) == 1
