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
    DuplicateCompletionError,
    MultiTaskMessageQueue,
)
from multi_task_scheduler.integration.verl.experimental_fully_async.rollouter import (
    MultiTaskFullyAsyncRollouter,
    _ContinuationAwareServer,
    _MultiTaskFullyAsyncLLMServerClient,
    _continuation_prefix_digest,
)
from multi_task_scheduler.integration.verl.experimental_fully_async.task_runner import (
    MultiTaskFullyAsyncTaskRunner,
)
from multi_task_scheduler.integration.verl.experimental_fully_async.trainer import (
    MultiTaskFullyAsyncTrainer,
)
from verl.single_controller.ray.base import _unwrap_ray_remote
from multi_task_scheduler.orchestration.contracts import (
    AttemptState,
    EvidenceType,
    ReplicaKey,
)
from multi_task_scheduler.rollout.load_balancer import MultiTaskGlobalRequestLoadBalancer

pytestmark = pytest.mark.native


def test_actor_subclasses_keep_native_business_methods():
    for extension, native, method in [
        (MultiTaskFullyAsyncTaskRunner, FullyAsyncTaskRunner, "_run_training_loop"),
        (MultiTaskFullyAsyncTrainer, FullyAsyncTrainer, "fit"),
        (MultiTaskFullyAsyncRollouter, FullyAsyncRollouter, "fit"),
    ]:
        extended_class = _unwrap_ray_remote(extension)
        native_class = _unwrap_ray_remote(native)
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


def test_late_release_settles_verified_continuation():
    server = object()
    key = ReplicaKey("task-a", "native-0")
    lb = MultiTaskGlobalRequestLoadBalancer(
        {"s": server},
        full_determinism=True,
        initial_routes={key: "s"},
    )
    lb.acquire_server("r")
    lb.begin_drain(key, "op")
    evidence = lb.confirm_continuation("r", "client-1", "prefix-1")
    assert evidence.type is EvidenceType.EXIT_READY
    assert evidence.operation_id == "op"
    assert lb.query_attempt("r") is AttemptState.TERMINATED
    lb.release_server("s", request_id="r")
    assert lb.query_attempt("r") is AttemptState.SETTLED


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

    assert lb.begin_drain(key, "op") == "s"
    assert "s" not in lb.get_all_servers()
    assert lb.requests_for_server("s") == ("r",)
    with pytest.raises(ValueError, match="requests remain admitted"):
        lb.finish_remove(key)

    lb.release_server("s", request_id="r")
    assert lb.query_attempt("r") is AttemptState.SETTLED
    lb.finish_remove(key)
    assert key not in lb.routes


def test_message_queue_deduplicates_completed_native_samples():
    queue_cls = _unwrap_ray_remote(MultiTaskMessageQueue)
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


class _AsyncRemote:
    def __init__(self, fn):
        self._fn = fn

    async def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def test_fully_async_client_retries_exact_partial_prefix_on_another_server():
    calls = []
    acquire_count = {"value": 0}

    class Output:
        def __init__(self, token_ids, stop_reason, global_steps):
            self.token_ids = list(token_ids)
            self.log_probs = []
            self.routed_experts = None
            self.num_preempted = 0
            self.stop_reason = stop_reason
            self.extra_fields = {"global_steps": global_steps}

    class Server:
        def __init__(self, name, output):
            self.name = name
            self.output = output
            self.generate = _AsyncRemote(self._generate)

        def _generate(self, **kwargs):
            calls.append(
                (
                    "generate",
                    self.name,
                    tuple(kwargs["prompt_ids"]),
                )
            )
            return self.output

    first = Server("target", Output([31, 32], "aborted", 1))
    second = Server("alternate", Output([33, 34], "stop", 2))

    class ImmediateRemote:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

    class LB:
        require_acquire_fields = _AsyncRemote(lambda: [])
        require_release_fields = _AsyncRemote(lambda: ["request_id"])

        async def _acquire(self, request_id, **kwargs):
            acquire_count["value"] += 1
            server = first if acquire_count["value"] == 1 else second
            calls.append(("acquire", request_id, server.name))
            return server.name, server

        acquire_server = type(
            "AcquireRemote",
            (),
            {"remote": lambda self, *args, **kwargs: LB()._acquire(*args, **kwargs)},
        )()
        release_server = ImmediateRemote(
            lambda server_id, request_id=None:
            calls.append(("release", server_id, request_id))
        )
        confirm_continuation = _AsyncRemote(
            lambda request_id, client_id, prefix_digest:
            calls.append(("confirm", request_id, client_id, prefix_digest))
        )

    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                name="vllm",
                full_determinism=False,
                response_length=16,
            )
        ),
        async_training=SimpleNamespace(partial_rollout=True),
    )
    client = _MultiTaskFullyAsyncLLMServerClient(
        config=config,
        load_balancer_handle=LB(),
        client_id="task-a",
    )

    output = asyncio.run(
        client.generate(
            request_id="logical-1",
            prompt_ids=[11, 12],
            sampling_params={"max_tokens": 4},
        )
    )

    assert output.token_ids == [31, 32, 33, 34]
    assert output.stop_reason == "length"
    assert ("generate", "target", (11, 12)) in calls
    assert ("generate", "alternate", (11, 12, 31, 32)) in calls
    assert any(item[0] == "confirm" for item in calls)
    assert acquire_count["value"] == 2


def test_continuation_proxy_records_aborted_prefix_before_native_retry():
    calls = []

    class Output:
        stop_reason = "aborted"
        token_ids = [31, 32]

    class Server:
        generate = _AsyncRemote(
            lambda **kwargs: calls.append(("generate", tuple(kwargs["prompt_ids"]))) or Output()
        )

    class LB:
        confirm_continuation = _AsyncRemote(
            lambda request_id, client_id, prefix_digest: calls.append(
                ("confirm", request_id, client_id, prefix_digest)
            )
        )

    proxy = _ContinuationAwareServer(Server(), LB(), "req-1", "task-a")
    output = asyncio.run(
        proxy.generate.remote(
            request_id="backend-id",
            prompt_ids=[11, 12],
            sampling_params={},
        )
    )

    assert output.stop_reason == "aborted"
    assert calls == [
        ("generate", (11, 12)),
        (
            "confirm",
            "req-1",
            "task-a",
            _continuation_prefix_digest([11, 12], [31, 32]),
        ),
    ]
