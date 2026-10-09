"""SERVER wiring regression scenarios; shared test fakes live in _wiring_support."""

import asyncio
import pytest
from _wiring_support import (
    isolated,
    http_server_class,
)


def test_runtime_health_fails_fast_when_vllm_sleep_api_is_incompatible():
    class Engine:
        async def check_health(self):
            return None

        async def is_sleeping(self):
            return False

        async def sleep(self):
            return None

        async def wake_up(self, tags=None):
            return None

        async def wait_for_requests_to_drain(self):
            return None

        async def reset_prefix_cache(self, reset_connector=False):
            return True

    server = http_server_class()()
    server.nnodes = 1
    server.node_rank = 0
    server._server_port = 12345
    server._server_task = type("Task", (), {"done": lambda self: False})()
    server.engine = Engine()

    with pytest.raises(RuntimeError, match=r"sleep\(level"):
        asyncio.run(server.runtime_health())


def test_http_server_health_and_shutdown_use_real_engine_boundaries():
    class Engine:
        def __init__(self):
            self.healthy = False
            self.drained = False
            self.shutdown_called = False

        async def check_health(self):
            self.healthy = True

        async def is_sleeping(self):
            return False

        async def sleep(self, level=1, mode="abort"):
            return None

        async def wake_up(self, tags=None):
            return None

        async def reset_prefix_cache(
            self,
            reset_running_requests=False,
            reset_connector=False,
        ):
            return True

        async def wait_for_requests_to_drain(self):
            self.drained = True

        def shutdown(self):
            self.shutdown_called = True

    fake_ray = type(
        "ServerRay",
        (),
        {
            "get_runtime_context": staticmethod(
                lambda: type("Context", (), {"get_node_id": lambda self: "node-a"})()
            )
        },
    )

    class Parent:
        pass

    cls = isolated(
        "rollout/http_server.py",
        "MultiTaskvLLMHttpServer",
        Parent,
        asyncio=asyncio,
        inspect=__import__("inspect"),
        ray=fake_ray,
    )

    async def scenario():
        server = cls()
        server.nnodes = 1
        server.node_rank = 0
        server.replica_rank = 5
        server._server_address = "127.0.0.1"
        server._server_port = 12345
        server.global_steps = None
        server._submission_paused = False
        server._resume_event = asyncio.Event()
        server._resume_event.set()
        server.engine = Engine()
        server._server_task = asyncio.create_task(asyncio.sleep(3600))

        health = await server.runtime_health()
        assert health["node_id"] == "node-a"
        assert server.engine.healthy is True

        engine = server.engine
        assert await server.shutdown_runtime() is None
        assert engine.drained is True
        assert engine.shutdown_called is True
        assert server.engine is None
        assert server._server_port is None
        assert server._server_task.done()

    asyncio.run(scenario())


def test_borrowed_shutdown_cancels_drained_vllm_output_handler_before_engine_core():
    """EngineCore teardown must not race vLLM's still-polling output handler."""
    events = []

    class Engine:
        def __init__(self):
            self.output_handler = None

        async def wait_for_requests_to_drain(self):
            events.append("drained")

        def shutdown(self):
            # Simulate an engine which would signal EngineDeadError to the
            # polling output handler if the core were shut down first.
            assert self.output_handler is not None
            assert self.output_handler.cancelled()
            events.append("engine-core-shutdown")

    class Parent:
        pass

    cls = isolated(
        "rollout/http_server.py",
        "MultiTaskvLLMHttpServer",
        Parent,
        asyncio=asyncio,
    )

    async def scenario():
        server = cls()
        server.nnodes = 1
        server.node_rank = 0
        server._server_port = 11223
        server._submission_paused = False
        server._resume_event = asyncio.Event()
        server._resume_event.set()
        server._server_task = asyncio.create_task(asyncio.sleep(3600))
        engine = Engine()
        server.engine = engine

        async def output_polling():
            try:
                await asyncio.Event().wait()
            finally:
                events.append("output-handler-cancelled")

        engine.output_handler = asyncio.create_task(output_polling())
        await asyncio.sleep(0)
        await server.shutdown_runtime()
        assert events == [
            "drained",
            "output-handler-cancelled",
            "engine-core-shutdown",
        ]
        assert server.engine is None
        assert server._server_task.done()

    asyncio.run(scenario())


def test_borrowed_shutdown_does_not_cancel_output_polling_before_drain():
    """A failed drain must not hide an active EngineCore error or kill requests."""
    events = []

    class Engine:
        def __init__(self):
            self.output_handler = None

        async def wait_for_requests_to_drain(self):
            events.append("drain")
            raise RuntimeError("requests still active")

        def shutdown(self):
            raise AssertionError("cannot destroy an undrained engine")

    cls = isolated(
        "rollout/http_server.py",
        "MultiTaskvLLMHttpServer",
        type("Parent", (), {}),
        asyncio=asyncio,
    )

    async def scenario():
        server = cls()
        server.nnodes = 1
        server.node_rank = 0
        server._submission_paused = False
        server._resume_event = asyncio.Event()
        server._resume_event.set()
        server.engine = Engine()
        server._server_task = asyncio.create_task(asyncio.sleep(3600))
        server.engine.output_handler = asyncio.create_task(asyncio.sleep(3600))
        try:
            with pytest.raises(RuntimeError, match="requests still active"):
                await server.shutdown_runtime()
            assert events == ["drain"]
            assert not server.engine.output_handler.cancelled()
        finally:
            server.engine.output_handler.cancel()
            server._server_task.cancel()
            await asyncio.gather(
                server.engine.output_handler,
                server._server_task,
                return_exceptions=True,
            )

    asyncio.run(scenario())


def test_standalone_server_waits_for_requests_already_past_local_gate():
    server = http_server_class()()
    server._admitting = 1

    async def exercise():
        async def finish_admission():
            await asyncio.sleep(0.02)
            server._admitting = 0

        task = asyncio.create_task(finish_admission())
        await server._wait_admission_barrier(timeout_s=0.2)
        await task

    asyncio.run(exercise())
    assert server._admitting == 0


def test_standalone_server_admission_barrier_timeout_is_not_success():
    server = http_server_class()()
    server._admitting = 1

    with pytest.raises(RuntimeError, match="admission barrier timed out"):
        asyncio.run(server._wait_admission_barrier(timeout_s=0.01))


def test_standalone_server_level2_sleep_and_two_phase_wake_keep_admission_fenced():
    calls = []

    class Event:
        def __init__(self):
            self.is_set = True

        def clear(self):
            self.is_set = False
            calls.append(("gate", "clear"))

        def set(self):
            self.is_set = True
            calls.append(("gate", "set"))

    class Engine:
        def __init__(self):
            self.sleeping = False

        async def wait_for_requests_to_drain(self):
            calls.append(("drain",))

        async def sleep(self, *, level, mode):
            calls.append(("sleep", level, mode))
            self.sleeping = True

        async def is_sleeping(self):
            calls.append(("is_sleeping", self.sleeping))
            return self.sleeping

        async def wake_up(self, *, tags=None):
            calls.append(("wake_up", tuple(tags) if tags is not None else None))
            if tags == ["weights"]:
                self.sleeping = True
            else:
                self.sleeping = False

        async def reset_prefix_cache(self, *, reset_connector):
            calls.append(("reset_prefix_cache", reset_connector))

        async def check_health(self):
            calls.append(("health",))

    server = http_server_class()()
    server.nnodes = 1
    server.node_rank = 0
    server.replica_rank = 3
    server.global_steps = 7
    server.config = type(
        "Config",
        (),
        {"enable_sleep_mode": True, "free_cache_engine": True},
    )()
    server.engine = Engine()
    server._resolve_sleep_level = lambda: 2
    server._submission_paused = False
    server._resume_event = Event()

    sleep_receipt = asyncio.run(server.sleep())
    assert sleep_receipt["sleep_level"] == 2
    assert sleep_receipt["sleeping"] is True
    assert server._submission_paused is True
    assert server._resume_event.is_set is False

    weights_receipt = asyncio.run(server.wake_up(tags=["weights"]))
    assert weights_receipt["fully_awake"] is False
    assert weights_receipt["sleeping"] is True
    assert server._submission_paused is True
    assert server._resume_event.is_set is False

    # VERL's inherited resume_kv_cache() restores KV/reset cache while the
    # MultiTask admission gate remains closed. Model that completed native step
    # here; this test only owns the final MultiTask admission commit.
    server.engine.sleeping = False

    wake_receipt = asyncio.run(server.wake_up())
    assert wake_receipt["fully_awake"] is True
    assert wake_receipt["sleeping"] is False
    assert server._submission_paused is False
    assert server._resume_event.is_set is True
    assert ("sleep", 2, "abort") in calls
    assert ("wake_up", ("weights",)) in calls
    assert ("reset_prefix_cache", True) not in calls
    assert calls[-2:] == [("health",), ("gate", "set")]


def test_standalone_server_npu_accepts_level1_sleep_without_mode():
    calls = []

    class Event:
        def __init__(self):
            self.is_set = True

        def clear(self):
            self.is_set = False

        def set(self):
            self.is_set = True

    class Engine:
        def __init__(self):
            self.sleeping = False

        async def wait_for_requests_to_drain(self):
            calls.append(("drain",))

        async def sleep(self, *, level):
            calls.append(("sleep", level))
            self.sleeping = True

        async def is_sleeping(self):
            return self.sleeping

    server = http_server_class("NPU")()
    server.nnodes = 1
    server.node_rank = 0
    server.replica_rank = 0
    server.global_steps = 1
    server.config = type(
        "Config",
        (),
        {"enable_sleep_mode": True, "free_cache_engine": True},
    )()
    server.engine = Engine()
    server._resolve_sleep_level = lambda: 1
    server._submission_paused = False
    server._resume_event = Event()
    server._admitting = 0

    receipt = asyncio.run(server.sleep())
    assert receipt["sleep_level"] == 1
    assert receipt["sleeping"] is True
    assert server._multitask_sleep_stage() == "level1"
    assert calls == [("drain",), ("sleep", 1)]


def test_standalone_server_weights_stage_rollback_returns_to_level2_sleep():
    calls = []

    class Event:
        def clear(self):
            calls.append(("gate", "clear"))

        def set(self):
            calls.append(("gate", "set"))

    class Engine:
        def __init__(self):
            self.sleeping = True

        async def is_sleeping(self):
            return self.sleeping

        async def wake_up(self, *, tags=None):
            calls.append(("wake_up", tuple(tags) if tags is not None else None))
            if tags == ["kv_cache"]:
                self.sleeping = False

        async def reset_prefix_cache(self, *, reset_connector):
            calls.append(("reset_prefix_cache", reset_connector))

        async def wait_for_requests_to_drain(self):
            calls.append(("drain",))

        async def sleep(self, *, level, mode):
            calls.append(("sleep", level, mode))
            self.sleeping = True

    server = http_server_class()()
    server.nnodes = 1
    server.node_rank = 0
    server.replica_rank = 0
    server.global_steps = 5
    server.config = type(
        "Config",
        (),
        {"enable_sleep_mode": True, "free_cache_engine": True},
    )()
    server.engine = Engine()
    server._resolve_sleep_level = lambda: 2
    server._multitask_sleep_stage_value = "weights"
    server._submission_paused = True
    server._resume_event = Event()
    server._admitting = 0

    receipt = asyncio.run(server.sleep())
    assert receipt["sleep_level"] == 2
    assert receipt["sleeping"] is True
    assert server._multitask_sleep_stage() == "level2"
    assert ("wake_up", ("kv_cache",)) in calls
    assert ("sleep", 2, "abort") in calls


def test_standalone_server_rejects_configs_that_cannot_use_level2_sleep():
    class Engine:
        async def wait_for_requests_to_drain(self):
            raise AssertionError("incompatible level must fail before runtime sleep")

    server = http_server_class()()
    server.nnodes = 1
    server.node_rank = 0
    server.replica_rank = 0
    server.global_steps = 1
    server.config = type(
        "Config",
        (),
        {"enable_sleep_mode": True, "free_cache_engine": True},
    )()
    class Event:
        def __init__(self):
            self.clear_calls = 0

        def clear(self):
            self.clear_calls += 1

    server.engine = Engine()
    server._resolve_sleep_level = lambda: 1
    server._submission_paused = False
    server._resume_event = Event()

    with pytest.raises(NotImplementedError, match="GPU DONATE requires vLLM sleep level 2"):
        asyncio.run(server.sleep())
    assert server._submission_paused is False
    assert server._resume_event.clear_calls == 0


def test_standalone_sleep_and_weights_wake_are_idempotent_by_stage():
    calls = []

    class Event:
        def clear(self):
            calls.append(("clear",))

        def set(self):
            calls.append(("set",))

    class Engine:
        def __init__(self):
            self.sleeping = False

        async def wait_for_requests_to_drain(self):
            calls.append(("drain",))

        async def sleep(self, *, level, mode):
            calls.append(("sleep", level, mode))
            self.sleeping = True

        async def is_sleeping(self):
            return self.sleeping

        async def wake_up(self, *, tags=None):
            calls.append(("wake", tuple(tags or ())))
            self.sleeping = tags == ["weights"]

    server = http_server_class()()
    server.nnodes = 1
    server.node_rank = 0
    server.replica_rank = 0
    server.global_steps = 4
    server.config = type(
        "Config",
        (),
        {"enable_sleep_mode": True, "free_cache_engine": True},
    )()
    server.engine = Engine()
    server._resolve_sleep_level = lambda: 2
    server._submission_paused = False
    server._resume_event = Event()

    first_sleep = asyncio.run(server.sleep())
    second_sleep = asyncio.run(server.sleep())
    assert second_sleep == first_sleep
    assert [call for call in calls if call[:1] == ("sleep",)] == [
        ("sleep", 2, "abort")
    ]

    first_weights = asyncio.run(server.wake_up(tags=["weights"]))
    second_weights = asyncio.run(server.wake_up(tags=["weights"]))
    assert second_weights == first_weights
    assert [call for call in calls if call[:1] == ("wake",)] == [
        ("wake", ("weights",))
    ]


def test_weights_only_wake_never_opens_admission_on_backend_anomaly():
    class Event:
        def __init__(self):
            self.set_calls = 0
            self.clear_calls = 0

        def clear(self):
            self.clear_calls += 1

        def set(self):
            self.set_calls += 1

    class Engine:
        def __init__(self):
            self.sleeping = True

        async def is_sleeping(self):
            return self.sleeping

        async def wake_up(self, *, tags=None):
            assert tags == ["weights"]
            self.sleeping = False

        async def reset_prefix_cache(self, *, reset_connector):
            raise AssertionError("weights-only anomaly must not reach full-wake cleanup")

        async def check_health(self):
            raise AssertionError("weights-only anomaly must not open admission")

    server = http_server_class()()
    server.nnodes = 1
    server.node_rank = 0
    server.replica_rank = 0
    server.global_steps = 5
    server.config = type(
        "Config",
        (),
        {"enable_sleep_mode": True, "free_cache_engine": True},
    )()
    server.engine = Engine()
    server._multitask_sleep_stage_value = "level2"
    server._submission_paused = True
    server._resume_event = Event()

    with pytest.raises(RuntimeError, match="weights-only wake unexpectedly"):
        asyncio.run(server.wake_up(tags=["weights"]))

    assert server._submission_paused is True
    assert server._resume_event.set_calls == 0
    assert server._multitask_sleep_stage() == "level2"


def test_full_wake_rejects_weights_stage_before_ce_kv_restore():
    class Event:
        def clear(self):
            return None

    class Engine:
        async def is_sleeping(self):
            return True

        async def wake_up(self, *, tags=None):
            raise AssertionError("full wake must fail before touching vLLM")

    server = http_server_class()()
    server.nnodes = 1
    server.node_rank = 0
    server.replica_rank = 0
    server.config = type(
        "Config",
        (),
        {"enable_sleep_mode": True, "free_cache_engine": True},
    )()
    server.engine = Engine()
    server._multitask_sleep_stage_value = "weights"
    server._submission_paused = True
    server._resume_event = Event()

    with pytest.raises(RuntimeError, match="requires CE KV restore"):
        asyncio.run(server.wake_up())


def test_full_wake_rejects_direct_level2_to_awake_skip():
    class Event:
        def clear(self):
            return None

    class Engine:
        async def is_sleeping(self):
            return True

        async def wake_up(self, *, tags=None):
            raise AssertionError("full wake must fail before touching vLLM")

    server = http_server_class()()
    server.nnodes = 1
    server.node_rank = 0
    server.replica_rank = 0
    server.config = type(
        "Config",
        (),
        {"enable_sleep_mode": True, "free_cache_engine": True},
    )()
    server.engine = Engine()
    server._multitask_sleep_stage_value = "level2"
    server._submission_paused = True
    server._resume_event = Event()

    with pytest.raises(RuntimeError, match="weights-only RESTORE preparation"):
        asyncio.run(server.wake_up())


def test_final_wake_can_commit_admission_after_ce_already_restored_kv():
    calls = []

    class Event:
        def __init__(self):
            self.is_set = False

        def clear(self):
            self.is_set = False

        def set(self):
            self.is_set = True
            calls.append(("gate", "set"))

    class Engine:
        async def is_sleeping(self):
            return False

        async def wake_up(self, *, tags=None):
            raise AssertionError("fully resident engine must not be woken twice")

        async def reset_prefix_cache(self, *, reset_connector):
            calls.append(("reset", reset_connector))

        async def check_health(self):
            calls.append(("health",))

    server = http_server_class()()
    server.nnodes = 1
    server.node_rank = 0
    server.replica_rank = 0
    server.global_steps = 9
    server.config = type(
        "Config",
        (),
        {"enable_sleep_mode": True, "free_cache_engine": True},
    )()
    server.engine = Engine()
    server._submission_paused = True
    server._resume_event = Event()

    receipt = asyncio.run(server.wake_up())
    assert receipt["fully_awake"] is True
    assert server._submission_paused is False
    assert server._resume_event.is_set is True
    assert calls == [("reset", True), ("health",), ("gate", "set")]
