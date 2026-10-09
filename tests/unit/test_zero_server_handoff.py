"""Zero-server lifecycle-handoff regression without Ray/NPU dependencies."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src/multi_task_scheduler"


def _extract_class(path, name, parent, **globals_):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    klass.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    klass.decorator_list = []
    ns = {"Parent": parent, **globals_}
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            klass,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), ns)
    return ns[name]


def _load_router():
    class Parent:
        def __init__(self, servers, **kwargs):
            self._servers = dict(servers)
            self._inflight_requests = {s: 0 for s in servers}

        def remove_servers(self, ids):
            for sid in ids:
                self._servers.pop(sid, None)
                self._inflight_requests.pop(sid, None)

        def add_servers(self, servers):
            for sid, handle in servers.items():
                self._servers[sid] = handle
                self._inflight_requests[sid] = 0

    class Key:
        def __init__(self, ident):
            self.ident = ident

        def __eq__(self, other):
            return type(other) is Key and other.ident == self.ident

        def __hash__(self):
            return hash(self.ident)

    class Evidence:
        @staticmethod
        def now(operation_id, kind):
            return (operation_id, kind)

    cls = _extract_class(
        "rollout/load_balancer.py", "MultiTaskGlobalRequestLoadBalancer",
        Parent, DEFAULT_ROUTING_CACHE_SIZE=1000, ReplicaKey=Key,
        OperationEvidence=Evidence, EvidenceType=SimpleNamespace(SERVICE_COMMITTED="service"),
    )
    return cls, Key


def test_zero_initial_lb_is_not_treated_as_handoff():
    cls, _ = _load_router()
    lb = cls({})
    assert lb.is_service_restore_pending() is False


def test_drained_last_server_waits_through_finish_remove_until_restore():
    cls, Key = _load_router()
    key = Key("native-0")
    lb = cls({"native-server": "handle"}, initial_routes={key: "native-server"})
    assert not lb.is_service_restore_pending()
    assert lb.begin_drain(key, "op-donate") == "native-server"
    assert lb.is_service_restore_pending()
    lb.finish_remove(key)
    assert lb.is_service_restore_pending()
    assert lb.commit_ready(key, "restored-server", "new-handle", "op-restore") == (
        "op-restore", "service"
    )
    assert not lb.is_service_restore_pending()
    assert lb._servers == {"restored-server": "new-handle"}


def test_draining_one_of_two_servers_does_not_trigger_wait():
    cls, Key = _load_router()
    key = Key("native-0")
    lb = cls({"native-0": "h0", "native-1": "h1"}, initial_routes={key: "native-0"})
    lb.begin_drain(key, "op-donate")
    assert not lb.is_service_restore_pending()
    lb.finish_remove(key)
    assert not lb.is_service_restore_pending()
    assert lb._servers == {"native-1": "h1"}


def _load_client(*, plan, pending, asyncio_module=asyncio):
    events = []

    class Remote:
        async def remote(self):
            events.append(("read-pending",))
            return pending

    class Balancer:
        is_service_restore_pending = Remote()

    class Parent:
        def __init__(self, *args, **kwargs):
            self.config = SimpleNamespace(
                multitask=SimpleNamespace(drain_timeout_s=5),
                async_training=SimpleNamespace(partial_rollout=False),
            )
            self._load_balancer = Balancer()

        async def _acquire_server(self, request_id, **extra):
            events.append(("acquire", request_id, dict(extra)))
            result = plan.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result

    cls = _extract_class(
        "integration/verl/experimental_fully_async/rollouter.py",
        "_MultiTaskFullyAsyncLLMServerClient", Parent,
        asyncio=asyncio_module, _ContinuationAwareServer=object,
    )
    return cls(client_id="task-1"), events


def test_client_retries_only_during_confirmed_handoff():
    client, events = _load_client(
        plan=[RuntimeError("No available servers in load balancer"), ("server-restored", "handle")],
        pending=True,
    )
    assert asyncio.run(client._acquire_server("request-123", prompt_ids=[1])) == (
        "server-restored", "handle"
    )
    assert [event[0] for event in events] == ["acquire", "read-pending", "acquire"]
    assert events[0][1:] == events[2][1:]


def test_client_fails_fast_when_lb_has_no_legitimate_handoff():
    client, events = _load_client(
        plan=[RuntimeError("No available servers in load balancer")],
        pending=False,
    )
    with pytest.raises(RuntimeError, match="No available servers"):
        asyncio.run(client._acquire_server("request-123"))
    assert [event[0] for event in events] == ["acquire", "read-pending"]


def test_client_does_not_retry_unrelated_errors():
    client, events = _load_client(
        plan=[RuntimeError("server crashed, cannot handle request")], pending=True
    )
    with pytest.raises(RuntimeError, match="server crashed"):
        asyncio.run(client._acquire_server("request-123"))
    assert [event[0] for event in events] == ["acquire"]


def test_client_handoff_has_finite_deadline():
    class FastClock:
        ticks = 0

        def time(self):
            self.ticks += 301.0
            return self.ticks

    async def no_sleep(_):
        return None

    clock = FastClock()
    fake_asyncio = SimpleNamespace(
        get_running_loop=lambda: clock,
        sleep=no_sleep,
    )
    client, events = _load_client(
        plan=[RuntimeError("No available servers in load balancer")],
        pending=True,
        asyncio_module=fake_asyncio,
    )
    with pytest.raises(TimeoutError, match="No rollout server restored"):
        asyncio.run(client._acquire_server("request-123"))
    assert [event[0] for event in events] == ["acquire", "read-pending"]
