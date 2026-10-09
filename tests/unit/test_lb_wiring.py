"""LB wiring regression scenarios; shared test fakes live in _wiring_support."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import pytest
import pytest
from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    AttemptState,
    OperationEvidence,
    ReplicaKey,
)
from _wiring_support import (
    load_balancer_class,
    isolated,
)


def test_admitted_request_blocks_removal_until_verified_continuation():
    key = ReplicaKey("task-a", "r0")
    lb = load_balancer_class()({"s0": object()}, initial_routes={key: "s0"})
    lb.acquire_server("request-1")
    lb.begin_drain(key, "op")

    # Draining closes admission by dropping the server from the routing pool.
    assert "s0" not in lb._servers

    # An admitted request still owns its generation, so the route must stay.
    assert lb.has_unsettled_requests("s0")
    with pytest.raises(ValueError, match="requests remain admitted"):
        lb.finish_remove(key)

    # A verified client continuation terminates the attempt on this server and
    # records the handoff proof; the request stops blocking the route.
    evidence = lb.confirm_continuation("request-1", "client-1", "prefix-1")
    assert evidence.type is EvidenceType.EXIT_READY
    assert evidence.operation_id == "op"
    assert lb.query_attempt("request-1") is AttemptState.TERMINATED
    assert not lb.has_unsettled_requests("s0")

    # ACK-loss replay is idempotent, but the accepted prefix/client identity
    # cannot be rewritten after the old attempt was terminated.
    assert lb.confirm_continuation("request-1", "client-1", "prefix-1") == evidence
    with pytest.raises(ValueError, match="conflicting continuation proof replay"):
        lb.confirm_continuation("request-1", "client-1", "prefix-other")
    with pytest.raises(ValueError, match="conflicting continuation proof replay"):
        lb.confirm_continuation("request-1", "client-other", "prefix-1")
    assert lb.query_attempt("request-1") is AttemptState.TERMINATED

    # A late native release settles the already-verified terminal attempt.
    lb.release_server("s0", request_id="request-1")
    assert lb.query_attempt("request-1") is AttemptState.SETTLED
    assert lb.requests_for_server("s0") == ()

    lb.finish_remove(key)
    assert key not in lb.routes


def test_settled_request_reuse_drops_previous_continuation_proof():
    key0 = ReplicaKey("task-a", "r0")
    key1 = ReplicaKey("task-a", "r1")
    lb = load_balancer_class()(
        {"s0": object(), "s1": object()},
        initial_routes={key0: "s0", key1: "s1"},
    )
    lb.acquire_server("request-1")
    old_server = lb.active_request_server["request-1"]
    old_key = key0 if old_server == "s0" else key1
    lb.begin_drain(old_key, "op-old")
    old_evidence = lb.confirm_continuation(
        "request-1", "client-1", "prefix-old"
    )
    lb.finish_remove(old_key)
    assert lb.query_attempt("request-1") is AttemptState.SETTLED
    assert lb.confirm_continuation(
        "request-1", "client-1", "prefix-old"
    ) == old_evidence

    # Reusing the logical request id creates a new attempt and invalidates the
    # old handoff receipt before any new continuation can be confirmed.
    new_server, _ = lb.acquire_server("request-1")
    assert new_server != old_server
    assert "request-1" not in lb.continuation_proofs
    assert lb.query_attempt("request-1") is AttemptState.ADMITTED


def test_failed_settled_request_readmission_preserves_previous_continuation_proof():
    key = ReplicaKey("task-a", "r0")
    lb = load_balancer_class()({"s0": object()}, initial_routes={key: "s0"})
    lb.acquire_server("request-1")
    lb.begin_drain(key, "op-old")
    old_evidence = lb.confirm_continuation(
        "request-1", "client-1", "prefix-old"
    )
    lb.finish_remove(key)
    assert lb.query_attempt("request-1") is AttemptState.SETTLED

    # Native LB has no active servers after drain/remove. A failed
    # replacement acquire must not erase the previous ACK-loss proof.
    with pytest.raises(RuntimeError, match="No available servers"):
        lb.acquire_server("request-1")

    assert lb.query_attempt("request-1") is AttemptState.SETTLED
    assert lb.confirm_continuation(
        "request-1", "client-1", "prefix-old"
    ) == old_evidence


def test_continuation_handoff_survives_readmission_until_remove_commit():
    key0 = ReplicaKey("task-a", "r0")
    key1 = ReplicaKey("task-a", "r1")
    lb = load_balancer_class()(
        {"s0": object(), "s1": object()},
        initial_routes={key0: "s0", key1: "s1"},
    )
    server_id, _ = lb.acquire_server("request-1")
    old_key = key0 if server_id == "s0" else key1
    lb.begin_drain(old_key, "op-force")
    lb.confirm_continuation("request-1", "client-1", "prefix-1")
    lb.release_server(server_id, request_id="request-1")

    lb.acquire_server("request-1")
    assert "request-1" not in lb.continuation_proofs
    assert lb.continuation_handoff_requests("op-force") == ("request-1",)

    lb.finish_remove(old_key)
    assert lb.continuation_handoff_requests("op-force") == ()


def test_load_balancer_bounds_settled_request_history():
    lb = load_balancer_class()({"s0": object()})
    lb._settled_retention = 2

    for index in range(5):
        request_id = f"request-{index}"
        server_id, _ = lb.acquire_server(request_id)
        lb.continuation_proofs[request_id] = (
            "client",
            f"digest-{index}",
            "op",
            OperationEvidence.now("op", EvidenceType.EXIT_READY),
        )
        lb.release_server(server_id, request_id=request_id)

    assert len(lb.attempt_state) <= 2
    assert set(lb.attempt_state) == {"request-3", "request-4"}
    assert set(lb.continuation_proofs) == {"request-3", "request-4"}
    assert all(state is AttemptState.SETTLED for state in lb.attempt_state.values())


def test_lb_commit_ready_is_atomic_and_idempotent():
    key = ReplicaKey("task-a", "borrowed-0")
    handle = object()
    lb = load_balancer_class()({})

    evidence = lb.commit_ready(key, "s-new", handle, "op-add")
    assert evidence.type is EvidenceType.SERVICE_COMMITTED
    assert evidence.operation_id == "op-add"
    assert lb.server_for_replica(key) == "s-new"
    assert lb._servers["s-new"] is handle
    assert lb.query_ready_operation("op-add") == evidence

    assert lb.commit_ready(key, "s-new", handle, "op-add") == evidence
    with pytest.raises(ValueError, match="conflicting ready operation replay"):
        lb.commit_ready(
            ReplicaKey("task-a", "other"),
            "s-other",
            object(),
            "op-add",
        )


    with pytest.raises(ValueError, match="requires server_handle"):
        lb.commit_ready(
            ReplicaKey("task-a", "none"),
            "s-none",
            None,
            "op-none",
        )

    lb.add_servers({"s-existing": handle})
    with pytest.raises(ValueError, match="another handle"):
        lb.commit_ready(
            ReplicaKey("task-a", "other-server"),
            "s-existing",
            object(),
            "op-other-server",
        )


def test_lb_finish_remove_clears_ready_operation_ledger():
    key = ReplicaKey("task-a", "borrowed-0")
    lb = load_balancer_class()({})
    lb.commit_ready(key, "s-new", object(), "op-add")

    lb.finish_remove(key)

    assert lb.server_for_replica(key) is None
    assert lb.query_ready_operation("op-add") is None


def test_continuation_requires_an_active_drain_and_does_not_mutate_on_reject():
    key = ReplicaKey("task-a", "r0")
    lb = load_balancer_class()({"s0": object()}, initial_routes={key: "s0"})
    lb.acquire_server("request-1")

    with pytest.raises(KeyError, match="active drain operation"):
        lb.confirm_continuation("request-1", "client-1", "prefix-1")

    assert lb.query_attempt("request-1") is AttemptState.ADMITTED
    assert lb.requests_for_server("s0") == ("request-1",)


def test_one_server_cannot_be_rebound_to_another_drain_operation():
    key = ReplicaKey("task-a", "r0")
    lb = load_balancer_class()({"s0": object()}, initial_routes={key: "s0"})
    lb.begin_drain(key, "op-1")

    with pytest.raises(ValueError, match="another operation"):
        lb.begin_drain(key, "op-2")


def test_finish_remove_settles_verified_continuation_without_late_release():
    key = ReplicaKey("task-a", "r0")
    lb = load_balancer_class()({"s0": object()}, initial_routes={key: "s0"})
    lb.acquire_server("request-1")
    lb.begin_drain(key, "op")
    lb.confirm_continuation("request-1", "client-1", "prefix-1")

    lb.finish_remove(key)

    assert lb.query_attempt("request-1") is AttemptState.SETTLED
    assert lb.requests_for_server("s0") == ()
    assert key not in lb.routes


def test_drained_server_leaves_the_routing_pool_for_new_requests():
    key = ReplicaKey("task-a", "r0")
    lb = load_balancer_class()(
        {"s0": object(), "s1": object()}, initial_routes={key: "s0"}
    )
    lb.begin_drain(key, "op")

    server_id, _handle = lb.acquire_server("request-2")
    assert server_id == "s1"
    assert lb.query_attempt("request-2") is AttemptState.ADMITTED


def test_wrong_server_release_does_not_change_native_inflight_count():
    lb = load_balancer_class()({"s0": object(), "s1": object()})
    lb.acquire_server("request-1")
    lb._inflight_requests["s1"] = 1

    with pytest.raises(ValueError, match="another server"):
        lb.release_server("s1", request_id="request-1")
    assert lb._inflight_requests["s1"] == 1
    assert lb.query_attempt("request-1") is AttemptState.ADMITTED


def test_lb_retains_settled_history_with_many_unsettled_requests():
    lb = load_balancer_class()({"s0": object()})
    lb._settled_retention = 2
    # In-flight requests must not cause a recent SETTLED proof to be evicted.
    for n in range(12):
        lb.acquire_server(f"inflight-{n}")
    server, _ = lb.acquire_server("settled-0")
    lb.release_server(server, request_id="settled-0")
    assert lb.query_attempt("settled-0") is AttemptState.SETTLED
    assert lb._settled_count == 1

    for n in range(1, 4):
        request_id = f"settled-{n}"
        server, _ = lb.acquire_server(request_id)
        lb.release_server(server, request_id=request_id)
    assert lb._settled_count == 2
    assert set(k for k, v in lb.attempt_state.items()
               if v is AttemptState.SETTLED) == {"settled-2", "settled-3"}
    assert all(lb.query_attempt(f"inflight-{n}") is AttemptState.ADMITTED
               for n in range(12))


def test_lb_gc_stale_release_preserves_native_inflight_counter():
    lb = load_balancer_class()({"s0": object()})
    lb._settled_retention = 1
    server, _ = lb.acquire_server("old")
    lb.release_server(server, request_id="old")
    server, _ = lb.acquire_server("completed")
    lb.release_server(server, request_id="completed")
    assert lb.query_attempt("old") is None  # bounded GC already evicted it

    server, _ = lb.acquire_server("active")
    count = lb._inflight_requests[server]
    lb.release_server(server, request_id="old")  # duplicate late ACK
    assert lb._inflight_requests[server] == count
    assert lb.query_attempt("active") is AttemptState.ADMITTED


def test_lb_settled_gc_count_survives_reacquire_and_forced_finish():
    key = ReplicaKey("task-a", "native-0")
    lb = load_balancer_class()({"s0": object(), "s1": object()},
                                initial_routes={key: "s0"})
    server, _ = lb.acquire_server("request")
    assert server == "s0"
    lb.begin_drain(key, "op")
    lb.confirm_continuation("request", "client", "prefix")
    lb.finish_remove(key)
    assert lb._settled_count == 1
    assert lb.query_attempt("request") is AttemptState.SETTLED
    lb.acquire_server("request")
    assert lb._settled_count == 0
    assert lb.query_attempt("request") is AttemptState.ADMITTED



# --- test_zero_server_handoff.py (consolidated boundary scenarios) ---

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

    cls = isolated(
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

    cls = isolated(
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
