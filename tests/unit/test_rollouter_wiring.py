"""ROLLOUTER wiring regression scenarios; shared test fakes live in _wiring_support."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import pytest
import asyncio
import pytest
from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    AttemptState,
    OperationEvidence,
    OperationRecord,
    OperationStatus,
    ReplicaKey,
    ReplicaKind,
    ReplicaState,
)
from _wiring_support import (
    INTEGRATION,
    isolated,
    AsyncRemoteMethod,
    rollouter_class,
)


def test_continuation_wrapper_reduces_remaining_min_tokens_after_abort():
    calls = []

    class Output:
        stop_reason = "abort"
        token_ids = [11, 12, 13]

    class Server:
        generate = AsyncRemoteMethod(lambda *args, **kwargs: Output())

    class LoadBalancer:
        confirm_continuation = AsyncRemoteMethod(
            lambda *args: calls.append(args) or object()
        )

    cls = isolated(
        f"{INTEGRATION}/rollouter.py",
        "_ContinuationAwareServer",
        object,
        _continuation_prefix_digest=lambda prompt_ids, token_ids: "digest",
    )
    wrapper = cls.__new__(cls)
    wrapper._server = Server()
    wrapper._load_balancer = LoadBalancer()
    wrapper._logical_request_id = "request-1"
    wrapper._client_id = "client-1"

    sampling_params = {
        "temperature": 0.0,
        "max_tokens": 128,
        "min_tokens": 128,
    }
    output = asyncio.run(
        wrapper._generate(
            request_id="attempt-1",
            prompt_ids=[1, 2],
            sampling_params=sampling_params,
            image_data=None,
        )
    )

    assert output.stop_reason == "abort"
    assert sampling_params["max_tokens"] == 128
    assert sampling_params["min_tokens"] == 125
    assert calls == [("request-1", "client-1", "digest")]


def test_multitask_client_waits_for_same_request_release_before_reacquire():
    class AwaitableRef:
        def __init__(self):
            self.event = asyncio.Event()

        def __await__(self):
            return self.event.wait().__await__()

    release_ref = AwaitableRef()
    calls = []

    class ReleaseRemote:
        def remote(self, **kwargs):
            calls.append(("release", kwargs))
            return release_ref

    class LoadBalancer:
        release_server = ReleaseRemote()

    class Parent:
        def __init__(self, *args, **kwargs):
            self.config = type(
                "Config",
                (),
                {
                    "async_training": type(
                        "Async",
                        (),
                        {"partial_rollout": False},
                    )()
                },
            )()
            self._load_balancer = LoadBalancer()
            self._lb_require_release_fields = ["request_id"]

        async def _acquire_server(self, request_id, **extra):
            calls.append(("acquire", request_id))
            return "s1", object()

    cls = isolated(
        f"{INTEGRATION}/rollouter.py",
        "_MultiTaskFullyAsyncLLMServerClient",
        Parent,
        asyncio=asyncio,
        _ContinuationAwareServer=object,
    )
    client = cls(client_id="client-1")

    async def scenario():
        client._release_server("s0", request_id="request-1")
        acquire = asyncio.create_task(client._acquire_server("request-1"))
        await asyncio.sleep(0)
        assert not acquire.done()
        assert [call[0] for call in calls] == ["release"]

        release_ref.event.set()
        await acquire
        assert [call[0] for call in calls] == ["release", "acquire"]

    asyncio.run(scenario())


def test_rollouter_clear_operation_binding_is_scoped_and_idempotent():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key0 = ReplicaKey("task-a", "borrowed-0")
    key1 = ReplicaKey("task-a", "borrowed-1")
    rollouter._pending_operation_targets = {
        "op-0": key0,
        "op-1": key1,
    }
    rollouter._force_exit_recovery = {
        "op-0": (("request-0",), 1),
        "op-1": (("request-1",), 1),
    }

    assert rollouter.clear_operation_binding("op-0") is True
    assert "op-0" not in rollouter._pending_operation_targets
    assert "op-0" not in rollouter._force_exit_recovery
    assert rollouter._pending_operation_targets["op-1"] == key1
    assert "op-1" in rollouter._force_exit_recovery
    assert rollouter.clear_operation_binding("op-0") is False


def test_rollouter_idle_detection_treats_unknown_capacity_as_zero():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    rollouter.max_concurrent_samples = None
    assert rollouter.collect_idle_candidates() == ()


def test_rollouter_idle_detection_preserves_committed_capacity_without_reading_lb():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    keys = [ReplicaKey("task-a", f"r{i}") for i in range(3)]
    rollouter.llm_server_manager = type(
        "M",
        (),
        {
            "replica_state": {key: ReplicaState.ACTIVE for key in keys},
            "replica_kind": {key: ReplicaKind.NATIVE for key in keys},
        },
    )()

    # committed C=8 and per-replica capacity=4 require two ACTIVE replicas.
    assert rollouter.collect_idle_candidates() == ((keys[0], ReplicaKind.NATIVE),)

    # With only the required replicas left there is no safe idle candidate.
    rollouter.llm_server_manager.replica_state.pop(keys[0])
    rollouter.llm_server_manager.replica_kind.pop(keys[0])
    assert rollouter.collect_idle_candidates() == ()

    rollouter.paused = False
    assert rollouter.collect_idle_candidates() == ()


def test_rollouter_submit_idle_report_forwards_paused_surplus_metadata():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    keys = [ReplicaKey("task-a", f"r{i}") for i in range(3)]
    rollouter.llm_server_manager = type(
        "M",
        (),
        {
            "replica_state": {key: ReplicaState.ACTIVE for key in keys},
            "replica_kind": {key: ReplicaKind.NATIVE for key in keys},
        },
    )()
    seen = []

    class GS:
        submit_idle_report = AsyncRemoteMethod(
            lambda report: seen.append(report)
            or {"accepted": True, "candidate_count": len(report["candidates"])}
        )

    rollouter.group_scheduler = GS()
    result = asyncio.run(rollouter.submit_idle_report())

    assert result == {"accepted": True, "candidate_count": 1}
    assert seen[0]["task_session"] == "task-a"
    assert seen[0]["candidates"] == (
        {"replica_key": keys[0], "kind": ReplicaKind.NATIVE.value},
    )


def test_rollouter_natural_drain_timeout_stays_draining_and_same_op_resumes():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "native-0")
    rollouter._natural_drain_timeout_s = 0.01
    state = {"unsettled": True, "begin_calls": 0}

    class LB:
        begin_drain = AsyncRemoteMethod(
            lambda target, operation_id:
            state.__setitem__("begin_calls", state["begin_calls"] + 1) or "s0"
        )
        has_unsettled_requests = AsyncRemoteMethod(
            lambda server_id: state["unsettled"]
        )

    class Manager:
        global_load_balancer = LB()

        def __init__(self):
            self.replica_state = {key: ReplicaState.ACTIVE}
            self.replica_kind = {key: ReplicaKind.NATIVE}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def transition_replica(self, target, new_state):
            self.replica_state[target] = new_state

    manager = Manager()
    rollouter.llm_server_manager = manager

    with pytest.raises(TimeoutError, match="natural drain"):
        asyncio.run(
            rollouter.prepare_exit(
                key,
                operation_id="op-timeout",
                force=False,
            )
        )

    assert manager.replica_state[key] is ReplicaState.DRAINING
    assert rollouter.get_pending_target("op-timeout") == key

    state["unsettled"] = False
    evidence = asyncio.run(
        rollouter.prepare_exit(
            key,
            operation_id="op-timeout",
            force=False,
        )
    )
    assert evidence.type is EvidenceType.EXIT_READY
    assert manager.replica_state[key] is ReplicaState.DRAINING
    assert state["begin_calls"] >= 2


def test_rollouter_add_commit_publishes_hidden_borrower_after_e_is_ready():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "borrowed-0")
    calls = []

    class Runtime:
        _server_address = "s-new"
        _server_handle = "h-new"

    runtime = Runtime()

    class LB:
        commit_ready = AsyncRemoteMethod(
            lambda target, server_id, handle, operation_id:
            calls.append(("commit_ready", target, server_id, handle, operation_id))
            or OperationEvidence(
                operation_id,
                EvidenceType.SERVICE_COMMITTED,
                1,
            )
        )

    class Manager:
        global_load_balancer = LB()

        def __init__(self):
            self.replica_state = {key: ReplicaState.CREATING}
            self.replica_kind = {key: ReplicaKind.BORROWED}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def inspect_runtime(self, target):
            return runtime

        def activate_service(self, target):
            calls.append(("activate_service", target))

        def deactivate_service(self, target):
            calls.append(("deactivate_service", target))

        def transition_replica(self, target, state):
            calls.append(("state", state))
            self.replica_state[target] = state

    manager = Manager()
    rollouter.llm_server_manager = manager
    rollouter._pending_operation_targets["op-add"] = key
    rollouter._update_max_concurrent_samples = lambda: calls.append(("capacity",))

    evidence = asyncio.run(
        rollouter.commit_service_change(
            OperationRecord("op-add", OperationStatus.RUNNING)
        )
    )

    assert evidence.type is EvidenceType.SERVICE_COMMITTED
    assert manager.replica_state[key] is ReplicaState.ACTIVE
    assert "op-add" not in rollouter._pending_operation_targets
    assert calls == [
        ("activate_service", key),
        ("capacity",),
        ("state", ReplicaState.ACTIVE),
        ("commit_ready", key, "s-new", "h-new", "op-add"),
    ]


def test_rollouter_add_definite_route_rejection_defers_destroy_until_e_is_removed():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "borrowed-0")
    calls = []

    class Runtime:
        _server_address = "s-new"
        _server_handle = "h-new"

    class LB:
        commit_ready = AsyncRemoteMethod(
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("route rejected")
            )
        )
        query_ready_operation = AsyncRemoteMethod(lambda operation_id: None)

    class Manager:
        global_load_balancer = LB()

        def __init__(self):
            self.replica_state = {key: ReplicaState.CREATING}
            self.replica_kind = {key: ReplicaKind.BORROWED}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def inspect_runtime(self, target):
            return Runtime()

        def activate_service(self, target):
            calls.append(("activate_service", target))

        def deactivate_service(self, target):
            calls.append(("deactivate_service", target))

        def transition_replica(self, target, state):
            calls.append(("state", state))
            self.replica_state[target] = state

        async def destroy(self, target, *, operation_id):
            calls.append(("destroy", target, operation_id))
            return OperationEvidence(
                operation_id,
                EvidenceType.RELEASED,
                3,
                ("u0",),
            )

    manager = Manager()
    rollouter.llm_server_manager = manager
    rollouter._pending_operation_targets["op-add"] = key
    rollouter._update_max_concurrent_samples = lambda: calls.append(("capacity",))

    evidence = asyncio.run(
        rollouter.commit_service_change(
            OperationRecord("op-add", OperationStatus.RUNNING)
        )
    )

    assert evidence is None
    assert manager.replica_state[key] is ReplicaState.DRAINING
    assert "op-add" in rollouter._pending_operation_targets
    assert calls == [
        ("activate_service", key),
        ("capacity",),
        ("state", ReplicaState.ACTIVE),
        ("state", ReplicaState.DRAINING),
        ("deactivate_service", key),
        ("capacity",),
    ]


def test_rollouter_add_pre_publish_rollback_destroys_creating_borrower():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "borrowed-0")
    calls = []

    class Manager:
        def __init__(self):
            self.replica_state = {key: ReplicaState.CREATING}
            self.replica_kind = {key: ReplicaKind.BORROWED}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        async def destroy(self, target, *, operation_id):
            calls.append(("destroy", target, operation_id))
            return OperationEvidence(
                operation_id,
                EvidenceType.RELEASED,
                1,
                ("u0",),
            )

        def transition_replica(self, target, state):
            calls.append(("state", state))
            self.replica_state[target] = state

    manager = Manager()
    rollouter.llm_server_manager = manager
    rollouter._pending_operation_targets["op-add"] = key

    evidence = asyncio.run(
        rollouter.finalize_release(
            OperationRecord("op-add", OperationStatus.RUNNING)
        )
    )

    assert evidence.type is EvidenceType.RELEASED
    assert manager.replica_state[key] is ReplicaState.RELEASED
    assert "op-add" not in rollouter._pending_operation_targets
    assert calls == [
        ("destroy", key, "op-add"),
        ("state", ReplicaState.RELEASED),
    ]


def test_rollouter_add_prepare_returns_release_when_manager_proves_cleanup():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "borrowed-0")

    class Manager:
        def __init__(self):
            self.replica_state = {key: ReplicaState.RELEASED}
            self.replica_kind = {key: ReplicaKind.BORROWED}

        async def create_borrowed_replica(self, spec):
            raise RuntimeError("create failed after verified cleanup")

    rollouter.llm_server_manager = Manager()
    spec = {
        "operation_id": "op-add",
        "borrower_task_id": "task-a",
        "borrower_replica_id": "borrowed-0",
        "claims": [{"gpu_uuid": "u0"}],
    }

    evidence = asyncio.run(
        rollouter.prepare_replica(
            key,
            operation_id="op-add",
            spec=spec,
        )
    )

    assert evidence.type is EvidenceType.RELEASED
    assert evidence.released_gpu_uuids == ("u0",)
    assert "op-add" not in rollouter._pending_operation_targets


def test_rollouter_prepare_replica_reuses_existing_entry_for_native_restore():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "native-0")
    calls = []

    class Manager:
        def __init__(self):
            self.replica_state = {key: ReplicaState.DORMANT}
            self.replica_kind = {key: ReplicaKind.NATIVE}

        def replica_meta(self, target):
            calls.append(("replica_meta", target))
            return self.replica_kind[target], self.replica_state[target]

    rollouter.llm_server_manager = Manager()
    result = asyncio.run(
        rollouter.prepare_replica(
            key,
            operation_id="op-restore",
            spec=None,
        )
    )
    assert result is None
    assert calls == [("replica_meta", key)]
    assert rollouter.get_pending_target("op-restore") == key


def test_rollouter_restore_commit_publishes_r_c_then_marks_native_active():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "native-0")
    calls = []

    class Runtime:
        _server_address = "s0"
        _server_handle = "h0"

        async def wake_up(self):
            calls.append(("wake_up",))
            return ({"sleeping": False, "fully_awake": True},)

    runtime = Runtime()

    class LB:
        commit_ready = AsyncRemoteMethod(
            lambda target, server_id, server_handle, operation_id:
            calls.append(("commit_ready", target, server_id, server_handle, operation_id))
            or OperationEvidence(
                operation_id,
                EvidenceType.SERVICE_COMMITTED,
                1,
            )
        )

    class Manager:
        global_load_balancer = LB()

        def __init__(self):
            self.replica_state = {key: ReplicaState.DORMANT}
            self.replica_kind = {key: ReplicaKind.NATIVE}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def inspect_runtime(self, target):
            return runtime

        def activate_service(self, target):
            calls.append(("activate_service", target))

        def transition_replica(self, target, state):
            calls.append(("state", state))
            self.replica_state[target] = state

    manager = Manager()
    rollouter.llm_server_manager = manager
    rollouter._pending_operation_targets["op-restore"] = key
    rollouter._update_max_concurrent_samples = lambda: calls.append(("capacity",))

    evidence = asyncio.run(
        rollouter.commit_service_change(
            OperationRecord("op-restore", OperationStatus.RUNNING)
        )
    )

    assert evidence.type is EvidenceType.SERVICE_COMMITTED
    assert manager.replica_state[key] is ReplicaState.ACTIVE
    assert "op-restore" not in rollouter._pending_operation_targets
    assert calls == [
        ("wake_up",),
        ("activate_service", key),
        ("capacity",),
        ("state", ReplicaState.ACTIVE),
        ("commit_ready", key, "s0", "h0", "op-restore"),
    ]


def test_rollouter_quarantines_dormant_restore_projection_only():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "native-0")

    class Manager:
        def __init__(self):
            self.replica_state = {key: ReplicaState.DORMANT}
            self.replica_kind = {key: ReplicaKind.NATIVE}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def transition_replica(self, target, state):
            self.replica_state[target] = state

    manager = Manager()
    rollouter.llm_server_manager = manager
    rollouter._pending_operation_targets["op-restore-quarantine"] = key

    assert rollouter.quarantine_dormant_restore("op-restore-quarantine") is True
    assert manager.replica_state[key] is ReplicaState.QUARANTINED


def test_rollouter_restore_ack_loss_reconciles_committed_route_without_sleep():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "native-0")
    calls = []
    committed = OperationEvidence(
        "op-restore",
        EvidenceType.SERVICE_COMMITTED,
        7,
    )

    class Runtime:
        _server_address = "s0"
        _server_handle = "h0"

        async def wake_up(self):
            calls.append(("wake_up",))
            return ({"sleeping": False, "fully_awake": True},)

        async def sleep(self):
            raise AssertionError("committed route must never be rolled back to sleep")

    class LB:
        commit_ready = AsyncRemoteMethod(
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("reply lost after commit")
            )
        )
        query_ready_operation = AsyncRemoteMethod(
            lambda operation_id: calls.append(("query_ready", operation_id))
            or committed
        )

    class Manager:
        global_load_balancer = LB()

        def __init__(self):
            self.replica_state = {key: ReplicaState.DORMANT}
            self.replica_kind = {key: ReplicaKind.NATIVE}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def inspect_runtime(self, target):
            return Runtime()

        def activate_service(self, target):
            calls.append(("activate_service", target))

        def transition_replica(self, target, state):
            calls.append(("state", state))
            self.replica_state[target] = state

    manager = Manager()
    rollouter.llm_server_manager = manager
    rollouter._pending_operation_targets["op-restore"] = key
    rollouter._update_max_concurrent_samples = lambda: calls.append(("capacity",))

    evidence = asyncio.run(
        rollouter.commit_service_change(
            OperationRecord("op-restore", OperationStatus.RUNNING)
        )
    )

    assert evidence == committed
    assert manager.replica_state[key] is ReplicaState.ACTIVE
    assert "op-restore" not in rollouter._pending_operation_targets
    assert calls == [
        ("wake_up",),
        ("activate_service", key),
        ("capacity",),
        ("state", ReplicaState.ACTIVE),
        ("query_ready", "op-restore"),
    ]


def test_rollouter_restore_unknown_route_outcome_keeps_runtime_active_and_awake():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "native-0")
    calls = []

    class Runtime:
        _server_address = "s0"
        _server_handle = "h0"

        async def wake_up(self):
            calls.append(("wake_up",))
            return ({"sleeping": False, "fully_awake": True},)

        async def sleep(self):
            raise AssertionError("unknown route outcome must not sleep the runtime")

    class LB:
        commit_ready = AsyncRemoteMethod(
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("reply lost")
            )
        )
        query_ready_operation = AsyncRemoteMethod(
            lambda operation_id: (_ for _ in ()).throw(
                RuntimeError("LB unavailable")
            )
        )

    class Manager:
        global_load_balancer = LB()

        def __init__(self):
            self.replica_state = {key: ReplicaState.DORMANT}
            self.replica_kind = {key: ReplicaKind.NATIVE}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def inspect_runtime(self, target):
            return Runtime()

        def activate_service(self, target):
            calls.append(("activate_service", target))

        def transition_replica(self, target, state):
            calls.append(("state", state))
            self.replica_state[target] = state

    manager = Manager()
    rollouter.llm_server_manager = manager
    rollouter._pending_operation_targets["op-restore"] = key
    rollouter._update_max_concurrent_samples = lambda: calls.append(("capacity",))

    with pytest.raises(RuntimeError, match="routing commit outcome is unknown"):
        asyncio.run(
            rollouter.commit_service_change(
                OperationRecord("op-restore", OperationStatus.RUNNING)
            )
        )

    assert manager.replica_state[key] is ReplicaState.ACTIVE
    assert calls == [
        ("wake_up",),
        ("activate_service", key),
        ("capacity",),
        ("state", ReplicaState.ACTIVE),
    ]


def test_rollouter_restore_definite_no_route_defers_sleep_until_e_is_removed():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "native-0")
    calls = []

    class Runtime:
        _server_address = "s0"
        _server_handle = "h0"

        async def wake_up(self):
            calls.append(("wake_up",))
            return ({"sleeping": False, "fully_awake": True},)

        async def sleep(self):
            raise AssertionError(
                "service publish failure after E commit must not re-sleep behind E"
            )

    class LB:
        commit_ready = AsyncRemoteMethod(
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("routing commit failed")
            )
        )
        query_ready_operation = AsyncRemoteMethod(lambda operation_id: None)

    class Manager:
        global_load_balancer = LB()

        def __init__(self):
            self.replica_state = {key: ReplicaState.DORMANT}
            self.replica_kind = {key: ReplicaKind.NATIVE}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def inspect_runtime(self, target):
            return Runtime()

        def activate_service(self, target):
            calls.append(("activate_service", target))

        def deactivate_service(self, target):
            calls.append(("deactivate_service", target))

        def transition_replica(self, target, state):
            calls.append(("state", state))
            self.replica_state[target] = state

    manager = Manager()
    rollouter.llm_server_manager = manager
    rollouter._pending_operation_targets["op-restore"] = key
    rollouter._update_max_concurrent_samples = lambda: calls.append(("capacity",))

    evidence = asyncio.run(
        rollouter.commit_service_change(
            OperationRecord("op-restore", OperationStatus.RUNNING)
        )
    )

    assert evidence is None
    assert manager.replica_state[key] is ReplicaState.DRAINING
    assert calls == [
        ("wake_up",),
        ("activate_service", key),
        ("capacity",),
        ("state", ReplicaState.ACTIVE),
        ("state", ReplicaState.DRAINING),
        ("deactivate_service", key),
        ("capacity",),
    ]


def test_rollouter_invalid_exit_state_does_not_bind_pending_operation():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "native-0")

    class Manager:
        global_load_balancer = object()

        def __init__(self):
            self.replica_state = {key: ReplicaState.DORMANT}
            self.replica_kind = {key: ReplicaKind.NATIVE}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

    rollouter.llm_server_manager = Manager()
    with pytest.raises(ValueError, match="ACTIVE/DRAINING"):
        asyncio.run(
            rollouter.prepare_exit(
                key,
                operation_id="op-invalid-exit",
            )
        )
    assert "op-invalid-exit" not in rollouter._pending_operation_targets


def test_rollouter_retries_transient_read_rpc_without_unknown_escalation():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "r0")
    attempts = {"count": 0}

    class LB:
        begin_drain = AsyncRemoteMethod(lambda target, operation_id: "s0")

        async def _has_unsettled(self, server_id):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("transient LB read failure")
            return False

        has_unsettled_requests = type(
            "Remote",
            (),
            {
                "remote": lambda self, server_id:
                LB()._has_unsettled(server_id)
            },
        )()

    class Manager:
        global_load_balancer = LB()

        def __init__(self):
            self.replica_state = {key: ReplicaState.ACTIVE}
            self.replica_kind = {key: ReplicaKind.NATIVE}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def transition_replica(self, target, state):
            self.replica_state[target] = state

    rollouter.llm_server_manager = Manager()
    evidence = asyncio.run(
        rollouter.prepare_exit(key, operation_id="op-read-retry")
    )
    assert evidence.type is EvidenceType.EXIT_READY
    assert attempts["count"] == 3


def test_rollouter_natural_exit_separates_drain_from_service_commit():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "r0")
    calls = []

    class LB:
        begin_drain = AsyncRemoteMethod(
            lambda target, operation_id: calls.append(("begin", target, operation_id)) or "s0"
        )
        has_unsettled_requests = AsyncRemoteMethod(lambda server_id: False)
        server_for_replica = AsyncRemoteMethod(lambda target: "s0")
        finish_remove = AsyncRemoteMethod(lambda target: calls.append(("finish", target)))

    class Manager:
        global_load_balancer = LB()

        def __init__(self):
            self.replica_state = {key: ReplicaState.ACTIVE}
            self.replica_kind = {key: ReplicaKind.NATIVE}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def transition_replica(self, target, state):
            self.replica_state[target] = state
            calls.append(("state", state))

        def deactivate_service(self, target):
            calls.append(("deactivate", target))

    manager = Manager()
    rollouter.llm_server_manager = manager
    rollouter._update_max_concurrent_samples = lambda: calls.append(("capacity",))

    exit_evidence = asyncio.run(rollouter.prepare_exit(key, operation_id="op"))
    assert exit_evidence.type is EvidenceType.EXIT_READY
    assert calls == [
        ("state", ReplicaState.DRAINING),
        ("begin", key, "op"),
    ]
    assert rollouter.get_pending_target("op") == key

    service_evidence = asyncio.run(
        rollouter.commit_service_change(OperationRecord("op", OperationStatus.RUNNING))
    )
    assert service_evidence.type is EvidenceType.SERVICE_COMMITTED
    assert manager.replica_state[key] is ReplicaState.DRAINING
    assert calls[-3:] == [
        ("finish", key),
        ("deactivate", key),
        ("capacity",),
    ]


def test_rollouter_natural_exit_retries_idempotent_begin_drain_after_lost_reply():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "r0")
    calls = []
    attempts = {"count": 0}

    class LB:
        async def _begin(self, target, operation_id):
            attempts["count"] += 1
            calls.append(("begin", attempts["count"]))
            if attempts["count"] == 1:
                raise RuntimeError("reply lost")
            return "s0"

        begin_drain = type(
            "Remote",
            (),
            {"remote": lambda self, target, operation_id: LB()._begin(target, operation_id)},
        )()
        has_unsettled_requests = AsyncRemoteMethod(lambda server_id: False)

    class Manager:
        global_load_balancer = LB()

        def __init__(self):
            self.replica_state = {key: ReplicaState.ACTIVE}
            self.replica_kind = {key: ReplicaKind.BORROWED}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def transition_replica(self, target, state):
            calls.append(("state", state))
            self.replica_state[target] = state

    manager = Manager()
    rollouter.llm_server_manager = manager

    evidence = asyncio.run(
        rollouter.prepare_exit(key, operation_id="op-retry")
    )
    assert evidence.type is EvidenceType.EXIT_READY
    assert manager.replica_state[key] is ReplicaState.DRAINING
    assert calls == [
        ("state", ReplicaState.DRAINING),
        ("begin", 1),
        ("begin", 2),
    ]


def test_rollouter_natural_exit_keeps_draining_when_drain_start_remains_unknown():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "r0")
    calls = []

    class LB:
        begin_drain = AsyncRemoteMethod(
            lambda target, operation_id: (_ for _ in ()).throw(
                RuntimeError("drain unavailable")
            )
        )

    class Manager:
        global_load_balancer = LB()

        def __init__(self):
            self.replica_state = {key: ReplicaState.ACTIVE}
            self.replica_kind = {key: ReplicaKind.BORROWED}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def transition_replica(self, target, state):
            calls.append(("state", state))
            self.replica_state[target] = state

    manager = Manager()
    rollouter.llm_server_manager = manager

    with pytest.raises(RuntimeError, match="drain unavailable"):
        asyncio.run(
            rollouter.prepare_exit(key, operation_id="op-fail")
        )
    assert manager.replica_state[key] is ReplicaState.DRAINING
    assert rollouter.get_pending_target("op-fail") == key
    assert calls == [
        ("state", ReplicaState.DRAINING),
    ]


def test_rollouter_exit_service_commit_failure_stays_draining_for_same_op_reconciliation():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "r0")

    class LB:
        server_for_replica = AsyncRemoteMethod(lambda target: "s0")
        has_unsettled_requests = AsyncRemoteMethod(lambda server_id: False)
        finish_remove = AsyncRemoteMethod(
            lambda target: (_ for _ in ()).throw(
                RuntimeError("route commit reply lost")
            )
        )

    class Manager:
        global_load_balancer = LB()

        def __init__(self):
            self.replica_state = {key: ReplicaState.DRAINING}
            self.replica_kind = {key: ReplicaKind.NATIVE}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def transition_replica(self, target, state):
            self.replica_state[target] = state

        def deactivate_service(self, target):
            raise AssertionError("failed finish_remove must not deactivate service")

    manager = Manager()
    rollouter.llm_server_manager = manager
    rollouter._pending_operation_targets["op"] = key

    with pytest.raises(RuntimeError, match="route commit reply lost"):
        asyncio.run(
            rollouter.commit_service_change(
                OperationRecord("op", OperationStatus.RUNNING)
            )
        )

    assert manager.replica_state[key] is ReplicaState.DRAINING


def test_rollouter_release_failure_quarantines_drained_replica():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "r0")

    class Manager:
        def __init__(self):
            self.replica_state = {key: ReplicaState.DRAINING}
            self.replica_kind = {key: ReplicaKind.NATIVE}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def transition_replica(self, target, state):
            self.replica_state[target] = state

        async def sleep(self, *args, **kwargs):
            raise NotImplementedError("verified sleep backend unavailable")

    rollouter.llm_server_manager = Manager()
    rollouter._pending_operation_targets["op"] = key

    with pytest.raises(NotImplementedError, match="sleep backend unavailable"):
        asyncio.run(
            rollouter.finalize_release(
                OperationRecord("op", OperationStatus.RUNNING)
            )
        )
    assert rollouter.llm_server_manager.replica_state[key] is ReplicaState.QUARANTINED


def test_rollouter_verified_native_release_commits_dormant():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "r0")

    class Manager:
        def __init__(self):
            self.replica_state = {key: ReplicaState.DRAINING}
            self.replica_kind = {key: ReplicaKind.NATIVE}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def transition_replica(self, target, state):
            self.replica_state[target] = state

        async def sleep(self, target, *, operation_id):
            return OperationEvidence(
                operation_id,
                EvidenceType.RELEASED,
                1,
                ("u0",),
            )

    rollouter.llm_server_manager = Manager()
    rollouter._pending_operation_targets["op"] = key

    evidence = asyncio.run(
        rollouter.finalize_release(
            OperationRecord("op", OperationStatus.RUNNING)
        )
    )
    assert evidence.type is EvidenceType.RELEASED
    assert rollouter.llm_server_manager.replica_state[key] is ReplicaState.DORMANT
    assert "op" not in rollouter._pending_operation_targets


def test_rollouter_verified_borrowed_destroy_commits_released():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "borrowed-0")

    class Manager:
        def __init__(self):
            self.replica_state = {key: ReplicaState.DRAINING}
            self.replica_kind = {key: ReplicaKind.BORROWED}
            self.destroy_calls = []

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def transition_replica(self, target, state):
            self.replica_state[target] = state

        async def destroy(self, target, *, operation_id):
            self.destroy_calls.append((target, operation_id))
            return OperationEvidence(
                operation_id,
                EvidenceType.RELEASED,
                1,
                ("u0",),
            )

    manager = Manager()
    rollouter.llm_server_manager = manager
    rollouter._pending_operation_targets["op"] = key

    evidence = asyncio.run(
        rollouter.finalize_release(
            OperationRecord("op", OperationStatus.RUNNING)
        )
    )

    assert evidence.type is EvidenceType.RELEASED
    assert manager.destroy_calls == [(key, "op")]
    assert manager.replica_state[key] is ReplicaState.RELEASED
    assert "op" not in rollouter._pending_operation_targets


def test_rollouter_force_requires_partial_rollout_before_mutating_m_or_r():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "borrowed-0")
    calls = []

    rollouter.config = type(
        "Config",
        (),
        {"async_training": type("Async", (), {"partial_rollout": False})()},
    )()

    class LB:
        def __getattr__(self, name):
            raise AssertionError(f"FORCE precondition must not touch LB: {name}")

    class Manager:
        global_load_balancer = LB()

        def __init__(self):
            self.replica_state = {key: ReplicaState.ACTIVE}
            self.replica_kind = {key: ReplicaKind.BORROWED}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def transition_replica(self, target, state):
            calls.append((target, state))
            self.replica_state[target] = state

    manager = Manager()
    rollouter.llm_server_manager = manager

    with pytest.raises(ValueError, match="partial_rollout=true"):
        asyncio.run(
            rollouter.prepare_exit(key, operation_id="op-force", force=True)
        )

    assert manager.replica_state[key] is ReplicaState.ACTIVE
    assert calls == []
    assert "op-force" not in rollouter._pending_operation_targets


def test_rollouter_force_aborts_target_replica_and_waits_for_continuation_handoff():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "borrowed-0")
    calls = []
    handoff_calls = {"count": 0}

    rollouter.config = type(
        "Config",
        (),
        {"async_training": type("Async", (), {"partial_rollout": True})()},
    )()

    class Runtime:
        async def abort_all_requests(self):
            calls.append(("abort_all_requests",))
            return {"aborted_count": 1, "request_ids": ["backend-1"]}

    class LB:
        server_for_replica = AsyncRemoteMethod(lambda target: "s0")
        get_all_servers = AsyncRemoteMethod(lambda: ["s0", "s1"])
        begin_drain = AsyncRemoteMethod(
            lambda target, operation_id:
            calls.append(("begin_drain", target, operation_id)) or "s0"
        )
        requests_for_server = AsyncRemoteMethod(lambda server_id: ("request-1",))
        query_attempt = AsyncRemoteMethod(lambda request_id: AttemptState.ADMITTED)
        has_unsettled_requests = AsyncRemoteMethod(lambda server_id: False)

        async def _handoffs(self, operation_id):
            handoff_calls["count"] += 1
            return ("request-1",)

        continuation_handoff_requests = type(
            "Remote",
            (),
            {"remote": lambda self, operation_id: LB()._handoffs(operation_id)},
        )()

    class Manager:
        global_load_balancer = LB()

        def __init__(self):
            self.replica_state = {key: ReplicaState.ACTIVE}
            self.replica_kind = {key: ReplicaKind.BORROWED}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def inspect_runtime(self, target):
            return Runtime()

        def transition_replica(self, target, state):
            calls.append(("state", state))
            self.replica_state[target] = state

    manager = Manager()
    rollouter.llm_server_manager = manager

    evidence = asyncio.run(
        rollouter.prepare_exit(key, operation_id="op-force", force=True)
    )

    assert evidence.type is EvidenceType.EXIT_READY
    assert manager.replica_state[key] is ReplicaState.DRAINING
    assert rollouter.get_pending_target("op-force") == key
    assert calls == [
        ("state", ReplicaState.DRAINING),
        ("begin_drain", key, "op-force"),
        ("abort_all_requests",),
    ]
    assert handoff_calls["count"] >= 1


def test_rollouter_force_timeout_same_op_resumes_without_repeating_abort():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "borrowed-0")
    rollouter._force_handoff_timeout_s = 0.01
    calls = []
    state = {"handoff": False, "unsettled": True}

    rollouter.config = type(
        "Config",
        (),
        {"async_training": type("Async", (), {"partial_rollout": True})()},
    )()

    class Runtime:
        async def abort_all_requests(self):
            calls.append(("abort_all_requests",))
            return {"aborted_count": 1, "request_ids": ["backend-1"]}

    class LB:
        server_for_replica = AsyncRemoteMethod(lambda target: "s0")
        get_all_servers = AsyncRemoteMethod(lambda: ["s1"])
        begin_drain = AsyncRemoteMethod(
            lambda target, operation_id:
            calls.append(("begin_drain", target, operation_id)) or "s0"
        )
        requests_for_server = AsyncRemoteMethod(lambda server_id: ("request-1",))
        query_attempt = AsyncRemoteMethod(lambda request_id: AttemptState.ADMITTED)
        has_unsettled_requests = AsyncRemoteMethod(
            lambda server_id: state["unsettled"]
        )
        continuation_handoff_requests = AsyncRemoteMethod(
            lambda operation_id: ("request-1",) if state["handoff"] else ()
        )

    class Manager:
        global_load_balancer = LB()

        def __init__(self):
            self.replica_state = {key: ReplicaState.ACTIVE}
            self.replica_kind = {key: ReplicaKind.BORROWED}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def inspect_runtime(self, target):
            return Runtime()

        def transition_replica(self, target, new_state):
            self.replica_state[target] = new_state

    manager = Manager()
    rollouter.llm_server_manager = manager

    with pytest.raises(TimeoutError, match="continuation handoff"):
        asyncio.run(
            rollouter.prepare_exit(key, operation_id="op-force-resume", force=True)
        )

    assert manager.replica_state[key] is ReplicaState.DRAINING
    assert calls.count(("abort_all_requests",)) == 1

    state["handoff"] = True
    state["unsettled"] = False
    evidence = asyncio.run(
        rollouter.prepare_exit(key, operation_id="op-force-resume", force=True)
    )

    assert evidence.type is EvidenceType.EXIT_READY
    assert manager.replica_state[key] is ReplicaState.DRAINING
    assert calls.count(("abort_all_requests",)) == 1


def test_rollouter_force_abort_ack_loss_recovers_from_full_continuation_proof_without_reabort():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "borrowed-0")
    rollouter._force_handoff_timeout_s = 0.01
    calls = []
    state = {"handoff": False, "unsettled": True}

    rollouter.config = type(
        "Config",
        (),
        {"async_training": type("Async", (), {"partial_rollout": True})()},
    )()

    class Runtime:
        async def abort_all_requests(self):
            calls.append(("abort_all_requests",))
            raise RuntimeError("abort reply lost")

    class LB:
        server_for_replica = AsyncRemoteMethod(lambda target: "s0")
        get_all_servers = AsyncRemoteMethod(lambda: ["s1"])
        begin_drain = AsyncRemoteMethod(lambda target, operation_id: "s0")
        requests_for_server = AsyncRemoteMethod(lambda server_id: ("request-1",))
        query_attempt = AsyncRemoteMethod(lambda request_id: AttemptState.ADMITTED)
        has_unsettled_requests = AsyncRemoteMethod(
            lambda server_id: state["unsettled"]
        )
        continuation_handoff_requests = AsyncRemoteMethod(
            lambda operation_id: ("request-1",) if state["handoff"] else ()
        )

    class Manager:
        global_load_balancer = LB()

        def __init__(self):
            self.replica_state = {key: ReplicaState.ACTIVE}
            self.replica_kind = {key: ReplicaKind.BORROWED}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def inspect_runtime(self, target):
            return Runtime()

        def transition_replica(self, target, new_state):
            self.replica_state[target] = new_state

    manager = Manager()
    rollouter.llm_server_manager = manager

    with pytest.raises(RuntimeError, match="abort reply lost"):
        asyncio.run(
            rollouter.prepare_exit(key, operation_id="op-force-ack-loss", force=True)
        )

    assert manager.replica_state[key] is ReplicaState.DRAINING
    assert calls == [("abort_all_requests",)]

    state["handoff"] = True
    state["unsettled"] = False
    evidence = asyncio.run(
        rollouter.prepare_exit(key, operation_id="op-force-ack-loss", force=True)
    )

    assert evidence.type is EvidenceType.EXIT_READY
    assert manager.replica_state[key] is ReplicaState.DRAINING
    assert calls == [("abort_all_requests",)]


def test_rollouter_force_rejects_native_before_backend_capability_gate():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "native-0")

    class Manager:
        global_load_balancer = object()

        def __init__(self):
            self.replica_state = {key: ReplicaState.ACTIVE}
            self.replica_kind = {key: ReplicaKind.NATIVE}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

    manager = Manager()
    rollouter.llm_server_manager = manager

    with pytest.raises(ValueError, match="borrowed-only"):
        asyncio.run(
            rollouter.prepare_exit(key, operation_id="op-force", force=True)
        )
    assert manager.replica_state[key] is ReplicaState.ACTIVE


@pytest.mark.parametrize("lost_retraction_ack", [False, True])
def test_rollouter_retracts_idle_candidates_on_resume(lost_retraction_ack):
    cls = rollouter_class()
    rollouter = cls(object(), object())
    rollouter.task_session = "task-a"
    rollouter.running = True
    rollouter.paused = False
    rollouter._idle_report_signature = (("task-a", "native-0", 0, "NATIVE"),)
    rollouter._idle_report_last_sent = 1.0

    received = []

    def report_to_gs(report):
        received.append(report)
        if lost_retraction_ack and len(received) == 1:
            raise TimeoutError("lost idle retraction ACK")
        return {"accepted": True, "candidate_count": 0}

    class GS:
        submit_idle_report = AsyncRemoteMethod(report_to_gs)

    rollouter.group_scheduler = GS()

    # Speed up only this AST-isolated class's 1s monitor tick.
    async def immediate_tick(_delay):
        await asyncio.sleep(0)

    cls._idle_report_loop.__globals__["asyncio"] = type(
        "TestAsyncio",
        (),
        {
            "sleep": staticmethod(immediate_tick),
            "get_running_loop": staticmethod(asyncio.get_running_loop),
            "CancelledError": asyncio.CancelledError,
        },
    )

    async def run():
        task = asyncio.create_task(rollouter._idle_report_loop())
        try:
            target_count = 2 if lost_retraction_ack else 1
            for _ in range(300):
                if len(received) >= target_count and rollouter._idle_report_signature is None:
                    break
                await asyncio.sleep(0)
            assert len(received) == target_count
            assert rollouter._idle_report_signature is None
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    assert all(report == {"task_session": "task-a", "candidates": ()}
               for report in received)



# --- test_reward_loop_isolation.py (consolidated boundary scenarios) ---

_REWARD_LOOP_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/multi_task_scheduler/integration/verl/experimental_fully_async/rollouter.py"
)


def _isolated_reward_manager():
    tree = ast.parse(_REWARD_LOOP_SOURCE.read_text(encoding="utf-8"))
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
    exec(compile(module, str(_REWARD_LOOP_SOURCE), "exec"), env)
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
    tree = ast.parse(_REWARD_LOOP_SOURCE.read_text(encoding="utf-8"))
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
                str(_REWARD_LOOP_SOURCE), "exec"),
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
