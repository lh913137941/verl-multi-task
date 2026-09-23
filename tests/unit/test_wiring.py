import ast
import asyncio
import threading
import time
from pathlib import Path

import pytest

from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    AttemptState,
    Lease,
    OperationCommand,
    OperationEvidence,
    OperationKind,
    OperationRecord,
    OperationStatus,
    ReplicaKey,
    ReplicaKind,
    ReplicaState,
)
from multi_task_scheduler.orchestration.operation_journal import OperationJournal

SOURCE = Path(__file__).resolve().parents[2] / "src/multi_task_scheduler"
INTEGRATION = "integration/verl/experimental_fully_async"


def isolated(relative, name, parent, **scope):
    path = SOURCE / relative
    tree = ast.parse(path.read_text())
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == name
    )
    node.bases = [ast.Name(id="Parent", ctx=ast.Load())]
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
    env = {"Parent": parent, **scope}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env[name]


class RemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class AsyncRemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    async def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class FakeRay:
    class exceptions:
        class GetTimeoutError(Exception):
            pass

        class RayActorError(Exception):
            pass

    @staticmethod
    def get(value, timeout=None):
        return value


def taskrunner_class():
    class Parent:
        def __init__(self):
            self.components = {}

    return isolated(
        f"{INTEGRATION}/task_runner.py",
        "MultiTaskFullyAsyncTaskRunner",
        Parent,
        OperationJournal=OperationJournal,
        Lease=Lease,
        OperationCommand=OperationCommand,
        OperationRecord=OperationRecord,
        OperationStatus=OperationStatus,
        OperationKind=OperationKind,
        OperationEvidence=OperationEvidence,
        EvidenceType=EvidenceType,
        threading=threading,
        ray=FakeRay,
        logger=type("Logger", (), {"exception": lambda *args, **kwargs: None})(),
    )


def load_balancer_class():
    class Parent:
        def __init__(self, servers, **kwargs):
            self._servers = dict(servers)
            self._inflight_requests = {server_id: 0 for server_id in servers}

        def acquire_server(self, request_id, **extra):
            server_id = next(iter(self._servers))
            self._inflight_requests[server_id] += 1
            return server_id, self._servers[server_id]

        def release_server(self, server_id, request_id=None):
            if server_id in self._inflight_requests:
                self._inflight_requests[server_id] -= 1

        def remove_servers(self, server_ids):
            for server_id in server_ids:
                self._servers.pop(server_id, None)
                self._inflight_requests.pop(server_id, None)

    return isolated(
        "rollout/load_balancer.py",
        "MultiTaskGlobalRequestLoadBalancer",
        Parent,
        DEFAULT_ROUTING_CACHE_SIZE=128,
        ReplicaKey=ReplicaKey,
        AttemptState=AttemptState,
        OperationEvidence=OperationEvidence,
        EvidenceType=EvidenceType,
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

    # A late native release settles the already-verified terminal attempt.
    lb.release_server("s0", request_id="request-1")
    assert lb.query_attempt("request-1") is AttemptState.SETTLED
    assert lb.requests_for_server("s0") == ()

    lb.finish_remove(key)
    assert key not in lb.routes


def test_continuation_requires_an_active_drain_and_does_not_mutate_on_reject():
    key = ReplicaKey("task-a", "r0")
    lb = load_balancer_class()({"s0": object()}, initial_routes={key: "s0"})
    lb.acquire_server("request-1")

    with pytest.raises(ValueError, match="active drain operation"):
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


def taskrunner_lease():
    return Lease(
        "l1",
        (
            {
                "claim_id": "claim-0",
                "source_lease_id": "source-lease-0",
                "donor_task_id": "donor-task",
                "donor_replica_rank": 0,
                "pg_id": "pg",
                "bundle_index": 0,
                "node_id": "n0",
                "gpu_uuid": "u0",
                "node_rank": 0,
                "local_rank": 0,
            },
        ),
        0,
    )


def test_taskrunner_minimal_journal_surface_and_replay_does_not_relaunch():
    cls = taskrunner_class()
    runner = cls()
    runner.task_session = "task-a"
    runner._control_ready = True
    launched = []
    runner._launch_operation = launched.append

    command = OperationCommand(
        "op",
        OperationKind.ADD,
        ReplicaKey("task-a", "r0"),
        "l1",
    )
    lease = taskrunner_lease()
    assert runner.submit_operation(command, lease=lease).status is OperationStatus.ACCEPTED
    assert runner.submit_operation(command, lease=lease).status is OperationStatus.ACCEPTED
    assert launched == ["op"]
    assert runner.query_operation("missing").status is OperationStatus.UNKNOWN


def test_taskrunner_executes_add_and_commits_terminal_record():
    cls = taskrunner_class()
    runner = cls()
    runner.task_session = "task-a"
    runner._control_ready = True
    runner._launch_operation = lambda operation_id: None

    prepared = []

    class Rollouter:
        prepare_replica = RemoteMethod(
            lambda target, **kwargs: prepared.append((target, kwargs)) or None
        )

    class Trainer:
        bootstrap_and_publish = RemoteMethod(
            lambda operation: OperationEvidence(
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                1,
            )
        )

    runner.components = {"rollouter": Rollouter(), "trainer": Trainer()}
    command = OperationCommand(
        "op",
        OperationKind.ADD,
        ReplicaKey("task-a", "r0"),
        "l1",
    )
    runner.submit_operation(command, lease=taskrunner_lease())
    runner._execute_operation("op")

    record = runner.query_operation("op")
    assert record.status is OperationStatus.SUCCEEDED
    assert record.result == EvidenceType.SERVICE_COMMITTED.value
    assert prepared[0][0] == command.target
    assert prepared[0][1]["operation_id"] == "op"
    spec = prepared[0][1]["spec"]
    assert spec["lease_id"] == "l1"
    assert spec["lease_ids"] == ["source-lease-0"]
    assert spec["borrower_task_id"] == "task-a"
    assert spec["borrower_replica_id"] == "r0"
    assert spec["selected_slots"][0]["rank"] == 0


def test_taskrunner_borrowed_spec_rebuilds_rank_view_from_claim_order():
    command = OperationCommand(
        "op-rank",
        OperationKind.ADD,
        ReplicaKey("task-a", "r0"),
        "l1",
    )
    claim = dict(taskrunner_lease().claims[0])
    claim["rank"] = 99
    claim["node_rank"] = 7
    claim["local_rank"] = 11
    lease = Lease("l1", (claim,), 0)

    spec = taskrunner_class()._build_borrowed_spec(command, lease)

    slot = spec["selected_slots"][0]
    assert slot["rank"] == 0
    assert slot["node_rank"] == 0
    assert slot["local_rank"] == 0

def test_taskrunner_add_requires_matching_lease_snapshot_before_launch():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    runner._launch_operation = lambda operation_id: None
    command = OperationCommand(
        "op-add",
        OperationKind.ADD,
        ReplicaKey("task-a", "r0"),
        "l1",
    )

    with pytest.raises(ValueError, match="Lease snapshot"):
        runner.submit_operation(command)

    wrong = Lease(
        "other",
        taskrunner_lease().claims,
        0,
    )
    with pytest.raises(ValueError, match="command.lease_id"):
        runner.submit_operation(command, lease=wrong)


def test_release_failure_after_service_commit_keeps_operation_unknown_and_fenced():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    runner._launch_operation = lambda operation_id: None
    key = ReplicaKey("task-a", "r0")

    class Rollouter:
        prepare_exit = RemoteMethod(
            lambda target, **kwargs: OperationEvidence(
                kwargs["operation_id"], EvidenceType.EXIT_READY, 1
            )
        )
        finalize_release = RemoteMethod(
            lambda operation: (_ for _ in ()).throw(
                NotImplementedError("native sleep not verified")
            )
        )

    class Trainer:
        remove_and_commit = RemoteMethod(
            lambda operation: OperationEvidence(
                operation.operation_id, EvidenceType.SERVICE_COMMITTED, 2
            )
        )

    runner.components = {"rollouter": Rollouter(), "trainer": Trainer()}
    runner.submit_operation(OperationCommand("op-1", OperationKind.DONATE, key, "l1"))
    runner._execute_operation("op-1")

    assert runner.query_operation("op-1").status is OperationStatus.UNKNOWN
    assert runner._ensure_journal().active_operation("task-a") == "op-1"
    with pytest.raises(Exception, match="another lifecycle operation"):
        runner.submit_operation(
            OperationCommand("op-2", OperationKind.DONATE, key, "l1")
        )


def test_manager_owns_state_kind_and_runtime_inventory_separately():
    class Parent:
        def __init__(self, *args):
            self.rollout_replicas = []
            self.server_addresses = []
            self.server_handles = []

    allowed = {
        ReplicaState.CREATING: {
            ReplicaState.ACTIVE,
            ReplicaState.RELEASED,
            ReplicaState.QUARANTINED,
        },
        ReplicaState.ACTIVE: {ReplicaState.DRAINING},
        ReplicaState.DRAINING: {
            ReplicaState.ACTIVE,
            ReplicaState.DORMANT,
            ReplicaState.RELEASED,
            ReplicaState.QUARANTINED,
        },
        ReplicaState.DORMANT: {
            ReplicaState.ACTIVE,
            ReplicaState.QUARANTINED,
        },
        ReplicaState.RELEASED: set(),
        ReplicaState.QUARANTINED: set(),
    }
    class FakePGId:
        def __init__(self, value):
            self.value = value

        def hex(self):
            return self.value

    class FakePG:
        def __init__(self, value, bundle_count=1):
            self.id = FakePGId(value)
            self.bundle_count = bundle_count

    fake_pg = FakePG("pg")
    fake_ray = type(
        "ManagerRay",
        (),
        {
            "util": type(
                "Util",
                (),
                {
                    "placement_group_table": staticmethod(
                        lambda: {
                            "pg": {
                                "state": "CREATED",
                                "name": "verl-pg-0",
                                "bundles": {0: {"CPU": 1, "GPU": 1}},
                                "bundles_to_node_id": {0: "n0"},
                            }
                        }
                    ),
                    "get_placement_group": staticmethod(
                        lambda name: fake_pg
                        if name == "verl-pg-0"
                        else (_ for _ in ()).throw(ValueError(name))
                    ),
                },
            )(),
            "get_runtime_context": staticmethod(
                lambda: type("RuntimeContext", (), {"namespace": "verl-test"})()
            ),
        },
    )
    cls = isolated(
        f"{INTEGRATION}/llm_server_manager.py",
        "MultiTaskLLMServerManager",
        Parent,
        MultiTaskvLLMReplica=object(),
        MultiTaskGlobalRequestLoadBalancer=object(),
        ReplicaKey=ReplicaKey,
        ReplicaKind=ReplicaKind,
        ReplicaState=ReplicaState,
        Lease=Lease,
        asyncio=asyncio,
        ray=fake_ray,
        time=time,
        _ALLOWED=allowed,
    )
    manager = cls(object())
    key = ReplicaKey("task-a", "r0")
    runtime = type("Runtime", (), {"_server_address": "s0", "_server_handle": "h0"})()
    manager.register_replica(key, ReplicaKind.NATIVE, state=ReplicaState.ACTIVE, runtime=runtime)
    manager.rollout_replicas.append(runtime)
    manager.server_addresses.append("s0")
    manager.server_handles.append("h0")

    manager.transition_replica(key, ReplicaState.DRAINING)
    assert manager.deactivate_service(key) is runtime
    assert manager.inspect_runtime(key) is runtime
    assert manager.rollout_replicas == []
    assert manager.server_addresses == []
    assert manager.server_handles == []

    manager.task_session = "task-a"
    valid_spec = {
        "operation_id": "op-add",
        "lease_id": "borrower-lease",
        "borrower_task_id": "task-a",
        "borrower_replica_id": "borrowed-0",
        "selected_slots": [
            {
                "claim_id": "claim-0",
                "source_lease_id": "source-lease-0",
                "donor_task_id": "donor-task",
                "donor_replica_rank": 0,
                "pg_id": "pg",
                "bundle_index": 0,
                "node_id": "n0",
                "gpu_uuid": "u0",
                "node_rank": 0,
                "local_rank": 0,
                "gpu_fraction": 1.0,
                "cpu_request": 1.0,
            }
        ],
        "world_size": 1,
        "max_colocate_count": 1,
        "replica_rank": None,
        "expires_at": 0,
        "placement_epoch": 0,
    }
    normalized = manager.validate_borrowed_spec(valid_spec)
    assert "selected_slots" not in normalized
    assert normalized["claims"][0]["rank"] == 0
    assert normalized["lease_ids"] == ["source-lease-0"]

    wrong_task = dict(valid_spec, borrower_task_id="task-b")
    with pytest.raises(ValueError, match="another task_session"):
        manager.validate_borrowed_spec(wrong_task)

    wrong_world = dict(valid_spec, world_size=2)
    with pytest.raises(ValueError, match="world_size"):
        manager.validate_borrowed_spec(wrong_world)

    expired = dict(valid_spec, expires_at=time.time() - 1)
    with pytest.raises(ValueError, match="expired"):
        manager.validate_borrowed_spec(expired)

    wrong_namespace = dict(valid_spec)
    wrong_namespace["selected_slots"] = [
        dict(valid_spec["selected_slots"][0], pg_namespace="other")
    ]
    with pytest.raises(ValueError, match="another Ray namespace"):
        manager._resolve_placement_groups(
            manager.validate_borrowed_spec(wrong_namespace)["claims"]
        )

    wrong_node = dict(valid_spec)
    wrong_node["selected_slots"] = [
        dict(valid_spec["selected_slots"][0], node_id="n9")
    ]
    with pytest.raises(ValueError, match="node_id does not match"):
        manager._resolve_placement_groups(
            manager.validate_borrowed_spec(wrong_node)["claims"]
        )

    resolved_pgs = manager._resolve_placement_groups(normalized["claims"])
    assert resolved_pgs == {"pg": fake_pg}

    with pytest.raises(NotImplementedError, match="PG/bundle actor backend"):
        asyncio.run(manager.create_borrowed_replica(valid_spec))

    record = manager.borrowed_operations["borrower-lease"]
    assert record["state"] == "FAILED"
    assert record["replica_rank"] == 0
    assert record["claim_ids"] == ["claim-0"]
    assert record["source_lease_ids"] == ["source-lease-0"]
    assert manager.next_replica_rank == 1

    # Exact replay returns the same failure boundary and does not allocate rank 1.
    with pytest.raises(NotImplementedError, match="PG/bundle actor backend"):
        asyncio.run(manager.create_borrowed_replica(valid_spec))
    assert manager.next_replica_rank == 1

    # A retry may carry the rank recovered from the manager's first record.
    resolved_retry = dict(valid_spec, replica_rank=0)
    with pytest.raises(NotImplementedError, match="PG/bundle actor backend"):
        asyncio.run(manager.create_borrowed_replica(resolved_retry))
    assert manager.next_replica_rank == 1

    wrong_rank = dict(valid_spec, replica_rank=7)
    with pytest.raises(ValueError, match="conflicting borrowed create replay"):
        asyncio.run(manager.create_borrowed_replica(wrong_rank))

    conflicting = dict(valid_spec, operation_id="op-other")
    with pytest.raises(ValueError, match="conflicting borrowed create replay"):
        asyncio.run(manager.create_borrowed_replica(conflicting))
    assert manager.next_replica_rank == 1


def rollouter_class():
    class Parent:
        def __init__(self, *args, **kwargs):
            self.paused = True
            self.max_concurrent_samples = 8
            self.concurrent_samples_per_replica = 4

    return isolated(
        f"{INTEGRATION}/rollouter.py",
        "MultiTaskFullyAsyncRollouter",
        Parent,
        ReplicaKey=ReplicaKey,
        ReplicaKind=ReplicaKind,
        ReplicaState=ReplicaState,
        OperationRecord=OperationRecord,
        OperationEvidence=OperationEvidence,
        EvidenceType=EvidenceType,
        asyncio=asyncio,
        ray=FakeRay,
    )


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

        def sleep(self, *args, **kwargs):
            raise NotImplementedError("verified sleep backend unavailable")

    rollouter.llm_server_manager = Manager()
    rollouter._pending_operation_targets["op"] = key

    with pytest.raises(NotImplementedError, match="sleep backend unavailable"):
        rollouter.finalize_release(OperationRecord("op", OperationStatus.RUNNING))
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

        def sleep(self, target, *, operation_id):
            return OperationEvidence(
                operation_id,
                EvidenceType.RELEASED,
                1,
                ("u0",),
            )

    rollouter.llm_server_manager = Manager()
    rollouter._pending_operation_targets["op"] = key

    evidence = rollouter.finalize_release(
        OperationRecord("op", OperationStatus.RUNNING)
    )
    assert evidence.type is EvidenceType.RELEASED
    assert rollouter.llm_server_manager.replica_state[key] is ReplicaState.DORMANT
    assert "op" not in rollouter._pending_operation_targets

def test_rollouter_force_fails_before_mutating_m_without_verified_backend():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "r0")

    class Manager:
        def __init__(self):
            self.replica_state = {key: ReplicaState.ACTIVE}
            self.replica_kind = {key: ReplicaKind.BORROWED}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def transition_replica(self, target, state):
            self.replica_state[target] = state

    manager = Manager()
    rollouter.llm_server_manager = manager
    with pytest.raises(NotImplementedError, match="targeted abort"):
        asyncio.run(rollouter.prepare_exit(key, operation_id="op", force=True))
    assert manager.replica_state[key] is ReplicaState.ACTIVE
