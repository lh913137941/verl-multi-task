import ast
import asyncio
import threading
import time
from pathlib import Path

import pytest

from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    AttemptState,
    FIRST_RELEASE_MAX_COLOCATE_COUNT,
    FIRST_RELEASE_RAY_GPU_FRACTION,
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
from multi_task_scheduler.orchestration.replica_sync_gate import GateKind, ReplicaSyncGate

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
        FIRST_RELEASE_MAX_COLOCATE_COUNT=FIRST_RELEASE_MAX_COLOCATE_COUNT,
        logger=type("Logger", (), {"exception": lambda *args, **kwargs: None})(),
    )


def trainer_class():
    class Parent:
        def __init__(self, *args, **kwargs):
            self.rollouter = None
            self.checkpoint_manager = None
            self.current_param_version = 0

    return isolated(
        f"{INTEGRATION}/trainer.py",
        "MultiTaskFullyAsyncTrainer",
        Parent,
        OperationRecord=OperationRecord,
        OperationEvidence=OperationEvidence,
        EvidenceType=EvidenceType,
        ReplicaKey=ReplicaKey,
        ReplicaKind=ReplicaKind,
        GateKind=GateKind,
        ReplicaSyncGate=ReplicaSyncGate,
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

        def add_servers(self, servers):
            for server_id, handle in dict(servers).items():
                self._servers[server_id] = handle
                self._inflight_requests.setdefault(server_id, 0)

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
                "gpu_fraction": FIRST_RELEASE_RAY_GPU_FRACTION,
                "cpu_request": 1.0,
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
        OperationKind.REMOVE,
        ReplicaKey("task-a", "borrowed-0"),
        "l1",
    )
    assert runner.submit_operation(command).status is OperationStatus.ACCEPTED
    assert runner.submit_operation(command).status is OperationStatus.ACCEPTED
    assert launched == ["op"]
    assert runner.query_operation("missing").status is OperationStatus.UNKNOWN


def test_taskrunner_add_fails_before_hidden_runtime_creation():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    launched = []
    runner._launch_operation = launched.append
    command = OperationCommand(
        "op-add",
        OperationKind.ADD,
        ReplicaKey("task-a", "borrowed-0"),
        "l1",
    )

    with pytest.raises(
        NotImplementedError,
        match="verified target-only parameter bootstrap",
    ):
        runner.submit_operation(command, lease=taskrunner_lease())

    assert launched == []
    assert runner._ensure_journal().query("op-add") is None


def test_taskrunner_executes_natural_borrowed_remove_and_advances_lease():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    runner._launch_operation = lambda operation_id: None
    key = ReplicaKey("task-a", "borrowed-0")
    calls = []

    class Rollouter:
        prepare_exit = RemoteMethod(
            lambda target, **kwargs: calls.append(
                ("prepare_exit", target, kwargs["operation_id"], kwargs["force"])
            )
            or OperationEvidence(
                kwargs["operation_id"], EvidenceType.EXIT_READY, 1
            )
        )
        finalize_release = RemoteMethod(
            lambda operation: calls.append(("finalize_release", operation.operation_id))
            or OperationEvidence(
                operation.operation_id,
                EvidenceType.RELEASED,
                3,
                ("u0",),
            )
        )

    class Trainer:
        remove_and_commit = RemoteMethod(
            lambda operation: calls.append(("remove_and_commit", operation.operation_id))
            or OperationEvidence(
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                2,
            )
        )

    class GroupScheduler:
        advance_lease = RemoteMethod(
            lambda lease_id, evidence: calls.append(
                ("advance_lease", lease_id, evidence.released_gpu_uuids)
            )
            or {"released": True}
        )

    runner.components = {"rollouter": Rollouter(), "trainer": Trainer()}
    runner.group_scheduler = GroupScheduler()
    runner.submit_operation(
        OperationCommand("op-remove", OperationKind.REMOVE, key, "l1")
    )
    runner._execute_operation("op-remove")

    record = runner.query_operation("op-remove")
    assert record.status is OperationStatus.SUCCEEDED
    assert record.result == EvidenceType.RELEASED.value
    assert calls == [
        ("prepare_exit", key, "op-remove", False),
        ("remove_and_commit", "op-remove"),
        ("finalize_release", "op-remove"),
        ("advance_lease", "l1", ("u0",)),
    ]


def test_taskrunner_restore_fails_before_journal_or_worker_launch():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    launched = []
    runner._launch_operation = launched.append
    key = ReplicaKey("task-a", "native-0")

    with pytest.raises(
        NotImplementedError,
        match="verified native wake/bootstrap backend",
    ):
        runner.submit_operation(
            OperationCommand("op-restore", OperationKind.RESTORE, key, "l1")
        )

    assert launched == []
    assert runner._ensure_journal().query("op-restore") is None


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


def test_taskrunner_unverified_native_donate_fails_before_drain_or_journal():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    launched = []
    runner._launch_operation = launched.append
    key = ReplicaKey("task-a", "native-0")

    with pytest.raises(
        NotImplementedError,
        match="verified native STANDALONE sleep backend",
    ):
        runner.submit_operation(
            OperationCommand("op-donate", OperationKind.DONATE, key, "l1")
        )

    assert launched == []
    assert runner._ensure_journal().query("op-donate") is None


def test_taskrunner_force_remove_fails_before_drain_or_journal():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    launched = []
    runner._launch_operation = launched.append
    key = ReplicaKey("task-a", "borrowed-0")

    with pytest.raises(
        NotImplementedError,
        match="verified targeted abort/continuation backend",
    ):
        runner.submit_operation(
            OperationCommand(
                "op-force",
                OperationKind.REMOVE,
                key,
                "l1",
                force=True,
            )
        )

    assert launched == []
    assert runner._ensure_journal().query("op-force") is None


def test_trainer_donate_parks_native_ce_member_in_existing_pending_set():
    key = ReplicaKey("task-a", "native-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.NATIVE})()
    calls = []

    class CE:
        def __init__(self):
            self.effective_replicas = {key: ((runtime,), 7)}
            self.pending_bootstrap = {}

        def remove_effective(self, target):
            calls.append(("remove_effective", target))
            self.effective_replicas.pop(target)

        def register_pending(self, target, replicas, *, operation_id):
            calls.append(("register_pending", target, operation_id))
            self.pending_bootstrap[target] = (tuple(replicas), operation_id)

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        commit_service_change = AsyncRemoteMethod(
            lambda operation: calls.append(("commit_service", operation.operation_id))
            or OperationEvidence(
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                1,
            )
        )

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    evidence = asyncio.run(
        trainer.remove_and_commit(
            OperationRecord("op-donate", OperationStatus.RUNNING)
        )
    )

    assert evidence.type is EvidenceType.SERVICE_COMMITTED
    assert trainer.checkpoint_manager.pending_bootstrap[key] == (
        (runtime,),
        "op-donate",
    )
    assert calls == [
        ("remove_effective", key),
        ("register_pending", key, "op-donate"),
        ("commit_service", "op-donate"),
    ]


def test_trainer_borrowed_remove_does_not_park_destroyed_ce_member():
    key = ReplicaKey("task-a", "borrowed-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.BORROWED})()

    class CE:
        def __init__(self):
            self.effective_replicas = {key: ((runtime,), 7)}
            self.pending_bootstrap = {}

        def remove_effective(self, target):
            self.effective_replicas.pop(target)

        def register_pending(self, *args, **kwargs):
            raise AssertionError("borrowed REMOVE must not park a destroyed runtime")

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        commit_service_change = AsyncRemoteMethod(
            lambda operation: OperationEvidence(
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                1,
            )
        )

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    evidence = asyncio.run(
        trainer.remove_and_commit(
            OperationRecord("op-remove", OperationStatus.RUNNING)
        )
    )
    assert evidence.type is EvidenceType.SERVICE_COMMITTED
    assert trainer.checkpoint_manager.pending_bootstrap == {}


def test_trainer_internal_restore_rebinds_parked_native_and_publishes_current_vpub():
    key = ReplicaKey("task-a", "native-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.NATIVE})()
    calls = []

    class CE:
        def __init__(self):
            self.pending_bootstrap = {key: ((runtime,), "op-donate")}

        def discard_pending(self, target):
            calls.append(("discard_pending", target))
            self.pending_bootstrap.pop(target, None)

        def register_pending(self, target, replicas, *, operation_id):
            calls.append(("register_pending", target, operation_id))
            self.pending_bootstrap[target] = (tuple(replicas), operation_id)

        async def bootstrap_target(self, target, *, operation_id, loaded_version):
            calls.append(("bootstrap_target", target, operation_id, loaded_version))
            return OperationEvidence(
                operation_id,
                EvidenceType.WEIGHT_READY,
                2,
            )

        def commit_pending(self, target, evidence, *, loaded_version):
            calls.append(("commit_pending", target, loaded_version))
            assert evidence.type is EvidenceType.WEIGHT_READY

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        commit_service_change = AsyncRemoteMethod(
            lambda operation: calls.append(("commit_service", operation.operation_id))
            or OperationEvidence(
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                3,
            )
        )

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    trainer.current_param_version = 11

    evidence = asyncio.run(
        trainer.restore_and_publish(
            OperationRecord("op-restore", OperationStatus.RUNNING)
        )
    )
    assert evidence.type is EvidenceType.SERVICE_COMMITTED
    assert calls == [
        ("discard_pending", key),
        ("register_pending", key, "op-restore"),
        ("bootstrap_target", key, "op-restore", 11),
        ("commit_service", "op-restore"),
        ("commit_pending", key, 11),
    ]


def test_trainer_restore_service_failure_leaves_ce_pending_not_effective():
    key = ReplicaKey("task-a", "native-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.NATIVE})()
    calls = []

    class CE:
        def __init__(self):
            self.pending_bootstrap = {key: ((runtime,), "op-donate")}
            self.effective_replicas = {}

        def discard_pending(self, target):
            calls.append(("discard_pending", target))
            self.pending_bootstrap.pop(target, None)

        def register_pending(self, target, replicas, *, operation_id):
            calls.append(("register_pending", target, operation_id))
            self.pending_bootstrap[target] = (tuple(replicas), operation_id)

        async def bootstrap_target(self, target, *, operation_id, loaded_version):
            calls.append(("bootstrap_target", target, loaded_version))
            return OperationEvidence(
                operation_id,
                EvidenceType.WEIGHT_READY,
                2,
            )

        def commit_pending(self, *args, **kwargs):
            raise AssertionError("failed service publish must not promote E")

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        commit_service_change = AsyncRemoteMethod(
            lambda operation: (_ for _ in ()).throw(
                RuntimeError("service publish failed")
            )
        )

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    trainer.current_param_version = 13

    with pytest.raises(RuntimeError, match="service publish failed"):
        asyncio.run(
            trainer.restore_and_publish(
                OperationRecord("op-restore", OperationStatus.RUNNING)
            )
        )

    assert key in trainer.checkpoint_manager.pending_bootstrap
    assert trainer.checkpoint_manager.effective_replicas == {}
    assert trainer.replica_sync_gate.health == "BLOCKED"


def test_manager_owns_state_kind_and_runtime_inventory_separately():
    class Parent:
        def __init__(self, *args):
            self.rollout_replicas = []
            self.server_addresses = []
            self.server_handles = []
            self.rollout_config = type(
                "RolloutConfig",
                (),
                {
                    "tensor_model_parallel_size": 1,
                    "data_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "n_gpus_per_node": 1,
                },
            )()
            self.model_config = object()

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
    class FakeBorrowedRuntime:
        def __init__(self, **kwargs):
            self.replica_rank = kwargs["replica_rank"]
            self.workers = ["worker-0"]
            self.servers = ["server-0"]
            self.borrowed_worker_names = ("worker-name",)
            self.borrowed_server_names = ("server-name",)
            self._server_address = "s-borrowed"
            self._server_handle = "h-borrowed"
            self.borrowed_cleanup_verified = False
            self.cleaned = False

        async def init_from_lease(self, spec, pg_by_id):
            assert pg_by_id == {"pg": fake_pg}
            return {
                "state": "RUNTIME_READY",
                "replica_rank": self.replica_rank,
                "server_address": self._server_address,
            }

        async def cleanup_borrowed_runtime(self):
            self.cleaned = True
            self.borrowed_cleanup_verified = True

    cls = isolated(
        f"{INTEGRATION}/llm_server_manager.py",
        "MultiTaskLLMServerManager",
        Parent,
        MultiTaskvLLMReplica=FakeBorrowedRuntime,
        MultiTaskGlobalRequestLoadBalancer=object(),
        ReplicaKey=ReplicaKey,
        ReplicaKind=ReplicaKind,
        ReplicaState=ReplicaState,
        Lease=Lease,
        EvidenceType=EvidenceType,
        OperationEvidence=OperationEvidence,
        FIRST_RELEASE_MAX_COLOCATE_COUNT=FIRST_RELEASE_MAX_COLOCATE_COUNT,
        FIRST_RELEASE_RAY_GPU_FRACTION=FIRST_RELEASE_RAY_GPU_FRACTION,
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
                "gpu_fraction": FIRST_RELEASE_RAY_GPU_FRACTION,
                "cpu_request": 1.0,
            }
        ],
        "world_size": 1,
        "max_colocate_count": FIRST_RELEASE_MAX_COLOCATE_COUNT,
        "replica_rank": None,
        "expires_at": 0,
        "placement_epoch": 0,
    }
    normalized = manager.validate_borrowed_spec(valid_spec)
    assert "selected_slots" not in normalized
    assert normalized["max_colocate_count"] == FIRST_RELEASE_MAX_COLOCATE_COUNT
    assert normalized["claims"][0]["rank"] == 0
    assert normalized["lease_ids"] == ["source-lease-0"]

    wrong_task = dict(valid_spec, borrower_task_id="task-b")
    with pytest.raises(ValueError, match="another task_session"):
        manager.validate_borrowed_spec(wrong_task)

    wrong_world = dict(valid_spec, world_size=2)
    with pytest.raises(ValueError, match="world_size"):
        manager.validate_borrowed_spec(wrong_world)

    manager.rollout_config.tensor_model_parallel_size = 2
    with pytest.raises(ValueError, match="parallel topology"):
        manager.validate_borrowed_spec(valid_spec)
    manager.rollout_config.tensor_model_parallel_size = 1

    wrong_m = dict(valid_spec, max_colocate_count=1)
    with pytest.raises(ValueError, match="max_colocate_count"):
        manager.validate_borrowed_spec(wrong_m)

    wrong_share = dict(valid_spec)
    wrong_share["selected_slots"] = [
        dict(valid_spec["selected_slots"][0], gpu_fraction=1.0)
    ]
    with pytest.raises(ValueError, match="Ray GPU accounting share"):
        manager.validate_borrowed_spec(wrong_share)

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

    receipt = asyncio.run(manager.create_borrowed_replica(valid_spec))

    record = manager.borrowed_operations["borrower-lease"]
    assert receipt["state"] == "RUNTIME_READY"
    assert receipt["server_address"] == "s-borrowed"
    assert record["state"] == "RUNTIME_READY"
    assert record["replica_rank"] == 0
    borrowed_key = ReplicaKey("task-a", "borrowed-0", 0)
    assert record["replica_key"] == borrowed_key
    assert manager.replica_kind[borrowed_key] is ReplicaKind.BORROWED
    assert manager.replica_state[borrowed_key] is ReplicaState.CREATING
    assert manager.inspect_runtime(borrowed_key) is record["replica"]
    assert record["created_actor_names"] == ["worker-name", "server-name"]
    assert record["claim_ids"] == ["claim-0"]
    assert record["source_lease_ids"] == ["source-lease-0"]
    assert manager.next_replica_rank == 1

    # Exact replay returns the same hidden runtime receipt without another rank.
    assert asyncio.run(manager.create_borrowed_replica(valid_spec)) == receipt
    assert manager.next_replica_rank == 1

    # A retry may carry the rank recovered from the manager's first record.
    resolved_retry = dict(valid_spec, replica_rank=0)
    assert asyncio.run(manager.create_borrowed_replica(resolved_retry)) == receipt
    assert manager.next_replica_rank == 1

    release = asyncio.run(
        manager.destroy(borrowed_key, operation_id="op-remove")
    )
    assert release.type is EvidenceType.RELEASED
    assert release.released_gpu_uuids == ("u0",)
    assert record["replica"].cleaned is True
    assert asyncio.run(
        manager.destroy(borrowed_key, operation_id="op-remove")
    ) == release

    wrong_rank = dict(valid_spec, replica_rank=7)
    with pytest.raises(ValueError, match="conflicting borrowed create replay"):
        asyncio.run(manager.create_borrowed_replica(wrong_rank))

    conflicting = dict(valid_spec, operation_id="op-other")
    with pytest.raises(ValueError, match="conflicting borrowed create replay"):
        asyncio.run(manager.create_borrowed_replica(conflicting))
    assert manager.next_replica_rank == 1


def test_http_server_health_and_shutdown_use_real_engine_boundaries():
    class Engine:
        def __init__(self):
            self.healthy = False
            self.drained = False
            self.shutdown_called = False

        async def check_health(self):
            self.healthy = True

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
        assert health["engine_ready"] is True
        assert server.engine.healthy is True

        engine = server.engine
        receipt = await server.shutdown_runtime()
        assert receipt["shutdown"] is True
        assert engine.drained is True
        assert engine.shutdown_called is True
        assert server.engine is None
        assert server._server_port is None
        assert server._server_task.done()

    asyncio.run(scenario())


def replica_class():
    class Parent:
        def __init__(
            self,
            replica_rank,
            config,
            model_config,
            gpus_per_node=8,
            is_reward_model=False,
            is_teacher_model=False,
            name_suffix="",
        ):
            self.replica_rank = replica_rank
            self.config = config
            self.model_config = model_config
            self.world_size = (
                config.tensor_model_parallel_size
                * config.data_parallel_size
                * config.pipeline_model_parallel_size
            )
            self.gpus_per_node = gpus_per_node
            self.gpus_per_replica_node = min(gpus_per_node, self.world_size)
            self.nnodes = self.world_size // self.gpus_per_replica_node
            self.is_reward_model = is_reward_model
            self.is_teacher_model = is_teacher_model
            self.name_suffix = name_suffix
            self.workers = []
            self.servers = []

    fake_ray = type("ReplicaRay", (), {"remote": staticmethod(lambda cls: cls)})
    return isolated(
        "rollout/replica.py",
        "MultiTaskvLLMReplica",
        Parent,
        ReplicaKind=ReplicaKind,
        FIRST_RELEASE_MAX_COLOCATE_COUNT=FIRST_RELEASE_MAX_COLOCATE_COUNT,
        RayClassWithInitArgs=object,
        RayWorkerGroup=object,
        ResourcePoolManager=object,
        RolloutMode=object,
        get_device_name=lambda: "cuda",
        MultiTaskCheckpointEngineWorker=object,
        MultiTaskvLLMHttpServer=object,
        hashlib=__import__("hashlib"),
        ray=fake_ray,
    )


def test_borrowed_worker_plan_is_deterministic_and_side_effect_free():
    config = type(
        "Config",
        (),
        {
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
        },
    )()
    cls = replica_class()
    replica = cls(
        replica_rank=7,
        config=config,
        model_config=object(),
        replica_kind=ReplicaKind.BORROWED,
        runtime_epoch=3,
    )
    spec = {
        "operation_id": "op-add",
        "lease_id": "lease-1",
        "replica_rank": 7,
        "placement_epoch": 3,
        "world_size": 1,
        "max_colocate_count": FIRST_RELEASE_MAX_COLOCATE_COUNT,
        "claims": [
            {
                "claim_id": "claim-0",
                "rank": 0,
                "pg_id": "pg",
                "bundle_index": 4,
                "node_id": "node-a",
                "gpu_uuid": "GPU-0",
                "node_rank": 0,
                "local_rank": 0,
                "gpu_fraction": FIRST_RELEASE_RAY_GPU_FRACTION,
                "cpu_request": 1.0,
            }
        ],
    }

    first = replica.build_borrowed_worker_plan(spec)
    second = replica.build_borrowed_worker_plan(spec)

    assert first == second
    assert first[0]["actor_name"].startswith("borrowed_ce_7_")
    assert first[0]["pg_id"] == "pg"
    assert first[0]["bundle_index"] == 4
    assert first[0]["num_gpus"] == FIRST_RELEASE_RAY_GPU_FRACTION
    assert first[0]["env_vars"] == {
        "WORLD_SIZE": "1",
        "RANK": "0",
        "RAY_LOCAL_WORLD_SIZE": "1",
        "WG_PREFIX": first[0]["env_vars"]["WG_PREFIX"],
        "WG_BACKEND": "ray",
    }
    assert replica.placement_claims[0]["claim_id"] == "claim-0"


def test_borrowed_worker_plan_rejects_donor_topology_or_wrong_identity():
    config = type(
        "Config",
        (),
        {
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
        },
    )()
    replica = replica_class()(
        replica_rank=2,
        config=config,
        model_config=object(),
        replica_kind=ReplicaKind.BORROWED,
        runtime_epoch=1,
    )
    base = {
        "operation_id": "op-add",
        "lease_id": "lease-1",
        "replica_rank": 2,
        "placement_epoch": 1,
        "world_size": 1,
        "max_colocate_count": FIRST_RELEASE_MAX_COLOCATE_COUNT,
        "claims": [
            {
                "claim_id": "claim-0",
                "rank": 0,
                "pg_id": "pg",
                "bundle_index": 0,
                "node_id": "node-a",
                "gpu_uuid": "GPU-0",
                "node_rank": 0,
                "local_rank": 0,
                "gpu_fraction": FIRST_RELEASE_RAY_GPU_FRACTION,
                "cpu_request": 1.0,
            }
        ],
    }

    wrong_rank = dict(base, replica_rank=3)
    with pytest.raises(ValueError, match="replica_rank"):
        replica.build_borrowed_worker_plan(wrong_rank)

    donor_layout = dict(base)
    donor_layout["claims"] = [dict(base["claims"][0], node_rank=9)]
    with pytest.raises(ValueError, match="one-node borrower rank layout"):
        replica.build_borrowed_worker_plan(donor_layout)


def test_worker_gpu_uuid_probe_maps_ray_index_without_guessing():
    class Parent:
        pass

    fake_subprocess = type(
        "Subprocess",
        (),
        {
            "check_output": staticmethod(
                lambda *args, **kwargs: "0, GPU-a\n1, GPU-b\n"
            )
        },
    )
    cls = isolated(
        "checkpoint/checkpoint_engine_worker.py",
        "MultiTaskCheckpointEngineWorker",
        Parent,
        subprocess=fake_subprocess,
        os=__import__("os"),
        ray=object(),
        get_resource_name=lambda: "GPU",
        get_visible_devices_keyword=lambda: "CUDA_VISIBLE_DEVICES",
    )

    assert cls._resolve_nvidia_gpu_uuid("1") == "GPU-b"
    assert cls._resolve_nvidia_gpu_uuid("GPU-direct") == "GPU-direct"
    with pytest.raises(NotImplementedError, match="MIG"):
        cls._resolve_nvidia_gpu_uuid("MIG-abc")
    with pytest.raises(RuntimeError, match="cannot map"):
        cls._resolve_nvidia_gpu_uuid("opaque-id")


def test_create_workers_from_claims_clones_actor_options_per_rank():
    class Parent:
        def __init__(
            self,
            replica_rank,
            config,
            model_config,
            gpus_per_node=8,
            is_reward_model=False,
            is_teacher_model=False,
            name_suffix="",
        ):
            self.replica_rank = replica_rank
            self.config = config
            self.model_config = model_config
            self.world_size = 1
            self.gpus_per_node = gpus_per_node
            self.gpus_per_replica_node = 1
            self.nnodes = 1
            self.is_reward_model = is_reward_model
            self.is_teacher_model = is_teacher_model
            self.name_suffix = name_suffix
            self.workers = []
            self.servers = []

    created = []
    killed = []

    class FakeHandle:
        def __init__(self, record, *, gpu_uuid="GPU-x"):
            self.record = record
            self.runtime_placement = AsyncRemoteMethod(
                lambda: {
                    "node_id": "node",
                    "accelerator_id": "0",
                    "gpu_uuid": gpu_uuid,
                    "visible_devices": "0",
                }
            )

    class FakeCIA:
        next_gpu_uuid = "GPU-x"

        def __init__(self, cls, *args, **kwargs):
            self.cls = cls
            self.args = args
            self.kwargs = kwargs
            self.options = {}

        def update_options(self, options):
            self.options.update(options)

        def __call__(self, **kwargs):
            handle = FakeHandle(
                {
                    "options": dict(self.options),
                    "placement": dict(kwargs),
                },
                gpu_uuid=type(self).next_gpu_uuid,
            )
            created.append(handle)
            return handle

    class FakeWG:
        def __init__(self, workers):
            self.workers = list(workers)

        @classmethod
        def from_detached(cls, *, worker_handles, **kwargs):
            return cls(worker_handles)

    fake_ray = type(
        "ReplicaRay",
        (),
        {
            "remote": staticmethod(lambda cls: cls),
            "kill": staticmethod(lambda worker, **kwargs: killed.append(worker)),
            "get_runtime_context": staticmethod(
                lambda: type("Context", (), {"namespace": "test"})()
            ),
        },
    )
    cls = isolated(
        "rollout/replica.py",
        "MultiTaskvLLMReplica",
        Parent,
        ReplicaKind=ReplicaKind,
        FIRST_RELEASE_MAX_COLOCATE_COUNT=FIRST_RELEASE_MAX_COLOCATE_COUNT,
        RayClassWithInitArgs=FakeCIA,
        RayWorkerGroup=FakeWG,
        ResourcePoolManager=object,
        RolloutMode=object,
        PlacementGroupSchedulingStrategy=object,
        get_master_addr_port=object,
        get_device_name=lambda: "cuda",
        MultiTaskCheckpointEngineWorker=object,
        MultiTaskvLLMHttpServer=object,
        hashlib=__import__("hashlib"),
        asyncio=asyncio,
        list_actors=lambda **kwargs: [],
        ray=fake_ray,
    )
    config = type(
        "Config",
        (),
        {
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
        },
    )()
    replica = cls(
        replica_rank=4,
        config=config,
        model_config=object(),
        replica_kind=ReplicaKind.BORROWED,
        runtime_epoch=0,
    )
    async def fake_master(pg, bundle_index):
        assert pg == "PG"
        assert bundle_index == 2
        return "127.0.0.1", "23456"

    replica._get_master_addr_port_for_slot = fake_master
    spec = {
        "operation_id": "op",
        "lease_id": "lease",
        "replica_rank": 4,
        "placement_epoch": 0,
        "world_size": 1,
        "max_colocate_count": FIRST_RELEASE_MAX_COLOCATE_COUNT,
        "claims": [
            {
                "claim_id": "claim-0",
                "rank": 0,
                "pg_id": "pg",
                "bundle_index": 2,
                "node_id": "node",
                "gpu_uuid": "GPU-x",
                "node_rank": 0,
                "local_rank": 0,
                "gpu_fraction": FIRST_RELEASE_RAY_GPU_FRACTION,
                "cpu_request": 1.0,
            }
        ],
    }

    group = asyncio.run(replica._create_workers_from_claims(spec, {"pg": "PG"}))

    assert group.workers == created
    assert len(created) == 1
    record = created[0].record
    assert record["placement"]["placement_group"] == "PG"
    assert record["placement"]["placement_group_bundle_idx"] == 2
    assert record["placement"]["num_gpus"] == FIRST_RELEASE_RAY_GPU_FRACTION
    env = record["options"]["runtime_env"]["env_vars"]
    assert env["MASTER_ADDR"] == "127.0.0.1"
    assert env["MASTER_PORT"] == "23456"
    assert env["WORLD_SIZE"] == "1"
    assert env["RANK"] == "0"
    assert replica.borrowed_worker_names == (
        record["options"]["name"],
    )
    assert replica.borrowed_worker_placement[0]["gpu_uuid"] == "GPU-x"
    assert killed == []

    FakeCIA.next_gpu_uuid = "GPU-wrong"
    failed = cls(
        replica_rank=4,
        config=config,
        model_config=object(),
        replica_kind=ReplicaKind.BORROWED,
        runtime_epoch=0,
    )
    failed._get_master_addr_port_for_slot = fake_master
    with pytest.raises(RuntimeError, match="unexpected GPU UUID"):
        asyncio.run(failed._create_workers_from_claims(spec, {"pg": "PG"}))
    assert killed
    assert failed.workers == []


def test_init_from_lease_reaches_runtime_ready_only_after_server_health():
    class Parent:
        def __init__(
            self,
            replica_rank,
            config,
            model_config,
            gpus_per_node=8,
            is_reward_model=False,
            is_teacher_model=False,
            name_suffix="",
        ):
            self.replica_rank = replica_rank
            self.config = config
            self.model_config = model_config
            self.world_size = 1
            self.gpus_per_node = gpus_per_node
            self.gpus_per_replica_node = 1
            self.nnodes = 1
            self.is_reward_model = is_reward_model
            self.is_teacher_model = is_teacher_model
            self.name_suffix = name_suffix
            self.workers = []
            self.servers = []
            self._server_handle = None
            self._server_address = None

        async def launch_servers(self):
            health = {
                "node_id": "node",
                "replica_rank": self.replica_rank,
                "node_rank": 0,
                "nnodes": 1,
                "server_address": "127.0.0.1",
                "server_port": 8000,
                "engine_ready": True,
                "global_steps": None,
            }
            server = type(
                "Server",
                (),
                {
                    "runtime_health": AsyncRemoteMethod(lambda: health),
                    "shutdown_runtime": AsyncRemoteMethod(lambda: {"shutdown": True}),
                },
            )()
            self.servers = [server]
            self._server_handle = server
            self._server_address = "127.0.0.1:8000"

    fake_ray = type("ReplicaRay", (), {"remote": staticmethod(lambda cls: cls)})
    cls = isolated(
        "rollout/replica.py",
        "MultiTaskvLLMReplica",
        Parent,
        ReplicaKind=ReplicaKind,
        FIRST_RELEASE_MAX_COLOCATE_COUNT=FIRST_RELEASE_MAX_COLOCATE_COUNT,
        RayClassWithInitArgs=object,
        RayWorkerGroup=object,
        ResourcePoolManager=object,
        RolloutMode=type("RolloutMode", (), {"STANDALONE": "standalone"}),
        PlacementGroupSchedulingStrategy=object,
        get_master_addr_port=object,
        get_device_name=lambda: "cuda",
        MultiTaskCheckpointEngineWorker=object,
        MultiTaskvLLMHttpServer=object,
        hashlib=__import__("hashlib"),
        asyncio=asyncio,
        list_actors=lambda **kwargs: [],
        ray=fake_ray,
    )
    config = type(
        "Config",
        (),
        {
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
        },
    )()
    replica = cls(
        replica_rank=6,
        config=config,
        model_config=object(),
        replica_kind=ReplicaKind.BORROWED,
        runtime_epoch=2,
    )
    worker = type(
        "Worker",
        (),
        {
            "runtime_placement": AsyncRemoteMethod(
                lambda: {
                    "node_id": "node",
                    "accelerator_id": "0",
                    "gpu_uuid": "GPU-x",
                    "visible_devices": "0",
                }
            )
        },
    )()

    async def fake_create(spec, pg_by_id):
        replica.build_borrowed_worker_plan(spec)
        replica.workers = [worker]
        replica.borrowed_worker_names = ("worker-0",)
        await replica.validate_worker_placement()
        return object()

    replica._create_workers_from_claims = fake_create
    spec = {
        "operation_id": "op",
        "lease_id": "lease",
        "replica_rank": 6,
        "placement_epoch": 2,
        "world_size": 1,
        "max_colocate_count": FIRST_RELEASE_MAX_COLOCATE_COUNT,
        "claims": [
            {
                "claim_id": "claim-0",
                "rank": 0,
                "pg_id": "pg",
                "bundle_index": 0,
                "node_id": "node",
                "gpu_uuid": "GPU-x",
                "node_rank": 0,
                "local_rank": 0,
                "gpu_fraction": FIRST_RELEASE_RAY_GPU_FRACTION,
                "cpu_request": 1.0,
            }
        ],
    }

    receipt = asyncio.run(replica.init_from_lease(spec, {"pg": "PG"}))

    assert receipt["state"] == "RUNTIME_READY"
    assert receipt["server_address"] == "127.0.0.1:8000"
    assert receipt["health"]["engine_ready"] is True
    assert replica.borrowed_runtime_state == "RUNTIME_READY"


def checkpoint_manager_class(**extra_scope):
    class Parent:
        def __init__(self, config=None, actor_wg=None, replicas=None):
            self.config = config
            self.backend = getattr(config, "backend", "nccl")
            self.actor_wg = actor_wg
            self.replicas = list(replicas or [])
            self.build_calls = []

        def remove_replicas(self, replicas):
            for replica in replicas:
                if replica in self.replicas:
                    self.replicas.remove(replica)

        def build_process_group(self, rollout):
            self.build_calls.append(rollout)

    scope = {
        "ReplicaKey": ReplicaKey,
        "OperationEvidence": OperationEvidence,
        "EvidenceType": EvidenceType,
        "asyncio": asyncio,
        **extra_scope,
    }
    return isolated(
        "checkpoint/checkpoint_engine_manager.py",
        "MultiTaskCheckpointEngineManager",
        Parent,
        **scope,
    )


def test_ce_pending_bootstrap_is_not_effective_until_weight_ready():
    ce = checkpoint_manager_class()(replicas=["native"])
    key = ReplicaKey("task-a", "borrowed-0")
    ce.register_pending(key, ["borrowed"], operation_id="op-add")

    assert ce.pending_bootstrap[key] == (("borrowed",), "op-add")
    assert key not in ce.effective_replicas
    assert ce.replicas == ["native"]

    with pytest.raises(ValueError, match="only from WEIGHT_READY"):
        ce.commit_pending(
            key,
            OperationEvidence("op-add", EvidenceType.SERVICE_COMMITTED, 1),
            loaded_version=7,
        )
    assert ce.replicas == ["native"]

    # Model the CE-owned confirmation; real transfer ordering is tested below.
    ce._bootstrap_ready()[key] = (
        "op-add", 7, OperationEvidence("op-add", EvidenceType.WEIGHT_READY, 2)
    )
    ce.commit_pending(
        key,
        OperationEvidence("op-add", EvidenceType.WEIGHT_READY, 2),
        loaded_version=7,
    )
    assert key not in ce.pending_bootstrap
    assert ce.effective_replicas[key] == (("borrowed",), 7)
    assert ce.replicas == ["native", "borrowed"]

    ce.commit_pending(
        key,
        OperationEvidence("op-add", EvidenceType.WEIGHT_READY, 2),
        loaded_version=7,
    )


def test_ce_pending_bootstrap_rejects_operation_rebind():
    ce = checkpoint_manager_class()(replicas=[])
    key = ReplicaKey("task-a", "borrowed-0")
    ce.register_pending(key, ["borrowed"], operation_id="op-1")

    with pytest.raises(ValueError, match="conflicting pending bootstrap"):
        ce.register_pending(key, ["borrowed"], operation_id="op-2")

    with pytest.raises(ValueError, match="another operation"):
        ce.commit_pending(
            key,
            OperationEvidence("op-2", EvidenceType.WEIGHT_READY, 1),
            loaded_version=3,
        )


def test_ce_target_bootstrap_syncs_only_pending_target_and_is_idempotent():
    calls = []

    class FakeRolloutWG:
        def __init__(self, workers):
            self.workers = workers
            self.world_size = len(workers)

        @classmethod
        def from_detached(cls, *, worker_handles, **kwargs):
            calls.append(("wrap", tuple(worker_handles)))
            return cls(list(worker_handles))

        def update_weights(self, *, global_steps):
            calls.append(("target-update", global_steps))
            return [("target", global_steps)]

        def execute_checkpoint_engine(self, methods):
            calls.append(("target-finalize", tuple(methods)))
            return [("target-finalize", len(methods))]

    class FakeCIA:
        def __init__(self, cls, *args, **kwargs):
            self.cls = cls

    class ActorWG:
        world_size = 1

        def update_weights(self, *, global_steps, mode):
            calls.append(("actor-update", global_steps, mode))
            return [("actor", global_steps)]

        def execute_checkpoint_engine(self, methods):
            calls.append(("actor-finalize", tuple(methods)))
            return [("actor-finalize", len(methods))]

    class FakeRay:
        @staticmethod
        def remote(cls):
            return cls

        @staticmethod
        def get(value):
            calls.append(("ray-get", tuple(value)))
            return list(value)

    class Replica:
        workers = ["worker-0"]

        async def release_kv_cache(self):
            calls.append(("release-kv",))

        async def resume_kv_cache(self):
            calls.append(("resume-kv",))

        async def validate_server_runtime(self):
            calls.append(("health",))
            return {"global_steps": 7}

    config = type("Config", (), {"backend": "nccl"})()
    cls = checkpoint_manager_class(
        ray=FakeRay,
        RayClassWithInitArgs=FakeCIA,
        RayWorkerGroup=FakeRolloutWG,
        MultiTaskCheckpointEngineWorker=object,
    )
    ce = cls(config=config, actor_wg=ActorWG(), replicas=["native"])
    key = ReplicaKey("task-a", "borrowed-0")
    replica = Replica()
    ce.register_pending(key, [replica], operation_id="op-add")

    first = asyncio.run(
        ce.bootstrap_target(key, operation_id="op-add", loaded_version=7)
    )
    assert first.type is EvidenceType.WEIGHT_READY
    assert key not in ce.effective_replicas
    assert ce.replicas == ["native"]
    assert ce.build_calls and ce.build_calls[0].workers == ["worker-0"]
    assert ("actor-update", 7, "nccl") in calls
    assert ("target-update", 7) in calls
    assert ("health",) in calls

    before = list(calls)
    second = asyncio.run(
        ce.bootstrap_target(key, operation_id="op-add", loaded_version=7)
    )
    assert second == first
    assert calls == before

    ce.commit_pending(key, first, loaded_version=7)
    assert ce.effective_replicas[key] == ((replica,), 7)
    assert ce.replicas == ["native", replica]

    with pytest.raises(ValueError, match="conflicting target bootstrap replay"):
        asyncio.run(
            ce.bootstrap_target(key, operation_id="op-add", loaded_version=8)
        )


def test_ce_target_bootstrap_requires_server_version_confirmation():
    class FakeRolloutWG:
        world_size = 1

        @classmethod
        def from_detached(cls, *, worker_handles, **kwargs):
            return cls()

        def update_weights(self, *, global_steps):
            return [None]

        def execute_checkpoint_engine(self, methods):
            return [None]

    class FakeCIA:
        def __init__(self, cls, *args, **kwargs):
            pass

    class ActorWG:
        world_size = 1

        def update_weights(self, *, global_steps, mode):
            return [None]

        def execute_checkpoint_engine(self, methods):
            return [None]

    class FakeRay:
        @staticmethod
        def remote(cls):
            return cls

        @staticmethod
        def get(value):
            return value

    class Replica:
        workers = ["worker-0"]

        async def release_kv_cache(self):
            return None

        async def resume_kv_cache(self):
            return None

        async def validate_server_runtime(self):
            return {"global_steps": 6}

    config = type("Config", (), {"backend": "nccl"})()
    cls = checkpoint_manager_class(
        ray=FakeRay,
        RayClassWithInitArgs=FakeCIA,
        RayWorkerGroup=FakeRolloutWG,
        MultiTaskCheckpointEngineWorker=object,
    )
    ce = cls(config=config, actor_wg=ActorWG(), replicas=[])
    key = ReplicaKey("task-a", "borrowed-0")
    ce.register_pending(key, [Replica()], operation_id="op-add")

    with pytest.raises(RuntimeError, match="published parameter version"):
        asyncio.run(
            ce.bootstrap_target(key, operation_id="op-add", loaded_version=7)
        )
    assert key not in ce._bootstrap_ready()



def test_ce_can_advance_one_effective_member_version():
    ce = checkpoint_manager_class()(replicas=[])
    key = ReplicaKey("task-a", "r0")
    ce.add_effective(key, ["replica"], loaded_version=2)

    ce.mark_loaded_version(key, 3)

    assert ce.effective_replicas[key] == (("replica",), 3)


def test_ce_rejects_unverified_weight_ready_without_changing_membership():
    ce = checkpoint_manager_class()(replicas=[])
    key = ReplicaKey("task-a", "borrowed-0")
    ce.register_pending(key, ["borrowed"], operation_id="op-add")
    with pytest.raises(ValueError, match="confirmed bootstrap"):
        ce.commit_pending(
            key, OperationEvidence("op-add", EvidenceType.WEIGHT_READY, 2),
            loaded_version=7,
        )
    assert key in ce.pending_bootstrap
    assert ce.effective_replicas == {}
    assert ce.replicas == []


@pytest.mark.parametrize("version", [6, 8, True])
def test_ce_commit_requires_confirmed_bootstrap_version(version):
    ce = checkpoint_manager_class()(replicas=[])
    key = ReplicaKey("task-a", "borrowed-0")
    evidence = OperationEvidence("op-add", EvidenceType.WEIGHT_READY, 2)
    ce.register_pending(key, ["borrowed"], operation_id="op-add")
    ce._bootstrap_ready()[key] = ("op-add", 7, evidence)
    with pytest.raises(ValueError):
        ce.commit_pending(key, evidence, loaded_version=version)
    assert ce.effective_replicas == {}
    assert ce.replicas == []


def test_ce_remove_invalidates_bootstrap_result_for_reused_key():
    ce = checkpoint_manager_class()(replicas=[])
    key = ReplicaKey("task-a", "borrowed-0")
    evidence = OperationEvidence("op-add", EvidenceType.WEIGHT_READY, 2)
    ce.register_pending(key, ["old-runtime"], operation_id="op-add")
    ce._bootstrap_ready()[key] = ("op-add", 7, evidence)
    ce.commit_pending(key, evidence, loaded_version=7)
    ce.remove_effective(key)
    ce.register_pending(key, ["new-runtime"], operation_id="op-next")
    assert key not in ce._bootstrap_ready()
    assert ce.replicas == []


def test_ce_committed_replay_rejects_replaced_evidence():
    ce = checkpoint_manager_class()(replicas=[])
    key = ReplicaKey("task-a", "borrowed-0")
    evidence = OperationEvidence("op-add", EvidenceType.WEIGHT_READY, 2)
    ce.register_pending(key, ["runtime"], operation_id="op-add")
    ce._bootstrap_ready()[key] = ("op-add", 7, evidence)
    ce.commit_pending(key, evidence, loaded_version=7)
    with pytest.raises(ValueError, match="confirmed bootstrap"):
        ce.commit_pending(
            key, OperationEvidence("op-add", EvidenceType.WEIGHT_READY, 3),
            loaded_version=7,
        )
    assert ce.effective_replicas[key] == (("runtime",), 7)


@pytest.mark.parametrize("stage", ["pending", "effective"])
def test_ce_runtime_cannot_belong_to_two_replica_keys(stage):
    ce = checkpoint_manager_class()(replicas=[])
    first = ReplicaKey("task-a", "first")
    second = ReplicaKey("task-a", "second")
    if stage == "pending":
        ce.register_pending(first, ["runtime"], operation_id="op-first")
    else:
        ce.add_effective(first, ["runtime"], loaded_version=1)
    with pytest.raises(ValueError):
        ce.register_pending(second, ["runtime"], operation_id="op-second")
    with pytest.raises(ValueError, match="another ReplicaKey"):
        ce.add_effective(second, ["runtime"], loaded_version=1)


def test_ce_rejects_duplicate_target_runtimes():
    ce = checkpoint_manager_class()(replicas=[])
    key = ReplicaKey("task-a", "borrowed-0")
    with pytest.raises(ValueError, match="duplicate"):
        ce.register_pending(key, ["runtime", "runtime"], operation_id="op-add")
    with pytest.raises(ValueError, match="duplicate"):
        ce.add_effective(key, ["runtime", "runtime"], loaded_version=1)


def test_rollouter_idle_detection_treats_unknown_capacity_as_zero():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    rollouter.max_concurrent_samples = None
    assert rollouter.committed_capacity == 0
    assert rollouter.collect_idle_candidates() == ()


def http_server_class():
    return isolated(
        "rollout/http_server.py",
        "MultiTaskvLLMHttpServer",
        object,
        asyncio=asyncio,
        ray=FakeRay,
    )


def replica_class():
    return isolated(
        "rollout/replica.py",
        "MultiTaskvLLMReplica",
        object,
        asyncio=asyncio,
        ReplicaKind=ReplicaKind,
        FIRST_RELEASE_MAX_COLOCATE_COUNT=FIRST_RELEASE_MAX_COLOCATE_COUNT,
    )


def native_manager_class():
    return isolated(
        f"{INTEGRATION}/llm_server_manager.py",
        "MultiTaskLLMServerManager",
        object,
        ReplicaKey=ReplicaKey,
        ReplicaKind=ReplicaKind,
        ReplicaState=ReplicaState,
        EvidenceType=EvidenceType,
        OperationEvidence=OperationEvidence,
        asyncio=asyncio,
    )


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

    weights_receipt = asyncio.run(server.wake_weights())
    assert weights_receipt["fully_awake"] is False
    assert weights_receipt["sleeping"] is True
    assert server._submission_paused is True
    assert server._resume_event.is_set is False

    wake_receipt = asyncio.run(server.wake_up())
    assert wake_receipt["fully_awake"] is True
    assert wake_receipt["sleeping"] is False
    assert server._submission_paused is False
    assert server._resume_event.is_set is True
    assert ("sleep", 2, "abort") in calls
    assert ("wake_up", ("weights",)) in calls
    assert ("reset_prefix_cache", True) in calls
    assert calls[-2:] == [("health",), ("gate", "set")]


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

    with pytest.raises(NotImplementedError, match="level-2 sleep"):
        asyncio.run(server.sleep())
    assert server._submission_paused is False
    assert server._resume_event.clear_calls == 0


def test_restore_partial_wake_makes_release_kv_cache_idempotent():
    class Engine:
        async def is_sleeping(self):
            return True

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

    receipt = asyncio.run(server.release_kv_cache())
    assert receipt == {"kv_cache_released": True, "already_sleeping": True}


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


def test_native_runtime_health_probe_confirms_server_and_ce_worker_node():
    health = {
        "node_id": "node-0",
        "replica_rank": 2,
        "node_rank": 0,
        "nnodes": 1,
        "server_address": "127.0.0.1",
        "server_port": 8000,
        "engine_ready": True,
        "global_steps": 12,
    }

    class Server:
        runtime_health = AsyncRemoteMethod(lambda: dict(health))

    class Worker:
        runtime_placement = AsyncRemoteMethod(
            lambda: {"node_id": "node-0", "gpu_uuid": "u0"}
        )

    cls = replica_class()
    replica = cls.__new__(cls)
    replica.replica_kind = ReplicaKind.NATIVE
    replica.replica_rank = 2
    replica.nnodes = 1
    replica.servers = [Server()]
    replica.workers = [Worker()]
    replica._server_handle = replica.servers[0]
    replica._server_address = "127.0.0.1:8000"

    result = asyncio.run(replica.validate_server_runtime())
    assert result["global_steps"] == 12
    assert result["node_id"] == "node-0"


def test_native_replica_requires_verified_server_receipts_for_sleep_and_wake():
    class Server:
        sleep = AsyncRemoteMethod(
            lambda: {
                "sleep_level": 2,
                "sleeping": True,
            }
        )
        wake_up = AsyncRemoteMethod(
            lambda *args, **kwargs: {
                "sleeping": False,
                "fully_awake": True,
            }
        )

    cls = replica_class()
    replica = cls.__new__(cls)
    replica.replica_kind = ReplicaKind.NATIVE
    replica.servers = [Server()]

    assert asyncio.run(replica.sleep())[0]["sleep_level"] == 2
    assert asyncio.run(replica.wake_up())[0]["fully_awake"] is True


def test_manager_native_sleep_binds_released_evidence_to_runtime_gpu_uuids():
    key = ReplicaKey("task-a", "native-0")

    class Worker:
        def __init__(self, gpu_uuid):
            self.runtime_placement = AsyncRemoteMethod(
                lambda: {"node_id": "n0", "gpu_uuid": gpu_uuid}
            )

    class Runtime:
        workers = [Worker("u0"), Worker("u1")]

        async def sleep(self):
            return (
                {"sleep_level": 2, "sleeping": True},
            )

    cls = native_manager_class()
    manager = cls.__new__(cls)
    manager.replica_kind = {key: ReplicaKind.NATIVE}
    manager.replica_state = {key: ReplicaState.DRAINING}
    manager._runtime_inventory = {key: Runtime()}

    evidence = asyncio.run(manager.sleep(key, operation_id="op-donate"))
    assert evidence.type is EvidenceType.RELEASED
    assert evidence.operation_id == "op-donate"
    assert evidence.released_gpu_uuids == ("u0", "u1")


def test_manager_native_wake_weights_keeps_dormant_runtime_fenced():
    key = ReplicaKey("task-a", "native-0")

    class Server:
        wake_weights = AsyncRemoteMethod(
            lambda: {"sleeping": True, "fully_awake": False}
        )

    runtime = type("Runtime", (), {"servers": [Server()]})()
    manager = native_manager_class().__new__(native_manager_class())
    manager.replica_kind = {key: ReplicaKind.NATIVE}
    manager.replica_state = {key: ReplicaState.DORMANT}
    manager._runtime_inventory = {key: runtime}

    receipts = asyncio.run(manager.wake_weights(key))
    assert receipts == ({"sleeping": True, "fully_awake": False},)


def test_group_scheduler_binds_donate_to_lease_donor_rank():
    path = SOURCE / "scheduler/group_scheduler.py"
    tree = ast.parse(path.read_text())
    scheduler = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GroupScheduler"
    )
    scheduler.decorator_list = []
    module = ast.Module(
        body=[scheduler],
        type_ignores=[],
    )
    env = {
        "ReplicaKey": ReplicaKey,
        "ActorHandle": object,
        "Lease": Lease,
        "OperationCommand": OperationCommand,
        "OperationEvidence": OperationEvidence,
        "OperationKind": OperationKind,
        "OperationRecord": OperationRecord,
        "RUNTIME_KIND": "test",
        "EvidenceType": EvidenceType,
        "time": time,
        "ray": type("Ray", (), {"remote": staticmethod(lambda **kwargs: (lambda cls: cls))}),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    cls = env["GroupScheduler"]
    lease = Lease(
        "l1",
        ({
            "claim_id": "claim-1",
            "source_lease_id": "source-1",
            "donor_task_id": "task-a",
            "donor_replica_rank": 1,
            "pg_id": "pg",
            "bundle_index": 1,
            "node_id": "n0",
            "gpu_uuid": "u1",
            "gpu_fraction": 0.5,
            "cpu_request": 1.0,
        },),
    )

    assert cls._target_matches_donor(ReplicaKey("task-a", "native-1"), lease)
    assert cls._target_matches_donor(ReplicaKey("task-a", "r1"), lease)
    assert not cls._target_matches_donor(ReplicaKey("task-a", "native-0"), lease)

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
            return self.replica_kind[target], self.replica_state[target]

        async def wake_weights(self, target):
            calls.append(("wake_weights", target))
            return ({"sleeping": True, "fully_awake": False},)

    rollouter.llm_server_manager = Manager()
    receipts = asyncio.run(
        rollouter.prepare_replica(
            key,
            operation_id="op-restore",
            spec=None,
        )
    )
    assert receipts[0]["fully_awake"] is False
    assert calls == [("wake_weights", key)]
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


def test_rollouter_restore_routing_failure_rolls_back_to_verified_dormant():
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
            calls.append(("sleep",))
            return ({"sleep_level": 2, "sleeping": True},)

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

    with pytest.raises(RuntimeError, match="routing commit failed"):
        asyncio.run(
            rollouter.commit_service_change(
                OperationRecord("op-restore", OperationStatus.RUNNING)
            )
        )
    assert manager.replica_state[key] is ReplicaState.DORMANT
    assert calls == [
        ("wake_up",),
        ("activate_service", key),
        ("capacity",),
        ("state", ReplicaState.ACTIVE),
        ("state", ReplicaState.DRAINING),
        ("deactivate_service", key),
        ("capacity",),
        ("sleep",),
        ("state", ReplicaState.DORMANT),
    ]


def test_rollouter_restore_unverified_rollback_quarantines_native():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "native-0")

    class Runtime:
        _server_address = "s0"
        _server_handle = "h0"

        async def wake_up(self):
            return ({"sleeping": False, "fully_awake": True},)

        async def sleep(self):
            raise RuntimeError("cannot prove re-sleep")

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
            return None

        def deactivate_service(self, target):
            return None

        def transition_replica(self, target, state):
            self.replica_state[target] = state

    manager = Manager()
    rollouter.llm_server_manager = manager
    rollouter._pending_operation_targets["op-restore"] = key
    rollouter._update_max_concurrent_samples = lambda: None

    with pytest.raises(
        RuntimeError,
        match="publish failed and rollback is unverified",
    ):
        asyncio.run(
            rollouter.commit_service_change(
                OperationRecord("op-restore", OperationStatus.RUNNING)
            )
        )
    assert manager.replica_state[key] is ReplicaState.QUARANTINED


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


def test_rollouter_force_is_fail_closed_before_mutating_m_or_r():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "borrowed-0")
    rollouter.config = type(
        "Config",
        (),
        {"async_training": type("Async", (), {"partial_rollout": True})()},
    )()
    rollouter._continuation_client_ready = True

    class LB:
        def __getattr__(self, name):
            raise AssertionError(f"FORCE fail-closed path must not touch LB: {name}")

    class Runtime:
        async def abort_all_requests(self):
            raise AssertionError("FORCE fail-closed path must not abort requests")

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
            raise AssertionError("FORCE fail-closed path must not mutate M")

    manager = Manager()
    rollouter.llm_server_manager = manager

    with pytest.raises(
        NotImplementedError,
        match="verified targeted abort/continuation backend",
    ):
        asyncio.run(
            rollouter.prepare_exit(key, operation_id="op-force", force=True)
        )

    assert manager.replica_state[key] is ReplicaState.ACTIVE
    assert "op-force" not in rollouter._pending_operation_targets


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
