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


def _test_require_evidence(value, operation_id, expected, label):
    if not isinstance(value, OperationEvidence):
        raise TypeError(f"{label} did not return OperationEvidence")
    if value.operation_id != operation_id:
        raise ValueError(f"{label} evidence belongs to another operation")
    if value.type is not expected:
        raise ValueError(f"expected {expected.value}, got {value.type.value}")
    return value


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
        _require_evidence=_test_require_evidence,
    )


def load_balancer_class():
    class Parent:
        def __init__(self, servers, **kwargs):
            self._servers = dict(servers)
            self._inflight_requests = {server_id: 0 for server_id in servers}

        def acquire_server(self, request_id, **extra):
            if not self._servers:
                raise RuntimeError("No available servers")
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
        lb.release_server(server_id, request_id=request_id)

    assert len(lb.attempt_state) <= 2
    assert set(lb.attempt_state) == {"request-3", "request-4"}
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
    missing = runner.query_operation("missing")
    assert missing.status is OperationStatus.UNKNOWN
    assert missing.result is None


def test_taskrunner_exact_unknown_remove_replay_relaunches_same_operation():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    command = OperationCommand(
        "op-remove-reconcile",
        OperationKind.REMOVE,
        ReplicaKey("task-a", "borrowed-0"),
        "l1",
    )
    runner._operation_journal.begin(command)
    runner._operation_journal.mark_running(command.operation_id)
    runner._operation_journal.finish(
        command.operation_id,
        OperationStatus.UNKNOWN,
        "release ACK unknown",
    )
    launched = []
    runner._launch_operation = launched.append

    replay = runner.submit_operation(command)

    assert replay.status is OperationStatus.UNKNOWN
    assert launched == [command.operation_id]


def test_taskrunner_exact_unknown_add_replay_relaunches_same_operation():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    command = OperationCommand(
        "op-add-reconcile",
        OperationKind.ADD,
        ReplicaKey("task-a", "borrowed-0"),
        "l1",
    )
    lease = taskrunner_lease()
    runner._operation_journal.begin(command)
    runner._operation_journal.mark_running(command.operation_id)
    runner._operation_journal.finish(
        command.operation_id,
        OperationStatus.UNKNOWN,
        "routing outcome unknown",
    )
    runner._operation_leases[command.operation_id] = lease
    launched = []
    runner._launch_operation = launched.append

    replay = runner.submit_operation(command, lease=lease)

    assert replay.status is OperationStatus.UNKNOWN
    assert launched == [command.operation_id]


def test_taskrunner_unknown_reconciliation_launch_failure_preserves_fence_and_lease():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    command = OperationCommand(
        "op-add-reconcile-launch-fail",
        OperationKind.ADD,
        ReplicaKey("task-a", "borrowed-0"),
        "l1",
    )
    lease = taskrunner_lease()
    runner._operation_journal.begin(command)
    runner._operation_journal.mark_running(command.operation_id)
    runner._operation_journal.finish(
        command.operation_id,
        OperationStatus.UNKNOWN,
        "owner outcome unknown",
    )
    runner._operation_leases[command.operation_id] = lease

    def fail_launch(operation_id):
        runner._operation_threads[operation_id] = object()
        raise RuntimeError("thread start failed")

    runner._launch_operation = fail_launch

    with pytest.raises(RuntimeError, match="thread start failed"):
        runner.submit_operation(command, lease=lease)

    record = runner.query_operation(command.operation_id)
    assert record.status is OperationStatus.UNKNOWN
    assert record.result == "owner outcome unknown"
    assert runner._operation_leases[command.operation_id] == lease
    assert command.operation_id not in runner._operation_threads


def test_taskrunner_add_is_admitted_with_matching_lease_snapshot():
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

    record = runner.submit_operation(command, lease=taskrunner_lease())

    assert record.status is OperationStatus.ACCEPTED
    assert launched == ["op-add"]
    assert runner._operation_leases["op-add"] == taskrunner_lease()


def test_taskrunner_verified_add_rollback_finishes_failed_and_advances_lease():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    runner._launch_operation = lambda operation_id: None
    key = ReplicaKey("task-a", "borrowed-0")
    calls = []

    class Rollouter:
        prepare_replica = RemoteMethod(
            lambda target, **kwargs: calls.append(
                ("prepare_replica", target, kwargs["operation_id"])
            )
        )

    class Trainer:
        bootstrap_and_publish = RemoteMethod(
            lambda operation: OperationEvidence(
                operation.operation_id,
                EvidenceType.RELEASED,
                2,
                ("u0",),
            )
        )

    class GroupScheduler:
        advance_lease = RemoteMethod(
            lambda lease_id, evidence: calls.append(
                ("advance_lease", lease_id, evidence.type, evidence.released_gpu_uuids)
            )
            or {"add_rolled_back": True}
        )

    runner.components = {"rollouter": Rollouter(), "trainer": Trainer()}
    runner.group_scheduler = GroupScheduler()
    runner.submit_operation(
        OperationCommand("op-add-rollback", OperationKind.ADD, key, "l1"),
        lease=taskrunner_lease(),
    )
    runner._execute_operation("op-add-rollback")

    record = runner.query_operation("op-add-rollback")
    assert record.status is OperationStatus.FAILED
    assert "verified RELEASED" in record.result
    assert calls == [
        ("prepare_replica", key, "op-add-rollback"),
        ("advance_lease", "l1", EvidenceType.RELEASED, ("u0",)),
    ]


def test_taskrunner_verified_add_prepare_failure_skips_trainer_and_finishes_failed():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    runner._launch_operation = lambda operation_id: None
    key = ReplicaKey("task-a", "borrowed-0")
    calls = []

    class Rollouter:
        prepare_replica = RemoteMethod(
            lambda target, **kwargs: OperationEvidence(
                kwargs["operation_id"],
                EvidenceType.RELEASED,
                2,
                ("u0",),
            )
        )

    class Trainer:
        bootstrap_and_publish = RemoteMethod(
            lambda operation: (_ for _ in ()).throw(
                AssertionError("known-safe prepare failure must not enter Trainer")
            )
        )

    class GroupScheduler:
        advance_lease = RemoteMethod(
            lambda lease_id, evidence: calls.append(
                ("advance_lease", lease_id, evidence.type, evidence.released_gpu_uuids)
            )
            or {"add_rolled_back": True}
        )

    runner.components = {"rollouter": Rollouter(), "trainer": Trainer()}
    runner.group_scheduler = GroupScheduler()
    runner.submit_operation(
        OperationCommand("op-add-prepare-fail", OperationKind.ADD, key, "l1"),
        lease=taskrunner_lease(),
    )
    runner._execute_operation("op-add-prepare-fail")

    record = runner.query_operation("op-add-prepare-fail")
    assert record.status is OperationStatus.FAILED
    assert record.result == "ADD prepare failed; no borrower runtime remains"
    assert calls == [
        ("advance_lease", "l1", EvidenceType.RELEASED, ("u0",)),
    ]


def test_taskrunner_replays_identical_lease_evidence_after_ack_loss():
    runner = taskrunner_class()()
    command = OperationCommand(
        "op-remove-ack-loss",
        OperationKind.REMOVE,
        ReplicaKey("task-a", "borrowed-0"),
        "l1",
    )
    evidence = OperationEvidence(
        command.operation_id,
        EvidenceType.RELEASED,
        4,
        ("u0",),
    )
    calls = []

    class GroupScheduler:
        def __init__(self):
            self.advance_lease = RemoteMethod(self._advance)

        def _advance(self, lease_id, received):
            calls.append((lease_id, received))
            if len(calls) == 1:
                raise TimeoutError("reply lost after commit")
            return {"released": True}

    runner.group_scheduler = GroupScheduler()
    runner._advance_lease(command, evidence)

    assert calls == [("l1", evidence), ("l1", evidence)]


def test_taskrunner_unknown_remove_replay_uses_verified_release_without_repeating_exit():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    runner._launch_operation = lambda operation_id: None
    key = ReplicaKey("task-a", "borrowed-0")
    command = OperationCommand("op-remove-release-lost", OperationKind.REMOVE, key, "l1")
    runner._operation_journal.begin(command)
    runner._operation_journal.mark_running(command.operation_id)
    runner._operation_journal.finish(
        command.operation_id,
        OperationStatus.UNKNOWN,
        "final release reply lost",
    )
    calls = []
    release = OperationEvidence(
        command.operation_id,
        EvidenceType.RELEASED,
        11,
        ("u0",),
    )

    class Rollouter:
        query_release_operation = RemoteMethod(
            lambda target, operation_id: calls.append(
                ("query_release", target, operation_id)
            ) or release
        )
        prepare_exit = RemoteMethod(
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("verified release replay must not drain again")
            )
        )

    class Trainer:
        reconcile_exit = RemoteMethod(
            lambda operation: (_ for _ in ()).throw(
                AssertionError("verified release replay must not enter Trainer")
            )
        )

    class GroupScheduler:
        advance_lease = RemoteMethod(
            lambda lease_id, evidence: calls.append(
                ("advance_lease", lease_id, evidence)
            ) or {"released": True}
        )

    runner.components = {"rollouter": Rollouter(), "trainer": Trainer()}
    runner.group_scheduler = GroupScheduler()
    runner._execute_operation(command.operation_id)

    record = runner.query_operation(command.operation_id)
    assert record.status is OperationStatus.SUCCEEDED
    assert record.result == EvidenceType.RELEASED.value
    assert calls == [
        ("query_release", key, command.operation_id),
        ("advance_lease", "l1", release),
    ]


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
    runner._operation_threads["op-remove"] = object()
    runner._execute_operation("op-remove")

    assert "op-remove" not in runner._operation_threads
    record = runner.query_operation("op-remove")
    assert record.status is OperationStatus.SUCCEEDED
    assert record.result == EvidenceType.RELEASED.value
    assert calls == [
        ("prepare_exit", key, "op-remove", False),
        ("remove_and_commit", "op-remove"),
        ("finalize_release", "op-remove"),
        ("advance_lease", "l1", ("u0",)),
    ]


def test_taskrunner_restore_is_admitted_and_launched():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    launched = []
    runner._launch_operation = launched.append
    key = ReplicaKey("task-a", "native-0")

    record = runner.submit_operation(
        OperationCommand("op-restore", OperationKind.RESTORE, key, "l1")
    )

    assert record.status is OperationStatus.ACCEPTED
    assert launched == ["op-restore"]
    assert runner._operation_journal.query("op-restore") is not None


def test_taskrunner_restore_closes_gs_lease_after_service_commit():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    runner._launch_operation = lambda operation_id: None
    key = ReplicaKey("task-a", "native-0")
    calls = []

    class Rollouter:
        prepare_replica = RemoteMethod(
            lambda target, **kwargs: calls.append(
                ("prepare_replica", target, kwargs["operation_id"], kwargs["spec"])
            )
        )

    class Trainer:
        restore_and_publish = RemoteMethod(
            lambda operation: calls.append(
                ("restore_and_publish", operation.operation_id)
            )
            or OperationEvidence(
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                7,
            )
        )

    class GroupScheduler:
        advance_lease = RemoteMethod(
            lambda lease_id, evidence: calls.append(
                ("advance_lease", lease_id, evidence.type)
            )
            or {"restored": True}
        )

    runner.components = {"rollouter": Rollouter(), "trainer": Trainer()}
    runner.group_scheduler = GroupScheduler()
    runner.submit_operation(
        OperationCommand("op-restore", OperationKind.RESTORE, key, "l1")
    )
    runner._execute_operation("op-restore")

    record = runner.query_operation("op-restore")
    assert record.status is OperationStatus.SUCCEEDED
    assert record.result == EvidenceType.SERVICE_COMMITTED.value
    assert calls == [
        ("prepare_replica", key, "op-restore", None),
        ("restore_and_publish", "op-restore"),
        ("advance_lease", "l1", EvidenceType.SERVICE_COMMITTED),
    ]


def test_taskrunner_verified_restore_rollback_finishes_failed_and_releases_reservation():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    runner._launch_operation = lambda operation_id: None
    key = ReplicaKey("task-a", "native-0")
    calls = []

    class Rollouter:
        prepare_replica = RemoteMethod(
            lambda target, **kwargs: calls.append(
                ("prepare_replica", target, kwargs["operation_id"], kwargs["spec"])
            )
        )

    class Trainer:
        restore_and_publish = RemoteMethod(
            lambda operation: OperationEvidence(
                operation.operation_id,
                EvidenceType.RELEASED,
                8,
                ("u0",),
            )
        )

    class GroupScheduler:
        advance_lease = RemoteMethod(
            lambda lease_id, evidence: calls.append(
                ("advance_lease", lease_id, evidence.type, evidence.released_gpu_uuids)
            )
            or {"restore_rolled_back": True}
        )

    runner.components = {"rollouter": Rollouter(), "trainer": Trainer()}
    runner.group_scheduler = GroupScheduler()
    runner.submit_operation(
        OperationCommand("op-restore-rollback", OperationKind.RESTORE, key, "l1")
    )
    runner._execute_operation("op-restore-rollback")

    record = runner.query_operation("op-restore-rollback")
    assert record.status is OperationStatus.FAILED
    assert "verified re-slept" in record.result
    assert calls == [
        ("prepare_replica", key, "op-restore-rollback", None),
        ("advance_lease", "l1", EvidenceType.RELEASED, ("u0",)),
    ]


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

    slot = spec["claims"][0]
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


def test_taskrunner_launch_failure_cleans_temp_thread_and_lease_state():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    command = OperationCommand(
        "op-remove-launch-fail",
        OperationKind.REMOVE,
        ReplicaKey("task-a", "borrowed-0"),
        "l1",
    )

    def fail_launch(operation_id):
        runner._operation_threads[operation_id] = object()
        runner._operation_leases[operation_id] = taskrunner_lease()
        raise RuntimeError("thread start failed")

    runner._launch_operation = fail_launch
    with pytest.raises(RuntimeError, match="thread start failed"):
        runner.submit_operation(command)

    assert "op-remove-launch-fail" not in runner._operation_threads
    assert "op-remove-launch-fail" not in runner._operation_leases
    record = runner.query_operation("op-remove-launch-fail")
    assert record.status is OperationStatus.FAILED


def test_taskrunner_executes_native_donate_and_advances_lease():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    runner._launch_operation = lambda operation_id: None
    key = ReplicaKey("task-a", "native-0")
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
                ("advance_lease", lease_id, evidence.type, evidence.released_gpu_uuids)
            )
            or {"released": True}
        )

    runner.components = {"rollouter": Rollouter(), "trainer": Trainer()}
    runner.group_scheduler = GroupScheduler()
    runner.submit_operation(
        OperationCommand("op-donate", OperationKind.DONATE, key, "l1")
    )
    runner._execute_operation("op-donate")

    record = runner.query_operation("op-donate")
    assert record.status is OperationStatus.SUCCEEDED
    assert record.result == EvidenceType.RELEASED.value
    assert calls == [
        ("prepare_exit", key, "op-donate", False),
        ("remove_and_commit", "op-donate"),
        ("finalize_release", "op-donate"),
        ("advance_lease", "l1", EvidenceType.RELEASED, ("u0",)),
    ]


def test_taskrunner_force_remove_is_admitted_and_dispatched():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    launched = []
    runner._launch_operation = launched.append
    key = ReplicaKey("task-a", "borrowed-0")

    record = runner.submit_operation(
        OperationCommand(
            "op-force",
            OperationKind.REMOVE,
            key,
            "l1",
            force=True,
        )
    )

    assert record.status is OperationStatus.ACCEPTED
    assert launched == ["op-force"]


def test_trainer_add_bootstraps_current_vpub_commits_e_then_publishes_service():
    key = ReplicaKey("task-a", "borrowed-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.BORROWED})()
    calls = []

    class CE:
        def __init__(self):
            self.pending_bootstrap = {}
            self.effective_replicas = {}

        def register_pending(self, target, replicas, *, operation_id):
            calls.append(("register_pending", target, operation_id))
            self.pending_bootstrap[target] = (tuple(replicas), operation_id)

        async def bootstrap_target(self, target, *, operation_id, loaded_version):
            calls.append(("bootstrap_target", target, operation_id, loaded_version))
            return OperationEvidence(operation_id, EvidenceType.WEIGHT_READY, 1)

        def commit_pending(self, target, evidence, *, loaded_version):
            calls.append(("commit_pending", target, loaded_version))
            replicas, _ = self.pending_bootstrap.pop(target)
            self.effective_replicas[target] = (replicas, loaded_version)

        def discard_pending(self, target):
            calls.append(("discard_pending", target))
            self.pending_bootstrap.pop(target, None)

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        get_pending_replicas = AsyncRemoteMethod(lambda operation_id: (runtime,))
        commit_service_change = AsyncRemoteMethod(
            lambda operation: calls.append(("commit_service", operation.operation_id))
            or OperationEvidence(
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                2,
            )
        )
        finalize_release = AsyncRemoteMethod(
            lambda operation: (_ for _ in ()).throw(
                AssertionError("successful ADD must not rollback runtime")
            )
        )

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    trainer.current_param_version = 9

    evidence = asyncio.run(
        trainer.bootstrap_and_publish(
            OperationRecord("op-add", OperationStatus.RUNNING)
        )
    )

    assert evidence.type is EvidenceType.SERVICE_COMMITTED
    assert trainer.checkpoint_manager.effective_replicas[key][1] == 9
    assert calls == [
        ("register_pending", key, "op-add"),
        ("bootstrap_target", key, "op-add", 9),
        ("commit_pending", key, 9),
        ("commit_service", "op-add"),
    ]


def test_trainer_add_definite_route_rejection_removes_effective_e_and_returns_release():
    key = ReplicaKey("task-a", "borrowed-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.BORROWED})()
    calls = []

    class CE:
        def __init__(self):
            self.pending_bootstrap = {}
            self.effective_replicas = {}

        def register_pending(self, target, replicas, *, operation_id):
            self.pending_bootstrap[target] = (tuple(replicas), operation_id)

        async def bootstrap_target(self, target, *, operation_id, loaded_version):
            return OperationEvidence(operation_id, EvidenceType.WEIGHT_READY, 1)

        def commit_pending(self, target, evidence, *, loaded_version):
            replicas, _ = self.pending_bootstrap.pop(target)
            self.effective_replicas[target] = (replicas, loaded_version)
            calls.append(("commit_pending", target, loaded_version))

        def remove_effective(self, target):
            calls.append(("remove_effective", target))
            self.effective_replicas.pop(target)

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        get_pending_replicas = AsyncRemoteMethod(lambda operation_id: (runtime,))
        commit_service_change = AsyncRemoteMethod(lambda operation: None)
        finalize_release = AsyncRemoteMethod(
            lambda operation: calls.append(("finalize_release", operation.operation_id))
            or OperationEvidence(
                operation.operation_id,
                EvidenceType.RELEASED,
                2,
                ("u0",),
            )
        )

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    trainer.current_param_version = 9

    evidence = asyncio.run(
        trainer.bootstrap_and_publish(
            OperationRecord("op-add", OperationStatus.RUNNING)
        )
    )

    assert evidence.type is EvidenceType.RELEASED
    assert evidence.released_gpu_uuids == ("u0",)
    assert trainer.checkpoint_manager.effective_replicas == {}
    assert trainer.replica_sync_gate.health == "HEALTHY"
    assert calls == [
        ("commit_pending", key, 9),
        ("remove_effective", key),
        ("finalize_release", "op-add"),
    ]


def test_trainer_add_bootstrap_failure_discards_pending_and_destroys_hidden_runtime():
    key = ReplicaKey("task-a", "borrowed-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.BORROWED})()
    calls = []

    class CE:
        def __init__(self):
            self.pending_bootstrap = {}

        def register_pending(self, target, replicas, *, operation_id):
            calls.append(("register_pending", target, operation_id))
            self.pending_bootstrap[target] = (tuple(replicas), operation_id)

        async def bootstrap_target(self, target, *, operation_id, loaded_version):
            calls.append(("bootstrap_target", target, loaded_version))
            raise RuntimeError("weight transfer failed")

        def discard_pending(self, target):
            calls.append(("discard_pending", target))
            self.pending_bootstrap.pop(target, None)

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        get_pending_replicas = AsyncRemoteMethod(lambda operation_id: (runtime,))
        finalize_release = AsyncRemoteMethod(
            lambda operation: calls.append(("finalize_release", operation.operation_id))
            or OperationEvidence(
                operation.operation_id,
                EvidenceType.RELEASED,
                2,
                ("u0",),
            )
        )

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    trainer.current_param_version = 9

    evidence = asyncio.run(
        trainer.bootstrap_and_publish(
            OperationRecord("op-add", OperationStatus.RUNNING)
        )
    )

    assert evidence.type is EvidenceType.RELEASED
    assert evidence.released_gpu_uuids == ("u0",)
    assert trainer.replica_sync_gate.health == "HEALTHY"
    assert trainer.checkpoint_manager.pending_bootstrap == {}
    assert calls == [
        ("register_pending", key, "op-add"),
        ("bootstrap_target", key, 9),
        ("discard_pending", key),
        ("finalize_release", "op-add"),
    ]


def test_trainer_blocked_add_replay_reconciles_committed_route_and_clears_g():
    key = ReplicaKey("task-a", "borrowed-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.BORROWED})()
    committed = OperationEvidence(
        "op-add",
        EvidenceType.SERVICE_COMMITTED,
        9,
    )

    class CE:
        effective_replicas = {key: ((runtime,), 7)}

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        commit_service_change = AsyncRemoteMethod(lambda operation: committed)

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    trainer.current_param_version = 7

    async def scenario():
        lease = await trainer.replica_sync_gate.acquire("op-add", GateKind.ADD)
        trainer.replica_sync_gate.block(lease.owner, "R outcome unknown")
        await lease.release()
        return await trainer.bootstrap_and_publish(
            OperationRecord("op-add", OperationStatus.RUNNING)
        )

    evidence = asyncio.run(scenario())

    assert evidence == committed
    assert trainer.replica_sync_gate.health == "HEALTHY"


def test_trainer_blocked_exit_replay_reconciles_service_commit_and_clears_g():
    key = ReplicaKey("task-a", "native-0")
    committed = OperationEvidence(
        "op-donate",
        EvidenceType.SERVICE_COMMITTED,
        12,
    )

    class CE:
        effective_replicas = {}

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        commit_service_change = AsyncRemoteMethod(lambda operation: committed)

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()

    async def scenario():
        lease = await trainer.replica_sync_gate.acquire(
            "op-donate",
            GateKind.REMOVE,
        )
        trainer.replica_sync_gate.block(lease.owner, "R/C outcome unknown")
        await lease.release()
        return await trainer.reconcile_exit(
            OperationRecord("op-donate", OperationStatus.RUNNING)
        )

    evidence = asyncio.run(scenario())

    assert evidence == committed
    assert trainer.replica_sync_gate.health == "HEALTHY"


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
        ("commit_pending", key, 11),
        ("commit_service", "op-restore"),
    ]


def test_trainer_restore_bootstrap_release_returns_compensation_without_blocking_g():
    key = ReplicaKey("task-a", "native-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.NATIVE})()
    calls = []

    class CE:
        def __init__(self):
            self.pending_bootstrap = {key: ((runtime,), "op-donate")}

        def discard_pending(self, target):
            self.pending_bootstrap.pop(target, None)

        def register_pending(self, target, replicas, *, operation_id):
            self.pending_bootstrap[target] = (tuple(replicas), operation_id)

        async def bootstrap_target(self, target, *, operation_id, loaded_version):
            calls.append(("bootstrap_target", loaded_version))
            return OperationEvidence(
                operation_id,
                EvidenceType.RELEASED,
                2,
                ("u0",),
            )

        def commit_pending(self, *args, **kwargs):
            raise AssertionError("compensated RESTORE must not enter E")

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        commit_service_change = AsyncRemoteMethod(
            lambda operation: (_ for _ in ()).throw(
                AssertionError("compensated RESTORE must not publish service")
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
    assert evidence.type is EvidenceType.RELEASED
    assert evidence.released_gpu_uuids == ("u0",)
    assert trainer.replica_sync_gate.health == "HEALTHY"
    assert calls == [("bootstrap_target", 11)]


def test_trainer_restore_definite_no_route_removes_e_then_resleeps():
    key = ReplicaKey("task-a", "native-0")
    runtime = type("Runtime", (), {"replica_kind": ReplicaKind.NATIVE})()
    calls = []

    class CE:
        def __init__(self):
            self.pending_bootstrap = {key: ((runtime,), "op-donate")}
            self.effective_replicas = {}

        def discard_pending(self, target):
            self.pending_bootstrap.pop(target, None)

        def register_pending(self, target, replicas, *, operation_id):
            calls.append(("register_pending", target, operation_id))
            self.pending_bootstrap[target] = (tuple(replicas), operation_id)

        async def bootstrap_target(self, target, *, operation_id, loaded_version):
            return OperationEvidence(operation_id, EvidenceType.WEIGHT_READY, 2)

        def commit_pending(self, target, evidence, *, loaded_version):
            self.pending_bootstrap.pop(target, None)
            self.effective_replicas[target] = ((runtime,), loaded_version)

        def remove_effective(self, target):
            calls.append(("remove_effective", target))
            self.effective_replicas.pop(target, None)
            self.pending_bootstrap.pop(target, None)

    class Rollouter:
        get_pending_target = AsyncRemoteMethod(lambda operation_id: key)
        commit_service_change = AsyncRemoteMethod(lambda operation: None)
        finalize_release = AsyncRemoteMethod(
            lambda operation: calls.append(("finalize_release", operation.operation_id))
            or OperationEvidence(
                operation.operation_id,
                EvidenceType.RELEASED,
                4,
                ("u0",),
            )
        )

    trainer = trainer_class()()
    trainer.rollouter = Rollouter()
    trainer.checkpoint_manager = CE()
    trainer.current_param_version = 13

    evidence = asyncio.run(
        trainer.restore_and_publish(
            OperationRecord("op-restore", OperationStatus.RUNNING)
        )
    )

    assert evidence.type is EvidenceType.RELEASED
    assert trainer.replica_sync_gate.health == "HEALTHY"
    assert key not in trainer.checkpoint_manager.effective_replicas
    assert key in trainer.checkpoint_manager.pending_bootstrap
    assert calls == [
        ("remove_effective", key),
        ("register_pending", key, "op-restore"),
        ("finalize_release", "op-restore"),
    ]


def test_trainer_restore_service_failure_keeps_e_effective_and_blocks_g():
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

        def commit_pending(self, target, evidence, *, loaded_version):
            calls.append(("commit_pending", target, loaded_version))
            self.pending_bootstrap.pop(target, None)
            self.effective_replicas[target] = ((runtime,), loaded_version)

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

    assert key not in trainer.checkpoint_manager.pending_bootstrap
    assert trainer.checkpoint_manager.effective_replicas[key][1] == 13
    assert ("commit_pending", key, 13) in calls
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
    manager.register_replica(key, ReplicaKind.NATIVE, state=ReplicaState.ACTIVE, runtime=runtime)
    with pytest.raises(ValueError, match="another runtime"):
        manager.register_replica(
            key,
            ReplicaKind.NATIVE,
            state=ReplicaState.ACTIVE,
            runtime=object(),
        )
    manager.rollout_replicas.append(runtime)
    manager.server_addresses.append("s0")
    manager.server_handles.append("h0")

    manager.transition_replica(key, ReplicaState.DRAINING)
    assert manager.deactivate_service(key) is runtime
    assert manager.inspect_runtime(key) is runtime
    assert manager.rollout_replicas == []
    assert manager.server_addresses == []
    assert manager.server_handles == []

    manager.replica_state[key] = ReplicaState.ACTIVE
    manager.rollout_replicas = []
    manager.server_addresses = ["s0"]
    manager.server_handles = ["wrong-handle"]
    with pytest.raises(RuntimeError, match="address/handle inventory"):
        manager.activate_service(key)
    manager.server_addresses = []
    manager.server_handles = []

    manager.task_session = "task-a"
    valid_spec = {
        "operation_id": "op-add",
        "lease_id": "borrower-lease",
        "borrower_task_id": "task-a",
        "borrower_replica_id": "borrowed-0",
        "claims": [
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
    assert "max_colocate_count" not in normalized
    assert normalized["claims"][0]["rank"] == 0

    wrong_task = dict(valid_spec, borrower_task_id="task-b")
    with pytest.raises(ValueError, match="another task_session"):
        manager.validate_borrowed_spec(wrong_task)

    manager.rollout_config.tensor_model_parallel_size = 2
    with pytest.raises(ValueError, match="TP=DP=PP=1"):
        manager.validate_borrowed_spec(valid_spec)
    manager.rollout_config.tensor_model_parallel_size = 1

    wrong_share = dict(valid_spec)
    wrong_share["claims"] = [
        dict(valid_spec["claims"][0], gpu_fraction=1.0)
    ]
    with pytest.raises(ValueError, match="Ray GPU accounting share"):
        manager.validate_borrowed_spec(wrong_share)

    expired = dict(valid_spec, expires_at=time.time() - 1)
    with pytest.raises(ValueError, match="expired"):
        manager.validate_borrowed_spec(expired)

    wrong_namespace = dict(valid_spec)
    wrong_namespace["claims"] = [
        dict(valid_spec["claims"][0], pg_namespace="other")
    ]
    with pytest.raises(ValueError, match="another Ray namespace"):
        manager._resolve_placement_groups(
            manager.validate_borrowed_spec(wrong_namespace)["claims"]
        )

    # Failure before a runtime object exists is a known zero-side-effect
    # landing: the Manager records RELEASED rather than quarantining the key.
    original_resolver = manager._resolve_placement_groups
    manager._resolve_placement_groups = lambda claims: (_ for _ in ()).throw(
        RuntimeError("placement lookup failed before runtime creation")
    )
    pre_runtime_failure = dict(
        valid_spec,
        operation_id="op-add-pre-runtime-failure",
        lease_id="borrower-lease-pre-runtime-failure",
        borrower_replica_id="borrowed-pre-runtime-failure",
        replica_rank=None,
    )
    with pytest.raises(RuntimeError, match="before runtime creation"):
        asyncio.run(manager.create_borrowed_replica(pre_runtime_failure))
    pre_runtime_key = ReplicaKey("task-a", "borrowed-pre-runtime-failure", 0)
    assert manager.replica_state[pre_runtime_key] is ReplicaState.RELEASED
    manager._resolve_placement_groups = original_resolver

    wrong_node = dict(valid_spec)
    wrong_node["claims"] = [
        dict(valid_spec["claims"][0], node_id="n9")
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
    assert record["resolved_spec"]["replica_rank"] == 0
    borrowed_key = ReplicaKey("task-a", "borrowed-0", 0)
    assert ReplicaKey(
        record["resolved_spec"]["borrower_task_id"],
        record["resolved_spec"]["borrower_replica_id"],
        record["resolved_spec"]["placement_epoch"],
    ) == borrowed_key
    assert manager.replica_kind[borrowed_key] is ReplicaKind.BORROWED
    assert manager.replica_state[borrowed_key] is ReplicaState.CREATING
    borrowed_runtime = manager.inspect_runtime(borrowed_key)
    assert borrowed_runtime is not None
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
    assert borrowed_runtime.cleaned is True
    assert asyncio.run(
        manager.destroy(borrowed_key, operation_id="op-remove")
    ) == release
    manager.transition_replica(borrowed_key, ReplicaState.RELEASED)
    assert asyncio.run(
        manager.destroy(borrowed_key, operation_id="op-remove")
    ) == release

    duplicate_identity = dict(
        valid_spec,
        operation_id="op-add-duplicate-key",
        lease_id="borrower-lease-duplicate-key",
    )
    next_rank_before = manager.next_replica_rank
    with pytest.raises(ValueError, match="new runtime_epoch"):
        asyncio.run(manager.create_borrowed_replica(duplicate_identity))
    assert manager.next_replica_rank == next_rank_before

    wrong_rank = dict(valid_spec, replica_rank=7)
    with pytest.raises(ValueError, match="conflicting borrowed create replay"):
        asyncio.run(manager.create_borrowed_replica(wrong_rank))

    conflicting = dict(valid_spec, operation_id="op-other")
    with pytest.raises(ValueError, match="conflicting borrowed create replay"):
        asyncio.run(manager.create_borrowed_replica(conflicting))
    assert manager.next_replica_rank == 1

    class InvalidReceiptRuntime(FakeBorrowedRuntime):
        async def init_from_lease(self, spec, pg_by_id):
            return {
                "state": "BROKEN",
                "replica_rank": self.replica_rank,
            }

    manager.rollout_replica_class = InvalidReceiptRuntime
    invalid_receipt = dict(
        valid_spec,
        operation_id="op-add-invalid-receipt",
        lease_id="borrower-lease-invalid-receipt",
        borrower_replica_id="borrowed-invalid-receipt",
        replica_rank=None,
    )
    with pytest.raises(RuntimeError, match="RUNTIME_READY"):
        asyncio.run(manager.create_borrowed_replica(invalid_receipt))
    invalid_key = ReplicaKey("task-a", "borrowed-invalid-receipt", 0)
    invalid_record = manager.borrowed_operations["borrower-lease-invalid-receipt"]
    assert manager.replica_state[invalid_key] is ReplicaState.RELEASED
    assert invalid_record["error"] is not None

    class CleanedFailureRuntime(FakeBorrowedRuntime):
        async def init_from_lease(self, spec, pg_by_id):
            self.borrowed_cleanup_verified = True
            raise RuntimeError("borrowed init failed after verified cleanup")

    manager.rollout_replica_class = CleanedFailureRuntime
    cleaned_failure = dict(
        valid_spec,
        operation_id="op-add-cleaned-failure",
        lease_id="borrower-lease-cleaned-failure",
        borrower_replica_id="borrowed-cleaned-failure",
        replica_rank=None,
    )
    with pytest.raises(RuntimeError, match="verified cleanup"):
        asyncio.run(manager.create_borrowed_replica(cleaned_failure))
    cleaned_key = ReplicaKey("task-a", "borrowed-cleaned-failure", 0)
    cleaned_record = manager.borrowed_operations["borrower-lease-cleaned-failure"]
    assert manager.replica_state[cleaned_key] is ReplicaState.RELEASED
    assert cleaned_record["error"] is not None
    with pytest.raises(RuntimeError, match="verified cleanup"):
        asyncio.run(manager.create_borrowed_replica(cleaned_failure))

    class UnverifiedFailureRuntime(FakeBorrowedRuntime):
        async def init_from_lease(self, spec, pg_by_id):
            raise RuntimeError("borrowed init failed before cleanup")

        async def cleanup_borrowed_runtime(self):
            self.borrowed_cleanup_verified = False
            raise RuntimeError("cleanup failed")

    manager.rollout_replica_class = UnverifiedFailureRuntime
    unverified_failure = dict(
        valid_spec,
        operation_id="op-add-unverified-failure",
        lease_id="borrower-lease-unverified-failure",
        borrower_replica_id="borrowed-unverified-failure",
        replica_rank=None,
    )
    with pytest.raises(RuntimeError, match="before cleanup"):
        asyncio.run(manager.create_borrowed_replica(unverified_failure))
    quarantined_key = ReplicaKey("task-a", "borrowed-unverified-failure", 0)
    quarantined_record = manager.borrowed_operations["borrower-lease-unverified-failure"]
    assert manager.replica_state[quarantined_key] is ReplicaState.QUARANTINED
    assert quarantined_record["error"] is not None


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
        assert server.engine.healthy is True

        engine = server.engine
        assert await server.shutdown_runtime() is None
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

        def _get_server_name_prefix(self):
            return "vllm_"

    fake_ray = type("ReplicaRay", (), {"remote": staticmethod(lambda cls: cls)})
    return isolated(
        "rollout/replica.py",
        "MultiTaskvLLMReplica",
        Parent,
        ReplicaKind=ReplicaKind,
        Lease=Lease,
        FIRST_RELEASE_MAX_COLOCATE_COUNT=FIRST_RELEASE_MAX_COLOCATE_COUNT,
        RayClassWithInitArgs=object,
        RayWorkerGroup=object,
        ResourcePoolManager=object,
        RolloutMode=object,
        get_device_name=lambda: "cuda",
        get_resource_name=lambda: "GPU",
        get_visible_devices_keyword=lambda: "CUDA_VISIBLE_DEVICES",
        MultiTaskvLLMHttpServer=object,
        os=__import__("os"),
        subprocess=__import__("subprocess"),
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
                "source_lease_id": "source-lease-0",
                "donor_task_id": "donor-task",
                "donor_replica_rank": 0,
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
    assert first["actor_name"].startswith("borrowed_ce_7_")
    assert first["pg_id"] == "pg"
    assert first["bundle_index"] == 4
    assert first["num_gpus"] == FIRST_RELEASE_RAY_GPU_FRACTION
    assert first["env_vars"] == {
        "WORLD_SIZE": "1",
        "RANK": "0",
        "RAY_LOCAL_WORLD_SIZE": "1",
        "WG_PREFIX": first["env_vars"]["WG_PREFIX"],
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
                "source_lease_id": "source-lease-0",
                "donor_task_id": "donor-task",
                "donor_replica_rank": 0,
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
    with pytest.raises(ValueError, match="rank=node_rank=local_rank=0"):
        replica.build_borrowed_worker_plan(donor_layout)


def test_worker_gpu_uuid_probe_uses_native_worker_ray_call_context():
    accelerator = {"value": "1"}

    class Context:
        def get_accelerator_ids(self):
            return {"GPU": [accelerator["value"]]}

        def get_node_id(self):
            return "node-a"

    fake_ray = type(
        "ProbeRay",
        (),
        {"get_runtime_context": staticmethod(lambda: Context())},
    )
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
        "rollout/replica.py",
        "MultiTaskvLLMReplica",
        object,
        ReplicaKind=ReplicaKind,
        Lease=Lease,
        FIRST_RELEASE_MAX_COLOCATE_COUNT=FIRST_RELEASE_MAX_COLOCATE_COUNT,
        MultiTaskvLLMHttpServer=object,
        os=__import__("os"),
        subprocess=fake_subprocess,
        ray=fake_ray,
        get_resource_name=lambda: "GPU",
        get_visible_devices_keyword=lambda: "CUDA_VISIBLE_DEVICES",
    )

    assert cls._runtime_placement_probe(None)["gpu_uuid"] == "GPU-b"
    accelerator["value"] = "GPU-direct"
    assert cls._runtime_placement_probe(None)["gpu_uuid"] == "GPU-direct"
    accelerator["value"] = "MIG-abc"
    with pytest.raises(NotImplementedError, match="MIG"):
        cls._runtime_placement_probe(None)
    accelerator["value"] = "opaque-id"
    with pytest.raises(RuntimeError, match="cannot map"):
        cls._runtime_placement_probe(None)


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

        def _get_server_name_prefix(self):
            return "vllm_"

        def get_ray_class_with_init_args(self):
            return FakeCIA(
                object,
                rollout_config=self.config,
                model_config=self.model_config,
                replica_rank=self.replica_rank,
            )

    created = []
    killed = []

    class FakeHandle:
        def __init__(self, record, *, gpu_uuid="GPU-x"):
            self.record = record
            self.__ray_call__ = AsyncRemoteMethod(
                lambda _probe: {
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
        Lease=Lease,
        FIRST_RELEASE_MAX_COLOCATE_COUNT=FIRST_RELEASE_MAX_COLOCATE_COUNT,
        RayClassWithInitArgs=FakeCIA,
        RayWorkerGroup=FakeWG,
        ResourcePoolManager=object,
        RolloutMode=object,
        PlacementGroupSchedulingStrategy=object,
        get_master_addr_port=object,
        get_device_name=lambda: "cuda",
        MultiTaskvLLMHttpServer=object,
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
                "source_lease_id": "source-lease-0",
                "donor_task_id": "donor-task",
                "donor_replica_rank": 0,
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

    assert asyncio.run(replica._create_workers_from_claims(spec, {"pg": "PG"})) is None

    assert replica.workers == created
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
    assert asyncio.run(replica.worker_placements())[0]["gpu_uuid"] == "GPU-x"
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

        def _get_server_name_prefix(self):
            return "vllm_"
            self._server_handle = None
            self._server_address = None

        async def launch_servers(self):
            health = {
                "node_id": "node",
                "replica_rank": self.replica_rank,
                "server_address": "127.0.0.1",
                "server_port": 8000,
                "global_steps": None,
            }
            server = type(
                "Server",
                (),
                {
                    "runtime_health": AsyncRemoteMethod(lambda: health),
                    "shutdown_runtime": AsyncRemoteMethod(lambda: None),
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
        Lease=Lease,
        FIRST_RELEASE_MAX_COLOCATE_COUNT=FIRST_RELEASE_MAX_COLOCATE_COUNT,
        RayClassWithInitArgs=object,
        RayWorkerGroup=object,
        ResourcePoolManager=object,
        RolloutMode=type("RolloutMode", (), {"STANDALONE": "standalone"}),
        PlacementGroupSchedulingStrategy=object,
        get_master_addr_port=object,
        get_device_name=lambda: "cuda",
        MultiTaskvLLMHttpServer=object,
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
            "__ray_call__": AsyncRemoteMethod(
                lambda _probe: {
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
                "source_lease_id": "source-lease-0",
                "donor_task_id": "donor-task",
                "donor_replica_rank": 0,
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
    assert receipt["health"]["server_port"] == 8000


def checkpoint_manager_class(**extra_scope):
    class Parent:
        def __init__(self, config=None, actor_wg=None, replicas=None):
            self.config = config
            self.backend = getattr(config, "backend", "nccl")
            self.actor_wg = actor_wg
            self.replicas = list(replicas or [])
            self.build_calls = []

        def add_replicas(self, replicas):
            self.replicas.extend(replicas)

        def remove_replicas(self, replicas):
            replicas = set(replicas)
            self.replicas = [r for r in self.replicas if r not in replicas]

        def build_process_group(self, rollout):
            self.build_calls.append(rollout)

    scope = {
        "ReplicaKey": ReplicaKey,
        "OperationEvidence": OperationEvidence,
        "EvidenceType": EvidenceType,
        "ReplicaKind": ReplicaKind,
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
    ce._bootstrap_ready_map[key] = (
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
        replica_kind = ReplicaKind.BORROWED
        workers = ["worker-0"]

        def get_ray_class_with_init_args(self):
            return object()

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
    assert calls.index(("release-kv",)) < calls.index(("target-update", 7))
    assert calls.index(("target-update", 7)) < calls.index(("resume-kv",))
    assert calls.index(("resume-kv",)) < calls.index(("health",))
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


def test_ce_native_restore_wakes_weights_under_bootstrap_and_restores_kv_before_health():
    calls = []

    class FakeRolloutWG:
        world_size = 1

        @classmethod
        def from_detached(cls, *, worker_handles, **kwargs):
            calls.append(("wrap", tuple(worker_handles)))
            return cls()

        def update_weights(self, *, global_steps):
            calls.append(("target-update", global_steps))
            return [None]

        def execute_checkpoint_engine(self, methods):
            calls.append(("target-finalize", tuple(methods)))
            return [None]

    class FakeCIA:
        def __init__(self, cls, *args, **kwargs):
            pass

    class ActorWG:
        world_size = 1

        def update_weights(self, *, global_steps, mode):
            calls.append(("actor-update", global_steps, mode))
            return [None]

        def execute_checkpoint_engine(self, methods):
            calls.append(("actor-finalize", tuple(methods)))
            return [None]

    class FakeRay:
        @staticmethod
        def remote(cls):
            return cls

        @staticmethod
        def get(value):
            return value

    class Replica:
        replica_kind = ReplicaKind.NATIVE
        workers = ["worker-0"]

        def get_ray_class_with_init_args(self):
            return object()

        async def wake_up(self, *, tags=None):
            calls.append(("wake", tuple(tags) if tags is not None else None))
            return ({"sleeping": True, "fully_awake": False},)

        async def release_kv_cache(self):
            calls.append(("release-kv",))

        async def resume_kv_cache(self):
            calls.append(("resume-kv",))

        async def validate_server_runtime(self):
            calls.append(("health",))
            return {"global_steps": 13}

        async def worker_placements(self):
            return ({"node_id": "n0", "gpu_uuid": "u0"},)

        async def sleep(self):
            raise AssertionError("successful RESTORE bootstrap must not rollback")

    config = type("Config", (), {"backend": "nccl"})()
    cls = checkpoint_manager_class(
        ray=FakeRay,
        RayClassWithInitArgs=FakeCIA,
        RayWorkerGroup=FakeRolloutWG,
            )
    ce = cls(config=config, actor_wg=ActorWG(), replicas=[])
    key = ReplicaKey("task-a", "native-0")
    replica = Replica()
    ce.register_pending(key, [replica], operation_id="op-restore")

    evidence = asyncio.run(
        ce.bootstrap_target(key, operation_id="op-restore", loaded_version=13)
    )
    assert evidence.type is EvidenceType.WEIGHT_READY
    assert calls.index(("wake", ("weights",))) < calls.index(("target-update", 13))
    assert calls.index(("target-update", 13)) < calls.index(("resume-kv",))
    assert calls.index(("resume-kv",)) < calls.index(("health",))


def test_ce_native_restore_failure_resleeps_before_escaping():
    calls = []

    class FakeRolloutWG:
        world_size = 1

        @classmethod
        def from_detached(cls, *, worker_handles, **kwargs):
            return cls()

        def update_weights(self, *, global_steps):
            calls.append(("target-update", global_steps))
            raise RuntimeError("target transfer failed")

        def execute_checkpoint_engine(self, methods):
            calls.append(("target-finalize", tuple(methods)))
            return [None]

    class FakeCIA:
        def __init__(self, cls, *args, **kwargs):
            pass

    class ActorWG:
        world_size = 1

        def update_weights(self, *, global_steps, mode):
            calls.append(("actor-update", global_steps, mode))
            return [None]

        def execute_checkpoint_engine(self, methods):
            calls.append(("actor-finalize", tuple(methods)))
            return [None]

    class FakeRay:
        @staticmethod
        def remote(cls):
            return cls

        @staticmethod
        def get(value):
            # Evaluate enough of the fake result list to surface transfer failure.
            return value

    class Replica:
        replica_kind = ReplicaKind.NATIVE
        workers = ["worker-0"]

        def get_ray_class_with_init_args(self):
            return object()

        async def wake_up(self, *, tags=None):
            calls.append(("wake", tuple(tags) if tags is not None else None))
            return ({"sleeping": True, "fully_awake": False},)

        async def release_kv_cache(self):
            calls.append(("release-kv",))

        async def resume_kv_cache(self):
            raise AssertionError("failed transfer must not resume KV")

        async def validate_server_runtime(self):
            raise AssertionError("failed transfer must not reach health")

        async def worker_placements(self):
            return ({"node_id": "n0", "gpu_uuid": "u0"},)

        async def sleep(self):
            calls.append(("sleep",))
            return ({"sleep_level": 2, "sleeping": True},)

    config = type("Config", (), {"backend": "nccl"})()
    cls = checkpoint_manager_class(
        ray=FakeRay,
        RayClassWithInitArgs=FakeCIA,
        RayWorkerGroup=FakeRolloutWG,
            )
    ce = cls(config=config, actor_wg=ActorWG(), replicas=[])
    key = ReplicaKey("task-a", "native-0")
    ce.register_pending(key, [Replica()], operation_id="op-restore")

    evidence = asyncio.run(
        ce.bootstrap_target(key, operation_id="op-restore", loaded_version=13)
    )
    assert evidence.type is EvidenceType.RELEASED
    assert evidence.released_gpu_uuids == ("u0",)
    assert ("wake", ("weights",)) in calls
    assert ("sleep",) in calls


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
        replica_kind = ReplicaKind.BORROWED
        workers = ["worker-0"]

        def get_ray_class_with_init_args(self):
            return object()

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
            )
    ce = cls(config=config, actor_wg=ActorWG(), replicas=[])
    key = ReplicaKey("task-a", "borrowed-0")
    ce.register_pending(key, [Replica()], operation_id="op-add")

    with pytest.raises(RuntimeError, match="published parameter version"):
        asyncio.run(
            ce.bootstrap_target(key, operation_id="op-add", loaded_version=7)
        )
    assert key not in ce._bootstrap_ready_map



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
    ce._bootstrap_ready_map[key] = ("op-add", 7, evidence)
    with pytest.raises(ValueError):
        ce.commit_pending(key, evidence, loaded_version=version)
    assert ce.effective_replicas == {}
    assert ce.replicas == []


def test_ce_remove_invalidates_bootstrap_result_for_reused_key():
    ce = checkpoint_manager_class()(replicas=[])
    key = ReplicaKey("task-a", "borrowed-0")
    evidence = OperationEvidence("op-add", EvidenceType.WEIGHT_READY, 2)
    ce.register_pending(key, ["old-runtime"], operation_id="op-add")
    ce._bootstrap_ready_map[key] = ("op-add", 7, evidence)
    ce.commit_pending(key, evidence, loaded_version=7)
    ce.remove_effective(key)
    ce.register_pending(key, ["new-runtime"], operation_id="op-next")
    assert key not in ce._bootstrap_ready_map
    assert ce.replicas == []


def test_ce_committed_replay_rejects_replaced_evidence():
    ce = checkpoint_manager_class()(replicas=[])
    key = ReplicaKey("task-a", "borrowed-0")
    evidence = OperationEvidence("op-add", EvidenceType.WEIGHT_READY, 2)
    ce.register_pending(key, ["runtime"], operation_id="op-add")
    ce._bootstrap_ready_map[key] = ("op-add", 7, evidence)
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
    assert rollouter.collect_idle_candidates() == ()


def http_server_class():
    class Parent:
        async def resume_kv_cache(self):
            await self.engine.wake_up(tags=["kv_cache"])
            await self.engine.reset_prefix_cache(reset_connector=True)

    return isolated(
        "rollout/http_server.py",
        "MultiTaskvLLMHttpServer",
        Parent,
        asyncio=asyncio,
        ray=FakeRay,
    )


def runtime_replica_class():
    return isolated(
        "rollout/replica.py",
        "MultiTaskvLLMReplica",
        object,
        asyncio=asyncio,
        ReplicaKind=ReplicaKind,
        Lease=Lease,
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

    with pytest.raises(NotImplementedError, match="level-2 sleep"):
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


def test_native_runtime_health_probe_confirms_server_and_ce_worker_node():
    health = {
        "node_id": "node-0",
        "replica_rank": 2,
        "server_address": "127.0.0.1",
        "server_port": 8000,
        "global_steps": 12,
    }

    class Server:
        runtime_health = AsyncRemoteMethod(lambda: dict(health))

    cls = runtime_replica_class()
    replica = cls.__new__(cls)
    replica.replica_kind = ReplicaKind.NATIVE
    replica.replica_rank = 2
    replica.nnodes = 1
    replica.servers = [Server()]
    replica.workers = [object()]

    async def worker_placements():
        return ({"node_id": "node-0", "gpu_uuid": "u0"},)

    replica.worker_placements = worker_placements
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

    cls = runtime_replica_class()
    replica = cls.__new__(cls)
    replica.replica_kind = ReplicaKind.NATIVE
    replica.servers = [Server()]

    assert asyncio.run(replica.sleep())[0]["sleep_level"] == 2
    assert asyncio.run(replica.wake_up())[0]["fully_awake"] is True


def test_manager_native_sleep_binds_released_evidence_to_runtime_gpu_uuids():
    key = ReplicaKey("task-a", "native-0")

    class Runtime:
        workers = [object()]

        async def worker_placements(self):
            return ({"node_id": "n0", "gpu_uuid": "u0"},)

        async def sleep(self):
            return (
                {"sleep_level": 2, "sleeping": True},
            )

    cls = native_manager_class()
    manager = cls.__new__(cls)
    manager.replica_kind = {key: ReplicaKind.NATIVE}
    manager.replica_state = {key: ReplicaState.DRAINING}
    manager._runtime_inventory = {key: Runtime()}
    manager._native_release_evidence = {}

    evidence = asyncio.run(manager.sleep(key, operation_id="op-donate"))
    assert evidence.type is EvidenceType.RELEASED
    assert evidence.operation_id == "op-donate"
    assert evidence.released_gpu_uuids == ("u0",)


def test_manager_release_evidence_query_reads_native_owner_ledger():
    key = ReplicaKey("task-a", "native-0")
    evidence = OperationEvidence(
        "op-donate",
        EvidenceType.RELEASED,
        3,
        ("u0",),
    )
    cls = native_manager_class()
    manager = cls.__new__(cls)
    manager.replica_kind = {key: ReplicaKind.NATIVE}
    manager._native_release_evidence = {"op-donate": (key, evidence)}

    assert manager.query_release_evidence(key, "op-donate") == evidence
    assert manager.query_release_evidence(key, "other") is None


def test_manager_native_release_replay_returns_identical_evidence():
    key = ReplicaKey("task-a", "native-0")
    sleep_calls = []

    class Runtime:
        workers = [object()]

        async def worker_placements(self):
            return ({"node_id": "n0", "gpu_uuid": "u0"},)

        async def sleep(self):
            sleep_calls.append("sleep")
            return ({"sleep_level": 2, "sleeping": True},)

    cls = native_manager_class()
    manager = cls.__new__(cls)
    manager.replica_kind = {key: ReplicaKind.NATIVE}
    manager.replica_state = {key: ReplicaState.DRAINING}
    manager._runtime_inventory = {key: Runtime()}
    manager._native_release_evidence = {}

    first = asyncio.run(manager.sleep(key, operation_id="op-donate"))
    manager.replica_state[key] = ReplicaState.DORMANT
    second = asyncio.run(manager.sleep(key, operation_id="op-donate"))

    assert second == first
    assert sleep_calls == ["sleep"]


def _isolated_group_scheduler_class():
    path = SOURCE / "scheduler/group_scheduler.py"
    tree = ast.parse(path.read_text())
    scheduler = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GroupScheduler"
    )
    scheduler.decorator_list = []
    module = ast.Module(body=[scheduler], type_ignores=[])
    env = {
        "ReplicaKey": ReplicaKey,
        "ReplicaKind": ReplicaKind,
        "ActorHandle": object,
        "Lease": Lease,
        "OperationCommand": OperationCommand,
        "OperationEvidence": OperationEvidence,
        "OperationKind": OperationKind,
        "OperationRecord": OperationRecord,
        "OperationStatus": OperationStatus,
        "RUNTIME_KIND": "test",
        "EvidenceType": EvidenceType,
        "_RELEASE_KINDS": {OperationKind.DONATE, OperationKind.REMOVE},
        "time": time,
        "ray": FakeRay,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env["GroupScheduler"]


def _scheduler_test_lease(lease_id="l1"):
    return Lease(
        lease_id,
        ({
            "claim_id": f"claim-{lease_id}",
            "source_lease_id": f"source-{lease_id}",
            "donor_task_id": "task-a",
            "donor_replica_rank": 0,
            "pg_id": "pg",
            "bundle_index": 0,
            "node_id": "n0",
            "gpu_uuid": "u0",
            "gpu_fraction": 0.5,
            "cpu_request": 1.0,
        },),
    )


def test_group_scheduler_stages_operation_intent_before_taskrunner_dispatch():
    cls = _isolated_group_scheduler_class()
    gs = cls()
    lease = _scheduler_test_lease()
    gs.open_lease(lease)
    command = OperationCommand(
        "op-donate-race",
        OperationKind.DONATE,
        ReplicaKey("task-a", "native-0"),
        lease.lease_id,
    )

    class Runner:
        submit_operation = RemoteMethod(
            lambda received, lease=None: (
                (_ for _ in ()).throw(AssertionError("GS intent was not staged"))
                if gs.operation_commands.get(received.operation_id) != received
                else OperationRecord(received.operation_id)
            )
        )

    gs.task_runners["task-a"] = Runner()
    record = gs.submit_operation(command)
    assert record.operation_id == command.operation_id
    assert gs.operation_commands[command.operation_id] == command


def test_group_scheduler_rolls_back_staged_add_state_when_taskrunner_rejects():
    cls = _isolated_group_scheduler_class()
    gs = cls()
    lease = _scheduler_test_lease()
    gs.open_lease(lease)
    gs.handoff_ready_leases.add(lease.lease_id)
    command = OperationCommand(
        "op-add-rejected",
        OperationKind.ADD,
        ReplicaKey("task-a", "borrowed-0"),
        lease.lease_id,
    )

    class Runner:
        submit_operation = RemoteMethod(
            lambda received, lease=None: (_ for _ in ()).throw(
                NotImplementedError("backend unavailable")
            )
        )
        query_operation = RemoteMethod(
            lambda operation_id: OperationRecord(
                operation_id,
                OperationStatus.UNKNOWN,
                None,
            )
        )

    gs.task_runners["task-a"] = Runner()
    with pytest.raises(NotImplementedError, match="backend unavailable"):
        gs.submit_operation(command)
    assert command.operation_id not in gs.operation_commands
    assert lease.lease_id not in gs.borrower_targets


def test_group_scheduler_rolls_back_restore_reservation_when_taskrunner_rejects():
    cls = _isolated_group_scheduler_class()
    gs = cls()
    lease = _scheduler_test_lease()
    gs.open_lease(lease)
    gs.active_gpu_owner.clear()
    gs.active_bundle_owner.clear()
    command = OperationCommand(
        "op-restore-rejected",
        OperationKind.RESTORE,
        ReplicaKey("task-a", "native-0"),
        lease.lease_id,
    )

    class Runner:
        submit_operation = RemoteMethod(
            lambda received, lease=None: (_ for _ in ()).throw(
                NotImplementedError("restore unavailable")
            )
        )
        query_operation = RemoteMethod(
            lambda operation_id: OperationRecord(
                operation_id,
                OperationStatus.UNKNOWN,
                None,
            )
        )

    gs.task_runners["task-a"] = Runner()
    with pytest.raises(NotImplementedError, match="restore unavailable"):
        gs.submit_operation(command)
    assert command.operation_id not in gs.operation_commands
    assert lease.lease_id not in gs.active_gpu_owner.values()
    assert lease.lease_id not in gs.active_bundle_owner.values()


def test_group_scheduler_preserves_staging_when_submission_outcome_is_ambiguous():
    cls = _isolated_group_scheduler_class()
    gs = cls()
    lease = _scheduler_test_lease()
    gs.open_lease(lease)
    command = OperationCommand(
        "op-donate-timeout",
        OperationKind.DONATE,
        ReplicaKey("task-a", "native-0"),
        lease.lease_id,
    )

    class Runner:
        submit_operation = RemoteMethod(
            lambda received, lease=None: (_ for _ in ()).throw(
                TimeoutError("reply lost")
            )
        )
        query_operation = RemoteMethod(
            lambda operation_id: OperationRecord(
                operation_id,
                OperationStatus.RUNNING,
            )
        )

    gs.task_runners["task-a"] = Runner()
    with pytest.raises(TimeoutError, match="reply lost"):
        gs.submit_operation(command)
    assert gs.operation_commands[command.operation_id] == command


def test_group_scheduler_rejects_second_add_with_new_operation_id():
    cls = _isolated_group_scheduler_class()
    gs = cls()
    lease = _scheduler_test_lease()
    gs.open_lease(lease)
    gs.handoff_ready_leases.add(lease.lease_id)

    class Runner:
        submit_operation = RemoteMethod(
            lambda command, lease=None: OperationRecord(command.operation_id)
        )

    gs.task_runners["task-a"] = Runner()
    target = ReplicaKey("task-a", "borrowed-0")
    first = OperationCommand("op-add-1", OperationKind.ADD, target, lease.lease_id)
    assert gs.submit_operation(first).operation_id == "op-add-1"

    with pytest.raises(ValueError, match="original operation_id"):
        gs.submit_operation(
            OperationCommand("op-add-2", OperationKind.ADD, target, lease.lease_id)
        )


def test_group_scheduler_add_release_compensation_unfreezes_claims():
    cls = _isolated_group_scheduler_class()
    gs = cls()
    lease = _scheduler_test_lease("l-add-fail")
    gs.open_lease(lease)
    gs.handoff_ready_leases.add(lease.lease_id)
    target = ReplicaKey("task-b", "borrowed-0")
    command = OperationCommand(
        "op-add-fail",
        OperationKind.ADD,
        target,
        lease.lease_id,
    )
    gs.operation_commands[command.operation_id] = command
    gs.borrower_targets[lease.lease_id] = target
    evidence = OperationEvidence(
        command.operation_id,
        EvidenceType.RELEASED,
        5,
        ("u0",),
    )

    result = gs.advance_lease(lease.lease_id, evidence)

    assert result["add_rolled_back"] is True
    assert lease.lease_id not in gs.handoff_ready_leases
    assert lease.lease_id not in gs.borrower_targets
    assert "u0" not in gs.active_gpu_owner
    assert ("pg", 0) not in gs.active_bundle_owner


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

def test_group_scheduler_restore_requires_original_donor_and_returned_claims():
    path = SOURCE / "scheduler/group_scheduler.py"
    tree = ast.parse(path.read_text())
    scheduler = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GroupScheduler"
    )
    scheduler.decorator_list = []
    module = ast.Module(body=[scheduler], type_ignores=[])
    env = {
        "ReplicaKey": ReplicaKey,
        "ReplicaKind": ReplicaKind,
        "ActorHandle": object,
        "Lease": Lease,
        "OperationCommand": OperationCommand,
        "OperationEvidence": OperationEvidence,
        "OperationKind": OperationKind,
        "OperationRecord": OperationRecord,
        "RUNTIME_KIND": "test",
        "EvidenceType": EvidenceType,
        "_RELEASE_KINDS": {OperationKind.DONATE, OperationKind.REMOVE},
        "time": time,
        "ray": FakeRay,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    cls = env["GroupScheduler"]
    gs = cls()
    lease = Lease(
        "l1",
        ({
            "claim_id": "claim-1",
            "source_lease_id": "source-1",
            "donor_task_id": "task-a",
            "donor_replica_rank": 0,
            "pg_id": "pg",
            "bundle_index": 0,
            "node_id": "n0",
            "gpu_uuid": "u0",
            "gpu_fraction": 0.5,
            "cpu_request": 1.0,
        },),
    )
    gs.open_lease(lease)

    class Runner:
        submit_operation = RemoteMethod(
            lambda command, lease=None: OperationRecord(command.operation_id)
        )

    gs.task_runners["task-a"] = Runner()

    with pytest.raises(ValueError, match="does not match the lease donor replica"):
        gs.submit_operation(
            OperationCommand(
                "op-wrong-replica",
                OperationKind.RESTORE,
                ReplicaKey("task-a", "native-1"),
                "l1",
            )
        )

    with pytest.raises(ValueError, match="does not own the lease claims"):
        gs.submit_operation(
            OperationCommand(
                "op-wrong-task",
                OperationKind.RESTORE,
                ReplicaKey("task-b", "native-0"),
                "l1",
            )
        )

    with pytest.raises(ValueError, match="claims to be fully returned"):
        gs.submit_operation(
            OperationCommand(
                "op-not-returned",
                OperationKind.RESTORE,
                ReplicaKey("task-a", "native-0"),
                "l1",
            )
        )

    # Simulate the already-verified borrowed REMOVE release boundary.
    gs.active_gpu_owner.pop("u0")
    gs.active_bundle_owner.pop(("pg", 0))

    # A different lease may claim the released physical slot before RESTORE.
    # The original donor must not wake onto somebody else's active ownership.
    gs.active_gpu_owner["u0"] = "l2"
    gs.active_bundle_owner[("pg", 0)] = "l2"
    with pytest.raises(ValueError, match="fully returned and unclaimed"):
        gs.submit_operation(
            OperationCommand(
                "op-reallocated",
                OperationKind.RESTORE,
                ReplicaKey("task-a", "native-0"),
                "l1",
            )
        )
    gs.active_gpu_owner.pop("u0")
    gs.active_bundle_owner.pop(("pg", 0))

    record = gs.submit_operation(
        OperationCommand(
            "op-restore",
            OperationKind.RESTORE,
            ReplicaKey("task-a", "native-0"),
            "l1",
        )
    )
    assert record.operation_id == "op-restore"
    assert gs.operation_commands["op-restore"].target == ReplicaKey(
        "task-a", "native-0"
    )
    assert gs.active_gpu_owner["u0"] == "l1"
    assert gs.active_bundle_owner[("pg", 0)] == "l1"

    restore_evidence = OperationEvidence(
        "op-restore",
        EvidenceType.SERVICE_COMMITTED,
        9,
    )
    result = gs.advance_lease("l1", restore_evidence)
    assert result == {
        "lease_id": "l1",
        "operation_id": "op-restore",
        "restored": True,
    }
    assert gs.advance_lease("l1", restore_evidence) == result
    assert "u0" not in gs.active_gpu_owner
    assert ("pg", 0) not in gs.active_bundle_owner

    # A failed RESTORE that is proven re-slept releases only its temporary
    # reservation and remains retryable under the same lease.
    retry_lease = _scheduler_test_lease("l-restore-retry")
    gs.open_lease(retry_lease)
    gs.active_gpu_owner.pop("u0")
    gs.active_bundle_owner.pop(("pg", 0))
    retry_command = OperationCommand(
        "op-restore-fail",
        OperationKind.RESTORE,
        ReplicaKey("task-a", "native-0"),
        retry_lease.lease_id,
    )
    gs.operation_commands[retry_command.operation_id] = retry_command
    gs.active_gpu_owner["u0"] = retry_lease.lease_id
    gs.active_bundle_owner[("pg", 0)] = retry_lease.lease_id
    rollback = OperationEvidence(
        retry_command.operation_id,
        EvidenceType.RELEASED,
        10,
        ("u0",),
    )
    rollback_result = gs.advance_lease(retry_lease.lease_id, rollback)
    assert rollback_result["restore_rolled_back"] is True
    assert gs.advance_lease(retry_lease.lease_id, rollback) == rollback_result
    assert "u0" not in gs.active_gpu_owner
    assert ("pg", 0) not in gs.active_bundle_owner

    with pytest.raises(ValueError, match="lifecycle is complete"):
        gs.submit_operation(
            OperationCommand(
                "op-reuse-old-lease",
                OperationKind.DONATE,
                ReplicaKey("task-a", "native-0"),
                "l1",
            )
        )

    next_lease = Lease(
        "l2",
        ({
            "claim_id": "claim-2",
            "source_lease_id": "source-2",
            "donor_task_id": "task-a",
            "donor_replica_rank": 0,
            "pg_id": "pg",
            "bundle_index": 0,
            "node_id": "n0",
            "gpu_uuid": "u0",
            "gpu_fraction": 0.5,
            "cpu_request": 1.0,
        },),
    )
    assert gs.open_lease(next_lease) == next_lease


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
        AttemptState=AttemptState,
        OperationRecord=OperationRecord,
        OperationEvidence=OperationEvidence,
        EvidenceType=EvidenceType,
        _require_evidence=_test_require_evidence,
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


def test_rollouter_natural_drain_timeout_quarantines_instead_of_polling_forever():
    cls = rollouter_class()
    rollouter = cls(object(), object())
    key = ReplicaKey("task-a", "native-0")
    rollouter._natural_drain_timeout_s = 0.01

    class LB:
        begin_drain = AsyncRemoteMethod(lambda target, operation_id: "s0")
        has_unsettled_requests = AsyncRemoteMethod(lambda server_id: True)

    class Manager:
        global_load_balancer = LB()

        def __init__(self):
            self.replica_state = {key: ReplicaState.ACTIVE}
            self.replica_kind = {key: ReplicaKind.NATIVE}

        def replica_meta(self, target):
            return self.replica_kind[target], self.replica_state[target]

        def transition_replica(self, target, state):
            self.replica_state[target] = state

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

    assert manager.replica_state[key] is ReplicaState.QUARANTINED


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


def test_rollouter_natural_exit_quarantines_when_drain_start_remains_unknown():
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
    assert manager.replica_state[key] is ReplicaState.QUARANTINED
    assert calls == [
        ("state", ReplicaState.DRAINING),
        ("state", ReplicaState.QUARANTINED),
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
