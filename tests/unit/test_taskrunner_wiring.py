"""TASKRUNNER wiring regression scenarios; shared test fakes live in _wiring_support."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import threading
import time
import pytest
from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    Lease,
    OperationCommand,
    OperationEvidence,
    OperationKind,
    OperationStatus,
    ReplicaKey,
)
from _wiring_support import (
    RemoteMethod,
    taskrunner_class,
    taskrunner_lease,
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


def test_taskrunner_terminal_finish_clears_rollouter_binding_but_unknown_keeps_it():
    runner = taskrunner_class()()
    cleared = []

    class Rollouter:
        clear_operation_binding = RemoteMethod(
            lambda operation_id: cleared.append(operation_id) or True
        )

    runner.components["rollouter"] = Rollouter()

    for operation_id, status in (
        ("op-success", OperationStatus.SUCCEEDED),
        ("op-failed", OperationStatus.FAILED),
        ("op-unknown", OperationStatus.UNKNOWN),
    ):
        command = OperationCommand(
            operation_id,
            OperationKind.REMOVE,
            ReplicaKey("task-a", "borrowed-0"),
            "l1",
        )
        runner._operation_journal.begin(command)
        runner._operation_journal.mark_running(operation_id)
        runner._finish(operation_id, status, status.value)

    assert cleared == ["op-success", "op-failed"]


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


def test_taskrunner_exit_preflight_rejection_finishes_failed_without_unknown_fence():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    runner._launch_operation = lambda operation_id: None
    key = ReplicaKey("task-a", "native-0")
    command = OperationCommand("op-donate-preflight", OperationKind.DONATE, key, "l1")

    class Rollouter:
        prepare_exit = RemoteMethod(
            lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError("target is not ACTIVE")
            )
        )
        get_pending_target = RemoteMethod(
            lambda operation_id: (_ for _ in ()).throw(
                KeyError(operation_id)
            )
        )

    class Trainer:
        remove_and_commit = RemoteMethod(
            lambda operation: (_ for _ in ()).throw(
                AssertionError("preflight failure must not enter Trainer")
            )
        )

    runner.components = {"rollouter": Rollouter(), "trainer": Trainer()}
    runner._operation_journal.begin(command)
    runner._execute_operation(command.operation_id)

    record = runner.query_operation(command.operation_id)
    assert record.status is OperationStatus.FAILED
    assert "before owner mutation" in record.result


def test_taskrunner_exit_failure_after_pending_binding_remains_unknown():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    runner._launch_operation = lambda operation_id: None
    key = ReplicaKey("task-a", "native-0")
    command = OperationCommand("op-donate-mutated", OperationKind.DONATE, key, "l1")

    class Rollouter:
        prepare_exit = RemoteMethod(
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("drain outcome unknown")
            )
        )
        get_pending_target = RemoteMethod(lambda operation_id: key)

    class Trainer:
        remove_and_commit = RemoteMethod(lambda operation: None)

    runner.components = {"rollouter": Rollouter(), "trainer": Trainer()}
    runner._operation_journal.begin(command)
    runner._execute_operation(command.operation_id)

    record = runner.query_operation(command.operation_id)
    assert record.status is OperationStatus.UNKNOWN
    assert "drain outcome unknown" in record.result


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


def test_taskrunner_exit_binding_query_timeout_keeps_unknown():
    # A failed owner query is never proof that prepare_exit had no side effects.
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    runner._launch_operation = lambda operation_id: None
    key = ReplicaKey("task-a", "native-0")
    command = OperationCommand("op-donate-query-timeout", OperationKind.DONATE, key, "l1")

    class Rollouter:
        prepare_exit = RemoteMethod(
            lambda *args, **kwargs: (_ for _ in ()).throw(
                TimeoutError("drain reply lost")
            )
        )
        get_pending_target = RemoteMethod(
            lambda operation_id: (_ for _ in ()).throw(
                TimeoutError("owner query timed out")
            )
        )

    runner.components = {"rollouter": Rollouter(), "trainer": object()}
    runner._operation_journal.begin(command)
    runner._execute_operation(command.operation_id)
    record = runner.query_operation(command.operation_id)
    assert record.status is OperationStatus.UNKNOWN
    assert "inconclusive" in record.result


def test_taskrunner_exit_binding_wrapped_keyerror_proves_preflight_rejection():
    # Ray wraps remote KeyError in RayTaskError with the original cause.
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    runner._launch_operation = lambda operation_id: None
    key = ReplicaKey("task-a", "native-0")
    command = OperationCommand("op-donate-query-no-target", OperationKind.DONATE, key, "l1")

    class RemoteKeyError(RuntimeError):
        def __init__(self):
            super().__init__("RayTaskError")
            self.cause = KeyError("no pending lifecycle target")

    class Rollouter:
        prepare_exit = RemoteMethod(
            lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError("preflight refused invalid state")
            )
        )
        get_pending_target = RemoteMethod(
            lambda operation_id: (_ for _ in ()).throw(RemoteKeyError())
        )

    runner.components = {"rollouter": Rollouter(), "trainer": object()}
    runner._operation_journal.begin(command)
    runner._execute_operation(command.operation_id)
    record = runner.query_operation(command.operation_id)
    assert record.status is OperationStatus.FAILED
    assert "before owner mutation" in record.result


def test_taskrunner_concurrent_unknown_replay_launches_only_one_worker():
    runner = taskrunner_class()()
    runner.task_session = "task-a"
    runner._control_ready = True
    key = ReplicaKey("task-a", "native-0")
    command = OperationCommand("op-racing-unknown", OperationKind.DONATE, key, "l1")
    runner._operation_journal.begin(command)
    runner._operation_journal.mark_running(command.operation_id)
    runner._operation_journal.finish(
        command.operation_id, OperationStatus.UNKNOWN, "lost acknowledgement"
    )
    launch_started = threading.Event()
    release_launch = threading.Event()
    second_started = threading.Event()
    launched = []
    returned = []

    def stalled_launch(operation_id):
        launched.append(operation_id)
        launch_started.set()
        assert release_launch.wait(timeout=5)
        runner._operation_threads[operation_id] = threading.current_thread()

    runner._launch_operation = stalled_launch

    def submit(*, second=False):
        if second:
            second_started.set()
        returned.append(runner.submit_operation(command))

    first = threading.Thread(target=submit)
    second = threading.Thread(target=lambda: submit(second=True))
    first.start()
    assert launch_started.wait(timeout=5)
    second.start()
    assert second_started.wait(timeout=5)
    # Give the concurrent retry an opportunity to observe the UNKNOWN record.
    time.sleep(0.05)
    release_launch.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert launched == [command.operation_id]
    assert len(returned) == 2



# --- test_taskrunner_attach_ack.py (consolidated boundary scenarios) ---

_ATTACH_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/multi_task_scheduler/integration/verl/experimental_fully_async/task_runner.py"
)


class GetTimeoutError(Exception):
    pass


class Ref:
    def __init__(self, kind):
        self.kind = kind


class _AttachRemoteMethod:
    def __init__(self, kind, calls):
        self.kind = kind
        self.calls = calls

    def remote(self, *args):
        self.calls.append((self.kind, args))
        return Ref(self.kind)


class FakeGS:
    def __init__(self, calls):
        self.attach_task = _AttachRemoteMethod("attach", calls)
        self.get_task_runners = _AttachRemoteMethod("ledger", calls)
        self.detach_task = _AttachRemoteMethod("detach", calls)


class _AttachFakeRay:
    exceptions = SimpleNamespace(GetTimeoutError=GetTimeoutError)

    def __init__(self, plan, actor):
        self.plan = {key: list(values) for key, values in plan.items()}
        self.actor = actor
        self.get_calls = []

    def get_runtime_context(self):
        return SimpleNamespace(current_actor=self.actor)

    def get(self, ref, *, timeout):
        self.get_calls.append((ref.kind, timeout))
        answers = self.plan[ref.kind]
        result = answers.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def instantiate(plan):
    tree = ast.parse(_ATTACH_SOURCE.read_text(encoding="utf-8"))
    parent = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MultiTaskFullyAsyncTaskRunner"
    )
    method = next(
        node for node in parent.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_attach_task_with_reconciliation"
    )
    isolated = ast.Module(
        body=[ast.ClassDef(
            name="IsolatedRunner",
            bases=[],
            keywords=[],
            body=[method],
            decorator_list=[],
        )],
        type_ignores=[],
    )
    events = []
    actor = SimpleNamespace(_actor_id="real-actor-A")
    ray = _AttachFakeRay(plan, actor)
    logs = []

    logger = SimpleNamespace(
        warning=lambda *args: logs.append(("warning", args)),
        info=lambda *args: logs.append(("info", args)),
    )
    scope = {
        "ray": ray,
        "logger": logger,
        "CONTROL_RPC_TIMEOUT_S": 1.0,
    }
    exec(compile(ast.fix_missing_locations(isolated), str(_ATTACH_SOURCE), "exec"), scope)
    runner = scope["IsolatedRunner"]()
    runner.task_session = "real-actor-A"
    runner.group_scheduler = FakeGS(events)
    runner._attached_to_gs = False
    return runner, ray, events, logs


def test_attach_acks_immediately_with_one_rpc():
    runner, ray, events, _ = instantiate({"attach": [None]})
    runner._attach_task_with_reconciliation()
    assert runner._attached_to_gs is True
    assert [event[0] for event in events] == ["attach"]
    assert [kind for kind, _ in ray.get_calls] == ["attach"]


def test_attach_waits_on_same_ref_after_delayed_ack():
    runner, ray, events, _ = instantiate({
        "attach": [GetTimeoutError(), GetTimeoutError(), None]
    })
    runner._attach_task_with_reconciliation()
    assert runner._attached_to_gs is True
    assert [event[0] for event in events] == ["attach"]
    assert [kind for kind, _ in ray.get_calls] == ["attach"] * 3


def test_attach_lost_ack_can_be_verified_by_actor_identity():
    runner, _, events, _ = instantiate({
        "attach": [GetTimeoutError()] * 3,
        "ledger": [{"real-actor-A": SimpleNamespace(_actor_id="real-actor-A")}],
    })
    runner._attach_task_with_reconciliation()
    assert runner._attached_to_gs is True
    assert [event[0] for event in events] == ["attach", "ledger"]


def test_attach_unconfirmed_is_compensated_and_fails_closed():
    runner, _, events, _ = instantiate({
        "attach": [GetTimeoutError()] * 3,
        "ledger": [{}],
        "detach": [None],
    })
    with pytest.raises(RuntimeError, match="compensating detach acknowledged"):
        runner._attach_task_with_reconciliation()
    assert runner._attached_to_gs is False
    assert [event[0] for event in events] == ["attach", "ledger", "detach"]


def test_attach_ledger_unavailable_compensates_without_claiming_success():
    runner, _, events, _ = instantiate({
        "attach": [GetTimeoutError()] * 3,
        "ledger": [GetTimeoutError()],
        "detach": [GetTimeoutError()],
    })
    with pytest.raises(RuntimeError, match="compensating detach also unconfirmed"):
        runner._attach_task_with_reconciliation()
    assert runner._attached_to_gs is False
    assert [event[0] for event in events] == ["attach", "ledger", "detach"]


def test_attach_conflicting_actor_identity_never_detaches_other_task():
    runner, _, events, _ = instantiate({
        "attach": [GetTimeoutError()] * 3,
        "ledger": [{"real-actor-A": SimpleNamespace(_actor_id="OTHER")}],
    })
    with pytest.raises(RuntimeError, match="different ActorHandle"):
        runner._attach_task_with_reconciliation()
    assert runner._attached_to_gs is False
    assert [event[0] for event in events] == ["attach", "ledger"]


def test_attach_remote_rejection_is_not_mistaken_for_timeout():
    runner, _, events, _ = instantiate({"attach": [ValueError("duplicate session")]})
    with pytest.raises(ValueError, match="duplicate session"):
        runner._attach_task_with_reconciliation()
    assert runner._attached_to_gs is False
    assert [event[0] for event in events] == ["attach"]



# --- test_trainer_pg_isolation.py (consolidated boundary scenarios) ---

_TRAINER_PG_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/multi_task_scheduler/integration/verl/experimental_fully_async/task_runner.py"
)


def _build_trainer_stubs():
    tree = ast.parse(_TRAINER_PG_SOURCE.read_text(encoding="utf-8"))
    klass = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MultiTaskFullyAsyncTaskRunner"
    )
    method = next(
        item for item in klass.body
        if isinstance(item, ast.FunctionDef) and item.name == "_create_trainer"
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ClassDef(
                    name="IsolatedTaskRunner",
                    bases=[],
                    keywords=[],
                    body=[method],
                    decorator_list=[],
                )
            ],
            type_ignores=[],
        )
    )
    created = []
    roles = SimpleNamespace(Rollout="rollout")
    class DummyTrainer:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            self.init_workers = SimpleNamespace(remote=lambda: "workers-started")

        @classmethod
        def remote(cls, **kwargs):
            instance = cls(kwargs)
            created.append(instance)
            return instance

    def create_resource_pool_manager(config, *, roles):
        assert "rollout" not in roles
        return SimpleNamespace(
            resource_pool_spec={"trainer_pool": [2], "other_pool": [1]},
            mapping={"actor": "trainer_pool", "ref": "trainer_pool",
                     "reward": "other_pool"},
            resource_pool_dict={},
        )

    scope = {
        "Role": roles,
        "MultiTaskFullyAsyncTrainer": DummyTrainer,
        "create_resource_pool_manager": create_resource_pool_manager,
        "ray": SimpleNamespace(get=lambda value: value),
    }
    exec(compile(module, str(_TRAINER_PG_SOURCE), "exec"), scope)
    return scope["IsolatedTaskRunner"], created


def _start_trainer(klass, session):
    runner = klass()
    runner.task_session = session
    runner.components = {
        "tokenizer": object(),
        "role_worker_mapping": {
            "actor": object(), "ref": object(),
            "reward": object(), "rollout": object(),
        },
        "ray_worker_group_cls": object(),
    }
    runner._create_trainer(SimpleNamespace(trainer=SimpleNamespace(device="npu")))
    return runner.components["trainer"].kwargs["resource_pool_manager"]


def test_independent_sessions_have_no_trainer_pg_name_collision():
    klass, created = _build_trainer_stubs()
    donor = _start_trainer(klass, "donor-actor")
    borrower = _start_trainer(klass, "borrower-actor")

    assert len(created) == 2
    assert set(donor.resource_pool_spec).isdisjoint(borrower.resource_pool_spec)
    assert donor.resource_pool_spec["trainer_pool_mt_donor-actor_"] == [2]
    assert borrower.resource_pool_spec["trainer_pool_mt_borrower-actor_"] == [2]
    # This is the exact PG-name construction used by RayResourcePool in VERL.
    donor_pg = f"{next(iter(donor.resource_pool_spec))}verl_group_1:0"
    borrower_pg = f"{next(iter(borrower.resource_pool_spec))}verl_group_1:0"
    assert donor_pg != borrower_pg
    assert "trainer_poolverl_group_1:0" not in (donor_pg, borrower_pg)


def test_all_roles_keep_same_pool_ownership_and_gpu_counts():
    klass, _ = _build_trainer_stubs()
    for session in ("task-a", "task-b"):
        pools = _start_trainer(klass, session)
        assert pools.mapping == {
            "actor": f"trainer_pool_mt_{session}_",
            "ref": f"trainer_pool_mt_{session}_",
            "reward": f"other_pool_mt_{session}_",
        }
        assert sum(sum(x) for x in pools.resource_pool_spec.values()) == 3
        assert pools.resource_pool_dict == {}


def test_missing_session_fails_before_resource_pool_or_trainer_creation():
    klass, created = _build_trainer_stubs()
    runner = klass()
    runner.task_session = None
    runner.components = {
        "tokenizer": object(),
        "role_worker_mapping": {"actor": object()},
        "ray_worker_group_cls": object(),
    }
    with pytest.raises(RuntimeError, match="require task_session"):
        runner._create_trainer(SimpleNamespace(trainer=SimpleNamespace(device="npu")))
    assert not created
