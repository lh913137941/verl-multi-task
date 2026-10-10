"""SCHEDULER wiring regression scenarios; shared test fakes live in _wiring_support."""

import time
import pytest
from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    Lease,
    OperationCommand,
    OperationEvidence,
    OperationKind,
    OperationRecord,
    OperationStatus,
    ReplicaKey,
    ReplicaKind,
)
from _wiring_support import (
    RemoteMethod,
    _isolated_group_scheduler_class,
    _scheduler_test_lease,
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
    gs.submit_idle_report({
        "task_session": "task-a",
        "candidates": ({"replica_key": command.target, "kind": ReplicaKind.NATIVE.value},),
    })
    record = gs.submit_operation(command)
    assert record.operation_id == command.operation_id
    assert gs.operation_commands[command.operation_id] == command


def test_group_scheduler_preserves_staged_add_state_when_taskrunner_rejects_unknown():
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
    assert gs.operation_commands[command.operation_id] == command
    assert gs.borrower_targets[lease.lease_id] == command.target


def test_group_scheduler_preserves_restore_reservation_when_taskrunner_rejects_unknown():
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
    assert gs.operation_commands[command.operation_id] == command
    assert gs.active_gpu_owner["u0"] == lease.lease_id
    assert gs.active_bundle_owner[("pg", 0)] == lease.lease_id


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
    gs.submit_idle_report({
        "task_session": "task-a",
        "candidates": ({"replica_key": command.target, "kind": ReplicaKind.NATIVE.value},),
    })
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


def test_group_scheduler_exact_donate_replay_survives_report_expiry():
    cls = _isolated_group_scheduler_class()
    gs = cls()
    lease = _scheduler_test_lease()
    gs.open_lease(lease)
    target = ReplicaKey("task-a", "native-0")
    calls = []

    class Runner:
        submit_operation = RemoteMethod(
            lambda command, lease=None: (
                calls.append(command.operation_id) or OperationRecord(command.operation_id)
            )
        )

    gs.task_runners["task-a"] = Runner()
    gs.submit_idle_report({
        "task_session": "task-a",
        "candidates": ({"replica_key": target, "kind": ReplicaKind.NATIVE.value},),
    })
    command = OperationCommand("op-donate-replay", OperationKind.DONATE, target, lease.lease_id)
    assert gs.submit_operation(command).operation_id == command.operation_id

    # Once accepted, exact replay reconciles the same operation even if its
    # advisory report has since aged out; it must not create a new operation.
    gs.idle_reports["task-a"]["observed_at"] = time.monotonic() - 11.0
    assert gs.submit_operation(command).operation_id == command.operation_id
    assert calls == ["op-donate-replay", "op-donate-replay"]
    assert tuple(gs.operation_commands) == ("op-donate-replay",)


def test_group_scheduler_rejects_stale_or_mismatched_idle_donate_when_report_exists():
    cls = _isolated_group_scheduler_class()
    gs = cls()
    lease = _scheduler_test_lease()
    gs.open_lease(lease)
    target = ReplicaKey("task-a", "native-0")

    class Runner:
        submit_operation = RemoteMethod(
            lambda command, lease=None: OperationRecord(command.operation_id)
        )

    gs.task_runners["task-a"] = Runner()
    gs.idle_reports["task-a"] = {
        "observed_at": time.monotonic() - 11.0,
        "candidates": ({"replica_key": target, "kind": ReplicaKind.NATIVE.value},),
    }
    with pytest.raises(ValueError, match="idle report is stale"):
        gs.submit_operation(
            OperationCommand("op-stale", OperationKind.DONATE, target, lease.lease_id)
        )

    gs.idle_reports["task-a"] = {
        "observed_at": time.monotonic(),
        "candidates": ({
            "replica_key": ReplicaKey("task-a", "native-1"),
            "kind": ReplicaKind.NATIVE.value,
        },),
    }
    with pytest.raises(ValueError, match="not a currently reported idle NATIVE"):
        gs.submit_operation(
            OperationCommand("op-mismatch", OperationKind.DONATE, target, lease.lease_id)
        )


def test_group_scheduler_requires_fresh_idle_report_for_donate():
    cls = _isolated_group_scheduler_class()
    gs = cls()
    lease = _scheduler_test_lease()
    gs.open_lease(lease)
    target = ReplicaKey("task-a", "native-0")

    class Runner:
        submit_operation = RemoteMethod(
            lambda command, lease=None: OperationRecord(command.operation_id)
        )

    gs.task_runners["task-a"] = Runner()

    with pytest.raises(ValueError, match="requires a fresh idle report"):
        gs.submit_operation(
            OperationCommand("op-no-report", OperationKind.DONATE, target, lease.lease_id)
        )

    # An empty report is an explicit retraction, not donation authorization.
    gs.submit_idle_report({"task_session": "task-a", "candidates": ()})
    with pytest.raises(ValueError, match="not a currently reported idle NATIVE"):
        gs.submit_operation(
            OperationCommand("op-empty-report", OperationKind.DONATE, target, lease.lease_id)
        )

    gs.submit_idle_report({
        "task_session": "task-a",
        "candidates": ({"replica_key": target, "kind": ReplicaKind.BORROWED.value},),
    })
    with pytest.raises(ValueError, match="not a currently reported idle NATIVE"):
        gs.submit_operation(
            OperationCommand("op-borrowed-report", OperationKind.DONATE, target, lease.lease_id)
        )

    assert "op-no-report" not in gs.operation_commands
    assert "op-empty-report" not in gs.operation_commands
    assert "op-borrowed-report" not in gs.operation_commands


def test_group_scheduler_binds_donate_to_lease_donor_rank():
    cls = _isolated_group_scheduler_class()
    lease = _scheduler_test_lease(
        "l1", claim_id="claim-1", source_lease_id="source-1",
        donor_replica_rank=1, bundle_index=1, gpu_uuid="u1",
    )

    assert cls._target_matches_donor(ReplicaKey("task-a", "native-1"), lease)
    assert not cls._target_matches_donor(ReplicaKey("task-a", "r1"), lease)
    assert not cls._target_matches_donor(ReplicaKey("task-a", "native-0"), lease)
    assert not cls._target_matches_donor(ReplicaKey("task-a", "native-1", 1), lease)


def test_group_scheduler_restore_requires_original_donor_and_returned_claims():
    cls = _isolated_group_scheduler_class()
    gs = cls()
    lease = _scheduler_test_lease(
        "l1", claim_id="claim-1", source_lease_id="source-1",
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


def test_gs_open_lease_copies_mutable_claims_and_defends_replay_return():
    gs = _isolated_group_scheduler_class()()
    external = _scheduler_test_lease()
    original = _scheduler_test_lease()
    opened = gs.open_lease(external)

    external.claims[0]["gpu_uuid"] = "GPU-forged"
    opened.claims[0]["pg_id"] = "PG-forged"
    assert gs.leases[original.lease_id] == original
    assert gs.active_gpu_owner == {"u0": original.lease_id}
    assert gs.active_bundle_owner == {("pg", 0): original.lease_id}

    replay = gs.open_lease(_scheduler_test_lease())
    assert replay == original
    assert replay is not gs.leases[original.lease_id]
    replay.claims[0]["claim_id"] = "claim-forged"
    assert gs.leases[original.lease_id].claim_ids == original.claim_ids
    with pytest.raises(ValueError, match="conflicting lease replay"):
        gs.open_lease(external)
