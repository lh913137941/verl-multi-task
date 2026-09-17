"""OperationJournal replay fencing and independent phase/status semantics."""

import pytest

from multi_task_scheduler.orchestration.operation_journal import (
    ExpiredLeaseError,
    IllegalOperationTransitionError,
    OperationIdentityError,
    OperationJournal,
    OperationKind,
    OperationStatus,
    Phase,
)


def test_begin_creates_validate_accepted_record():
    journal = OperationJournal()
    record = journal.begin("op-1", 0, "r1", OperationKind.ADD)
    assert record.phase is Phase.VALIDATE
    assert record.status is OperationStatus.ACCEPTED
    assert record.phase_revision == 0


def test_exact_replay_is_idempotent_but_identity_changes_are_rejected():
    journal = OperationJournal()
    first = journal.begin(
        "op-1", 0, "r1", OperationKind.ADD, payload_digest="d1", command_seq=3
    )
    second = journal.begin(
        "op-1", 0, "r1", OperationKind.ADD, payload_digest="d1", command_seq=3
    )
    assert first is second

    for kwargs in (
        {"replica_id": "r2"},
        {"kind": OperationKind.REMOVE},
        {"payload_digest": "other"},
        {"command_seq": 4},
    ):
        values = dict(
            operation_id="op-1",
            lease_epoch=0,
            replica_id="r1",
            kind=OperationKind.ADD,
            payload_digest="d1",
            command_seq=3,
        )
        values.update(kwargs)
        with pytest.raises(OperationIdentityError):
            journal.begin(**values)


def test_operation_id_cannot_be_reused_across_lease_epochs():
    journal = OperationJournal()
    journal.begin("op-1", 1, "r1", OperationKind.ADD)
    with pytest.raises(ExpiredLeaseError):
        journal.begin("op-1", 0, "r1", OperationKind.ADD)
    with pytest.raises(OperationIdentityError):
        journal.begin("op-1", 2, "r1", OperationKind.ADD)
    assert journal.query("op-1").lease_epoch == 1


def test_add_happy_path_finishes_only_with_explicit_succeeded_status():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationKind.ADD)
    for phase in (
        Phase.CREATE,
        Phase.WAIT_GATE,
        Phase.LOAD_WEIGHTS,
        Phase.JOIN_CE,
        Phase.COMMIT_SERVICE,
    ):
        journal.transition("op-1", phase)
    with pytest.raises(IllegalOperationTransitionError, match="explicit"):
        journal.transition("op-1", Phase.DONE)
    journal.transition("op-1", Phase.DONE, status=OperationStatus.SUCCEEDED)
    record = journal.query("op-1")
    assert record.phase is Phase.DONE
    assert record.status is OperationStatus.SUCCEEDED


def test_failed_and_succeeded_are_terminal_only_at_done():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationKind.ADD)
    with pytest.raises(IllegalOperationTransitionError, match="requires phase DONE"):
        journal.transition("op-1", Phase.CREATE, status=OperationStatus.FAILED)


def test_reconcile_is_unknown_and_can_finish_unknown():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationKind.ADD)
    journal.transition("op-1", Phase.RECONCILE)
    assert journal.query("op-1").status is OperationStatus.UNKNOWN
    journal.transition("op-1", Phase.DONE, status=OperationStatus.UNKNOWN)
    assert journal.query("op-1").status is OperationStatus.UNKNOWN


def test_donate_can_enter_drain_and_illegal_skips_raise():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationKind.DONATE)
    journal.transition("op-1", Phase.DRAIN)
    assert journal.query("op-1").status is OperationStatus.RUNNING

    other = OperationJournal()
    other.begin("op-2", 0, "r2", OperationKind.ADD)
    with pytest.raises(IllegalOperationTransitionError):
        other.transition("op-2", Phase.COMMIT_SERVICE)


def test_require_rejects_wrong_epoch():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationKind.ADD)
    with pytest.raises(ExpiredLeaseError):
        journal.require("op-1", 1)


def test_phase_result_timing_and_error_revision_the_record():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationKind.ADD)
    proof = object()
    error = object()
    record = journal.transition(
        "op-1",
        Phase.CREATE,
        phase_result=proof,
        elapsed_ms=12,
        error=error,
        detail={"step": "created"},
    )
    assert record.phase_results[Phase.CREATE] is proof
    assert record.phase_timings_ms[Phase.CREATE] == 12
    assert record.error is error
    assert record.detail == {"step": "created"}
    assert record.phase_revision == 1
