"""OperationJournal replay fencing and simplified §6.1 phase/status contract."""

import pytest

from multi_task_scheduler.orchestration.operation_journal import (
    ExpiredLeaseError,
    IllegalOperationTransitionError,
    OperationIdentityError,
    OperationJournal,
    OperationKind,
    OperationStatus,
    Phase,
    phase_to_state,
)


def test_begin_creates_validate_accepted_record():
    journal = OperationJournal()
    record = journal.begin("op-1", 0, "r1", OperationKind.ADD)
    assert record.phase is Phase.VALIDATE
    assert record.status is OperationStatus.ACCEPTED
    assert record.phase_revision == 0


def test_idempotent_replay_returns_existing():
    journal = OperationJournal()
    first = journal.begin("op-1", 0, "r1", OperationKind.ADD, payload_digest="d1")
    second = journal.begin("op-1", 0, "r1", OperationKind.ADD, payload_digest="d1")
    assert first is second


def test_older_epoch_is_rejected():
    journal = OperationJournal()
    journal.begin("op-1", 1, "r1", OperationKind.ADD)
    with pytest.raises(ExpiredLeaseError):
        journal.begin("op-1", 0, "r1", OperationKind.ADD)


def test_conflicting_replay_is_rejected():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationKind.ADD, payload_digest="d1")
    with pytest.raises(OperationIdentityError):
        journal.begin("op-1", 0, "r1", OperationKind.ADD, payload_digest="other")
    with pytest.raises(OperationIdentityError):
        journal.begin("op-1", 0, "r2", OperationKind.ADD, payload_digest="d1")
    with pytest.raises(OperationIdentityError):
        journal.begin("op-1", 0, "r1", OperationKind.REMOVE, payload_digest="d1")


def test_newer_epoch_cannot_reuse_existing_operation_id():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationKind.ADD)
    with pytest.raises(OperationIdentityError):
        journal.begin("op-1", 1, "r1", OperationKind.ADD)
    assert journal.query("op-1").lease_epoch == 0


@pytest.mark.parametrize("digest,seq", [(None, 3), ("d1", 4)])
def test_replay_cannot_drop_digest_or_change_command_sequence(digest, seq):
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationKind.ADD, payload_digest="d1", command_seq=3)
    with pytest.raises(OperationIdentityError):
        journal.begin(
            "op-1", 0, "r1", OperationKind.ADD,
            payload_digest=digest, command_seq=seq,
        )


def test_add_happy_path_finishes_with_explicit_succeeded_status():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationKind.ADD)
    for phase in [
        Phase.CREATE,
        Phase.WAIT_GATE,
        Phase.LOAD_WEIGHTS,
        Phase.JOIN_CE,
        Phase.COMMIT_SERVICE,
    ]:
        journal.transition("op-1", phase)
    journal.transition("op-1", Phase.DONE, status=OperationStatus.SUCCEEDED)
    record = journal.query("op-1")
    assert record.phase is Phase.DONE
    assert record.status is OperationStatus.SUCCEEDED
    assert record.phase_revision == 6


def test_donate_starts_drain_from_validate():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationKind.DONATE)
    journal.transition("op-1", Phase.DRAIN)
    assert journal.query("op-1").phase is Phase.DRAIN
    assert journal.query("op-1").status is OperationStatus.RUNNING


def test_illegal_transition_raises():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationKind.ADD)
    with pytest.raises(IllegalOperationTransitionError):
        journal.transition("op-1", Phase.COMMIT_SERVICE)


def test_require_rejects_wrong_epoch():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationKind.ADD)
    with pytest.raises(ExpiredLeaseError):
        journal.require("op-1", 1)


def test_done_does_not_infer_success_or_failure_from_phase():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationKind.ADD)
    with pytest.raises(IllegalOperationTransitionError, match="explicit"):
        journal.transition("op-1", Phase.DONE)
    with pytest.raises(ValueError, match="explicit"):
        phase_to_state(Phase.DONE)
    assert phase_to_state(
        Phase.DONE, terminal_status=OperationStatus.FAILED
    ) is OperationStatus.FAILED


def test_reconcile_is_unknown_and_can_finish_unknown():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationKind.ADD)
    journal.transition("op-1", Phase.RECONCILE)
    assert journal.query("op-1").status is OperationStatus.UNKNOWN
    journal.transition("op-1", Phase.DONE, status=OperationStatus.UNKNOWN)
    assert journal.query("op-1").status is OperationStatus.UNKNOWN


def test_phase_result_and_timing_are_saved_by_phase_and_revisioned():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationKind.ADD)
    proof = object()
    record = journal.transition(
        "op-1", Phase.CREATE, phase_result=proof, elapsed_ms=12
    )
    assert record.phase_results[Phase.CREATE] is proof
    assert record.phase_timings_ms[Phase.CREATE] == 12
    assert record.phase_revision == 1
