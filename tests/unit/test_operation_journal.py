"""OperationJournal: replay idempotency, epoch fencing, phase transitions."""

import pytest

from multi_task_scheduler.orchestration.operation_journal import (
    ExpiredLeaseError,
    IllegalOperationTransitionError,
    OperationIdentityError,
    OperationJournal,
    OperationPhase,
    OperationState,
    OperationType,
    phase_to_state,
)


def test_begin_creates_accepted_record():
    journal = OperationJournal()
    record = journal.begin("op-1", 0, "r1", OperationType.ADD)
    assert record.phase is OperationPhase.ACCEPTED
    assert record.state is OperationState.ACCEPTED


def test_idempotent_replay_returns_existing():
    journal = OperationJournal()
    first = journal.begin("op-1", 0, "r1", OperationType.ADD, payload_digest="d1")
    second = journal.begin("op-1", 0, "r1", OperationType.ADD, payload_digest="d1")
    assert first is second


def test_older_epoch_is_rejected():
    journal = OperationJournal()
    journal.begin("op-1", 1, "r1", OperationType.ADD)
    with pytest.raises(ExpiredLeaseError):
        journal.begin("op-1", 0, "r1", OperationType.ADD)


def test_conflicting_replay_is_rejected():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationType.ADD, payload_digest="d1")
    with pytest.raises(OperationIdentityError):
        journal.begin("op-1", 0, "r1", OperationType.ADD, payload_digest="other")
    with pytest.raises(OperationIdentityError):
        journal.begin("op-1", 0, "r2", OperationType.ADD, payload_digest="d1")
    with pytest.raises(OperationIdentityError):
        journal.begin("op-1", 0, "r1", OperationType.REMOVE, payload_digest="d1")


def test_newer_epoch_cannot_overwrite_an_existing_operation():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationType.ADD)
    with pytest.raises(OperationIdentityError):
        journal.begin("op-1", 1, "r1", OperationType.ADD)
    assert journal.query("op-1").lease_epoch == 0


@pytest.mark.parametrize("digest,seq", [(None, 3), ("d1", 4)])
def test_replay_cannot_drop_digest_or_change_command_sequence(digest, seq):
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationType.ADD, payload_digest="d1", command_seq=3)
    with pytest.raises(OperationIdentityError):
        journal.begin("op-1", 0, "r1", OperationType.ADD, payload_digest=digest, command_seq=seq)


def test_happy_path_transition_chain():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationType.ADD)
    for phase in [
        OperationPhase.PREPARING,
        OperationPhase.WAIT_GATE,
        OperationPhase.APPLYING,
        OperationPhase.COMMITTED,
    ]:
        journal.transition("op-1", phase)
    assert journal.query("op-1").state is OperationState.COMMITTED


def test_donate_goes_accepted_to_draining():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationType.DONATE)
    journal.transition("op-1", OperationPhase.DRAINING)
    assert journal.query("op-1").phase is OperationPhase.DRAINING


def test_illegal_transition_raises():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationType.ADD)
    with pytest.raises(IllegalOperationTransitionError):
        journal.transition("op-1", OperationPhase.COMMITTED)


def test_require_rejects_wrong_epoch():
    journal = OperationJournal()
    journal.begin("op-1", 0, "r1", OperationType.ADD)
    with pytest.raises(ExpiredLeaseError):
        journal.require("op-1", 1)


@pytest.mark.parametrize(
    "phase,state",
    [
        (OperationPhase.ACCEPTED, OperationState.ACCEPTED),
        (OperationPhase.PREPARING, OperationState.RUNNING),
        (OperationPhase.DRAINING, OperationState.RUNNING),
        (OperationPhase.WAIT_GATE, OperationState.RUNNING),
        (OperationPhase.APPLYING, OperationState.RUNNING),
        (OperationPhase.COMMITTED, OperationState.COMMITTED),
        (OperationPhase.ROLLED_BACK, OperationState.ROLLED_BACK),
        (OperationPhase.RECONCILING, OperationState.RECONCILING),
        (OperationPhase.QUARANTINED, OperationState.QUARANTINED),
    ],
)
def test_phase_to_state_mapping(phase, state):
    assert phase_to_state(phase) is state
