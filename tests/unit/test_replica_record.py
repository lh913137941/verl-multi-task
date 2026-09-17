"""ReplicaRecord lifecycle: borrowed path, donor path, illegal transitions."""

import pytest

from multi_task_scheduler.orchestration.replica_record import (
    IllegalReplicaTransitionError,
    ReplicaKind,
    ReplicaRecord,
    ReplicaState,
)


def test_default_state_is_creating_and_borrowed():
    record = ReplicaRecord(replica_id="r1")
    assert record.kind is ReplicaKind.BORROWED
    assert record.state is ReplicaState.CREATING


@pytest.mark.parametrize(
    "path",
    [
        [
            ReplicaState.HIDDEN,
            ReplicaState.BOOTSTRAPPING,
            ReplicaState.CE_EFFECTIVE,
            ReplicaState.ACTIVE,
            ReplicaState.DRAINING,
            ReplicaState.CE_REMOVED,
            ReplicaState.DESTROYING,
            ReplicaState.DESTROYED,
        ],
    ],
)
def test_borrowed_lifecycle_advances_to_destroyed(path):
    record = ReplicaRecord(replica_id="r1")
    for state in path:
        record.transition_to(state)
    assert record.state is ReplicaState.DESTROYED


def test_donor_sleep_wake_path():
    record = ReplicaRecord(replica_id="r1", kind=ReplicaKind.NATIVE)
    for state in [
        ReplicaState.ACTIVE,
        ReplicaState.DRAINING,
        ReplicaState.CE_REMOVED,
        ReplicaState.SLEEPING,
        ReplicaState.DORMANT,
        ReplicaState.WAKING_WEIGHTS,
        ReplicaState.BOOTSTRAPPING,
        ReplicaState.CE_EFFECTIVE,
        ReplicaState.ACTIVE,
    ]:
        record.transition_to(state)
    assert record.state is ReplicaState.ACTIVE


def test_creating_to_failed_hidden():
    record = ReplicaRecord(replica_id="r1")
    record.transition_to(ReplicaState.FAILED_HIDDEN)
    assert record.state is ReplicaState.FAILED_HIDDEN


def test_illegal_transition_raises():
    record = ReplicaRecord(replica_id="r1")
    with pytest.raises(IllegalReplicaTransitionError):
        record.transition_to(ReplicaState.ACTIVE)  # cannot skip HIDDEN


def test_transition_to_same_state_is_noop():
    record = ReplicaRecord(replica_id="r1")
    record.transition_to(ReplicaState.CREATING)
    assert record.state is ReplicaState.CREATING


def test_quarantined_is_terminal():
    record = ReplicaRecord(replica_id="r1")
    record.transition_to(ReplicaState.HIDDEN)
    record.transition_to(ReplicaState.QUARANTINED)
    with pytest.raises(IllegalReplicaTransitionError):
        record.transition_to(ReplicaState.DESTROYED)


def test_bootstrap_rollback_to_hidden():
    record = ReplicaRecord(replica_id="r1")
    record.transition_to(ReplicaState.HIDDEN)
    record.transition_to(ReplicaState.BOOTSTRAPPING)
    record.transition_to(ReplicaState.HIDDEN)  # rollback edge
    assert record.state is ReplicaState.HIDDEN
