"""Public eight-state replica lifecycle and kind-specific terminal rules."""

import pytest

from multi_task_scheduler.orchestration.replica_record import (
    IllegalReplicaTransitionError,
    ReplicaKind,
    ReplicaRecord,
    ReplicaState,
)


def test_borrowed_defaults_to_preparing():
    record = ReplicaRecord(replica_id="r1")
    assert record.kind is ReplicaKind.BORROWED
    assert record.state is ReplicaState.PREPARING
    assert record.revision == 0


def test_borrowed_lifecycle_advances_to_destroyed():
    record = ReplicaRecord(replica_id="r1")
    for state in [
        ReplicaState.ACTIVE,
        ReplicaState.DRAINING,
        ReplicaState.DETACHED,
        ReplicaState.DESTROYED,
    ]:
        record.transition_to(state)
    assert record.state is ReplicaState.DESTROYED
    assert record.revision == 4


def test_native_sleep_restore_path_keeps_same_runtime_identity():
    record = ReplicaRecord(replica_id="r1", kind=ReplicaKind.NATIVE, runtime_epoch=3)
    for state in [
        ReplicaState.DRAINING,
        ReplicaState.DETACHED,
        ReplicaState.DORMANT,
        ReplicaState.RESTORING,
        ReplicaState.ACTIVE,
    ]:
        record.transition_to(state)
    assert record.state is ReplicaState.ACTIVE
    assert record.runtime_epoch == 3


def test_failed_hidden_create_can_only_become_destroyed_with_cleanup_or_quarantined():
    cleaned = ReplicaRecord(replica_id="r1")
    cleaned.transition_to(ReplicaState.DESTROYED)
    assert cleaned.state is ReplicaState.DESTROYED

    unknown = ReplicaRecord(replica_id="r2")
    unknown.transition_to(ReplicaState.QUARANTINED)
    assert unknown.state is ReplicaState.QUARANTINED


def test_illegal_transition_raises():
    record = ReplicaRecord(replica_id="r1")
    with pytest.raises(IllegalReplicaTransitionError):
        record.transition_to(ReplicaState.DETACHED)


def test_transition_to_same_state_is_noop_and_does_not_bump_revision():
    record = ReplicaRecord(replica_id="r1")
    record.transition_to(ReplicaState.PREPARING)
    assert record.revision == 0


def test_quarantined_is_terminal():
    record = ReplicaRecord(replica_id="r1")
    record.transition_to(ReplicaState.QUARANTINED)
    with pytest.raises(IllegalReplicaTransitionError):
        record.transition_to(ReplicaState.DESTROYED)


def test_drain_can_be_safely_cancelled_before_service_detach():
    record = ReplicaRecord(replica_id="r1")
    record.transition_to(ReplicaState.ACTIVE)
    record.transition_to(ReplicaState.DRAINING)
    record.transition_to(ReplicaState.ACTIVE)
    assert record.state is ReplicaState.ACTIVE


def test_native_never_destroys_and_borrowed_never_becomes_dormant():
    native = ReplicaRecord(replica_id="n1", kind=ReplicaKind.NATIVE)
    native.transition_to(ReplicaState.DRAINING)
    native.transition_to(ReplicaState.DETACHED)
    with pytest.raises(IllegalReplicaTransitionError):
        native.transition_to(ReplicaState.DESTROYED)

    borrowed = ReplicaRecord(replica_id="b1")
    borrowed.transition_to(ReplicaState.ACTIVE)
    borrowed.transition_to(ReplicaState.DRAINING)
    borrowed.transition_to(ReplicaState.DETACHED)
    with pytest.raises(IllegalReplicaTransitionError):
        borrowed.transition_to(ReplicaState.DORMANT)
