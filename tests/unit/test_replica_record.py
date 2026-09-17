"""Public eight-state replica lifecycle with full current identity."""

import pytest

from multi_task_scheduler.orchestration.contracts import NodePlacement, PlacementSpec, ReplicaKey
from multi_task_scheduler.orchestration.replica_record import (
    IllegalReplicaTransitionError,
    ReplicaKind,
    ReplicaRecord,
    ReplicaState,
)


def _placement():
    return PlacementSpec(
        node=NodePlacement(
            node_id="n1",
            gpu_uuids=("u0",),
            physical_gpu_ids=(0,),
            global_ranks=(0,),
            local_ranks=(0,),
        ),
        tp=1,
        dp=1,
        pp=1,
        model_signature="sig",
        placement_digest="placement-1",
    )


def _record(*, replica_id="r1", runtime_epoch=0, kind=ReplicaKind.BORROWED, state=None):
    if state is None:
        state = ReplicaState.ACTIVE if kind is ReplicaKind.NATIVE else ReplicaState.PREPARING
    return ReplicaRecord(
        key=ReplicaKey(task_session="s1", replica_id=replica_id, runtime_epoch=runtime_epoch),
        kind=kind,
        state=state,
        revision=0,
        placement=_placement(),
    )


def test_borrowed_starts_preparing_and_advances_to_destroyed():
    record = _record()
    assert record.state is ReplicaState.PREPARING
    for state in (
        ReplicaState.ACTIVE,
        ReplicaState.DRAINING,
        ReplicaState.DETACHED,
        ReplicaState.DESTROYED,
    ):
        record.transition_to(state)
    assert record.state is ReplicaState.DESTROYED
    assert record.revision == 4


def test_native_sleep_restore_keeps_same_replica_key():
    record = _record(kind=ReplicaKind.NATIVE, runtime_epoch=3)
    original_key = record.key
    for state in (
        ReplicaState.DRAINING,
        ReplicaState.DETACHED,
        ReplicaState.DORMANT,
        ReplicaState.RESTORING,
        ReplicaState.ACTIVE,
    ):
        record.transition_to(state)
    assert record.state is ReplicaState.ACTIVE
    assert record.key is original_key
    assert record.key.runtime_epoch == 3


def test_failed_borrowed_prepare_can_destroy_or_quarantine():
    cleaned = _record()
    cleaned.transition_to(ReplicaState.DESTROYED)
    assert cleaned.state is ReplicaState.DESTROYED

    unknown = _record(replica_id="r2")
    unknown.transition_to(ReplicaState.QUARANTINED)
    assert unknown.state is ReplicaState.QUARANTINED


def test_illegal_transition_and_terminal_quarantine_raise():
    record = _record()
    with pytest.raises(IllegalReplicaTransitionError):
        record.transition_to(ReplicaState.DETACHED)
    record.transition_to(ReplicaState.QUARANTINED)
    with pytest.raises(IllegalReplicaTransitionError):
        record.transition_to(ReplicaState.DESTROYED)


def test_same_state_is_noop_and_drain_can_cancel_before_detach():
    record = _record()
    record.transition_to(ReplicaState.PREPARING)
    assert record.revision == 0
    record.transition_to(ReplicaState.ACTIVE)
    record.transition_to(ReplicaState.DRAINING)
    record.transition_to(ReplicaState.ACTIVE)
    assert record.state is ReplicaState.ACTIVE


def test_native_never_destroys_and_borrowed_never_sleeps():
    native = _record(kind=ReplicaKind.NATIVE)
    native.transition_to(ReplicaState.DRAINING)
    native.transition_to(ReplicaState.DETACHED)
    with pytest.raises(IllegalReplicaTransitionError):
        native.transition_to(ReplicaState.DESTROYED)

    borrowed = _record()
    borrowed.transition_to(ReplicaState.ACTIVE)
    borrowed.transition_to(ReplicaState.DRAINING)
    borrowed.transition_to(ReplicaState.DETACHED)
    with pytest.raises(IllegalReplicaTransitionError):
        borrowed.transition_to(ReplicaState.DORMANT)
