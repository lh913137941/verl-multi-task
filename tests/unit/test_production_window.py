"""Production-window sensing emits canonical evidence-bearing candidates."""

from dataclasses import replace

from multi_task_scheduler.orchestration.contracts import ReplicaKey
from multi_task_scheduler.orchestration.production_window import (
    ProductionWindow,
    ReplicaObservation,
    ReplicaView,
    advance_source_seq,
    candidate,
    select_idle_candidates,
)


def _window(epoch=3, source_seq=7, revision=11):
    return ProductionWindow(
        production_epoch=epoch,
        source_seq=source_seq,
        production_revision=revision,
        eligible_pending=0,
        held=0,
        closed_by_staleness=True,
    )


def _observation(replica_id="r1", epoch=3):
    return ReplicaObservation(
        key=ReplicaKey(task_session="s1", replica_id=replica_id, runtime_epoch=0),
        engine_seq=5,
        observed_age_ms=10,
        engine_digest=f"engine-{replica_id}",
        in_flight=0,
        admitting=0,
        queued=0,
        running=0,
        pending_admissions=0,
        production_epoch=epoch,
        all_backends_observed=True,
    )


def _view(replica_id="r1", epoch=3, gpu_count=1):
    return ReplicaView(
        observation=_observation(replica_id, epoch),
        manager_revision=2,
        lb_revision=4,
        placement_digest=f"placement-{replica_id}",
        stable_idle_ms=500,
        gpu_count=gpu_count,
    )


def test_window_idle_requires_known_zero_counts_and_closed_or_exhausted_reason():
    assert _window().idle is True
    assert _window().idle_reason == "CLOSED_STALENESS"
    assert not ProductionWindow(closed_by_staleness=True).idle
    assert not ProductionWindow(
        eligible_pending=0, held=0, policy_refresh_inflight=False
    ).idle
    exhausted = ProductionWindow(
        eligible_pending=0, held=0, exhausted_this_round=True
    )
    assert exhausted.idle_reason == "EXHAUSTED"


def test_candidate_rejects_stale_epoch_unknown_activity_and_concurrent_transfer():
    window = _window()
    assert not candidate(
        window,
        _view(),
        observations_fresh=False,
        keeps_min_active_gpus=True,
        leaves_routable=True,
    )
    assert not candidate(
        window,
        _view(epoch=2),
        observations_fresh=True,
        keeps_min_active_gpus=True,
        leaves_routable=True,
    )
    unknown = replace(_observation(), queued=None)
    assert not candidate(
        window,
        replace(_view(), observation=unknown),
        observations_fresh=True,
        keeps_min_active_gpus=True,
        leaves_routable=True,
    )
    transferring = replace(_observation(), transfer_inflight=True)
    assert not candidate(
        window,
        replace(_view(), observation=transferring),
        observations_fresh=True,
        keeps_min_active_gpus=True,
        leaves_routable=True,
    )


def test_selector_emits_full_idle_candidate_identity_and_evidence():
    result = select_idle_candidates(
        _window(),
        [_view("r1", gpu_count=4), _view("r2", gpu_count=4)],
        observations_fresh=True,
        min_active_gpus=4,
        current_active_gpus=8,
        routable_count=3,
    )
    assert tuple(item.key.replica_id for item in result) == ("r1", "r2")
    first = result[0]
    assert first.key.task_session == "s1"
    assert first.production_epoch == 3
    assert first.source_seq == 7
    assert first.manager_revision == 2
    assert first.lb_revision == 4
    assert first.engine_digest == "engine-r1"
    assert first.reason == "CLOSED_STALENESS"
    assert first.stable_idle_ms == 500
    assert first.observed_age_ms == 10
    assert first.gpu_count == 4
    assert first.placement_digest == "placement-r1"
    assert first.evidence_digest


def test_selector_respects_capacity_and_routable_lower_bounds():
    assert select_idle_candidates(
        _window(),
        [_view("r1", gpu_count=4)],
        observations_fresh=True,
        min_active_gpus=8,
        current_active_gpus=8,
        routable_count=2,
    ) == ()
    assert select_idle_candidates(
        _window(),
        [_view("r1", gpu_count=1)],
        observations_fresh=True,
        min_active_gpus=0,
        current_active_gpus=1,
        routable_count=1,
    ) == ()


def test_engine_observation_epoch_is_never_relabelled_to_current_window():
    assert select_idle_candidates(
        _window(epoch=4),
        [_view("r1", epoch=3)],
        observations_fresh=True,
        min_active_gpus=0,
        current_active_gpus=2,
        routable_count=2,
    ) == ()


def test_source_seq_advance_invalidates_previous_candidates_without_changing_epoch():
    window = _window(source_seq=7)
    advanced = advance_source_seq(window)
    assert advanced.source_seq == 8
    assert advanced.production_epoch == window.production_epoch
    assert advanced.production_revision == window.production_revision
