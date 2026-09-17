"""Production window and idle-candidate selection (section 5.1)."""

import pytest
from dataclasses import replace

from multi_task_scheduler.orchestration.production_window import (
    CandidateSet,
    ProductionWindow,
    ReplicaObservation,
    ReplicaView,
    advance_source_seq,
    candidate,
    select_idle_candidates,
)


def _idle_window(epoch=3, source_seq=7):
    return ProductionWindow(
        production_epoch=epoch,
        source_seq=source_seq,
        eligible_pending=0,
        held=0,
        closed_by_staleness=True,
    )


def _quiet_active_native():
    return ReplicaObservation(
        replica_id="r1", is_active_native=True, in_flight=0, admitting=0, queued=0,
        running=0, pending_admissions=0, production_epoch=3, all_backends_observed=True,
    )


def test_window_idle_requires_closed_or_exhausted_and_zero_p_h():
    assert _idle_window().idle is True
    assert ProductionWindow(closed_by_staleness=True, eligible_pending=1).idle is False
    assert ProductionWindow(closed_by_staleness=True, held=1).idle is False
    assert ProductionWindow(eligible_pending=0, held=0).idle is False  # still open
    assert ProductionWindow(exhausted_this_round=True, eligible_pending=0, held=0).idle


def test_candidate_rejects_stale_observations():
    window = _idle_window(epoch=3)
    obs = _quiet_active_native()
    assert candidate(
        window, obs, observations_fresh=False, observations_epoch=3,
        keeps_min_active_gpus=True, leaves_routable=True,
    ) is False
    assert candidate(
        window, obs, observations_fresh=True, observations_epoch=2,
        keeps_min_active_gpus=True, leaves_routable=True,
    ) is False


def test_candidate_rejects_non_quiet_replica():
    window = _idle_window()
    obs = ReplicaObservation(replica_id="r1", in_flight=1)
    assert candidate(
        window, obs, observations_fresh=True, observations_epoch=3,
        keeps_min_active_gpus=True, leaves_routable=True,
    ) is False


def test_candidate_rejects_concurrent_transfer():
    window = _idle_window()
    obs = ReplicaObservation(replica_id="r1", concurrent_transfer=True)
    assert candidate(
        window, obs, observations_fresh=True, observations_epoch=3,
        keeps_min_active_gpus=True, leaves_routable=True,
    ) is False


def test_candidate_requires_aggregate_constraints():
    window = _idle_window()
    obs = _quiet_active_native()
    assert candidate(
        window, obs, observations_fresh=True, observations_epoch=3,
        keeps_min_active_gpus=False, leaves_routable=True,
    ) is False
    assert candidate(
        window, obs, observations_fresh=True, observations_epoch=3,
        keeps_min_active_gpus=True, leaves_routable=False,
    ) is False


def test_select_idle_candidates_respects_min_active_and_routable():
    window = _idle_window()
    replicas = [
        ReplicaView(_quiet_active_native(), gpus=4),
        ReplicaView(ReplicaObservation(replica_id="r2", in_flight=5), gpus=4),
        ReplicaView(replace(_quiet_active_native(), replica_id="r3"), gpus=4),
    ]
    result = select_idle_candidates(
        window,
        replicas,
        observations_fresh=True,
        min_active_gpus=8,
        current_active_gpus=12,
        routable_count=3,
    )
    assert result.candidate_ids == ("r1", "r3")
    assert isinstance(result, CandidateSet)
    assert result.source_seq == window.source_seq
    assert result.production_epoch == window.production_epoch


def test_select_idle_candidates_empty_when_removal_drops_below_min():
    window = _idle_window()
    replicas = [
        ReplicaView(ReplicaObservation(replica_id="r1"), gpus=4),
        ReplicaView(ReplicaObservation(replica_id="r2"), gpus=4),
    ]
    # Removing either leaves 4 < min_active_gpus=8, so no candidate.
    result = select_idle_candidates(
        window,
        replicas,
        observations_fresh=True,
        min_active_gpus=8,
        current_active_gpus=8,
        routable_count=2,
    )
    assert result.candidate_ids == ()


def test_select_idle_candidates_empty_when_only_one_routable():
    window = _idle_window()
    replicas = [ReplicaView(ReplicaObservation(replica_id="r1"), gpus=4)]
    result = select_idle_candidates(
        window,
        replicas,
        observations_fresh=True,
        min_active_gpus=0,
        current_active_gpus=4,
        routable_count=1,
    )
    assert result.candidate_ids == ()


def test_advance_source_seq_invalidates_candidates():
    window = _idle_window(source_seq=7)
    advanced = advance_source_seq(window)
    assert advanced.source_seq == 8
    assert advanced.production_epoch == window.production_epoch


def test_missing_production_counts_and_backend_observations_never_mean_idle():
    assert not ProductionWindow(closed_by_staleness=True).idle
    assert not ReplicaObservation(replica_id="r1").quiet
    assert not replace(_quiet_active_native(), all_backends_observed=False).quiet
    assert not replace(_quiet_active_native(), pending_admissions=1).quiet
    assert not replace(_idle_window(), policy_refresh_inflight=True).idle


def test_selector_does_not_relabel_an_old_engine_observation_with_current_epoch():
    result = select_idle_candidates(
        _idle_window(epoch=4), [ReplicaView(_quiet_active_native(), gpus=1)],
        observations_fresh=True, min_active_gpus=1, current_active_gpus=2, routable_count=2,
    )
    assert result.candidate_ids == ()
