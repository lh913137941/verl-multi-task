"""Production-window sensing uses the v3 window/activity contracts."""

from dataclasses import replace

import pytest

from multi_task_scheduler.orchestration.contracts import ReplicaKey
from multi_task_scheduler.orchestration.production_window import (
    ProductionWindow,
    ReplicaView,
    ServerActivity,
    WindowState,
    candidate,
    select_idle_candidates,
)


def _window(
    *,
    task_session="s1",
    epoch=3,
    revision=11,
    state=WindowState.CLOSED_STALENESS,
    eligible_pending=0,
    held_samples=0,
):
    return ProductionWindow(
        task_session=task_session,
        epoch=epoch,
        revision=revision,
        state=state,
        eligible_pending=eligible_pending,
        held_samples=held_samples,
        active_samples=2,
        output_queue_size=1,
        max_queue_size=8,
        producer_exhausted=state is WindowState.EXHAUSTED,
        policy_refresh_inflight=False,
    )


def _activity(replica_id="r1", task_session="s1"):
    return ServerActivity(
        key=ReplicaKey(
            task_session=task_session,
            replica_id=replica_id,
            runtime_epoch=0,
        ),
        engine_seq=5,
        observed_age_ms=10,
        admitting=0,
        queued=0,
        running=0,
        pending_admissions=0,
        transfer_inflight=False,
        all_backends_observed=True,
        engine_digest=f"engine-{replica_id}",
    )


def _view(replica_id="r1", task_session="s1", gpu_count=1):
    return ReplicaView(
        activity=_activity(replica_id, task_session),
        manager_revision=2,
        lb_revision=4,
        placement_digest=f"placement-{replica_id}",
        stable_idle_ms=500,
        gpu_count=gpu_count,
    )


def test_window_idle_uses_explicit_state_and_known_zero_p_h():
    assert _window().idle is True
    assert _window().idle_reason == "CLOSED_STALENESS"
    assert _window(state=WindowState.CLOSED_BACKPRESSURE).idle_reason == "CLOSED_BACKPRESSURE"
    assert _window(state=WindowState.EXHAUSTED).idle_reason == "EXHAUSTED"

    assert not _window(state=WindowState.OPEN).idle
    assert not _window(state=WindowState.UNKNOWN).idle
    assert not _window(eligible_pending=None).idle
    assert not _window(held_samples=None).idle
    assert not replace(_window(), policy_refresh_inflight=True).idle


def test_window_validates_native_fact_shape_without_reimplementing_thresholds():
    with pytest.raises(ValueError, match="output_queue_size"):
        replace(_window(), output_queue_size=9, max_queue_size=8)
    with pytest.raises(ValueError, match="task_session"):
        replace(_window(), task_session="")


def test_candidate_rejects_stale_unknown_or_conflicting_owner_facts():
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
        _view(task_session="other"),
        observations_fresh=True,
        keeps_min_active_gpus=True,
        leaves_routable=True,
    )

    unknown = replace(_activity(), queued=None)
    assert not candidate(
        window,
        replace(_view(), activity=unknown),
        observations_fresh=True,
        keeps_min_active_gpus=True,
        leaves_routable=True,
    )

    transferring = replace(_activity(), transfer_inflight=True)
    assert not candidate(
        window,
        replace(_view(), activity=transferring),
        observations_fresh=True,
        keeps_min_active_gpus=True,
        leaves_routable=True,
    )

    blocked_sync = replace(_view(), sync_healthy=False)
    assert not candidate(
        window,
        blocked_sync,
        observations_fresh=True,
        keeps_min_active_gpus=True,
        leaves_routable=True,
    )

    unsettled = replace(_view(), attempts_settled=False)
    assert not candidate(
        window,
        unsettled,
        observations_fresh=True,
        keeps_min_active_gpus=True,
        leaves_routable=True,
    )


def test_selector_emits_full_idle_candidate_with_lb_source_seq():
    result = select_idle_candidates(
        _window(),
        [_view("r1", gpu_count=4), _view("r2", gpu_count=4)],
        source_seq=7,
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
        source_seq=1,
        observations_fresh=True,
        min_active_gpus=8,
        current_active_gpus=8,
        routable_count=2,
    ) == ()
    assert select_idle_candidates(
        _window(),
        [_view("r1", gpu_count=1)],
        source_seq=2,
        observations_fresh=True,
        min_active_gpus=0,
        current_active_gpus=1,
        routable_count=1,
    ) == ()


def test_source_seq_is_external_to_production_window():
    window = _window(epoch=4, revision=12)
    first = select_idle_candidates(
        window,
        [_view()],
        source_seq=10,
        observations_fresh=True,
        min_active_gpus=0,
        current_active_gpus=2,
        routable_count=2,
    )[0]
    second = select_idle_candidates(
        window,
        [_view()],
        source_seq=11,
        observations_fresh=True,
        min_active_gpus=0,
        current_active_gpus=2,
        routable_count=2,
    )[0]
    assert window.epoch == 4 and window.revision == 12
    assert first.source_seq == 10
    assert second.source_seq == 11
    assert first.evidence_digest != second.evidence_digest
