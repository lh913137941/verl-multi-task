"""Production-window sensing uses the compact 0918 public contracts."""

from dataclasses import replace

import pytest

from multi_task_scheduler.orchestration.contracts import ReplicaKey
from multi_task_scheduler.orchestration.production_window import (
    ProductionWindow,
    ReplicaView,
    ServerActivity,
    WindowState,
    select_idle_candidates,
)


def _window(*, task_session="s1", epoch=3, revision=11, state=WindowState.CLOSED_STALENESS, eligible_pending=0, held_samples=0):
    return ProductionWindow(task_session, epoch, revision, state, eligible_pending, held_samples)


def _activity(replica_id="r1", task_session="s1", *, engine_seq=5):
    return ServerActivity(
        key=ReplicaKey(task_session, replica_id, 0),
        engine_seq=engine_seq,
        observed_age_ms=10,
        admitting=0,
        queued=0,
        running=0,
        pending_admissions=0,
        all_backends_observed=True,
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


def _select(view, *, window=None, fresh=True, minimum=0, active=2, routable=2):
    return select_idle_candidates(
        window or _window(),
        [view],
        observations_fresh=fresh,
        min_active_gpus=minimum,
        current_active_gpus=active,
        routable_count=routable,
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


def test_window_public_shape_does_not_duplicate_native_algorithm_inputs():
    window = _window()
    assert not hasattr(window, "active_samples")
    assert not hasattr(window, "output_queue_size")
    assert not hasattr(window, "producer_exhausted")
    with pytest.raises(ValueError, match="task_session"):
        replace(window, task_session="")


@pytest.mark.parametrize(
    ("view", "fresh"),
    [
        (_view(), False),
        (_view(task_session="other"), True),
        (replace(_view(), activity=replace(_activity(), queued=None)), True),
        (replace(_view(), transfer_quiet=False), True),
        (replace(_view(), sync_healthy=False), True),
        (replace(_view(), attempts_settled=False), True),
    ],
)
def test_selector_rejects_stale_unknown_or_conflicting_owner_facts(view, fresh):
    assert _select(view, fresh=fresh) == ()


def test_selector_emits_lightweight_candidate_only():
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
    assert first.production_epoch == 3
    assert first.reason == "CLOSED_STALENESS"
    assert first.evidence_digest
    assert not hasattr(first, "source_seq")
    assert not hasattr(first, "manager_revision")
    assert not hasattr(first, "placement_digest")


def test_candidate_digest_changes_when_owner_observation_changes():
    first = _select(replace(_view(), activity=_activity(engine_seq=5)))[0]
    second = _select(replace(_view(), activity=_activity(engine_seq=6)))[0]
    assert first.evidence_digest != second.evidence_digest


def test_selector_respects_capacity_and_routable_lower_bounds():
    assert _select(_view(gpu_count=4), minimum=8, active=8) == ()
    assert _select(_view(), active=1, routable=1) == ()
