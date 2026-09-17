"""Idle-bubble sensing aligned with the simplified fusion design §6.6.

ProductionWindow is the Rollouter-owned task production fact. ServerActivity is
the per-runtime engine observation. LB combines those facts with Manager/LB
revisions and capacity/sync checks to freeze an IdleCandidate. ``source_seq``
is intentionally supplied by LB and is not a ProductionWindow field.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .contracts import IdleCandidate, ReplicaKey


def _digest(*parts: object) -> str:
    payload = "|".join(repr(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class WindowState(str, Enum):
    OPEN = "OPEN"
    CLOSED_STALENESS = "CLOSED_STALENESS"
    CLOSED_BACKPRESSURE = "CLOSED_BACKPRESSURE"
    EXHAUSTED = "EXHAUSTED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ProductionWindow:
    """Rollouter-owned versioned production facts for one task session."""

    task_session: str
    epoch: int
    revision: int
    state: WindowState
    eligible_pending: int | None
    held_samples: int | None
    active_samples: int
    output_queue_size: int
    max_queue_size: int
    producer_exhausted: bool
    policy_refresh_inflight: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", WindowState(self.state))
        if not self.task_session:
            raise ValueError("task_session must be nonempty")
        if self.epoch < 0 or self.revision < 0:
            raise ValueError("production epoch/revision must be nonnegative")
        for value in (self.eligible_pending, self.held_samples):
            if value is not None and value < 0:
                raise ValueError("P/H must be nonnegative when observed")
        for value in (
            self.active_samples,
            self.output_queue_size,
            self.max_queue_size,
        ):
            if value < 0:
                raise ValueError("production counters must be nonnegative")
        if self.output_queue_size > self.max_queue_size:
            raise ValueError("output_queue_size cannot exceed max_queue_size")

    @property
    def idle(self) -> bool:
        """Whether the task window supports considering an idle replica.

        The state itself is produced by the native Rollouter logic; this helper
        does not reimplement staleness/backpressure thresholds. P/H must be
        explicitly observed as zero and a policy refresh cannot be in flight.
        """
        return (
            self.state
            in {
                WindowState.CLOSED_STALENESS,
                WindowState.CLOSED_BACKPRESSURE,
                WindowState.EXHAUSTED,
            }
            and self.eligible_pending == 0
            and self.held_samples == 0
            and not self.policy_refresh_inflight
        )

    @property
    def idle_reason(self) -> str | None:
        return self.state.value if self.idle else None


@dataclass(frozen=True)
class ServerActivity:
    """All-backend activity observation for one exact runtime."""

    key: ReplicaKey
    engine_seq: int
    observed_age_ms: int
    admitting: int | None
    queued: int | None
    running: int | None
    pending_admissions: int | None
    transfer_inflight: bool
    all_backends_observed: bool

    def __post_init__(self) -> None:
        if self.engine_seq < 0 or self.observed_age_ms < 0:
            raise ValueError("engine_seq and observed_age_ms must be nonnegative")
        for value in (
            self.admitting,
            self.queued,
            self.running,
            self.pending_admissions,
        ):
            if value is not None and value < 0:
                raise ValueError("activity counters must be nonnegative when observed")

    @property
    def quiet(self) -> bool:
        return (
            self.all_backends_observed
            and self.admitting == 0
            and self.queued == 0
            and self.running == 0
            and self.pending_admissions == 0
            and not self.transfer_inflight
        )

    @property
    def digest(self) -> str:
        """Stable digest of the complete aggregate engine observation."""
        return _digest(
            "SERVER_ACTIVITY",
            self.key,
            self.engine_seq,
            self.observed_age_ms,
            self.admitting,
            self.queued,
            self.running,
            self.pending_admissions,
            self.transfer_inflight,
            self.all_backends_observed,
        )


@dataclass(frozen=True)
class ReplicaView:
    """LB-local aggregate of the cross-owner facts needed for donation sensing.

    This is an implementation helper, not a public wire type. The booleans are
    already-validated facts from Manager/capacity/sync/attempt owners; they are
    kept outside ServerActivity so that the public ServerActivity shape remains
    exactly the design contract.
    """

    activity: ServerActivity
    manager_revision: int
    lb_revision: int
    placement_digest: str
    stable_idle_ms: int
    gpu_count: int = 1
    active_native: bool = True
    sync_healthy: bool = True
    attempts_settled: bool = True
    lifecycle_conflict: bool = False

    def __post_init__(self) -> None:
        for value in (
            self.manager_revision,
            self.lb_revision,
            self.stable_idle_ms,
        ):
            if value < 0:
                raise ValueError("candidate revisions/timing must be nonnegative")
        if self.gpu_count <= 0:
            raise ValueError("gpu_count must be positive")
        if not self.placement_digest:
            raise ValueError("placement_digest must be nonempty")

    @property
    def eligible(self) -> bool:
        return (
            self.active_native
            and self.sync_healthy
            and self.attempts_settled
            and not self.lifecycle_conflict
        )


def candidate(
    window: ProductionWindow,
    view: ReplicaView,
    *,
    observations_fresh: bool,
    keeps_min_active_gpus: bool,
    leaves_routable: bool,
) -> bool:
    activity = view.activity
    if activity.key.task_session != window.task_session:
        return False
    if not observations_fresh or not window.idle:
        return False
    if not activity.quiet or not view.eligible:
        return False
    if not keeps_min_active_gpus or not leaves_routable:
        return False
    return True


def select_idle_candidates(
    window: ProductionWindow,
    replicas: Sequence[ReplicaView],
    *,
    source_seq: int,
    observations_fresh: bool,
    min_active_gpus: int,
    current_active_gpus: int,
    routable_count: int,
) -> tuple[IdleCandidate, ...]:
    """Freeze one complete candidate set using the LB-owned observation seq."""
    if source_seq < 0:
        raise ValueError("source_seq must be nonnegative")
    if min_active_gpus < 0 or current_active_gpus < 0 or routable_count < 0:
        raise ValueError("capacity counts must be nonnegative")

    result: list[IdleCandidate] = []
    reason = window.idle_reason
    for view in replicas:
        keeps_min = current_active_gpus - view.gpu_count >= min_active_gpus
        leaves_routable = routable_count - 1 >= 1
        if not candidate(
            window,
            view,
            observations_fresh=observations_fresh,
            keeps_min_active_gpus=keeps_min,
            leaves_routable=leaves_routable,
        ):
            continue

        activity = view.activity
        engine_digest = activity.digest
        evidence_digest = _digest(
            activity.key,
            window.task_session,
            window.epoch,
            window.revision,
            window.state,
            window.eligible_pending,
            window.held_samples,
            window.active_samples,
            window.output_queue_size,
            window.max_queue_size,
            window.producer_exhausted,
            window.policy_refresh_inflight,
            source_seq,
            view.manager_revision,
            view.lb_revision,
            activity.engine_seq,
            engine_digest,
            reason,
            view.stable_idle_ms,
            activity.observed_age_ms,
            view.gpu_count,
            view.placement_digest,
        )
        result.append(
            IdleCandidate(
                key=activity.key,
                production_epoch=window.epoch,
                source_seq=source_seq,
                manager_revision=view.manager_revision,
                lb_revision=view.lb_revision,
                engine_digest=engine_digest,
                reason=reason,
                stable_idle_ms=view.stable_idle_ms,
                observed_age_ms=activity.observed_age_ms,
                gpu_count=view.gpu_count,
                evidence_digest=evidence_digest,
                placement_digest=view.placement_digest,
            )
        )
    return tuple(result)
