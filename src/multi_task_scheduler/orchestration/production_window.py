"""Idle-bubble sensing for the current simplified fusion contract.

ProductionWindow is the compact Rollouter-owned public fact. Detailed native
queue/producer inputs stay inside Rollouter. ServerActivity is the Server-owned
request observation, while CE transfer quiescence is supplied separately by the
LB-local ReplicaView used to freeze an IdleCandidate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .contracts import IdleCandidate, ReplicaKey


def _digest(domain: str, *parts: object) -> str:
    payload = "|".join((domain, *(repr(part) for part in parts))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class WindowState(str, Enum):
    OPEN = "OPEN"
    CLOSED_STALENESS = "CLOSED_STALENESS"
    CLOSED_BACKPRESSURE = "CLOSED_BACKPRESSURE"
    EXHAUSTED = "EXHAUSTED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ProductionWindow:
    """Compact public production fact; native algorithm inputs are not copied."""

    task_session: str
    epoch: int
    revision: int
    state: WindowState
    eligible_pending: int | None
    held_samples: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", WindowState(self.state))
        if not self.task_session:
            raise ValueError("task_session must be nonempty")
        if self.epoch < 0 or self.revision < 0:
            raise ValueError("production epoch/revision must be nonnegative")
        for value in (self.eligible_pending, self.held_samples):
            if value is not None and value < 0:
                raise ValueError("P/H must be nonnegative when observed")

    @property
    def idle(self) -> bool:
        return (
            self.state
            in {
                WindowState.CLOSED_STALENESS,
                WindowState.CLOSED_BACKPRESSURE,
                WindowState.EXHAUSTED,
            }
            and self.eligible_pending == 0
            and self.held_samples == 0
        )

    @property
    def idle_reason(self) -> str | None:
        return self.state.value if self.idle else None


@dataclass(frozen=True)
class ServerActivity:
    """All-backend request activity observation for one exact runtime."""

    key: ReplicaKey
    engine_seq: int
    observed_age_ms: int
    admitting: int | None
    queued: int | None
    running: int | None
    pending_admissions: int | None
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
        )


@dataclass(frozen=True)
class ReplicaView:
    """LB-local owner facts needed for donation sensing; not a wire type."""

    activity: ServerActivity
    manager_revision: int
    lb_revision: int
    placement_digest: str
    stable_idle_ms: int
    gpu_count: int = 1
    active_native: bool = True
    sync_healthy: bool = True
    transfer_quiet: bool = True
    attempts_settled: bool = True
    lifecycle_conflict: bool = False

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (self.manager_revision, self.lb_revision, self.stable_idle_ms)
        ):
            raise ValueError("candidate revisions/timing must be nonnegative")
        if self.gpu_count <= 0:
            raise ValueError("gpu_count must be positive")
        if not self.placement_digest:
            raise ValueError("placement_digest must be nonempty")


def select_idle_candidates(
    window: ProductionWindow,
    replicas: Sequence[ReplicaView],
    *,
    observations_fresh: bool,
    min_active_gpus: int,
    current_active_gpus: int,
    routable_count: int,
) -> tuple[IdleCandidate, ...]:
    """Freeze one complete candidate set from real owner observations."""
    if min_active_gpus < 0 or current_active_gpus < 0 or routable_count < 0:
        raise ValueError("capacity counts must be nonnegative")
    if not observations_fresh or not window.idle or routable_count <= 1:
        return ()

    result: list[IdleCandidate] = []
    reason = window.idle_reason
    for view in replicas:
        activity = view.activity
        if activity.key.task_session != window.task_session or not activity.quiet:
            continue
        if (
            not view.active_native
            or not view.sync_healthy
            or not view.transfer_quiet
            or not view.attempts_settled
            or view.lifecycle_conflict
        ):
            continue
        if current_active_gpus - view.gpu_count < min_active_gpus:
            continue

        evidence_digest = _digest(
            "IDLE_CANDIDATE_V1",
            activity.key.task_session,
            activity.key.replica_id,
            activity.key.runtime_epoch,
            window.epoch,
            window.revision,
            window.state.value,
            window.eligible_pending,
            window.held_samples,
            view.manager_revision,
            view.lb_revision,
            activity.engine_seq,
            activity.observed_age_ms,
            activity.admitting,
            activity.queued,
            activity.running,
            activity.pending_admissions,
            activity.all_backends_observed,
            view.stable_idle_ms,
            view.gpu_count,
            view.placement_digest,
            reason,
        )
        result.append(
            IdleCandidate(
                key=activity.key,
                production_epoch=window.epoch,
                reason=reason,
                evidence_digest=evidence_digest,
            )
        )
    return tuple(result)
