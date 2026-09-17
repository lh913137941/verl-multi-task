"""Idle-bubble sensing that emits canonical ``IdleCandidate`` evidence.

Missing observations are UNKNOWN, never zero. A candidate is emitted only when
production, Manager, LB and engine facts are all present, fresh and mutually
consistent. The result is directly suitable for ``IdleCandidateReport``; no
ID-only candidate wrapper exists.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from .contracts import IdleCandidate, ReplicaKey


def _digest(*parts: object) -> str:
    payload = "|".join(repr(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ReplicaObservation:
    """Versioned Server/LB activity facts for one exact runtime."""

    key: ReplicaKey
    engine_seq: int
    observed_age_ms: int
    engine_digest: str
    in_flight: int | None = None
    admitting: int | None = None
    queued: int | None = None
    running: int | None = None
    pending_admissions: int | None = None
    production_epoch: int | None = None
    all_backends_observed: bool = False
    is_active_native: bool = True
    transfer_inflight: bool = False
    concurrent_validation: bool = False
    concurrent_scaling: bool = False

    def __post_init__(self) -> None:
        if self.engine_seq < 0 or self.observed_age_ms < 0:
            raise ValueError("engine_seq and observed_age_ms must be nonnegative")
        if not self.engine_digest:
            raise ValueError("engine_digest must be nonempty")
        for value in (
            self.in_flight,
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
            and self.pending_admissions == 0
            and self.in_flight == 0
            and self.admitting == 0
            and self.queued == 0
            and self.running == 0
        )

    @property
    def scalable(self) -> bool:
        return (
            self.is_active_native
            and not self.transfer_inflight
            and not self.concurrent_validation
            and not self.concurrent_scaling
        )


@dataclass(frozen=True)
class ProductionWindow:
    """Versioned production-side facts used by the idle-candidate predicate."""

    production_epoch: int = 0
    source_seq: int = 0
    production_revision: int = 0
    eligible_pending: int | None = None
    held: int | None = None
    policy_refresh_inflight: bool = False
    closed_by_staleness: bool = False
    closed_by_backpressure: bool = False
    exhausted_this_round: bool = False

    def __post_init__(self) -> None:
        if self.production_epoch < 0 or self.source_seq < 0 or self.production_revision < 0:
            raise ValueError("production revisions must be nonnegative")
        for value in (self.eligible_pending, self.held):
            if value is not None and value < 0:
                raise ValueError("production counts must be nonnegative when observed")

    @property
    def idle(self) -> bool:
        if self.policy_refresh_inflight:
            return False
        if self.eligible_pending != 0 or self.held != 0:
            return False
        return (
            self.closed_by_staleness
            or self.closed_by_backpressure
            or self.exhausted_this_round
        )

    @property
    def idle_reason(self) -> str | None:
        """Return the canonical reason or None when the window is not idle."""
        if not self.idle:
            return None
        if self.closed_by_staleness:
            return "CLOSED_STALENESS"
        if self.closed_by_backpressure:
            return "CLOSED_BACKPRESSURE"
        if self.exhausted_this_round:
            return "EXHAUSTED"
        return None


@dataclass(frozen=True)
class ReplicaView:
    """Cross-owner facts required to freeze one ``IdleCandidate``."""

    observation: ReplicaObservation
    manager_revision: int
    lb_revision: int
    placement_digest: str
    stable_idle_ms: int
    gpu_count: int = 1

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


def candidate(
    window: ProductionWindow,
    view: ReplicaView,
    *,
    observations_fresh: bool,
    keeps_min_active_gpus: bool,
    leaves_routable: bool,
) -> bool:
    observation = view.observation
    if not observations_fresh:
        return False
    if observation.production_epoch != window.production_epoch:
        return False
    if not window.idle or window.idle_reason is None:
        return False
    if not observation.quiet or not observation.scalable:
        return False
    if not keeps_min_active_gpus or not leaves_routable:
        return False
    return True


def select_idle_candidates(
    window: ProductionWindow,
    replicas: Sequence[ReplicaView],
    *,
    observations_fresh: bool,
    min_active_gpus: int,
    current_active_gpus: int,
    routable_count: int,
) -> tuple[IdleCandidate, ...]:
    """Return a complete set of evidence-bearing idle candidates."""
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
        observation = view.observation
        evidence_digest = _digest(
            observation.key,
            window.production_epoch,
            window.source_seq,
            window.production_revision,
            view.manager_revision,
            view.lb_revision,
            observation.engine_seq,
            observation.engine_digest,
            reason,
            view.stable_idle_ms,
            observation.observed_age_ms,
            view.gpu_count,
            view.placement_digest,
        )
        result.append(
            IdleCandidate(
                key=observation.key,
                production_epoch=window.production_epoch,
                source_seq=window.source_seq,
                manager_revision=view.manager_revision,
                lb_revision=view.lb_revision,
                engine_digest=observation.engine_digest,
                reason=reason,
                stable_idle_ms=view.stable_idle_ms,
                observed_age_ms=observation.observed_age_ms,
                gpu_count=view.gpu_count,
                evidence_digest=evidence_digest,
                placement_digest=view.placement_digest,
            )
        )
    return tuple(result)


def advance_source_seq(window: ProductionWindow) -> ProductionWindow:
    """Invalidate previous candidate observations without changing their epoch."""
    return ProductionWindow(
        production_epoch=window.production_epoch,
        source_seq=window.source_seq + 1,
        production_revision=window.production_revision,
        eligible_pending=window.eligible_pending,
        held=window.held,
        policy_refresh_inflight=window.policy_refresh_inflight,
        closed_by_staleness=window.closed_by_staleness,
        closed_by_backpressure=window.closed_by_backpressure,
        exhausted_this_round=window.exhausted_this_round,
    )
