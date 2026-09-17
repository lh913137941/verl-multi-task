"""Idle-bubble sensing (section 5.1) independent of any runtime.

``ProductionWindow`` carries the versioned facts a Rollouter merges into the
load balancer: the production ``epoch`` (logical generation), the ``source_seq``
revision that invalidates older candidate sets, and the two drain counts
``P`` (eligible pending) and ``H`` (held by a processor, not yet committed).

A replica is an idle *candidate* only when the window is genuinely idle, the
replica's four request counters are all zero, it is an ACTIVE native with no
concurrent transfer/validation/scaling, and removing it still leaves the task
with at least ``min_active_gpus`` and one routable replica.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass(frozen=True)
class ReplicaObservation:
    """Versioned request/engine facts for one replica (section 5.1)."""

    replica_id: str
    in_flight: int | None = None       # I; absent facts are UNKNOWN
    admitting: int | None = None       # A
    queued: int | None = None          # Q (engine queued)
    running: int | None = None         # R (engine running)
    pending_admissions: int | None = None
    production_epoch: int | None = None
    all_backends_observed: bool = False
    is_active_native: bool = True
    concurrent_transfer: bool = False
    concurrent_validation: bool = False
    concurrent_scaling: bool = False

    @property
    def quiet(self) -> bool:
        """I = A = Q = R = 0."""
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
        """ACTIVE native with no concurrent transfer/validation/scaling."""
        return (
            self.is_active_native
            and not self.concurrent_transfer
            and not self.concurrent_validation
            and not self.concurrent_scaling
        )


@dataclass(frozen=True)
class ProductionWindow:
    """Versioned production-side facts (section 5.1)."""

    production_epoch: int = 0
    source_seq: int = 0
    eligible_pending: int | None = None      # P; None is not zero
    held: int | None = None                  # H
    policy_refresh_inflight: bool = False
    closed_by_staleness: bool = False
    closed_by_backpressure: bool = False
    exhausted_this_round: bool = False

    @property
    def closed(self) -> bool:
        return self.closed_by_staleness or self.closed_by_backpressure

    @property
    def idle(self) -> bool:
        """Window closed or exhausted, with nothing pending or held."""
        if self.policy_refresh_inflight or self.eligible_pending != 0 or self.held != 0:
            return False
        return self.closed or self.exhausted_this_round


@dataclass(frozen=True)
class CandidateSet:
    """An LB-published candidate collection with a monotonic source_seq."""

    source_seq: int
    production_epoch: int
    candidate_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ReplicaView:
    observation: ReplicaObservation
    gpus: int = 1


def candidate(
    window: ProductionWindow,
    observation: ReplicaObservation,
    *,
    observations_fresh: bool,
    observations_epoch: int,
    keeps_min_active_gpus: bool,
    leaves_routable: bool,
) -> bool:
    """Single-replica candidate predicate (section 5.1 formula)."""
    if not observations_fresh:
        return False
    if observations_epoch != window.production_epoch:
        return False
    if not window.idle:
        return False
    if not observation.quiet:
        return False
    if not observation.scalable:
        return False
    if not keeps_min_active_gpus:
        return False
    if not leaves_routable:
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
) -> CandidateSet:
    """Build the candidate set with the aggregate removal constraints.

    ``current_active_gpus`` and ``routable_count`` are the pre-removal totals;
    each candidate must leave at least ``min_active_gpus`` active and at least
    one routable replica behind.
    """
    ids = []
    for view in replicas:
        obs = view.observation
        keeps_min = (current_active_gpus - view.gpus) >= min_active_gpus
        leaves_routable = (routable_count - 1) >= 1
        if candidate(
            window,
            obs,
            observations_fresh=observations_fresh,
            observations_epoch=obs.production_epoch,
            keeps_min_active_gpus=keeps_min,
            leaves_routable=leaves_routable,
        ):
            ids.append(obs.replica_id)
    return CandidateSet(
        source_seq=window.source_seq,
        production_epoch=window.production_epoch,
        candidate_ids=tuple(ids),
    )


def advance_source_seq(window: ProductionWindow) -> ProductionWindow:
    """Invalidate stale candidates by bumping the revision (section 5.1)."""
    return ProductionWindow(
        production_epoch=window.production_epoch,
        source_seq=window.source_seq + 1,
        eligible_pending=window.eligible_pending,
        held=window.held,
        policy_refresh_inflight=window.policy_refresh_inflight,
        closed_by_staleness=window.closed_by_staleness,
        closed_by_backpressure=window.closed_by_backpressure,
        exhausted_this_round=window.exhausted_this_round,
    )
