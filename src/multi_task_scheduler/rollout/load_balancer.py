"""Native routing subclass plus the simplified R-view commit protocol.

Dynamic service commits use full ReplicaKey identity and typed RouteEntry state.
LB owns IdleCandidateReport sequencing; a candidate itself deliberately carries
no report-level source_seq/TTL or detailed owner observations.
"""

from __future__ import annotations

import hashlib
import uuid

import ray
from verl.workers.rollout.llm_server import DEFAULT_ROUTING_CACHE_SIZE, GlobalRequestLoadBalancer

from multi_task_scheduler.orchestration.contracts import (
    IdleCandidateReport,
    RouteEntry,
    RouteState,
    ServiceAction,
)
from multi_task_scheduler.orchestration.production_window import (
    ProductionWindow,
    select_idle_candidates,
)
from multi_task_scheduler.orchestration.receipts import (
    Ack,
    CommitOwner,
    CommitReceipt,
    DrainTicket,
    EvidenceHeader,
)


def _digest(*parts: object) -> str:
    data = "|".join(repr(part) for part in parts).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class MultiTaskGlobalRequestLoadBalancer(GlobalRequestLoadBalancer):
    """LB owns R, its attempt ledger, and idle-candidate report sequencing."""

    def __init__(
        self,
        servers,
        max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
        full_determinism=False,
        *,
        group_scheduler=None,
    ):
        self.group_scheduler = group_scheduler
        super().__init__(servers, max_cache_size=max_cache_size, full_determinism=full_determinism)
        self.lb_revision = 0
        self.sync_epoch = 0
        self.routes: dict[object, RouteEntry] = {}
        self.draining = {}
        self.attempts = {}
        self._commit_receipts = {}
        self._idle_source_seq = -1
        self._last_idle_report = None

    @property
    def last_idle_report(self) -> IdleCandidateReport | None:
        return self._last_idle_report

    def _next_route_epoch(self, key) -> int:
        existing = self.routes.get(key)
        return 1 if existing is None else existing.replica_route_epoch + 1

    def build_idle_candidate_report(
        self,
        window: ProductionWindow,
        replicas,
        *,
        gs_epoch: str,
        lb_session: str,
        valid_for_ms: int,
        observations_fresh: bool,
        min_active_gpus: int,
        current_active_gpus: int,
        routable_count: int,
    ) -> IdleCandidateReport:
        """Freeze one new complete candidate observation with an LB-owned seq."""
        if not isinstance(window, ProductionWindow):
            raise TypeError("idle reporting requires ProductionWindow")
        source_seq = self._idle_source_seq + 1
        candidates = select_idle_candidates(
            window,
            replicas,
            observations_fresh=observations_fresh,
            min_active_gpus=min_active_gpus,
            current_active_gpus=current_active_gpus,
            routable_count=routable_count,
        )
        report = IdleCandidateReport(
            task_session=window.task_session,
            gs_epoch=gs_epoch,
            lb_session=lb_session,
            source_seq=source_seq,
            production_revision=window.revision,
            valid_for_ms=valid_for_ms,
            candidates=candidates,
        )
        self._idle_source_seq = source_seq
        self._last_idle_report = report
        return report

    def report_idle_candidates(self, report: IdleCandidateReport) -> Ack:
        """Send one already-frozen report to GS; retries reuse the same object."""
        if not isinstance(report, IdleCandidateReport):
            raise TypeError("report_idle_candidates requires IdleCandidateReport")
        if self.group_scheduler is None:
            raise RuntimeError("GroupScheduler handle is required for idle reporting")
        if self._last_idle_report is None or report != self._last_idle_report:
            raise ValueError("LB may report only its latest frozen candidate set")
        ack = ray.get(
            self.group_scheduler.report_idle_candidates.remote(report),
            timeout=30,
        )
        if not isinstance(ack, Ack):
            raise TypeError("GroupScheduler returned a non-Ack idle-report response")
        return ack

    def close_for_exit(
        self,
        ctx,
        key,
        *,
        phase_revision: int,
        server_admission_epoch: int,
    ) -> DrainTicket:
        """Move one ROUTABLE entry to DRAINING and fence its old route tickets."""
        if key.task_session != ctx.task_session:
            raise ValueError("drain target does not belong to operation task_session")
        existing_ticket = self.draining.get((ctx.identity, key))
        if existing_ticket is not None:
            return existing_ticket

        route = self.routes.get(key)
        if route is None:
            raise ValueError("drain target has no RouteEntry")
        if route.state is not RouteState.ROUTABLE:
            raise ValueError(f"drain target route is {route.state.value}, not ROUTABLE")

        route_epoch = self._next_route_epoch(key)
        self.lb_revision += 1
        self.routes[key] = RouteEntry(
            key=key,
            head_server=route.head_server,
            state=RouteState.DRAINING,
            replica_route_epoch=route_epoch,
            sync_epoch=route.sync_epoch,
            serving_version=route.serving_version,
            commit_operation_id=ctx.operation_id,
        )
        drain_id = f"drain-{uuid.uuid4().hex}"
        ticket = DrainTicket(
            header=EvidenceHeader(
                ctx=ctx,
                key=key,
                phase_revision=phase_revision,
                digest=_digest("DRAIN", ctx.identity, key, drain_id, route_epoch),
            ),
            drain_id=drain_id,
            route_epoch=route_epoch,
            server_admission_epoch=server_admission_epoch,
        )
        self.draining[(ctx.identity, key)] = ticket
        return ticket

    def commit_routable(self, ctx, prepared, weight, ce_commit) -> CommitReceipt:
        """Commit R=ROUTABLE after valid weight and CE ADD evidence."""
        key = prepared.key
        if key.task_session != ctx.task_session or weight.header.ctx != ctx or weight.header.key != key:
            raise ValueError("LB ADD evidence identity mismatch")
        if (
            ce_commit.header.ctx != ctx
            or ce_commit.header.key != key
            or ce_commit.owner is not CommitOwner.CE
            or ce_commit.action is not ServiceAction.ADD
            or ce_commit.version != weight.version
        ):
            raise ValueError("LB ADD requires matching CE ADD commit")
        cache_key = (ctx.identity, key, ServiceAction.ADD)
        cached = self._commit_receipts.get(cache_key)
        if cached is not None:
            return cached

        existing = self.routes.get(key)
        if existing is not None and existing.state in {
            RouteState.ROUTABLE,
            RouteState.DRAINING,
            RouteState.QUARANTINED,
        }:
            raise ValueError(
                f"cannot commit route from existing state {existing.state.value}"
            )

        route_epoch = self._next_route_epoch(key)
        self.lb_revision += 1
        self.routes[key] = RouteEntry(
            key=key,
            head_server=prepared.head_server,
            state=RouteState.ROUTABLE,
            replica_route_epoch=route_epoch,
            sync_epoch=self.sync_epoch,
            serving_version=weight.version,
            commit_operation_id=ctx.operation_id,
        )
        self.draining.pop((ctx.identity, key), None)
        digest = _digest(
            "LB",
            "ADD",
            ctx.identity,
            key,
            self.lb_revision,
            route_epoch,
            weight.header.digest,
            ce_commit.header.digest,
        )
        receipt = CommitReceipt(
            header=EvidenceHeader(
                ctx=ctx,
                key=key,
                phase_revision=max(
                    weight.header.phase_revision, ce_commit.header.phase_revision
                ) + 1,
                digest=digest,
            ),
            owner=CommitOwner.LB,
            action=ServiceAction.ADD,
            revision=self.lb_revision,
            version=weight.version,
            route_epoch=route_epoch,
        )
        self._commit_receipts[cache_key] = receipt
        return receipt

    def _has_unsettled_attempts(self, key) -> bool:
        attempts = self.attempts.get(key)
        if attempts is None:
            return False
        if isinstance(attempts, dict):
            terminal = {"TERMINAL", "RELEASED"}
            for value in attempts.values():
                state = getattr(value, "state", value)
                state = getattr(state, "value", state)
                if state not in terminal:
                    return True
            return False
        return bool(attempts)

    def finish_remove(self, ctx, proof, ce_commit) -> CommitReceipt:
        """Commit R=REMOVED only for this operation's active, settled drain."""
        key = proof.header.key
        if proof.header.ctx != ctx:
            raise ValueError("LB REMOVE exit evidence belongs to another operation")
        if (
            ce_commit.header.ctx != ctx
            or ce_commit.header.key != key
            or ce_commit.owner is not CommitOwner.CE
            or ce_commit.action is not ServiceAction.REMOVE
        ):
            raise ValueError("LB REMOVE requires matching CE REMOVE commit")

        cache_key = (ctx.identity, key, ServiceAction.REMOVE)
        cached = self._commit_receipts.get(cache_key)
        if cached is not None:
            return cached

        ticket = self.draining.get((ctx.identity, key))
        if ticket is None:
            raise ValueError("LB REMOVE requires the active drain ticket")
        if proof.drain_id != ticket.drain_id:
            raise ValueError("ExitEvidence drain_id does not match active drain ticket")
        if proof.header.phase_revision < ticket.header.phase_revision:
            raise ValueError("ExitEvidence predates the active drain ticket")
        if self._has_unsettled_attempts(key):
            raise ValueError("cannot remove route while attempts remain unsettled")

        route = self.routes.get(key)
        if route is None or route.state is not RouteState.DRAINING:
            raise ValueError("LB REMOVE requires a DRAINING RouteEntry")
        if route.replica_route_epoch != ticket.route_epoch:
            raise ValueError("active RouteEntry epoch does not match drain ticket")

        route_epoch = self._next_route_epoch(key)
        self.lb_revision += 1
        self.routes[key] = RouteEntry(
            key=key,
            head_server=route.head_server,
            state=RouteState.REMOVED,
            replica_route_epoch=route_epoch,
            sync_epoch=route.sync_epoch,
            serving_version=route.serving_version,
            commit_operation_id=ctx.operation_id,
        )
        self.draining.pop((ctx.identity, key), None)
        digest = _digest(
            "LB",
            "REMOVE",
            ctx.identity,
            key,
            self.lb_revision,
            route_epoch,
            proof.header.digest,
            ce_commit.header.digest,
        )
        receipt = CommitReceipt(
            header=EvidenceHeader(
                ctx=ctx,
                key=key,
                phase_revision=max(
                    proof.header.phase_revision, ce_commit.header.phase_revision
                ) + 1,
                digest=digest,
            ),
            owner=CommitOwner.LB,
            action=ServiceAction.REMOVE,
            revision=self.lb_revision,
            version=None,
            route_epoch=route_epoch,
        )
        self._commit_receipts[cache_key] = receipt
        return receipt

    def set_sync_barrier(self, token, direction, version=None):
        """Canonical PAUSE/RESUME entry; requires real Server barrier wiring."""
        raise NotImplementedError(
            "LB.set_sync_barrier requires verified task-wide admission barrier wiring"
        )

    def query_phase(self, ctx, phase):
        """Owner-side query must be backed by a real phase journal before use."""
        raise NotImplementedError("LB query_phase requires owner-side journal wiring")
