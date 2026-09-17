"""Native routing subclass plus the simplified R-view commit protocol.

Dynamic service commits use full ReplicaKey identity and return CommitReceipt.
No bool/int compatibility result is exposed. Native request selection remains
inherited until dynamic routing is wired end-to-end.
"""

from __future__ import annotations

import hashlib
import uuid

from verl.workers.rollout.llm_server import DEFAULT_ROUTING_CACHE_SIZE, GlobalRequestLoadBalancer

from multi_task_scheduler.orchestration.contracts import ServiceAction
from multi_task_scheduler.orchestration.receipts import (
    CommitOwner,
    CommitReceipt,
    DrainTicket,
    EvidenceHeader,
)


def _digest(*parts: object) -> str:
    data = "|".join(repr(part) for part in parts).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class MultiTaskGlobalRequestLoadBalancer(GlobalRequestLoadBalancer):
    """LB owns R and its attempt ledger; native selection stays inherited."""

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
        self.routing_epoch = 0
        self.lb_revision = 0
        self.routes = {}
        self.draining = {}
        self.attempts = {}
        self._commit_receipts = {}

    def close_for_exit(
        self,
        ctx,
        key,
        *,
        phase_revision: int,
        server_admission_epoch: int,
    ) -> DrainTicket:
        """Internal RO primitive: close one dynamic route and create a drain ticket."""
        if key.task_session != ctx.task_session:
            raise ValueError("drain target does not belong to operation task_session")
        existing = self.draining.get((ctx.identity, key))
        if existing is not None:
            return existing
        self.routing_epoch += 1
        self.lb_revision += 1
        self.routes.pop(key, None)
        drain_id = f"drain-{uuid.uuid4().hex}"
        ticket = DrainTicket(
            header=EvidenceHeader(
                ctx=ctx,
                key=key,
                phase_revision=phase_revision,
                digest=_digest("DRAIN", ctx.identity, key, drain_id, self.routing_epoch),
            ),
            drain_id=drain_id,
            route_epoch=self.routing_epoch,
            server_admission_epoch=server_admission_epoch,
        )
        self.draining[(ctx.identity, key)] = ticket
        return ticket

    def commit_routable(self, ctx, prepared, weight, ce_commit) -> CommitReceipt:
        """Commit R=ROUTABLE after valid weight and CE ADD evidence."""
        key = prepared.key
        if prepared.ctx != ctx or weight.header.ctx != ctx or weight.header.key != key:
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

        self.routing_epoch += 1
        self.lb_revision += 1
        self.routes[key] = {
            "head_server": prepared.head_server,
            "version": weight.version,
            "ce_revision": ce_commit.revision,
            "route_epoch": self.routing_epoch,
        }
        self.draining.pop((ctx.identity, key), None)
        digest = _digest(
            "LB",
            "ADD",
            ctx.identity,
            key,
            self.lb_revision,
            self.routing_epoch,
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
            route_epoch=self.routing_epoch,
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

        self.routing_epoch += 1
        self.lb_revision += 1
        self.routes.pop(key, None)
        self.draining.pop((ctx.identity, key), None)
        self.attempts.pop(key, None)
        digest = _digest(
            "LB",
            "REMOVE",
            ctx.identity,
            key,
            self.lb_revision,
            self.routing_epoch,
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
            route_epoch=self.routing_epoch,
        )
        self._commit_receipts[cache_key] = receipt
        return receipt

    def query_phase(self, ctx, phase):
        """Owner-side query must be backed by a real phase journal before use."""
        raise NotImplementedError("LB query_phase requires owner-side journal wiring")
