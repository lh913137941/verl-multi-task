"""Checkpoint Engine owner for the simplified E view.

E membership is keyed by ReplicaKey and stores only the receiver projection
needed for normal parameter synchronization. Every committed mutation returns
an idempotent CommitReceipt. Target-only bootstrap remains unavailable until
the native transfer backend can return real WeightEvidence.
"""

from __future__ import annotations

import hashlib

from verl.checkpoint_engine.base import CheckpointEngineManager

from multi_task_scheduler.orchestration.contracts import ServiceAction
from multi_task_scheduler.orchestration.effective_replica import EffectiveReplicaEntry
from multi_task_scheduler.orchestration.receipts import (
    CommitOwner,
    CommitReceipt,
    EvidenceHeader,
)


def _digest(*parts: object) -> str:
    data = "|".join(repr(part) for part in parts).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class MultiTaskCheckpointEngineManager(CheckpointEngineManager):
    """Trainer-owned CE manager; native whole-set synchronization stays inherited."""

    def _effective_replicas(self) -> dict:
        if not hasattr(self, "_effective_replica_map"):
            self._effective_replica_map = {}
        return self._effective_replica_map

    def _ensure_effective_revision(self) -> int:
        if not hasattr(self, "_effective_replica_revision"):
            self._effective_replica_revision = 0
        return self._effective_replica_revision

    def _commit_cache(self) -> dict:
        if not hasattr(self, "_effective_commit_receipts"):
            self._effective_commit_receipts = {}
        return self._effective_commit_receipts

    def _active_transfers(self) -> set:
        """Return CE-owned runtime keys with a parameter transfer still in flight.

        The verified native transfer backend must maintain this set around actual
        send/load work. Until that backend is wired, REMOVE is conservative:
        any key present here is fenced from leaving E.
        """
        if not hasattr(self, "_effective_transfer_inflight"):
            self._effective_transfer_inflight = set()
        return self._effective_transfer_inflight

    @property
    def effective_replicas(self) -> dict:
        return self._effective_replicas()

    @property
    def effective_revision(self) -> int:
        return self._ensure_effective_revision()

    def add_effective(self, ctx, prepared, weight) -> CommitReceipt:
        """Commit E ADD idempotently after verified WeightEvidence."""
        if prepared.ctx != ctx or weight.header.ctx != ctx:
            raise ValueError("CE ADD evidence belongs to a different operation")
        if prepared.key != weight.header.key:
            raise ValueError("CE ADD runtime identity mismatch")
        expected_receivers = {receiver.receiver_id for receiver in prepared.receivers}
        if set(weight.receiver_versions) != expected_receivers:
            raise ValueError("WeightEvidence does not cover prepared receivers")
        if any(version != weight.version for version in weight.receiver_versions.values()):
            raise ValueError("WeightEvidence receiver versions must match loaded version")

        cache_key = (ctx.identity, prepared.key, ServiceAction.ADD)
        cached = self._commit_cache().get(cache_key)
        if cached is not None:
            return cached

        members = self._effective_replicas()
        known_signatures = {entry.model_signature for entry in members.values()}
        if known_signatures and prepared.model_signature not in known_signatures:
            raise ValueError("prepared target model signature conflicts with effective CE set")

        entry = EffectiveReplicaEntry(
            key=prepared.key,
            receivers=prepared.receivers,
            loaded_version=weight.version,
            model_signature=prepared.model_signature,
            membership_operation_id=ctx.operation_id,
        )
        existing = members.get(prepared.key)
        if existing is not None and existing != entry:
            raise ValueError("ReplicaKey already has conflicting CE membership")
        if existing is None:
            members[prepared.key] = entry
            self._effective_replica_revision = self._ensure_effective_revision() + 1

        revision = self._ensure_effective_revision()
        digest = _digest(
            "CE",
            "ADD",
            ctx.identity,
            prepared.key,
            revision,
            weight.header.digest,
        )
        receipt = CommitReceipt(
            header=EvidenceHeader(
                ctx=ctx,
                key=prepared.key,
                phase_revision=weight.header.phase_revision + 1,
                digest=digest,
            ),
            owner=CommitOwner.CE,
            action=ServiceAction.ADD,
            revision=revision,
            version=weight.version,
            route_epoch=None,
        )
        self._commit_cache()[cache_key] = receipt
        return receipt

    def remove_effective(self, ctx, key, exit) -> CommitReceipt:
        """Commit E REMOVE idempotently after exit and transfer-quiescence proof."""
        if exit.header.ctx != ctx or exit.header.key != key:
            raise ValueError("CE REMOVE exit evidence identity mismatch")
        cache_key = (ctx.identity, key, ServiceAction.REMOVE)
        cached = self._commit_cache().get(cache_key)
        if cached is not None:
            return cached
        if key in self._active_transfers():
            raise ValueError("cannot remove CE member while parameter transfer is in flight")

        members = self._effective_replicas()
        if key in members:
            members.pop(key)
            self._effective_replica_revision = self._ensure_effective_revision() + 1

        revision = self._ensure_effective_revision()
        digest = _digest("CE", "REMOVE", ctx.identity, key, revision, exit.header.digest)
        receipt = CommitReceipt(
            header=EvidenceHeader(
                ctx=ctx,
                key=key,
                phase_revision=exit.header.phase_revision + 1,
                digest=digest,
            ),
            owner=CommitOwner.CE,
            action=ServiceAction.REMOVE,
            revision=revision,
            version=None,
            route_epoch=None,
        )
        self._commit_cache()[cache_key] = receipt
        return receipt

    def bootstrap_target(self, ctx, prepared, snapshot):
        """Load one immutable published snapshot and return real WeightEvidence."""
        if prepared.ctx != ctx:
            raise ValueError("PreparedReplica belongs to a different operation")
        if not snapshot.snapshot_id or not snapshot.manifest_digest:
            raise ValueError("bootstrap_target requires immutable published snapshot evidence")
        if snapshot.model_signature != prepared.model_signature:
            raise ValueError("snapshot model signature does not match prepared target")
        raise NotImplementedError(
            "CE.bootstrap_target requires verified target-only native backend"
        )
