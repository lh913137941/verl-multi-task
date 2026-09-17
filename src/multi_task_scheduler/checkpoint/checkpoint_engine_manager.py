"""Native checkpoint coordinator extension with an effective-membership overlay.

E is owned only here.  Its revision is a monotonic commit revision, not the
current member count: removals must never make an old revision look newer.
Target-only bootstrap remains an explicit failure until the native NCCL backend
is verified.
"""

from verl.checkpoint_engine.base import CheckpointEngineManager


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

    @property
    def effective_replicas(self) -> dict:
        return self._effective_replicas()

    @property
    def effective_revision(self) -> int:
        return self._ensure_effective_revision()

    def add_effective_replica(self, ctx, prepared) -> int:
        """Commit an E member idempotently after verified target bootstrap."""
        members = self._effective_replicas()
        current = members.get(prepared.replica_id)
        if current == prepared:
            return self._ensure_effective_revision()
        members[prepared.replica_id] = prepared
        self._effective_replica_revision = self._ensure_effective_revision() + 1
        return self._effective_replica_revision

    def remove_effective_replica(self, ctx, replica_id) -> int:
        """Remove an E member idempotently; the commit revision only increases."""
        members = self._effective_replicas()
        if replica_id not in members:
            return self._ensure_effective_revision()
        members.pop(replica_id)
        self._effective_replica_revision = self._ensure_effective_revision() + 1
        return self._effective_replica_revision

    def bootstrap_target(self, prepared, snapshot) -> object:
        """Transfer one immutable published snapshot onto one hidden target."""
        if not getattr(snapshot, "snapshot_id", None) or not getattr(snapshot, "manifest_digest", None):
            raise ValueError("bootstrap_target requires immutable published snapshot evidence")
        raise NotImplementedError(
            "target-only NCCL bootstrap requires verified native backend"
        )
