"""Native checkpoint coordinator extension with an effective-membership overlay.

The native ``update_weights`` stays inherited (section 6 publishes under the
Trainer's replica-sync gate, not here). This overlay adds the section 4.1
effective replica set ``E`` and the target-only bootstrap entry.

``bootstrap_target`` is an NCCL transfer to a *single* target replica and is an
explicit failure until the native backend is verified: it must never fall back
to a fake or to the whole-set native sync.
"""

from verl.checkpoint_engine.base import CheckpointEngineManager


class MultiTaskCheckpointEngineManager(CheckpointEngineManager):
    """Trainer creates this ordinary object; native weight synchronization stays."""

    def _effective_replicas(self) -> dict:
        # Lazy so the inherited native ``__init__`` is preserved unchanged.
        if not hasattr(self, "_effective_replica_map"):
            self._effective_replica_map = {}
        return self._effective_replica_map

    @property
    def effective_replicas(self) -> dict:
        return self._effective_replicas()

    def add_effective_replica(self, ctx, prepared) -> int:
        """Record a CE member only after a completed bootstrap (section 4.1)."""
        self._effective_replicas()[prepared.replica_id] = prepared
        return len(self._effective_replicas())

    def remove_effective_replica(self, ctx, replica_id) -> int:
        """Drop a CE member; requires the replica-sync gate (section 4.1)."""
        self._effective_replicas().pop(replica_id, None)
        return len(self._effective_replicas())

    def bootstrap_target(self, prepared, snapshot) -> object:
        """Transfer the pinned published snapshot onto one target replica.

        The verified sender -> target receiver group is built here; until the
        NCCL backend is verified this raises rather than pretending success.
        """
        raise NotImplementedError(
            "target-only NCCL bootstrap requires verified native backend"
        )
