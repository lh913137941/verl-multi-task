"""Native receiver extension; ServerAdapter and transport remain native.

The worker is the receiver side of both native sync (whole effective set) and
the target-only bootstrap. ``replay_published`` is the target-only replay of a
``PublishedWeightSnapshot`` and stays an explicit failure until verified; the
native ``update_weights`` receiver path is inherited unchanged.
"""

from verl.checkpoint_engine.base import CheckpointEngineWorker


class MultiTaskCheckpointEngineWorker(CheckpointEngineWorker):
    """RayWorkerGroup creates this selected class; native method metadata is inherited."""

    def replay_published(self, snapshot, transfer_id: str) -> object:
        """Replay one pinned published snapshot onto this receiver only.

        Communication groups are isolated by (task_session, transfer_id); this
        is an interface boundary, not a GPU capability yet.
        """
        raise NotImplementedError(
            "single-target published-weight replay requires verified native backend"
        )
