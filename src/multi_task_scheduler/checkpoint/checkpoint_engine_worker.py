"""Native CheckpointEngine receiver extension for target-only bootstrap."""

from __future__ import annotations

from verl.checkpoint_engine.base import CheckpointEngineWorker


class MultiTaskCheckpointEngineWorker(CheckpointEngineWorker):
    """Keep transfer replay facts local to the checkpoint owner.

    Replay readiness is reported only once a native backend can prove it moved
    the weights; until then both hooks fail explicitly rather than synthesising
    a READY record.
    """

    def replay_current_weights(self, transfer_id: str):
        if not isinstance(transfer_id, str) or not transfer_id:
            raise ValueError("transfer_id must be a nonempty string")
        raise NotImplementedError(
            "weight replay requires a verified native checkpoint transfer backend"
        )

    def replay_status(self, transfer_id: str):
        raise NotImplementedError(
            "replay status requires a verified native checkpoint transfer backend"
        )
