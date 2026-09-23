"""Native CheckpointEngine receiver extension for target-only bootstrap."""

from __future__ import annotations

from verl.checkpoint_engine.base import CheckpointEngineWorker


class MultiTaskCheckpointEngineWorker(CheckpointEngineWorker):
    """Keep transfer replay facts local to the checkpoint owner.

    The worker records replay readiness only. Actual backend-specific weight
    movement remains owned by the native VERL checkpoint implementation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._replay_records: dict[str, dict[str, str]] = {}

    def replay_current_weights(self, transfer_id: str):
        if not isinstance(transfer_id, str) or not transfer_id:
            raise ValueError("transfer_id must be a nonempty string")

        record = {
            "transfer_id": transfer_id,
            "status": "READY",
        }
        self._replay_records[transfer_id] = record
        return record.copy()

    def replay_status(self, transfer_id: str):
        return self._replay_records.get(transfer_id)
