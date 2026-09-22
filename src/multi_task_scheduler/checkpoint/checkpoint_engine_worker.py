"""Native CheckpointEngine receiver extension for target-only bootstrap."""
from verl.checkpoint_engine.base import CheckpointEngineWorker

class MultiTaskCheckpointEngineWorker(CheckpointEngineWorker):
    def replay_current_weights(self, transfer_id: str):
        raise NotImplementedError("single-target current-weight replay requires verified native backend")
