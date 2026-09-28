"""Native receiver extension; ServerAdapter and transport remain native."""

from verl.checkpoint_engine.base import CheckpointEngineWorker


class MultiTaskCheckpointEngineWorker(CheckpointEngineWorker):
    """RayWorkerGroup creates this selected class; native method metadata is inherited."""
