"""Checkpoint Engine owner for the simplified E view."""

from __future__ import annotations

from verl.checkpoint_engine.base import CheckpointEngineManager

from multi_task_scheduler.orchestration.contracts import ReplicaKey


class MultiTaskCheckpointEngineManager(CheckpointEngineManager):
    """Keep only effective receiver membership as CE-owned mutable truth."""

    def _members(self) -> dict:
        if not hasattr(self, "_effective_replica_map"):
            self._effective_replica_map = {}
        return self._effective_replica_map

    @property
    def effective_replicas(self) -> dict:
        return self._members()

    def add_effective(self, key: ReplicaKey, replicas, *, loaded_version: int) -> None:
        if not isinstance(key, ReplicaKey):
            raise TypeError("key must be ReplicaKey")
        if type(loaded_version) is not int or loaded_version < 0:
            raise ValueError("loaded_version must be a nonnegative integer")

        replicas = tuple(replicas)
        if not replicas:
            raise ValueError("effective membership requires at least one replica")
        entry = (replicas, loaded_version)
        existing = self._members().get(key)
        if existing is not None and existing != entry:
            raise ValueError("ReplicaKey already has conflicting CE membership")

        for replica in replicas:
            if replica not in self.replicas:
                self.replicas.append(replica)
        self._members()[key] = entry

    def remove_effective(self, key: ReplicaKey) -> None:
        entry = self._members().pop(key, None)
        if entry is None:
            return
        replicas, _loaded_version = entry
        super().remove_replicas(list(replicas))

    def mark_all_loaded_version(self, loaded_version: int) -> None:
        if type(loaded_version) is not int or loaded_version < 0:
            raise ValueError("loaded_version must be a nonnegative integer")
        for key, (replicas, _old_version) in tuple(self._members().items()):
            self._members()[key] = (replicas, loaded_version)

    def bootstrap_target(self, *args, **kwargs):
        raise NotImplementedError("CE.bootstrap_target requires verified target-only native backend")
