"""Checkpoint Engine owner for the simplified E view."""

from __future__ import annotations

from verl.checkpoint_engine.base import CheckpointEngineManager

from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    OperationEvidence,
    ReplicaKey,
)


class MultiTaskCheckpointEngineManager(CheckpointEngineManager):
    """Keep only effective receiver membership as CE-owned mutable truth."""

    def _members(self) -> dict:
        if not hasattr(self, "_effective_replica_map"):
            self._effective_replica_map = {}
        return self._effective_replica_map

    @property
    def effective_replicas(self) -> dict:
        return self._members()

    def _pending_members(self) -> dict:
        if not hasattr(self, "_pending_bootstrap_map"):
            self._pending_bootstrap_map = {}
        return self._pending_bootstrap_map

    def _bootstrap_commits(self) -> dict:
        if not hasattr(self, "_bootstrap_commit_map"):
            self._bootstrap_commit_map = {}
        return self._bootstrap_commit_map

    @property
    def pending_bootstrap(self) -> dict:
        return self._pending_members()

    def register_pending(
        self,
        key: ReplicaKey,
        replicas,
        *,
        operation_id: str,
    ) -> None:
        """Register a runtime for target-only bootstrap without making it effective."""
        if not isinstance(key, ReplicaKey):
            raise TypeError("key must be ReplicaKey")
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id must be a nonempty string")
        replicas = tuple(replicas)
        if not replicas:
            raise ValueError("pending bootstrap requires at least one replica")

        effective = self._members().get(key)
        if effective is not None:
            if effective[0] == replicas and self._bootstrap_commits().get(key) == operation_id:
                return
            raise ValueError("ReplicaKey is already an effective CE member")

        entry = (replicas, operation_id)
        existing = self._pending_members().get(key)
        if existing is not None:
            if existing != entry:
                raise ValueError("ReplicaKey already has conflicting pending bootstrap")
            return

        # Parent replicas remains the effective set used by native full sync.
        # Pending borrowed runtimes stay out until WEIGHT_READY is committed.
        if any(replica in self.replicas for replica in replicas):
            raise ValueError("pending replica is already part of native effective membership")
        self._pending_members()[key] = entry

    def commit_pending(
        self,
        key: ReplicaKey,
        evidence: OperationEvidence,
        *,
        loaded_version: int,
    ) -> None:
        """Promote one pending target only after matching WEIGHT_READY evidence."""
        if not isinstance(evidence, OperationEvidence):
            raise TypeError("commit_pending requires OperationEvidence")
        if evidence.type is not EvidenceType.WEIGHT_READY:
            raise ValueError("pending bootstrap may commit only from WEIGHT_READY")

        existing_commit = self._bootstrap_commits().get(key)
        if existing_commit is not None:
            if existing_commit != evidence.operation_id:
                raise ValueError("ReplicaKey bootstrap was committed by another operation")
            member = self._members().get(key)
            if member is None or member[1] != loaded_version:
                raise ValueError("conflicting bootstrap commit replay")
            return

        try:
            replicas, operation_id = self._pending_members()[key]
        except KeyError as exc:
            raise KeyError(f"no pending bootstrap for {key!r}") from exc
        if operation_id != evidence.operation_id:
            raise ValueError("WEIGHT_READY evidence belongs to another operation")

        self.add_effective(key, replicas, loaded_version=loaded_version)
        self._pending_members().pop(key, None)
        self._bootstrap_commits()[key] = operation_id

    def discard_pending(self, key: ReplicaKey) -> None:
        self._pending_members().pop(key, None)

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
        self._pending_members().pop(key, None)
        self._bootstrap_commits().pop(key, None)
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
