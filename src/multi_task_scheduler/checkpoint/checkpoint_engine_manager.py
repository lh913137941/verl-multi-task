"""Checkpoint Engine owner for the simplified E view."""

from __future__ import annotations

import asyncio

import ray
from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup

from multi_task_scheduler.checkpoint.checkpoint_engine_worker import (
    MultiTaskCheckpointEngineWorker,
)

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

    def _bootstrap_ready(self) -> dict:
        if not hasattr(self, "_bootstrap_ready_map"):
            self._bootstrap_ready_map = {}
        return self._bootstrap_ready_map

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
        self._bootstrap_ready().pop(key, None)

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

    def mark_loaded_version(self, key: ReplicaKey, loaded_version: int) -> None:
        """Advance one CE member after a target-only WEIGHT_READY commit."""
        if not isinstance(key, ReplicaKey):
            raise TypeError("key must be ReplicaKey")
        if type(loaded_version) is not int or loaded_version < 0:
            raise ValueError("loaded_version must be a nonnegative integer")
        try:
            replicas, _old_version = self._members()[key]
        except KeyError as exc:
            raise KeyError(f"unknown effective replica {key!r}") from exc
        self._members()[key] = (replicas, loaded_version)


    async def bootstrap_target(
        self,
        key: ReplicaKey,
        *,
        operation_id: str,
        loaded_version: int,
    ) -> OperationEvidence:
        """Synchronize only one hidden pending target using the native CE protocol."""
        if not isinstance(key, ReplicaKey):
            raise TypeError("key must be ReplicaKey")
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id must be a nonempty string")
        if type(loaded_version) is not int or loaded_version < 0:
            raise ValueError("loaded_version must be a nonnegative integer")
        if self.backend == "naive":
            raise NotImplementedError(
                "first-release target-only bootstrap requires non-naive checkpoint engine"
            )

        ready = self._bootstrap_ready().get(key)
        if ready is not None:
            ready_operation, ready_version, evidence = ready
            if ready_operation != operation_id or ready_version != loaded_version:
                raise ValueError("conflicting target bootstrap replay")
            return evidence

        try:
            replicas, pending_operation = self._pending_members()[key]
        except KeyError as exc:
            raise KeyError(f"no pending bootstrap for {key!r}") from exc
        if pending_operation != operation_id:
            raise ValueError("pending target belongs to another operation")

        workers = []
        for replica in replicas:
            workers.extend(replica.workers)
        if not workers:
            raise ValueError("pending target has no checkpoint-engine workers")

        rollout = RayWorkerGroup.from_detached(
            worker_handles=workers,
            ray_cls_with_init=RayClassWithInitArgs(
                cls=ray.remote(MultiTaskCheckpointEngineWorker)
            ),
            name_prefix=f"bootstrap_{operation_id}_",
            use_gpu=True,
        )
        actor_wg = self.actor_wg
        topology_started = False
        finalized = False

        try:
            # Target is hidden, so active replicas must not be aborted or touched.
            await asyncio.gather(
                *[replica.release_kv_cache() for replica in replicas]
            )

            topology_started = True
            self.build_process_group(rollout)

            ray.get(
                actor_wg.update_weights(
                    global_steps=loaded_version,
                    mode=self.backend,
                )
                + rollout.update_weights(global_steps=loaded_version)
            )

            ray.get(
                actor_wg.execute_checkpoint_engine(
                    ["finalize"] * actor_wg.world_size
                )
                + rollout.execute_checkpoint_engine(
                    ["finalize"] * rollout.world_size
                )
            )
            finalized = True

            await asyncio.gather(
                *[replica.resume_kv_cache() for replica in replicas]
            )
            health = await asyncio.gather(
                *[replica.validate_server_runtime() for replica in replicas]
            )
            for item in health:
                if item.get("global_steps") != loaded_version:
                    raise RuntimeError(
                        "target server did not confirm the published parameter version"
                    )

            evidence = OperationEvidence.now(
                operation_id,
                EvidenceType.WEIGHT_READY,
            )
            self._bootstrap_ready()[key] = (
                operation_id,
                loaded_version,
                evidence,
            )
            return evidence
        except BaseException as exc:
            if topology_started and not finalized:
                try:
                    ray.get(
                        actor_wg.execute_checkpoint_engine(
                            ["finalize"] * actor_wg.world_size
                        )
                        + rollout.execute_checkpoint_engine(
                            ["finalize"] * rollout.world_size
                        )
                    )
                except BaseException as finalize_exc:
                    raise RuntimeError(
                        "target bootstrap failed and checkpoint topology cleanup is unverified"
                    ) from finalize_exc
            raise exc
