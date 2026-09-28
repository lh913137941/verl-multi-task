"""Checkpoint Engine owner for the simplified E view."""

from __future__ import annotations

import asyncio

import ray
from verl.checkpoint_engine.base import CheckpointEngineManager, CheckpointEngineWorker
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup

from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    OperationEvidence,
    ReplicaKey,
    ReplicaKind,
)


class MultiTaskCheckpointEngineManager(CheckpointEngineManager):
    """Keep only effective receiver membership as CE-owned mutable truth."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._effective_replica_map = {}
        self._pending_bootstrap_map = {}
        self._bootstrap_ready_map = {}

    @property
    def effective_replicas(self) -> dict:
        return self._effective_replica_map

    @property
    def pending_bootstrap(self) -> dict:
        return self._pending_bootstrap_map

    def _validate_runtime_membership(self, key: ReplicaKey, replicas: tuple) -> None:
        for index, replica in enumerate(replicas):
            if replica in replicas[:index]:
                raise ValueError("duplicate runtime in CE membership")
        for entries in (self._effective_replica_map, self._pending_bootstrap_map):
            for other_key, (other_replicas, _) in entries.items():
                if other_key != key and any(r in other_replicas for r in replicas):
                    raise ValueError("runtime already belongs to another ReplicaKey")

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
        self._validate_runtime_membership(key, replicas)

        effective = self._effective_replica_map.get(key)
        if effective is not None:
            ready = self._bootstrap_ready_map.get(key)
            if effective[0] == replicas and ready is not None and ready[0] == operation_id:
                return
            raise ValueError("ReplicaKey is already an effective CE member")

        entry = (replicas, operation_id)
        existing = self._pending_bootstrap_map.get(key)
        if existing is not None:
            if existing != entry:
                raise ValueError("ReplicaKey already has conflicting pending bootstrap")
            return

        # Parent replicas remains the effective set used by native full sync.
        # Pending borrowed runtimes stay out until WEIGHT_READY is committed.
        if any(replica in self.replicas for replica in replicas):
            raise ValueError("pending replica is already part of native effective membership")
        self._pending_bootstrap_map[key] = entry

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
        if type(loaded_version) is not int or loaded_version < 0:
            raise ValueError("loaded_version must be a nonnegative integer")

        confirmed = self._bootstrap_ready_map.get(key)
        member = self._effective_replica_map.get(key)
        if (
            key not in self._pending_bootstrap_map
            and confirmed == (evidence.operation_id, loaded_version, evidence)
            and member is not None
            and member[1] == loaded_version
        ):
            return

        try:
            replicas, operation_id = self._pending_bootstrap_map[key]
        except KeyError as exc:
            raise KeyError(f"no pending bootstrap for {key!r}") from exc
        if operation_id != evidence.operation_id:
            raise ValueError("WEIGHT_READY evidence belongs to another operation")
        if self._bootstrap_ready_map.get(key) != (operation_id, loaded_version, evidence):
            raise ValueError("WEIGHT_READY does not match confirmed bootstrap")

        self.add_effective(key, replicas, loaded_version=loaded_version)
        self._pending_bootstrap_map.pop(key, None)

    def discard_pending(self, key: ReplicaKey) -> None:
        self._pending_bootstrap_map.pop(key, None)
        self._bootstrap_ready_map.pop(key, None)

    def add_effective(self, key: ReplicaKey, replicas, *, loaded_version: int) -> None:
        if not isinstance(key, ReplicaKey):
            raise TypeError("key must be ReplicaKey")
        if type(loaded_version) is not int or loaded_version < 0:
            raise ValueError("loaded_version must be a nonnegative integer")

        replicas = tuple(replicas)
        if not replicas:
            raise ValueError("effective membership requires at least one replica")
        self._validate_runtime_membership(key, replicas)
        entry = (replicas, loaded_version)
        existing = self._effective_replica_map.get(key)
        if existing is not None and existing != entry:
            raise ValueError("ReplicaKey already has conflicting CE membership")

        super().add_replicas(
            [replica for replica in replicas if replica not in self.replicas]
        )
        self._effective_replica_map[key] = entry

    def remove_effective(self, key: ReplicaKey) -> None:
        self.discard_pending(key)
        entry = self._effective_replica_map.pop(key, None)
        if entry is None:
            return
        replicas, _loaded_version = entry
        super().remove_replicas(list(replicas))

    def mark_all_loaded_version(self, loaded_version: int) -> None:
        if type(loaded_version) is not int or loaded_version < 0:
            raise ValueError("loaded_version must be a nonnegative integer")
        for key, (replicas, _old_version) in tuple(self._effective_replica_map.items()):
            self._effective_replica_map[key] = (replicas, loaded_version)

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

        ready = self._bootstrap_ready_map.get(key)
        if ready is not None:
            ready_operation, ready_version, evidence = ready
            if ready_operation != operation_id or ready_version != loaded_version:
                raise ValueError("conflicting target bootstrap replay")
            return evidence

        try:
            replicas, pending_operation = self._pending_bootstrap_map[key]
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
                cls=ray.remote(CheckpointEngineWorker)
            ),
            name_prefix=f"bootstrap_{operation_id}_",
            use_gpu=True,
        )
        actor_wg = self.actor_wg
        topology_started = False
        finalized = False
        native_restore = False
        native_wake_started = False

        replica_kinds = {
            getattr(replica, "replica_kind", None)
            for replica in replicas
        }
        if len(replica_kinds) != 1:
            raise ValueError("pending target contains inconsistent replica kinds")
        replica_kind = next(iter(replica_kinds))
        if replica_kind not in {ReplicaKind.NATIVE, ReplicaKind.BORROWED}:
            raise ValueError("pending target has unsupported replica kind")
        native_restore = replica_kind is ReplicaKind.NATIVE

        try:
            # Native RESTORE allocates only weight memory under the same G that
            # serializes parameter publication. ADD targets are already resident.
            if native_restore:
                native_wake_started = True
                await asyncio.gather(
                    *[replica.wake_up(tags=["weights"]) for replica in replicas]
                )

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
            self._bootstrap_ready_map[key] = (
                operation_id,
                loaded_version,
                evidence,
            )
            return evidence
        except BaseException as exc:
            cleanup_error = None
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
                    cleanup_error = finalize_exc

            # A failed native RESTORE must not escape with a partially awake
            # retained runtime.  Re-enter proven level-2 sleep while G is still
            # held; server admission never opens on this path.
            if native_restore and native_wake_started:
                try:
                    receipts = await asyncio.gather(
                        *[replica.sleep() for replica in replicas]
                    )
                    for replica_receipts in receipts:
                        if not isinstance(replica_receipts, (tuple, list)) or not replica_receipts:
                            raise RuntimeError("native RESTORE rollback returned no sleep receipts")
                        if any(
                            not isinstance(receipt, dict)
                            or receipt.get("sleep_level") != 2
                            or receipt.get("sleeping") is not True
                            for receipt in replica_receipts
                        ):
                            raise RuntimeError("native RESTORE rollback did not confirm level-2 sleep")
                except BaseException as rollback_exc:
                    cleanup_error = rollback_exc

            if cleanup_error is not None:
                raise RuntimeError(
                    "target bootstrap failed and runtime cleanup is unverified"
                ) from cleanup_error
            raise exc
