# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Native Fully Async Trainer plus the task-local G and published snapshot holder.

Native training stays inherited. Lifecycle commits and native weight sync share
one gate. A version integer is compatibility/telemetry metadata only;
ADD/RESTORE must use ``published_snapshot`` and cannot manufacture weight
contents from a version number.
"""

import ray

from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.utils.config import omega_conf_to_dataclass

from multi_task_scheduler.checkpoint.checkpoint_engine_manager import MultiTaskCheckpointEngineManager
from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.contracts import PublishedWeightSnapshot
from multi_task_scheduler.orchestration.replica_sync_gate import GateKind, ReplicaSyncGate


@ray.remote(num_cpus=10)
class MultiTaskFullyAsyncTrainer(unwrap_native_actor_class(FullyAsyncTrainer)):
    """Own CE, G and Vpub evidence while inheriting native training behavior."""

    async def _setup_checkpoint_manager(self):
        replicas = await self.rollouter.get_replicas.remote()
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = MultiTaskCheckpointEngineManager(
            config=checkpoint_engine_config, actor_wg=self.actor_wg, replicas=replicas
        )
        print(f"[FullyAsyncTrainer] Checkpoint manager initialized (backend={checkpoint_engine_config.backend})")

    def _ensure_gate(self) -> ReplicaSyncGate:
        if not hasattr(self, "_replica_sync_gate"):
            self._replica_sync_gate = ReplicaSyncGate()
        return self._replica_sync_gate

    @property
    def replica_sync_gate(self) -> ReplicaSyncGate:
        return self._ensure_gate()

    async def _fit_update_weights(self):
        """Delegate native synchronization under G; uncertainty blocks followups."""
        if self.local_trigger_step != 1:
            return None
        gate = self.replica_sync_gate
        lease = await gate.acquire(f"native-sync:{self.current_param_version}", GateKind.NATIVE_SYNC)
        try:
            result = await lease.guard(super()._fit_update_weights)
            # Native VERL currently returns timing/results rather than the
            # immutable replayable snapshot required by ADD/RESTORE. Do not
            # fabricate PublishedWeightSnapshot here.
            self._last_native_serving_version = self.current_param_version
            return result
        except BaseException as exc:
            gate.block(lease.owner, f"Native synchronization outcome unknown: {type(exc).__name__}")
            raise
        finally:
            await lease.release()

    @property
    def published_snapshot(self) -> PublishedWeightSnapshot | None:
        return getattr(self, "_published_snapshot", None)

    def current_snapshot(self) -> PublishedWeightSnapshot | None:
        return self.published_snapshot

    def publish_snapshot(self, snapshot: PublishedWeightSnapshot) -> None:
        if not isinstance(snapshot, PublishedWeightSnapshot):
            raise TypeError("publish_snapshot requires PublishedWeightSnapshot")
        current = self.published_snapshot
        if current is not None and snapshot.version < current.version:
            raise ValueError("published snapshot version cannot move backwards")
        self._published_snapshot = snapshot
        self._last_native_serving_version = snapshot.version

    def publish(self, snapshot: PublishedWeightSnapshot) -> None:
        self.publish_snapshot(snapshot)

    @property
    def published_serving_version(self) -> int:
        snapshot = self.published_snapshot
        if snapshot is not None:
            return snapshot.version
        return getattr(self, "_last_native_serving_version", 0)

    def publish_serving_version(self, version: int) -> None:
        """Compatibility telemetry only; this does not create bootstrap evidence."""
        if version < 0:
            raise ValueError("version must be nonnegative")
        self._last_native_serving_version = version

    async def bootstrap_and_publish(self, ctx, prepared):
        """ADD coordination entry: load Vpub, join E and commit service under G."""
        raise NotImplementedError("ADD transaction requires verified native backend")

    async def remove_and_commit(self, ctx, proof):
        """REMOVE/DONATE service-exit entry; physical release happens afterwards."""
        raise NotImplementedError("REMOVE transaction requires verified native backend")

    async def restore_and_publish(self, ctx, key, fence_satisfied=None):
        """RESTORE native runtime using the donor task's current published snapshot.

        ``fence_satisfied`` is accepted only for compatibility with the previous
        binding signature; the simplified contract carries release authority in
        the GS operation/lease evidence rather than as a free boolean.
        """
        raise NotImplementedError("RESTORE transaction requires verified native backend")
