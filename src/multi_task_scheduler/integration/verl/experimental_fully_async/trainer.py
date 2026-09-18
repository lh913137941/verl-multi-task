# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Native Fully Async Trainer plus task-local G and immutable Vpub evidence.

A version integer is never treated as published-weight evidence. ADD/RESTORE may
proceed only from an actual PublishedWeightSnapshot produced by a verified
native synchronization backend.
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
    """Own CE, G and the exact current PublishedWeightSnapshot."""

    async def _setup_checkpoint_manager(self):
        replicas = await self.rollouter.get_replicas.remote()
        checkpoint_engine_config = omega_conf_to_dataclass(
            self.config.actor_rollout_ref.rollout.checkpoint_engine
        )
        self.checkpoint_manager = MultiTaskCheckpointEngineManager(
            config=checkpoint_engine_config, actor_wg=self.actor_wg, replicas=replicas
        )
        print(
            "[FullyAsyncTrainer] Checkpoint manager initialized "
            f"(backend={checkpoint_engine_config.backend})"
        )

    def _ensure_gate(self) -> ReplicaSyncGate:
        if not hasattr(self, "_replica_sync_gate"):
            self._replica_sync_gate = ReplicaSyncGate()
        return self._replica_sync_gate

    @property
    def replica_sync_gate(self) -> ReplicaSyncGate:
        return self._ensure_gate()

    async def _fit_update_weights(self):
        """Delegate native synchronization under G without fabricating Vpub."""
        if self.local_trigger_step != 1:
            return None
        gate = self.replica_sync_gate
        lease = await gate.acquire(
            f"native-sync:{self.current_param_version}", GateKind.NATIVE_SYNC
        )
        try:
            return await lease.guard(super()._fit_update_weights)
        except BaseException as exc:
            gate.block(
                lease.owner,
                f"Native synchronization outcome unknown: {type(exc).__name__}",
            )
            raise
        finally:
            await lease.release()

    @property
    def published_snapshot(self) -> PublishedWeightSnapshot | None:
        return getattr(self, "_published_snapshot", None)

    def current_snapshot(self) -> PublishedWeightSnapshot | None:
        return self.published_snapshot

    def publish_snapshot(self, snapshot: PublishedWeightSnapshot) -> None:
        """Install only verified immutable published-weight evidence."""
        if not isinstance(snapshot, PublishedWeightSnapshot):
            raise TypeError("publish_snapshot requires PublishedWeightSnapshot")
        current = self.published_snapshot
        if current is not None and snapshot.version < current.version:
            raise ValueError("published snapshot version cannot move backwards")
        self._published_snapshot = snapshot

    async def bootstrap_and_publish(self, ctx, prepared):
        """ADD coordinator: Vpub bootstrap -> CE ADD -> RO service commit."""
        raise NotImplementedError(
            "bootstrap_and_publish requires verified native runtime/CE evidence wiring"
        )

    async def remove_and_commit(self, ctx, proof):
        """Exit coordinator: internal revalidation -> CE REMOVE -> service removal."""
        raise NotImplementedError(
            "remove_and_commit requires verified native runtime/CE evidence wiring"
        )

    async def restore_and_publish(self, ctx, key):
        """RESTORE coordinator using the donor task's current PublishedWeightSnapshot."""
        raise NotImplementedError(
            "restore_and_publish requires verified native wake/bootstrap wiring"
        )
