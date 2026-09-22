# Copyright 2025 Meituan Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Native Fully Async Trainer plus the single task-local synchronization gate G."""

import ray
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.utils.config import omega_conf_to_dataclass
from multi_task_scheduler.checkpoint.checkpoint_engine_manager import MultiTaskCheckpointEngineManager
from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.contracts import OperationRecord
from multi_task_scheduler.orchestration.replica_sync_gate import GateKind, ReplicaSyncGate


@ray.remote(num_cpus=10)
class MultiTaskFullyAsyncTrainer(unwrap_native_actor_class(FullyAsyncTrainer)):
    async def _setup_checkpoint_manager(self):
        replicas = await self.rollouter.get_replicas.remote()
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = MultiTaskCheckpointEngineManager(config=checkpoint_engine_config, actor_wg=self.actor_wg, replicas=replicas)
        print(f"[FullyAsyncTrainer] Checkpoint manager initialized (backend={checkpoint_engine_config.backend})")

    def _ensure_gate(self) -> ReplicaSyncGate:
        if not hasattr(self, "_replica_sync_gate"):
            self._replica_sync_gate = ReplicaSyncGate()
        return self._replica_sync_gate

    @property
    def replica_sync_gate(self) -> ReplicaSyncGate:
        return self._ensure_gate()

    async def _fit_update_weights(self):
        if self.local_trigger_step != 1:
            return None
        gate = self.replica_sync_gate
        lease = await gate.acquire(f"native-sync:{self.current_param_version}", GateKind.NATIVE_SYNC)
        try:
            return await lease.guard(super()._fit_update_weights)
        except BaseException as exc:
            gate.block(lease.owner, f"Native synchronization outcome unknown: {type(exc).__name__}")
            raise
        finally:
            await lease.release()

    async def bootstrap_and_publish(self, operation: OperationRecord):
        if not isinstance(operation, OperationRecord):
            raise TypeError("bootstrap_and_publish requires OperationRecord")
        raise NotImplementedError("bootstrap_and_publish requires verified target-only parameter bootstrap")

    async def remove_and_commit(self, operation: OperationRecord):
        if not isinstance(operation, OperationRecord):
            raise TypeError("remove_and_commit requires OperationRecord")
        raise NotImplementedError("remove_and_commit requires verified exit revalidation and owner commit wiring")

    async def restore_and_publish(self, operation: OperationRecord):
        if not isinstance(operation, OperationRecord):
            raise TypeError("restore_and_publish requires OperationRecord")
        raise NotImplementedError("restore_and_publish requires verified native wake/bootstrap wiring")
