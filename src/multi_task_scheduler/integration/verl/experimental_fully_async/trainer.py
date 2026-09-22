# Copyright 2025 Meituan Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Native Fully Async Trainer plus the single task-local synchronization gate G."""

import ray
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.utils.config import omega_conf_to_dataclass

from multi_task_scheduler.checkpoint.checkpoint_engine_manager import MultiTaskCheckpointEngineManager
from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    OperationEvidence,
    OperationRecord,
    ReplicaKey,
)
from multi_task_scheduler.orchestration.replica_sync_gate import GateKind, ReplicaSyncGate


@ray.remote(num_cpus=10)
class MultiTaskFullyAsyncTrainer(unwrap_native_actor_class(FullyAsyncTrainer)):
    def __init__(self, *args, task_session=None, **kwargs):
        self.task_session = task_session
        super().__init__(*args, **kwargs)

    async def _setup_checkpoint_manager(self):
        replicas = await self.rollouter.get_replicas.remote()
        checkpoint_engine_config = omega_conf_to_dataclass(
            self.config.actor_rollout_ref.rollout.checkpoint_engine
        )
        self.checkpoint_manager = MultiTaskCheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_wg,
            replicas=replicas,
        )
        if not self.task_session:
            raise RuntimeError("Trainer requires task_session before CE membership setup")

        for index, replica in enumerate(replicas):
            rank = getattr(replica, "replica_rank", index)
            key = ReplicaKey(self.task_session, f"native-{rank}", 0)
            self.checkpoint_manager.add_effective(
                key,
                [replica],
                loaded_version=self.current_param_version,
            )
        print(
            f"[FullyAsyncTrainer] Checkpoint manager initialized "
            f"(backend={checkpoint_engine_config.backend}, effective={len(replicas)})"
        )

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
        lease = await gate.acquire(
            f"native-sync:{self.current_param_version}",
            GateKind.NATIVE_SYNC,
        )
        try:
            result = await lease.guard(super()._fit_update_weights)
            if result is not None and self.checkpoint_manager is not None:
                await lease.guard(
                    self.checkpoint_manager.mark_all_loaded_version,
                    self.current_param_version,
                )
            return result
        except BaseException as exc:
            gate.block(
                lease.owner,
                f"Native synchronization outcome unknown: {type(exc).__name__}",
            )
            raise
        finally:
            await lease.release()

    async def bootstrap_and_publish(self, operation: OperationRecord):
        if not isinstance(operation, OperationRecord):
            raise TypeError("bootstrap_and_publish requires OperationRecord")
        raise NotImplementedError(
            "bootstrap_and_publish requires verified target-only parameter bootstrap"
        )

    async def remove_and_commit(self, operation: OperationRecord) -> OperationEvidence:
        """Remove E and commit R/C while holding the same gate as native sync."""
        if not isinstance(operation, OperationRecord):
            raise TypeError("remove_and_commit requires OperationRecord")
        if self.rollouter is None or self.checkpoint_manager is None:
            raise RuntimeError("Trainer owner dependencies are not initialized")

        gate = self.replica_sync_gate
        lease = await gate.acquire(operation.operation_id, GateKind.REMOVE)
        mutated = False
        try:
            target = await self.rollouter.get_pending_target.remote(operation.operation_id)
            member = self.checkpoint_manager.effective_replicas.get(target)
            if member is None:
                raise KeyError(f"replica {target!r} is not an effective CE member")

            await lease.guard(self.checkpoint_manager.remove_effective, target)
            mutated = True
            evidence = await lease.guard(
                self.rollouter.commit_service_change.remote,
                operation,
            )
            if not isinstance(evidence, OperationEvidence):
                raise TypeError("service commit did not return OperationEvidence")
            if evidence.operation_id != operation.operation_id:
                raise ValueError("service evidence belongs to another operation")
            if evidence.type is not EvidenceType.SERVICE_COMMITTED:
                raise ValueError(
                    f"expected SERVICE_COMMITTED, got {evidence.type.value}"
                )
            return evidence
        except BaseException as exc:
            if mutated:
                gate.block(
                    lease.owner,
                    f"Exit service commit outcome unknown: {type(exc).__name__}",
                )
            raise
        finally:
            await lease.release()

    async def restore_and_publish(self, operation: OperationRecord):
        if not isinstance(operation, OperationRecord):
            raise TypeError("restore_and_publish requires OperationRecord")
        raise NotImplementedError(
            "restore_and_publish requires verified native wake/bootstrap wiring"
        )
