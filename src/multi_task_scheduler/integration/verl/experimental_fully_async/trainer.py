# Copyright 2025 Meituan Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Native Fully Async Trainer plus the single task-local synchronization gate G."""

import ray
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.utils.config import omega_conf_to_dataclass

from multi_task_scheduler.checkpoint.checkpoint_engine_manager import MultiTaskCheckpointEngineManager
from verl.single_controller.ray.base import _unwrap_ray_remote
from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    OperationEvidence,
    OperationRecord,
    ReplicaKey,
    ReplicaKind,
)
from multi_task_scheduler.orchestration.replica_sync_gate import GateKind, ReplicaSyncGate


def _require_evidence(value, operation_id: str, expected: EvidenceType, label: str):
    if not isinstance(value, OperationEvidence):
        raise TypeError(f"{label} did not return OperationEvidence")
    if value.operation_id != operation_id:
        raise ValueError(f"{label} evidence belongs to another operation")
    if value.type is not expected:
        raise ValueError(f"expected {expected.value}, got {value.type.value}")
    return value


@ray.remote(num_cpus=10)
class MultiTaskFullyAsyncTrainer(_unwrap_ray_remote(FullyAsyncTrainer)):
    def __init__(self, *args, task_session=None, **kwargs):
        self.task_session = task_session
        self._replica_sync_gate = ReplicaSyncGate()
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

    @property
    def replica_sync_gate(self) -> ReplicaSyncGate:
        return self._replica_sync_gate

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

    async def bootstrap_and_publish(self, operation: OperationRecord) -> OperationEvidence:
        """Bootstrap one hidden borrowed target at current Vpub, then publish it."""
        if not isinstance(operation, OperationRecord):
            raise TypeError("bootstrap_and_publish requires OperationRecord")
        if self.rollouter is None or self.checkpoint_manager is None:
            raise RuntimeError("Trainer owner dependencies are not initialized")
        if type(self.current_param_version) is not int or self.current_param_version < 0:
            raise RuntimeError("current parameter version is unavailable for ADD")

        gate = self.replica_sync_gate
        lease = await gate.acquire(operation.operation_id, GateKind.ADD)
        pending_registered = False
        e_committed = False
        try:
            target = await self.rollouter.get_pending_target.remote(operation.operation_id)
            replicas = tuple(
                await self.rollouter.get_pending_replicas.remote(operation.operation_id)
            )
            if not replicas:
                raise RuntimeError("ADD target has no prepared runtime")
            if {
                getattr(replica, "replica_kind", None)
                for replica in replicas
            } != {ReplicaKind.BORROWED}:
                raise ValueError("ADD requires a prepared BORROWED runtime")

            await lease.guard(
                self.checkpoint_manager.register_pending,
                target,
                replicas,
                operation_id=operation.operation_id,
            )
            pending_registered = True

            weight_evidence = await lease.guard(
                self.checkpoint_manager.bootstrap_target,
                target,
                operation_id=operation.operation_id,
                loaded_version=self.current_param_version,
            )
            _require_evidence(
                weight_evidence,
                operation.operation_id,
                EvidenceType.WEIGHT_READY,
                "ADD bootstrap",
            )

            await lease.guard(
                self.checkpoint_manager.commit_pending,
                target,
                weight_evidence,
                loaded_version=self.current_param_version,
            )
            e_committed = True

            service_evidence = await lease.guard(
                self.rollouter.commit_service_change.remote,
                operation,
            )
            return _require_evidence(
                service_evidence,
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                "ADD service commit",
            )
        except BaseException as exc:
            if not e_committed:
                cleanup_error = None
                if pending_registered:
                    try:
                        await lease.guard(
                            self.checkpoint_manager.discard_pending,
                            target,
                        )
                    except BaseException as discard_exc:
                        cleanup_error = discard_exc
                try:
                    release_evidence = await lease.guard(
                        self.rollouter.finalize_release.remote,
                        operation,
                    )
                    _require_evidence(
                        release_evidence,
                        operation.operation_id,
                        EvidenceType.RELEASED,
                        "ADD rollback release",
                    )
                except BaseException as release_exc:
                    cleanup_error = release_exc
                if cleanup_error is not None:
                    gate.block(
                        lease.owner,
                        f"ADD rollback outcome unknown: {type(cleanup_error).__name__}",
                    )
                    raise RuntimeError(
                        "ADD bootstrap failed and prepared runtime cleanup is unverified"
                    ) from cleanup_error
            else:
                gate.block(
                    lease.owner,
                    f"ADD publish outcome unknown: {type(exc).__name__}",
                )
            raise
        finally:
            await lease.release()

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
            replicas, _loaded_version = member
            replica_kinds = {
                getattr(replica, "replica_kind", None)
                for replica in replicas
            }
            if len(replica_kinds) != 1:
                raise ValueError("CE member contains inconsistent replica kinds")
            replica_kind = next(iter(replica_kinds))
            if replica_kind not in {ReplicaKind.NATIVE, ReplicaKind.BORROWED}:
                raise ValueError("CE member has unsupported replica kind")

            await lease.guard(self.checkpoint_manager.remove_effective, target)
            mutated = True
            # Reuse CE's existing pending set as the parked receiver reference
            # for retained native DONATE. Borrowed REMOVE is destroyed later and
            # therefore must not leave a stale pending runtime behind.
            if replica_kind is ReplicaKind.NATIVE:
                await lease.guard(
                    self.checkpoint_manager.register_pending,
                    target,
                    replicas,
                    operation_id=operation.operation_id,
                )
            evidence = await lease.guard(
                self.rollouter.commit_service_change.remote,
                operation,
            )
            return _require_evidence(
                evidence,
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                "service commit",
            )
        except BaseException as exc:
            if mutated:
                gate.block(
                    lease.owner,
                    f"Exit service commit outcome unknown: {type(exc).__name__}",
                )
            raise
        finally:
            await lease.release()

    async def restore_and_publish(self, operation: OperationRecord) -> OperationEvidence:
        """Restore current Vpub into one parked native runtime and republish it.

        TaskRunner admission remains fail-closed until this path is validated on
        the real CUDA/vLLM backend.  This method intentionally reuses the
        existing CE pending/bootstrap state instead of introducing RESTORE-only
        interfaces or lifecycle DTOs.
        """
        if not isinstance(operation, OperationRecord):
            raise TypeError("restore_and_publish requires OperationRecord")
        if self.rollouter is None or self.checkpoint_manager is None:
            raise RuntimeError("Trainer owner dependencies are not initialized")
        if type(self.current_param_version) is not int or self.current_param_version < 0:
            raise RuntimeError("current parameter version is unavailable for RESTORE")

        gate = self.replica_sync_gate
        lease = await gate.acquire(operation.operation_id, GateKind.RESTORE)
        mutated = False
        try:
            target = await self.rollouter.get_pending_target.remote(operation.operation_id)
            parked = self.checkpoint_manager.pending_bootstrap.get(target)
            if parked is None:
                raise KeyError(f"no parked native CE runtime for {target!r}")
            replicas, _parked_operation = parked
            replica_kinds = {
                getattr(replica, "replica_kind", None)
                for replica in replicas
            }
            if replica_kinds != {ReplicaKind.NATIVE}:
                raise ValueError("RESTORE requires a parked NATIVE CE runtime")

            # Rebind the already-retained receiver to this RESTORE operation
            # using CE's existing pending API; no new lifecycle state is added.
            await lease.guard(self.checkpoint_manager.discard_pending, target)
            mutated = True
            await lease.guard(
                self.checkpoint_manager.register_pending,
                target,
                replicas,
                operation_id=operation.operation_id,
            )

            weight_evidence = await lease.guard(
                self.checkpoint_manager.bootstrap_target,
                target,
                operation_id=operation.operation_id,
                loaded_version=self.current_param_version,
            )
            _require_evidence(
                weight_evidence,
                operation.operation_id,
                EvidenceType.WEIGHT_READY,
                "RESTORE bootstrap",
            )

            # Promote the proven current-Vpub receiver into E before
            # opening R. A routed replica must never be absent from native
            # effective weight-sync membership. If service publication later
            # fails or is uncertain, G is fenced below for reconciliation.
            await lease.guard(
                self.checkpoint_manager.commit_pending,
                target,
                weight_evidence,
                loaded_version=self.current_param_version,
            )

            service_evidence = await lease.guard(
                self.rollouter.commit_service_change.remote,
                operation,
            )
            return _require_evidence(
                service_evidence,
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                "RESTORE service commit",
            )
        except BaseException as exc:
            if mutated:
                gate.block(
                    lease.owner,
                    f"RESTORE outcome unknown: {type(exc).__name__}",
                )
            raise
        finally:
            await lease.release()
