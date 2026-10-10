# Copyright 2025 Meituan Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Native Fully Async Trainer plus the single task-local synchronization gate G."""

import json

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
    require_operation_evidence as _require_evidence,
    ReplicaKind,
    native_replica_key,
)
from multi_task_scheduler.orchestration.replica_sync_gate import GateKind, ReplicaSyncGate



@ray.remote(num_cpus=10)
class MultiTaskFullyAsyncTrainer(unwrap_native_actor_class(FullyAsyncTrainer)):
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
            key = native_replica_key(self.task_session, rank)
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
                # E records a published version only after all enabled
                # receiver/source checks have succeeded. On validation failure
                # the gate stays BLOCKED and E must not claim this version.
                if getattr(
                    self.checkpoint_manager,
                    "parameter_validation_enabled",
                    False,
                ):
                    effective = [
                        replica
                        for replicas, _loaded_version
                        in self.checkpoint_manager.effective_replicas.values()
                        for replica in replicas
                    ]
                    source_manifest = (
                        await lease.guard(
                            self.checkpoint_manager._get_source_manifest
                        )
                        if getattr(
                            self.checkpoint_manager,
                            "source_validation_enabled",
                            False,
                        )
                        else None
                    )
                    validation = await lease.guard(
                        self.checkpoint_manager.validate_parameter_sync,
                        effective,
                        self.current_param_version,
                        source_manifest,
                    )
                    print(
                        "CE_PARAMETER_VALIDATION "
                        + json.dumps(validation, sort_keys=True)
                    )
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

    @staticmethod
    def _gate_fences(gate: ReplicaSyncGate, operation_id: str) -> bool:
        return (
            gate.health == "BLOCKED"
            and gate.blocked_operation_id == operation_id
        )

    async def _reconcile_blocked_publish(
        self,
        operation: OperationRecord,
        *,
        restore_native: bool,
    ) -> OperationEvidence | None:
        """Resolve a BLOCKED ADD/RESTORE publication from owner facts under G."""
        gate = self.replica_sync_gate
        if not self._gate_fences(gate, operation.operation_id):
            return None

        target = await self.rollouter.get_pending_target.remote(operation.operation_id)
        member = self.checkpoint_manager.effective_replicas.get(target)
        if member is None:
            # This was not a post-E publication failure; keep the existing fence.
            return None

        replicas, loaded_version = member
        result = {}

        async def reconcile():
            service_evidence = await self.rollouter.commit_service_change.remote(
                operation
            )
            if service_evidence is None:
                self.checkpoint_manager.remove_effective(target)
                if restore_native:
                    self.checkpoint_manager.register_pending(
                        target,
                        replicas,
                        operation_id=operation.operation_id,
                    )
                release_evidence = await self.rollouter.finalize_release.remote(
                    operation
                )
                result["evidence"] = _require_evidence(
                    release_evidence,
                    operation.operation_id,
                    EvidenceType.RELEASED,
                    "publication rollback release",
                )
                return

            if (
                isinstance(service_evidence, OperationEvidence)
                and service_evidence.type is EvidenceType.RELEASED
            ):
                # Compatibility with a worker that completed rollback before the
                # Trainer observed the response.
                self.checkpoint_manager.remove_effective(target)
                result["evidence"] = _require_evidence(
                    service_evidence,
                    operation.operation_id,
                    EvidenceType.RELEASED,
                    "publication rollback",
                )
                return

            result["evidence"] = _require_evidence(
                service_evidence,
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                "publication reconciliation",
            )

        await gate.reconcile(operation.operation_id, reconcile)
        evidence = result.get("evidence")
        if evidence is None:
            raise RuntimeError("publication reconciliation returned no evidence")
        return evidence

    async def bootstrap_and_publish(self, operation: OperationRecord) -> OperationEvidence:
        """Bootstrap one hidden borrowed target at current Vpub, then publish it."""
        if not isinstance(operation, OperationRecord):
            raise TypeError("bootstrap_and_publish requires OperationRecord")
        if self.rollouter is None or self.checkpoint_manager is None:
            raise RuntimeError("Trainer owner dependencies are not initialized")
        if type(self.current_param_version) is not int or self.current_param_version < 0:
            raise RuntimeError("current parameter version is unavailable for ADD")

        gate = self.replica_sync_gate
        reconciled = await self._reconcile_blocked_publish(
            operation,
            restore_native=False,
        )
        if reconciled is not None:
            return reconciled
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
            if service_evidence is None:
                # R is definitely absent. Remove E before physical destroy.
                await lease.guard(
                    self.checkpoint_manager.remove_effective,
                    target,
                )
                release_evidence = await lease.guard(
                    self.rollouter.finalize_release.remote,
                    operation,
                )
                _require_evidence(
                    release_evidence,
                    operation.operation_id,
                    EvidenceType.RELEASED,
                    "ADD no-route rollback",
                )
                e_committed = False
                return release_evidence
            if (
                isinstance(service_evidence, OperationEvidence)
                and service_evidence.type is EvidenceType.RELEASED
            ):
                _require_evidence(
                    service_evidence,
                    operation.operation_id,
                    EvidenceType.RELEASED,
                    "ADD service rollback",
                )
                await lease.guard(
                    self.checkpoint_manager.remove_effective,
                    target,
                )
                e_committed = False
                return service_evidence
            return _require_evidence(
                service_evidence,
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                "ADD service commit",
            )
        except BaseException as exc:
            # Keep this import in the method body as well: the isolated wiring
            # test compiles the class without module-level imports.
            import logging

            logging.getLogger(__name__).exception(
                "ADD bootstrap/publish failed before rollback: operation_id=%s target=%s e_committed=%s",
                operation.operation_id,
                operation.target,
                e_committed,
            )
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
                # The target never entered E/R/C/M and the hidden runtime is
                # proven RELEASED. Return that existing evidence so TaskRunner
                # can close this operation as FAILED and let GS roll back only
                # the staged borrower target while preserving lease handoff.
                return release_evidence
            gate.block(
                lease.owner,
                f"ADD publish outcome unknown: {type(exc).__name__}",
            )
            raise
        finally:
            await lease.release()

    async def _reconcile_blocked_exit(
        self,
        operation: OperationRecord,
    ) -> OperationEvidence | None:
        """Resolve a BLOCKED DONATE/REMOVE service commit from owner facts under G."""
        gate = self.replica_sync_gate
        if not self._gate_fences(gate, operation.operation_id):
            return None

        target = await self.rollouter.get_pending_target.remote(operation.operation_id)
        if target in self.checkpoint_manager.effective_replicas:
            # E still owns the replica, so this is not a post-remove uncertainty.
            return None

        result = {}

        async def reconcile():
            service_evidence = await self.rollouter.commit_service_change.remote(
                operation
            )
            result["evidence"] = _require_evidence(
                service_evidence,
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                "exit service reconciliation",
            )

        await gate.reconcile(operation.operation_id, reconcile)
        evidence = result.get("evidence")
        if evidence is None:
            raise RuntimeError("exit reconciliation returned no evidence")
        return evidence

    async def reconcile_exit(
        self,
        operation: OperationRecord,
    ) -> OperationEvidence | None:
        """Exact-op recovery hook used only after TaskRunner recorded UNKNOWN."""
        if not isinstance(operation, OperationRecord):
            raise TypeError("reconcile_exit requires OperationRecord")
        if self.rollouter is None or self.checkpoint_manager is None:
            raise RuntimeError("Trainer owner dependencies are not initialized")
        return await self._reconcile_blocked_exit(operation)

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

        This method reuses the existing CE pending/bootstrap state instead of
        introducing RESTORE-only interfaces or lifecycle DTOs. Runtime success
        is still evidence-driven: unsupported or unverifiable GPU behavior fails
        the operation instead of synthesizing service success.
        """
        if not isinstance(operation, OperationRecord):
            raise TypeError("restore_and_publish requires OperationRecord")
        if self.rollouter is None or self.checkpoint_manager is None:
            raise RuntimeError("Trainer owner dependencies are not initialized")
        if type(self.current_param_version) is not int or self.current_param_version < 0:
            raise RuntimeError("current parameter version is unavailable for RESTORE")

        gate = self.replica_sync_gate
        reconciled = await self._reconcile_blocked_publish(
            operation,
            restore_native=True,
        )
        if reconciled is not None:
            return reconciled
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
            if (
                isinstance(weight_evidence, OperationEvidence)
                and weight_evidence.type is EvidenceType.RELEASED
            ):
                return _require_evidence(
                    weight_evidence,
                    operation.operation_id,
                    EvidenceType.RELEASED,
                    "RESTORE bootstrap rollback",
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
            if service_evidence is None:
                await lease.guard(
                    self.checkpoint_manager.remove_effective,
                    target,
                )
                await lease.guard(
                    self.checkpoint_manager.register_pending,
                    target,
                    replicas,
                    operation_id=operation.operation_id,
                )
                release_evidence = await lease.guard(
                    self.rollouter.finalize_release.remote,
                    operation,
                )
                return _require_evidence(
                    release_evidence,
                    operation.operation_id,
                    EvidenceType.RELEASED,
                    "RESTORE no-route rollback",
                )
            return _require_evidence(
                service_evidence,
                operation.operation_id,
                EvidenceType.SERVICE_COMMITTED,
                "RESTORE service commit",
            )
        except BaseException as exc:
            if mutated:
                quarantine_error = None
                try:
                    await self.rollouter.quarantine_dormant_restore.remote(
                        operation.operation_id
                    )
                except BaseException as projection_exc:
                    quarantine_error = projection_exc
                reason = f"RESTORE outcome unknown: {type(exc).__name__}"
                if quarantine_error is not None:
                    reason += "; M projection reconciliation failed"
                gate.block(lease.owner, reason)
            raise
        finally:
            await lease.release()
