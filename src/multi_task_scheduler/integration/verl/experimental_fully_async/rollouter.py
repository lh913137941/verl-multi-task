"""Experimental Fully Async Rollouter aligned with the 092203 owner boundaries."""

from __future__ import annotations

import asyncio

import ray
from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncAgentLoopManager,
    FullyAsyncRollouter,
)
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    OperationEvidence,
    OperationRecord,
    ReplicaKey,
    ReplicaKind,
    ReplicaState,
)

from .llm_server_manager import MultiTaskLLMServerManager


@ray.remote(num_cpus=10, max_concurrency=100)
class MultiTaskFullyAsyncRollouter(unwrap_native_actor_class(FullyAsyncRollouter)):
    def __init__(
        self,
        config,
        tokenizer,
        processor=None,
        device_name=None,
        *,
        group_scheduler=None,
        task_session=None,
    ):
        self.group_scheduler = group_scheduler
        self.task_session = task_session
        self._pending_operation_targets: dict[str, ReplicaKey] = {}
        super().__init__(
            config,
            tokenizer,
            processor=processor,
            device_name=device_name,
        )

    async def _init_async_rollout_manager(self):
        enable_agent_reward_loop = (
            not self.use_rm or self.config.reward.reward_model.enable_resource_pool
        )
        reward_loop_worker_handles = (
            self.reward_loop_manager.reward_loop_workers
            if enable_agent_reward_loop
            else None
        )
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        self.async_rollout_mode = True
        self.llm_server_manager = await MultiTaskLLMServerManager.create(
            config=self.config,
            worker_group=self.get_hybrid_worker_group(),
            task_session=self.task_session,
        )
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(
                client_cls=FullyAsyncLLMServerClient
            ),
            reward_loop_worker_handles=reward_loop_worker_handles,
            teacher_client=(
                self.teacher_model_manager.get_client()
                if self.teacher_model_manager
                else None
            ),
        )

    @property
    def committed_capacity(self) -> int:
        value = getattr(self, "max_concurrent_samples", None)
        return 0 if value is None else int(value)

    @property
    def production_window_open(self) -> bool:
        return not bool(getattr(self, "paused", False))

    def collect_idle_candidates(self) -> tuple[tuple[ReplicaKey, ReplicaKind], ...]:
        """Return only ACTIVE replicas whose removal preserves committed C.

        Native ``max_concurrent_samples`` is capped by ``max_required_samples``.
        Compute how many ACTIVE replicas are required to preserve the currently
        committed capacity and report only the surplus. This stays Rollouter-local
        and deliberately does not read LB/R.
        """
        if self.production_window_open or self.committed_capacity <= 0:
            return ()
        manager = getattr(self, "llm_server_manager", None)
        if manager is None:
            return ()

        per_replica = int(getattr(self, "concurrent_samples_per_replica", 0))
        if per_replica <= 0:
            return ()

        active = [
            (key, manager.replica_kind[key])
            for key, state in manager.replica_state.items()
            if state is ReplicaState.ACTIVE
        ]
        required_active = (self.committed_capacity + per_replica - 1) // per_replica
        surplus_count = max(0, len(active) - required_active)
        return tuple(active[:surplus_count])

    def submit_idle_report(self):
        if self.group_scheduler is None:
            raise RuntimeError("GroupScheduler handle is required for idle reporting")
        candidates = self.collect_idle_candidates()
        if not candidates:
            return None
        task_sessions = {key.task_session for key, _ in candidates}
        if len(task_sessions) != 1:
            raise ValueError(
                "one Rollouter may report candidates for only one task_session"
            )
        report = {
            "task_session": next(iter(task_sessions)),
            "candidates": tuple(
                {"replica_key": key, "kind": kind.value}
                for key, kind in candidates
            ),
        }
        return ray.get(
            self.group_scheduler.submit_idle_report.remote(report),
            timeout=30,
        )

    async def create_borrowed_replica(self, spec: dict) -> dict:
        """Thin runtime entry; Manager owns placement validation and creation."""
        manager = getattr(self, "llm_server_manager", None)
        if manager is None:
            raise RuntimeError("LLM server manager is not initialized")
        return await manager.create_borrowed_replica(spec)

    async def prepare_replica(
        self,
        replica_key: ReplicaKey,
        *,
        operation_id: str,
        spec: dict,
    ):
        if not isinstance(replica_key, ReplicaKey):
            raise TypeError("prepare_replica requires ReplicaKey")
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("prepare_replica requires operation_id")
        if not isinstance(spec, dict):
            raise TypeError("prepare_replica requires placement spec")
        if spec.get("operation_id") != operation_id:
            raise ValueError("placement spec belongs to another operation")
        if spec.get("borrower_task_id") != replica_key.task_session:
            raise ValueError("placement spec belongs to another task_session")
        if spec.get("borrower_replica_id") != replica_key.replica_id:
            raise ValueError("placement spec belongs to another replica")

        previous_target = self._pending_operation_targets.get(operation_id)
        if previous_target is not None and previous_target != replica_key:
            raise ValueError("operation_id is already bound to another replica")
        self._pending_operation_targets[operation_id] = replica_key
        return await self.create_borrowed_replica(spec)

    async def prepare_exit(
        self,
        replica_key: ReplicaKey,
        *,
        operation_id: str,
        force: bool = False,
    ) -> OperationEvidence:
        """Close admission and wait for a natural drain without holding Trainer G."""
        if not isinstance(replica_key, ReplicaKey):
            raise TypeError("prepare_exit requires ReplicaKey")
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("prepare_exit requires operation_id")
        if type(force) is not bool:
            raise TypeError("force must be bool")

        manager = getattr(self, "llm_server_manager", None)
        if manager is None:
            raise RuntimeError("LLM server manager is not initialized")
        kind, state = manager.replica_meta(replica_key)

        if force:
            if kind is not ReplicaKind.BORROWED:
                raise ValueError("FORCE REMOVE is borrowed-only")
            if state is not ReplicaState.ACTIVE:
                raise ValueError(
                    f"FORCE prepare_exit requires ACTIVE replica, got {state.value}"
                )
            raise NotImplementedError(
                "FORCE prepare_exit requires verified targeted abort/continuation backend"
            )

        previous_target = self._pending_operation_targets.get(operation_id)
        if previous_target is not None and previous_target != replica_key:
            raise ValueError("operation_id is already bound to another replica")
        self._pending_operation_targets[operation_id] = replica_key

        if state is ReplicaState.ACTIVE:
            manager.transition_replica(replica_key, ReplicaState.DRAINING)
        elif state is not ReplicaState.DRAINING:
            raise ValueError(
                f"prepare_exit requires ACTIVE/DRAINING replica, got {state.value}"
            )

        lb = manager.global_load_balancer
        server_id = await lb.begin_drain.remote(replica_key, operation_id)

        while await lb.has_unsettled_requests.remote(server_id):
            await asyncio.sleep(0.1)

        return OperationEvidence.now(operation_id, EvidenceType.EXIT_READY)

    def get_pending_target(self, operation_id: str) -> ReplicaKey:
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id must be a nonempty string")
        try:
            return self._pending_operation_targets[operation_id]
        except KeyError as exc:
            raise KeyError(f"no pending lifecycle target for {operation_id!r}") from exc

    async def commit_service_change(self, operation: OperationRecord) -> OperationEvidence:
        """Commit the R/C part of a drained exit while Manager keeps M=DRAINING."""
        if not isinstance(operation, OperationRecord):
            raise TypeError("commit_service_change requires OperationRecord")
        target = self.get_pending_target(operation.operation_id)
        manager = self.llm_server_manager
        _kind, state = manager.replica_meta(target)
        if state is not ReplicaState.DRAINING:
            raise ValueError(
                f"service exit commit requires DRAINING replica, got {state.value}"
            )

        lb = manager.global_load_balancer
        try:
            server_id = await lb.server_for_replica.remote(target)
            if server_id is not None and await lb.has_unsettled_requests.remote(server_id):
                raise ValueError("cannot commit service exit while requests remain unsettled")
            await lb.finish_remove.remote(target)
            manager.deactivate_service(target)
            self._update_max_concurrent_samples()
        except BaseException:
            if manager.replica_state.get(target) is ReplicaState.DRAINING:
                manager.transition_replica(target, ReplicaState.QUARANTINED)
            raise

        return OperationEvidence.now(
            operation.operation_id,
            EvidenceType.SERVICE_COMMITTED,
        )

    async def finalize_release(self, operation: OperationRecord) -> OperationEvidence:
        """Close M only from verified runtime release evidence.

        R/C/E were already committed before this phase. If the runtime backend
        cannot prove sleep/destroy, M must not remain deceptively DRAINING: the
        target is quarantined and the lifecycle operation stays unresolved.
        """
        if not isinstance(operation, OperationRecord):
            raise TypeError("finalize_release requires OperationRecord")

        target = self.get_pending_target(operation.operation_id)
        manager = self.llm_server_manager
        kind, state = manager.replica_meta(target)
        if state is not ReplicaState.DRAINING:
            raise ValueError(
                f"finalize_release requires DRAINING replica, got {state.value}"
            )

        try:
            if kind is ReplicaKind.NATIVE:
                evidence = manager.sleep(
                    target,
                    operation_id=operation.operation_id,
                )
                target_state = ReplicaState.DORMANT
            else:
                evidence = await manager.destroy(
                    target,
                    operation_id=operation.operation_id,
                )
                target_state = ReplicaState.RELEASED

            if not isinstance(evidence, OperationEvidence):
                raise TypeError("runtime release did not return OperationEvidence")
            if evidence.operation_id != operation.operation_id:
                raise ValueError("release evidence belongs to another operation")
            if evidence.type is not EvidenceType.RELEASED:
                raise ValueError(
                    f"expected RELEASED evidence, got {evidence.type.value}"
                )

            manager.transition_replica(target, target_state)
            self._pending_operation_targets.pop(operation.operation_id, None)
            return evidence
        except BaseException:
            if manager.replica_state.get(target) is ReplicaState.DRAINING:
                manager.transition_replica(target, ReplicaState.QUARANTINED)
            raise
