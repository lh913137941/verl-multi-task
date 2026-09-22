"""Experimental Fully Async Rollouter aligned with the 092203 owner boundaries."""

from __future__ import annotations

import time

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
            group_scheduler=self.group_scheduler,
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
        return int(getattr(self, "max_concurrent_samples", 0))

    @property
    def production_window_open(self) -> bool:
        return not bool(getattr(self, "paused", False))

    def collect_idle_candidates(self) -> tuple[tuple[ReplicaKey, ReplicaKind], ...]:
        if self.production_window_open or self.committed_capacity <= 0:
            return ()
        manager = getattr(self, "llm_server_manager", None)
        if manager is None:
            return ()
        return tuple(
            (key, manager.replica_kind[key])
            for key, state in manager.replica_state.items()
            if state is ReplicaState.ACTIVE
        )

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

    def prepare_replica(self, replica_key: ReplicaKey):
        if not isinstance(replica_key, ReplicaKey):
            raise TypeError("prepare_replica requires ReplicaKey")
        raise NotImplementedError(
            "prepare_replica requires verified borrowed-runtime backend"
        )

    def prepare_exit(
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
            # Do not close admission or mutate M before the validated continuation
            # backend exists. A placeholder must never strand a live replica.
            raise NotImplementedError(
                "FORCE prepare_exit requires verified targeted abort/continuation backend"
            )

        if state is ReplicaState.ACTIVE:
            manager.transition_replica(replica_key, ReplicaState.DRAINING)
        elif state is not ReplicaState.DRAINING:
            raise ValueError(
                f"prepare_exit requires ACTIVE/DRAINING replica, got {state.value}"
            )

        lb = manager.global_load_balancer
        server_id = ray.get(lb.begin_drain.remote(replica_key), timeout=30)

        # Natural draining has no success-by-timeout rule. Keep waiting until every
        # request admitted on the target reaches a legal terminal state. Final route
        # deletion belongs to the later service commit under Trainer G.
        while ray.get(lb.has_unsettled_requests.remote(server_id), timeout=30):
            time.sleep(0.1)

        return OperationEvidence.now(operation_id, EvidenceType.EXIT_READY)

    def commit_service_change(self, operation: OperationRecord):
        if not isinstance(operation, OperationRecord):
            raise TypeError("commit_service_change requires OperationRecord")
        raise NotImplementedError(
            "commit_service_change requires verified R/C/M commit wiring"
        )

    def finalize_release(self, operation: OperationRecord):
        if not isinstance(operation, OperationRecord):
            raise TypeError("finalize_release requires OperationRecord")
        raise NotImplementedError(
            "finalize_release requires verified sleep/destroy and exact GPU release evidence"
        )
