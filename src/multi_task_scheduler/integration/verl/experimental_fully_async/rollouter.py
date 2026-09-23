"""Experimental Fully Async Rollouter aligned with the 092203 owner boundaries."""

from __future__ import annotations

import asyncio
import hashlib

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


def _continuation_prefix_digest(prompt_ids, token_ids) -> str:
    """Digest the exact token prefix the native FullyAsync client will retry."""
    digest = hashlib.sha256()
    for token_id in tuple(prompt_ids or ()) + tuple(token_ids or ()):
        digest.update(str(int(token_id)).encode("ascii"))
        digest.update(b",")
    return digest.hexdigest()


class _ContinuationAwareGenerate:
    def __init__(self, owner):
        self._owner = owner

    def remote(self, *args, **kwargs):
        return self._owner._generate(*args, **kwargs)


class _ContinuationAwareServer:
    """Observe native aborted outputs and record FORCE handoff evidence."""

    def __init__(self, server, load_balancer, logical_request_id: str, client_id: str):
        self._server = server
        self._load_balancer = load_balancer
        self._logical_request_id = logical_request_id
        self._client_id = client_id
        self.generate = _ContinuationAwareGenerate(self)

    async def _generate(self, *args, **kwargs):
        output = await self._server.generate.remote(*args, **kwargs)
        if getattr(output, "stop_reason", None) not in {"abort", "aborted"}:
            return output

        # FullyAsyncLLMServerClient will retry exactly prompt_ids + token_ids.
        prefix_digest = _continuation_prefix_digest(
            kwargs.get("prompt_ids", ()),
            getattr(output, "token_ids", ()),
        )
        try:
            await self._load_balancer.confirm_continuation.remote(
                self._logical_request_id,
                self._client_id,
                prefix_digest,
            )
        except Exception as exc:
            # Native Fully Async also aborts for ordinary weight-sync/rebalance.
            # Only an active lifecycle drain needs continuation evidence.
            if "not part of an active drain operation" not in str(exc):
                raise
        return output


class _MultiTaskFullyAsyncLLMServerClient(FullyAsyncLLMServerClient):
    """Reuse VERL abort-resume and add evidence for FORCE REMOVE only."""

    def __init__(self, *args, client_id: str, **kwargs):
        if not isinstance(client_id, str) or not client_id:
            raise ValueError("continuation client_id must be a nonempty string")
        super().__init__(*args, **kwargs)
        self._continuation_client_id = client_id
        async_training = getattr(self.config, "async_training", None)
        self._continuation_enabled = bool(
            getattr(async_training, "partial_rollout", False)
        )

    async def _acquire_server(self, request_id: str, **extra):
        server_id, server = await super()._acquire_server(request_id, **extra)
        if not self._continuation_enabled:
            return server_id, server
        return server_id, _ContinuationAwareServer(
            server,
            self._load_balancer,
            request_id,
            self._continuation_client_id,
        )


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
        self._continuation_client_ready = False
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
                client_cls=_MultiTaskFullyAsyncLLMServerClient,
                client_id=self.task_session,
            ),
            reward_loop_worker_handles=reward_loop_worker_handles,
            teacher_client=(
                self.teacher_model_manager.get_client()
                if self.teacher_model_manager
                else None
            ),
        )
        self._continuation_client_ready = True

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

        previous_target = self._pending_operation_targets.get(operation_id)
        if previous_target is not None and previous_target != replica_key:
            raise ValueError("operation_id is already bound to another replica")

        lb = manager.global_load_balancer
        if force:
            if kind is not ReplicaKind.BORROWED:
                raise ValueError("FORCE REMOVE is borrowed-only")
            if state is not ReplicaState.ACTIVE:
                raise ValueError(
                    f"FORCE prepare_exit requires ACTIVE replica, got {state.value}"
                )

            async_training = getattr(self.config, "async_training", None)
            if not bool(getattr(async_training, "partial_rollout", False)):
                raise RuntimeError(
                    "FORCE REMOVE requires async_training.partial_rollout=true"
                )
            if not self._continuation_client_ready:
                raise RuntimeError(
                    "FORCE REMOVE continuation-aware client is not ready"
                )

            runtime = manager.inspect_runtime(replica_key)
            if runtime is None or not callable(
                getattr(runtime, "abort_all_requests", None)
            ):
                raise RuntimeError(
                    "FORCE REMOVE target runtime cannot abort in-flight requests"
                )

            target_server = await lb.server_for_replica.remote(replica_key)
            if not target_server:
                raise RuntimeError("FORCE REMOVE target has no routable server")
            available_servers = set(await lb.get_all_servers.remote())
            if target_server not in available_servers:
                raise RuntimeError(
                    "FORCE REMOVE target server is not active in the load balancer"
                )
            if not (available_servers - {target_server}):
                raise RuntimeError(
                    "FORCE REMOVE requires another healthy rollout server"
                )

            self._pending_operation_targets[operation_id] = replica_key
            manager.transition_replica(replica_key, ReplicaState.DRAINING)
            try:
                server_id = await lb.begin_drain.remote(replica_key, operation_id)
                if server_id != target_server:
                    raise RuntimeError(
                        "FORCE drain target changed during preparation"
                    )

                # VERL vLLM abort produces stop_reason=aborted/abort. The
                # continuation-aware client records confirm_continuation()
                # before native release_server(), then the native FullyAsync
                # retry loop re-acquires another server with prompt+partial tokens.
                await runtime.abort_all_requests()

                while await lb.has_unsettled_requests.remote(server_id):
                    await asyncio.sleep(0.1)
                return OperationEvidence.now(
                    operation_id,
                    EvidenceType.EXIT_READY,
                )
            except BaseException:
                if manager.replica_state.get(replica_key) is ReplicaState.DRAINING:
                    manager.transition_replica(
                        replica_key,
                        ReplicaState.QUARANTINED,
                    )
                raise

        self._pending_operation_targets[operation_id] = replica_key
        if state is ReplicaState.ACTIVE:
            manager.transition_replica(replica_key, ReplicaState.DRAINING)
        elif state is not ReplicaState.DRAINING:
            raise ValueError(
                f"prepare_exit requires ACTIVE/DRAINING replica, got {state.value}"
            )

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
