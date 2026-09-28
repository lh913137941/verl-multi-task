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

from verl.single_controller.ray.base import _unwrap_ray_remote
from multi_task_scheduler.orchestration.contracts import (
    AttemptState,
    EvidenceType,
    OperationEvidence,
    OperationRecord,
    ReplicaKey,
    ReplicaKind,
    ReplicaState,
)

from .llm_server_manager import MultiTaskLLMServerManager


def _require_evidence(value, operation_id: str, expected: EvidenceType, label: str):
    if not isinstance(value, OperationEvidence):
        raise TypeError(f"{label} did not return OperationEvidence")
    if value.operation_id != operation_id or value.type is not expected:
        raise ValueError(f"{label} evidence does not match {operation_id}/{expected.value}")
    return value


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
            # Ray may surface a remote KeyError as RayTaskError(cause=KeyError).
            cause = getattr(exc, "cause", None)
            if not isinstance(exc, KeyError) and not isinstance(cause, KeyError):
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
class MultiTaskFullyAsyncRollouter(_unwrap_ray_remote(FullyAsyncRollouter)):
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

    def collect_idle_candidates(self) -> tuple[tuple[ReplicaKey, ReplicaKind], ...]:
        """Return only ACTIVE replicas whose removal preserves committed C.

        Native ``max_concurrent_samples`` is capped by ``max_required_samples``.
        Compute how many ACTIVE replicas are required to preserve the currently
        committed capacity and report only the surplus. This stays Rollouter-local
        and deliberately does not read LB/R.
        """
        committed_capacity = int(getattr(self, "max_concurrent_samples", 0) or 0)
        if not getattr(self, "paused", False) or committed_capacity <= 0:
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
        required_active = (committed_capacity + per_replica - 1) // per_replica
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

    async def prepare_replica(
        self,
        replica_key: ReplicaKey,
        *,
        operation_id: str,
        spec: dict | None,
    ):
        """Prepare ADD placement or stage a retained native runtime for RESTORE."""
        if not isinstance(replica_key, ReplicaKey):
            raise TypeError("prepare_replica requires ReplicaKey")
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("prepare_replica requires operation_id")

        previous_target = self._pending_operation_targets.get(operation_id)
        if previous_target is not None and previous_target != replica_key:
            raise ValueError("operation_id is already bound to another replica")

        manager = getattr(self, "llm_server_manager", None)
        if manager is None:
            raise RuntimeError("LLM server manager is not initialized")
        if spec is None:
            kind, state = manager.replica_meta(replica_key)
            if kind is not ReplicaKind.NATIVE or state is not ReplicaState.DORMANT:
                raise ValueError(
                    "spec-free prepare_replica is valid only for DORMANT native RESTORE"
                )
            self._pending_operation_targets[operation_id] = replica_key
            # RESTORE GPU/weight mutation must happen under Trainer G.  This
            # pre-step only binds the operation to its retained DORMANT target.
            return None

        if not isinstance(spec, dict):
            raise TypeError("prepare_replica requires placement spec or None for RESTORE")
        if spec.get("operation_id") != operation_id:
            raise ValueError("placement spec belongs to another operation")
        if spec.get("borrower_task_id") != replica_key.task_session:
            raise ValueError("placement spec belongs to another task_session")
        if spec.get("borrower_replica_id") != replica_key.replica_id:
            raise ValueError("placement spec belongs to another replica")

        self._pending_operation_targets[operation_id] = replica_key
        return await manager.create_borrowed_replica(spec)

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
                raise ValueError("FORCE REMOVE requires async_training.partial_rollout=true")

            server_id = await lb.server_for_replica.remote(replica_key)
            if not isinstance(server_id, str) or not server_id:
                raise RuntimeError("FORCE target has no active route")
            active_servers = tuple(await lb.get_all_servers.remote())
            if not any(candidate != server_id for candidate in active_servers):
                raise RuntimeError("FORCE REMOVE requires another active rollout server")

            runtime = manager.inspect_runtime(replica_key)
            if runtime is None:
                raise RuntimeError("FORCE target runtime is unavailable")

            manager.transition_replica(replica_key, ReplicaState.DRAINING)
            self._pending_operation_targets[operation_id] = replica_key
            try:
                try:
                    drained_server = await lb.begin_drain.remote(
                        replica_key, operation_id
                    )
                except BaseException:
                    drained_server = await lb.begin_drain.remote(
                        replica_key, operation_id
                    )
                if drained_server != server_id:
                    raise RuntimeError("FORCE drain bound a different server")

                admitted = tuple(
                    request_id
                    for request_id in await lb.requests_for_server.remote(server_id)
                    if await lb.query_attempt.remote(request_id)
                    is AttemptState.ADMITTED
                )
                abort_result = await runtime.abort_all_requests()
                if not isinstance(abort_result, dict):
                    raise TypeError("FORCE abort returned a non-dict result")
                aborted_count = abort_result.get("aborted_count")
                if type(aborted_count) is not int or aborted_count < 0:
                    raise RuntimeError("FORCE abort did not report aborted_count")
                if aborted_count > len(admitted):
                    raise RuntimeError(
                        "FORCE abort count exceeds LB requests owned by target"
                    )

                loop = asyncio.get_running_loop()
                deadline = loop.time() + 30.0
                while True:
                    handoffs = set(
                        await lb.continuation_handoff_requests.remote(operation_id)
                    )
                    unsettled = await lb.has_unsettled_requests.remote(server_id)
                    if not unsettled and len(handoffs.intersection(admitted)) >= aborted_count:
                        break
                    if loop.time() >= deadline:
                        raise TimeoutError(
                            "FORCE continuation handoff did not complete before timeout"
                        )
                    await asyncio.sleep(0.05)

                return OperationEvidence.now(operation_id, EvidenceType.EXIT_READY)
            except BaseException:
                if manager.replica_state.get(replica_key) is ReplicaState.DRAINING:
                    manager.transition_replica(replica_key, ReplicaState.QUARANTINED)
                raise

        if state not in {ReplicaState.ACTIVE, ReplicaState.DRAINING}:
            raise ValueError(
                f"prepare_exit requires ACTIVE/DRAINING replica, got {state.value}"
            )
        if state is ReplicaState.ACTIVE:
            manager.transition_replica(replica_key, ReplicaState.DRAINING)
        self._pending_operation_targets[operation_id] = replica_key

        try:
            try:
                server_id = await lb.begin_drain.remote(replica_key, operation_id)
            except BaseException:
                # begin_drain is idempotent for the same operation. One retry
                # reconciles the common case where R committed but the reply was
                # lost; a second failure leaves the drain outcome unverified.
                server_id = await lb.begin_drain.remote(replica_key, operation_id)

            while await lb.has_unsettled_requests.remote(server_id):
                await asyncio.sleep(0.1)
        except BaseException:
            if manager.replica_state.get(replica_key) is ReplicaState.DRAINING:
                manager.transition_replica(replica_key, ReplicaState.QUARANTINED)
            raise

        return OperationEvidence.now(operation_id, EvidenceType.EXIT_READY)

    def get_pending_target(self, operation_id: str) -> ReplicaKey:
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id must be a nonempty string")
        try:
            return self._pending_operation_targets[operation_id]
        except KeyError as exc:
            raise KeyError(f"no pending lifecycle target for {operation_id!r}") from exc

    def get_pending_replicas(self, operation_id: str):
        """Return the prepared runtime bound to one lifecycle operation."""
        target = self.get_pending_target(operation_id)
        runtime = self.llm_server_manager.inspect_runtime(target)
        if runtime is None:
            raise RuntimeError("pending lifecycle target has no runtime")
        return (runtime,)

    async def commit_service_change(self, operation: OperationRecord) -> OperationEvidence:
        """Commit the R/C part of a drained exit while Manager keeps M=DRAINING."""
        if not isinstance(operation, OperationRecord):
            raise TypeError("commit_service_change requires OperationRecord")
        target = self.get_pending_target(operation.operation_id)
        manager = self.llm_server_manager
        kind, state = manager.replica_meta(target)
        lb = manager.global_load_balancer

        if state is ReplicaState.CREATING:
            if kind is not ReplicaKind.BORROWED:
                raise ValueError("only BORROWED replicas may publish from CREATING")
            runtime = manager.inspect_runtime(target)
            if runtime is None:
                raise RuntimeError("borrowed runtime is unavailable for ADD commit")
            server_id = getattr(runtime, "_server_address", None)
            server_handle = getattr(runtime, "_server_handle", None)
            if not isinstance(server_id, str) or not server_id or server_handle is None:
                raise RuntimeError("borrowed ADD runtime lacks routable server identity")

            activated = False
            try:
                manager.activate_service(target)
                activated = True
                self._update_max_concurrent_samples()
                manager.transition_replica(target, ReplicaState.ACTIVE)
                try:
                    evidence = await lb.commit_ready.remote(
                        target,
                        server_id,
                        server_handle,
                        operation.operation_id,
                    )
                    _require_evidence(
                        evidence,
                        operation.operation_id,
                        EvidenceType.SERVICE_COMMITTED,
                        "ADD routing commit",
                    )
                except BaseException as commit_exc:
                    try:
                        reconciled = await lb.query_ready_operation.remote(
                            operation.operation_id
                        )
                    except BaseException as reconcile_exc:
                        raise RuntimeError(
                            "ADD routing commit outcome is unknown"
                        ) from reconcile_exc
                    if reconciled is not None:
                        evidence = _require_evidence(
                            reconciled,
                            operation.operation_id,
                            EvidenceType.SERVICE_COMMITTED,
                            "ADD routing ledger",
                        )
                    else:
                        manager.transition_replica(target, ReplicaState.DRAINING)
                        if activated:
                            manager.deactivate_service(target)
                            self._update_max_concurrent_samples()
                        manager.transition_replica(target, ReplicaState.QUARANTINED)
                        raise commit_exc

                self._pending_operation_targets.pop(operation.operation_id, None)
                return evidence
            except BaseException:
                if manager.replica_state.get(target) is ReplicaState.CREATING:
                    if activated:
                        try:
                            manager.deactivate_service(target)
                            self._update_max_concurrent_samples()
                        except BaseException:
                            pass
                    manager.transition_replica(target, ReplicaState.QUARANTINED)
                raise

        # RESTORE reuses the same service-commit boundary after Trainer has
        # proved current Vpub under G.  The vLLM engine may already be fully
        # resident because CE resumed KV cache; runtime.wake_up() then acts as
        # the final local-admission + health commit before R/C/M are published.
        if state is ReplicaState.DORMANT:
            if kind is not ReplicaKind.NATIVE:
                raise ValueError("only NATIVE replicas may restore from DORMANT")
            runtime = None
            activated = False
            try:
                runtime = manager.inspect_runtime(target)
                if runtime is None:
                    raise RuntimeError("native runtime is unavailable for RESTORE commit")
                await runtime.wake_up()

                server_id = getattr(runtime, "_server_address", None)
                server_handle = getattr(runtime, "_server_handle", None)
                if not isinstance(server_id, str) or not server_id or server_handle is None:
                    raise RuntimeError("native RESTORE runtime lacks routable server identity")

                # Trainer has already committed current-Vpub membership into E.
                # Prepare owner-local service state before the externally visible
                # R commit, but never re-sleep from this point without also
                # rolling E back under G.
                manager.activate_service(target)
                activated = True
                self._update_max_concurrent_samples()
                manager.transition_replica(target, ReplicaState.ACTIVE)

                try:
                    evidence = await lb.commit_ready.remote(
                        target,
                        server_id,
                        server_handle,
                        operation.operation_id,
                    )
                    _require_evidence(
                        evidence,
                        operation.operation_id,
                        EvidenceType.SERVICE_COMMITTED,
                        "RESTORE routing commit",
                    )
                except BaseException as commit_exc:
                    try:
                        reconciled = await lb.query_ready_operation.remote(
                            operation.operation_id
                        )
                    except BaseException as reconcile_exc:
                        # R may already be open. Keep the current-Vpub runtime
                        # awake and M/C ACTIVE; Trainer fences G on this error.
                        raise RuntimeError(
                            "RESTORE routing commit outcome is unknown"
                        ) from reconcile_exc

                    if reconciled is not None:
                        try:
                            evidence = _require_evidence(
                                reconciled,
                                operation.operation_id,
                                EvidenceType.SERVICE_COMMITTED,
                                "RESTORE routing ledger",
                            )
                        except (TypeError, ValueError) as exc:
                            raise RuntimeError(
                                "RESTORE routing ledger conflicts with the operation"
                            ) from exc
                    else:
                        # E is already effective, so a definite R failure cannot
                        # safely re-sleep here. Retract local service/capacity and
                        # quarantine M; Trainer will block G for reconciliation.
                        manager.transition_replica(target, ReplicaState.DRAINING)
                        if activated:
                            manager.deactivate_service(target)
                            activated = False
                            self._update_max_concurrent_samples()
                        manager.transition_replica(target, ReplicaState.QUARANTINED)
                        raise commit_exc

                self._pending_operation_targets.pop(operation.operation_id, None)
                return evidence
            except BaseException:
                # Failures before ACTIVE/R publication still happen after E was
                # committed. Do not manufacture DORMANT by sleeping behind E's
                # back; quarantine the local runtime projection instead.
                if manager.replica_state.get(target) is ReplicaState.DORMANT:
                    if activated:
                        try:
                            manager.deactivate_service(target)
                            self._update_max_concurrent_samples()
                        except BaseException:
                            pass
                    manager.transition_replica(target, ReplicaState.QUARANTINED)
                raise

        if state is not ReplicaState.DRAINING:
            raise ValueError(
                f"service exit commit requires DRAINING replica, got {state.value}"
            )

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
        add_rollback = kind is ReplicaKind.BORROWED and state is ReplicaState.CREATING
        if not add_rollback and state is not ReplicaState.DRAINING:
            raise ValueError(
                f"finalize_release requires DRAINING or borrowed CREATING replica, got {state.value}"
            )

        try:
            if kind is ReplicaKind.NATIVE:
                evidence = await manager.sleep(
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

            _require_evidence(
                evidence,
                operation.operation_id,
                EvidenceType.RELEASED,
                "runtime release",
            )
            manager.transition_replica(target, target_state)
            self._pending_operation_targets.pop(operation.operation_id, None)
            return evidence
        except BaseException:
            if manager.replica_state.get(target) in {
                ReplicaState.CREATING,
                ReplicaState.DRAINING,
            }:
                manager.transition_replica(target, ReplicaState.QUARANTINED)
            raise
