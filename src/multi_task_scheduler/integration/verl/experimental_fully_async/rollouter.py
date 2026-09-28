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
            try:
                cause = exc.cause
            except Exception:
                cause = None
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
        # FORCE keeps only operation-local recovery facts: the LB requests that
        # were ADMITTED before abort and the backend-confirmed abort count.
        # None means the abort RPC outcome itself was not proven.
        self._force_exit_recovery: dict[
            str, tuple[tuple[str, ...], int | None]
        ] = {}
        self._force_handoff_timeout_s = 30.0
        multitask_config = getattr(config, "multitask", None)
        if multitask_config is None and isinstance(config, dict):
            multitask_config = config.get("multitask")
        timeout_value = (
            multitask_config.get("drain_timeout_s", 300.0)
            if multitask_config is not None
            else 300.0
        )
        self._natural_drain_timeout_s = float(timeout_value)
        self._idle_report_signature = None
        self._idle_report_last_sent = None
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

    async def submit_idle_report(self):
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
        return await self.group_scheduler.submit_idle_report.remote(report)

    async def _idle_report_loop(self):
        """Emit one metadata report per distinct paused surplus set."""
        while True:
            await asyncio.sleep(1.0)
            # fit() owns cancellation of this reporter. Do not exit merely
            # because native super().fit() has not set running=True yet.
            if not getattr(self, "running", False):
                self._idle_report_signature = None
                continue
            if not getattr(self, "paused", False) or self.group_scheduler is None:
                self._idle_report_signature = None
                self._idle_report_last_sent = None
                continue

            candidates = self.collect_idle_candidates()
            signature = tuple(
                (key.task_session, key.replica_id, key.runtime_epoch, kind.value)
                for key, kind in candidates
            )
            if not signature:
                continue
            now = asyncio.get_running_loop().time()
            if (
                signature == self._idle_report_signature
                and self._idle_report_last_sent is not None
                and now - self._idle_report_last_sent < 5.0
            ):
                continue
            try:
                await self.submit_idle_report()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Idle reports are advisory metadata. Failure must not stop
                # rollout production; retry on the next monitor tick.
                print(
                    "[MultiTaskRollouter] idle report failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                self._idle_report_signature = signature
                self._idle_report_last_sent = now

    async def fit(self):
        reporter = asyncio.create_task(self._idle_report_loop())
        try:
            return await super().fit()
        finally:
            reporter.cancel()
            await asyncio.gather(reporter, return_exceptions=True)

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
            if kind is not ReplicaKind.NATIVE:
                raise ValueError(
                    "spec-free prepare_replica is valid only for native RESTORE"
                )
            if state is ReplicaState.DORMANT:
                self._pending_operation_targets[operation_id] = replica_key
                # RESTORE GPU/weight mutation must happen under Trainer G. This
                # pre-step only binds the operation to its retained DORMANT target.
                return None
            if (
                previous_target == replica_key
                and state is ReplicaState.ACTIVE
            ):
                # Exact UNKNOWN replay: runtime/E may already be active while R
                # publication is unresolved. Trainer G owns the reconciliation.
                return None
            raise ValueError(
                "spec-free prepare_replica requires DORMANT or same-operation ACTIVE reconciliation"
            )

        if not isinstance(spec, dict):
            raise TypeError("prepare_replica requires placement spec or None for RESTORE")
        if spec.get("operation_id") != operation_id:
            raise ValueError("placement spec belongs to another operation")
        if spec.get("borrower_task_id") != replica_key.task_session:
            raise ValueError("placement spec belongs to another task_session")
        if spec.get("borrower_replica_id") != replica_key.replica_id:
            raise ValueError("placement spec belongs to another replica")

        self._pending_operation_targets[operation_id] = replica_key
        try:
            return await manager.create_borrowed_replica(spec)
        except BaseException:
            state = manager.replica_state.get(replica_key)
            kind = manager.replica_kind.get(replica_key)
            # create_borrowed_replica reaches RELEASED only after verified
            # cleanup. A missing M entry means validation failed before runtime
            # registration, which is also a zero-side-effect failure.
            if state is None or (
                kind is ReplicaKind.BORROWED and state is ReplicaState.RELEASED
            ):
                gpu_uuids = tuple(
                    dict.fromkeys(
                        claim.get("gpu_uuid")
                        for claim in tuple(spec.get("claims") or ())
                        if isinstance(claim, dict)
                        and isinstance(claim.get("gpu_uuid"), str)
                        and claim["gpu_uuid"]
                    )
                )
                if gpu_uuids:
                    self._pending_operation_targets.pop(operation_id, None)
                    return OperationEvidence.now(
                        operation_id,
                        EvidenceType.RELEASED,
                        released_gpu_uuids=gpu_uuids,
                    )
            raise

    def quarantine_dormant_restore(self, operation_id: str) -> bool:
        """Project an uncertain partially-awake RESTORE into M=QUARANTINED."""
        target = self.get_pending_target(operation_id)
        manager = self.llm_server_manager
        kind, state = manager.replica_meta(target)
        if kind is not ReplicaKind.NATIVE:
            raise ValueError("RESTORE quarantine is valid only for NATIVE replicas")
        if state is ReplicaState.QUARANTINED:
            return True
        if state is ReplicaState.DORMANT:
            manager.transition_replica(target, ReplicaState.QUARANTINED)
            return True
        # ACTIVE/DRAINING carry stronger publication/rollback facts and must be
        # reconciled by the existing service path rather than overwritten here.
        return False

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
            same_operation_resume = (
                state is ReplicaState.DRAINING
                and previous_target == replica_key
            )
            if state is not ReplicaState.ACTIVE and not same_operation_resume:
                raise ValueError(
                    "FORCE prepare_exit requires ACTIVE replica or same-operation "
                    f"DRAINING reconciliation, got {state.value}"
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

            if state is ReplicaState.ACTIVE:
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

                recovery = self._force_exit_recovery.get(operation_id)
                if recovery is None:
                    admitted_list = []
                    for request_id in await lb.requests_for_server.remote(server_id):
                        if (
                            await lb.query_attempt.remote(request_id)
                            is AttemptState.ADMITTED
                        ):
                            admitted_list.append(request_id)
                    admitted = tuple(admitted_list)
                    # Persist the pre-abort request set before invoking the
                    # backend. If its result is lost, same-op replay can remain
                    # fail-closed without re-aborting already handed-off work.
                    self._force_exit_recovery[operation_id] = (admitted, None)

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
                    self._force_exit_recovery[operation_id] = (
                        admitted,
                        aborted_count,
                    )
                else:
                    admitted, aborted_count = recovery

                loop = asyncio.get_running_loop()
                deadline = loop.time() + self._force_handoff_timeout_s
                while True:
                    handoffs = set(
                        await lb.continuation_handoff_requests.remote(operation_id)
                    )
                    unsettled = await lb.has_unsettled_requests.remote(server_id)
                    confirmed = len(handoffs.intersection(admitted))
                    if aborted_count is None:
                        # The abort ACK itself was lost. Do not repeat abort.
                        # Only a continuation proof for every request that was
                        # ADMITTED at the boundary can recover this conservatively.
                        handoff_complete = confirmed == len(admitted)
                    else:
                        handoff_complete = confirmed >= aborted_count
                    if not unsettled and handoff_complete:
                        break
                    if loop.time() >= deadline:
                        raise TimeoutError(
                            "FORCE continuation handoff did not complete before timeout"
                        )
                    await asyncio.sleep(0.05)

                return OperationEvidence.now(operation_id, EvidenceType.EXIT_READY)
            except BaseException:
                # Once FORCE closes admission or abort may have happened, keep
                # DRAINING and preserve the same operation's request facts.
                # Replays resume this operation; they never reset M to ACTIVE
                # and never blindly repeat an already-issued abort.
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

            loop = asyncio.get_running_loop()
            if self._natural_drain_timeout_s <= 0:
                raise ValueError("multitask.drain_timeout_s must be positive")
            deadline = loop.time() + self._natural_drain_timeout_s
            next_log = loop.time() + 10.0
            while await lb.has_unsettled_requests.remote(server_id):
                now = loop.time()
                if now >= deadline:
                    raise TimeoutError(
                        "natural drain did not settle before "
                        f"{self._natural_drain_timeout_s:.1f}s deadline"
                    )
                if now >= next_log:
                    print(
                        "[MultiTaskRollouter] waiting for natural drain: "
                        f"operation={operation_id}, server={server_id}, "
                        f"remaining_budget={deadline - now:.1f}s"
                    )
                    next_log = now + 10.0
                await asyncio.sleep(min(0.1, max(0.0, deadline - now)))
        except BaseException:
            # Natural drain timeout/ACK uncertainty is resumable. Preserve M as
            # DRAINING and the same operation binding; do not silently upgrade
            # to FORCE or quarantine away the only safe replay path.
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

    def query_release_operation(
        self,
        target: ReplicaKey,
        operation_id: str,
    ) -> OperationEvidence | None:
        """Read Manager-owned RELEASED evidence for exact lifecycle replay."""
        if not isinstance(target, ReplicaKey):
            raise TypeError("query_release_operation requires ReplicaKey")
        return self.llm_server_manager.query_release_evidence(target, operation_id)

    async def commit_service_change(
        self, operation: OperationRecord
    ) -> OperationEvidence | None:
        """Commit the R/C part of a drained exit while Manager keeps M=DRAINING."""
        if not isinstance(operation, OperationRecord):
            raise TypeError("commit_service_change requires OperationRecord")
        target = self.get_pending_target(operation.operation_id)
        manager = self.llm_server_manager
        kind, state = manager.replica_meta(target)
        lb = manager.global_load_balancer

        if state is ReplicaState.ACTIVE:
            # The only lifecycle path that leaves ACTIVE together with a pending
            # operation is an ADD/RESTORE publication whose R outcome was
            # ambiguous. Query R; never infer success from local M/C.
            try:
                reconciled = await lb.query_ready_operation.remote(
                    operation.operation_id
                )
            except BaseException as exc:
                raise RuntimeError(
                    "service publication remains unknown during reconciliation"
                ) from exc
            if reconciled is not None:
                evidence = _require_evidence(
                    reconciled,
                    operation.operation_id,
                    EvidenceType.SERVICE_COMMITTED,
                    "routing reconciliation",
                )
                self._pending_operation_targets.pop(operation.operation_id, None)
                return evidence

            # R authoritatively reports no publish. Retract owner-local service
            # state while Trainer still owns G; Trainer will remove E before
            # finalize_release performs destroy/re-sleep.
            manager.transition_replica(target, ReplicaState.DRAINING)
            manager.deactivate_service(target)
            self._update_max_concurrent_samples()
            return None

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
                        # R authoritatively reports no publish. Retract C/M but
                        # do not destroy while E still says the runtime is an
                        # effective receiver. Trainer owns G and will remove E
                        # before calling finalize_release().
                        manager.transition_replica(target, ReplicaState.DRAINING)
                        if activated:
                            manager.deactivate_service(target)
                            activated = False
                            self._update_max_concurrent_samples()
                        return None

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
                        # R is definitely absent. Retract C/M while Trainer still
                        # owns G, then return a definite no-publish marker. Trainer
                        # removes E before finalize_release() re-enters level-2 sleep.
                        manager.transition_replica(target, ReplicaState.DRAINING)
                        if activated:
                            manager.deactivate_service(target)
                            activated = False
                            self._update_max_concurrent_samples()
                        return None

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
            # E was already removed by Trainer under G. Keep the conservative
            # DRAINING projection so the same operation can reconcile R/C under
            # the BLOCKED gate. Physical release still quarantines separately
            # when it cannot be proved.
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
            self._force_exit_recovery.pop(operation.operation_id, None)
            return evidence
        except BaseException:
            if manager.replica_state.get(target) in {
                ReplicaState.CREATING,
                ReplicaState.DRAINING,
            }:
                manager.transition_replica(target, ReplicaState.QUARANTINED)
            raise
