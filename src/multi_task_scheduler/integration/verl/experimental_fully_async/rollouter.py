"""Experimental Fully Async Rollouter aligned with the 092203 owner boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import json

import ray
from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncAgentLoopManager,
    FullyAsyncRollouter,
)
from verl.experimental.reward_loop import RewardLoopManager
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.contracts import (
    AttemptState,
    EvidenceType,
    OperationEvidence,
    OperationRecord,
    ReplicaKey,
    require_operation_evidence as _require_evidence,
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

        # VERL rewrites the remaining max_tokens budget after an aborted
        # attempt, but currently leaves min_tokens at its original logical
        # request value. Keep the same semantics for the minimum budget:
        # tokens already produced before FORCE handoff count toward min_tokens.
        sampling_params = kwargs.get("sampling_params")
        emitted_tokens = len(getattr(output, "token_ids", ()) or ())
        if (
            isinstance(sampling_params, dict)
            and emitted_tokens > 0
            and isinstance(sampling_params.get("min_tokens"), int)
            and not isinstance(sampling_params.get("min_tokens"), bool)
        ):
            sampling_params["min_tokens"] = max(
                0,
                sampling_params["min_tokens"] - emitted_tokens,
            )

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
        self._release_fences: dict[str, object] = {}

    async def _clear_release_fence(self, request_id: str, release_ref) -> None:
        try:
            await release_ref
        finally:
            if self._release_fences.get(request_id) is release_ref:
                self._release_fences.pop(request_id, None)

    def _release_server(self, server_id: str, request_id: str | None = None) -> None:
        # Preserve VERL's fire-and-forget finally path, but retain the Ray
        # ObjectRef so a same-request retry cannot overtake its own release.
        pool = {"request_id": request_id}
        fields = {
            name: pool[name]
            for name in self._lb_require_release_fields
            if name in pool
        }
        release_ref = self._load_balancer.release_server.remote(
            server_id=server_id,
            **fields,
        )
        if not request_id:
            return
        self._release_fences[request_id] = release_ref
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._clear_release_fence(request_id, release_ref))

    async def _acquire_server(self, request_id: str, **extra):
        release_ref = self._release_fences.get(request_id)
        if release_ref is not None:
            await release_ref
            if self._release_fences.get(request_id) is release_ref:
                self._release_fences.pop(request_id, None)

        # Native FullyAsyncLLMServerClient retries empty routing only when
        # initialized in only-hybrid mode. MultiTask may instead temporarily
        # remove the LAST standalone server during DONATE/REMOVE, then restore
        # it via ADD/RESTORE. Do not let a valid no-service handoff kill the
        # streaming generation worker; retry only with LB-confirmed evidence.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(
            300.0,
            float(getattr(getattr(self.config, "multitask", None),
                          "drain_timeout_s", 300.0)),
        )
        waiting = False
        while True:
            try:
                server_id, server = await super()._acquire_server(request_id, **extra)
                break
            except RuntimeError as exc:
                if "No available servers in load balancer" not in str(exc):
                    raise
                if not await self._load_balancer.is_service_restore_pending.remote():
                    # Empty at initial boot or due to unrelated failure: fail
                    # fast, do not mask incorrect routing setup.
                    raise
                if not waiting:
                    print(
                        "[MultiTaskRollouter] no active rollout server during "
                        "lifecycle handoff; waiting for ADD/RESTORE",
                        flush=True,
                    )
                    waiting = True
                if loop.time() >= deadline:
                    raise TimeoutError(
                        "No rollout server restored before lifecycle wait deadline; "
                        "check DONATE/REMOVE/ADD/RESTORE operation evidence"
                    ) from exc
                await asyncio.sleep(0.5)
        if not self._continuation_enabled:
            return server_id, server
        return server_id, _ContinuationAwareServer(
            server,
            self._load_balancer,
            request_id,
            self._continuation_client_id,
        )


class _TaskScopedRewardLoopManager(RewardLoopManager):
    """Retain native reward calculation, but scope its named Ray workers to a task."""

    def __init__(self, *args, task_session: str, **kwargs):
        if not isinstance(task_session, str) or not task_session:
            raise ValueError("RewardLoop workers require task_session")
        self._task_session = task_session
        super().__init__(*args, **kwargs)

    def _init_reward_loop_workers(self):
        # Native VERL hardcodes reward_loop_worker_{i}. This task-local override
        # changes ONLY the name while preserving native node affinity and
        # RewardLoopWorker constructor/compute_score_batch behavior.
        self.reward_loop_workers = []
        node_ids = [
            node["NodeID"]
            for node in ray.nodes()
            if node["Alive"] and node["Resources"].get("CPU", 0) > 0
        ]
        for i in range(self.config.reward.num_workers):
            node_id = node_ids[i % len(node_ids)]
            self.reward_loop_workers.append(
                self.reward_loop_workers_class.options(
                    name=f"reward_loop_worker_{i}_mt_{self._task_session}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=True,
                    ),
                ).remote(self.config, self.reward_router_address)
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
        self._idle_report_task = None
        super().__init__(
            config,
            tokenizer,
            processor=processor,
            device_name=device_name,
        )

    async def _read_rpc(self, remote_method, *args, attempts: int = 3):
        """Retry read-only owner queries; never retry lifecycle mutations here."""
        if attempts <= 0:
            raise ValueError("read RPC attempts must be positive")
        delay = 0.05
        for attempt in range(attempts):
            try:
                return await remote_method.remote(*args)
            except Exception:
                if attempt + 1 >= attempts:
                    raise
                await asyncio.sleep(delay)
                delay *= 2

    async def _create_reward_loop_manager(self):
        """Use native reward functionality with a task-scoped worker namespace."""
        if not self.task_session:
            raise RuntimeError("RewardLoop manager requires task_session")
        loop = asyncio.get_running_loop()
        self.reward_loop_manager = await loop.run_in_executor(
            None,
            lambda: _TaskScopedRewardLoopManager(
                config=self.config,
                rm_resource_pool=None,
                task_session=self.task_session,
            ),
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

        # Keep native FullyAsyncRollouter.fit() untouched.  This existing
        # multitask-only initialization hook is the narrow place to attach the
        # advisory idle reporter; it exits after the native fit lifecycle has
        # been observed running and then stopped.
        if self.group_scheduler is not None and (
            self._idle_report_task is None or self._idle_report_task.done()
        ):
            self._idle_report_task = asyncio.create_task(self._idle_report_loop())

    async def native_placement_candidates(self) -> tuple[dict, ...]:
        """Read-only snapshot from owned native replica and live CE worker.

        Never infer GPU/NPU identity from a rank or from another task's PG.
        Candidate eligibility (idle/bubble/lease ownership) remains a separate
        GroupScheduler decision; this method does not donate any resource.
        """
        manager = getattr(self, "llm_server_manager", None)
        if manager is None or not self.task_session:
            raise RuntimeError("native placement is unavailable before rollout initialization")
        table = ray.util.placement_group_table()
        candidates = []
        for key, kind in manager.replica_kind.items():
            if (
                key.task_session != self.task_session
                or kind is not ReplicaKind.NATIVE
                or manager.replica_state[key] is not ReplicaState.ACTIVE
            ):
                continue
            runtime = manager.inspect_runtime(key)
            if runtime is None or runtime.world_size != 1 or len(runtime.workers) != 1:
                raise RuntimeError(f"native replica {key} lacks a verifiable TP=1 CE worker")
            pool = runtime.resource_pool
            pgs = getattr(pool, "pgs", None)
            if not pgs or len(pgs) != 1 or pgs[0].bundle_count != 1:
                raise RuntimeError(f"native replica {key} lacks a unique single-bundle PG")
            pg = pgs[0]
            pg_id = pg.id.hex()
            info = table.get(pg_id)
            if not info or info.get("state") != "CREATED":
                raise RuntimeError(f"native replica {key} PG is not CREATED: {pg_id}")
            name = info.get("name")
            if not isinstance(name, str) or not name:
                raise RuntimeError(f"native replica {key} PG is not named")
            if ray.util.get_placement_group(name).id.hex() != pg_id:
                raise RuntimeError(f"native replica {key} named PG identity changed")
            placements = await runtime.worker_placements()
            if len(placements) != 1:
                raise RuntimeError(f"native replica {key} has ambiguous CE placement")
            physical = placements[0]
            if physical.get("pg_id") != pg_id:
                raise RuntimeError(
                    f"native CE worker is not a member of its replica-owned PG: {key}"
                )
            bundle = (info.get("bundles") or {})
            bundle = bundle.get(0, bundle.get("0"))
            nodes = info.get("bundles_to_node_id") or {}
            node = nodes.get(0, nodes.get("0"))
            device = physical.get("resource_name")
            if device not in ("GPU", "NPU"):
                raise RuntimeError(f"unsupported CE resource type: {device!r}")
            if not isinstance(bundle, dict) or float(bundle.get(device, 0)) != 1.0:
                raise RuntimeError(f"native PG bundle lacks one whole {device} allocation")
            if float(bundle.get("CPU", 0)) < 1.0:
                raise RuntimeError("native PG bundle lacks CPU for a borrowed CE actor")
            if node != physical.get("node_id"):
                raise RuntimeError(f"native CE node does not match PG bundle: {key}")
            physical_id = physical.get("gpu_uuid")
            if not isinstance(physical_id, str) or not physical_id:
                raise RuntimeError(f"native CE did not prove physical accelerator identity: {key}")
            if not isinstance(key.replica_id, str) or not key.replica_id.startswith("native-"):
                raise RuntimeError(f"unexpected native replica ID: {key.replica_id!r}")
            rank = runtime.replica_rank
            if key.replica_id != f"native-{rank}":
                raise RuntimeError(f"native replica rank mismatch: {key}")
            candidates.append({
                "donor_replica_rank": rank,
                "pg_id": pg_id,
                "pg_name": name,
                "node_id": node,
                "gpu_uuid": physical_id,
                "bundle_index": 0,
                "resource_name": device,
            })
        return tuple(sorted(candidates, key=lambda item: item["donor_replica_rank"]))

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
        """Emit metadata while native fit is active, then exit with that lifecycle."""
        observed_running = False
        while True:
            await asyncio.sleep(1.0)
            running = bool(getattr(self, "running", False))
            if running:
                observed_running = True
            elif observed_running:
                self._idle_report_signature = None
                self._idle_report_last_sent = None
                return
            else:
                # The reporter is created during async-manager initialization,
                # which precedes native fit() setting running=True.
                self._idle_report_signature = None
                self._idle_report_last_sent = None
                continue
            if self.group_scheduler is None:
                self._idle_report_signature = None
                self._idle_report_last_sent = None
                continue

            # collect_idle_candidates() is already empty when native production
            # resumes or the surplus disappears. Retract the previous report
            # immediately rather than leaving GS a still-fresh (<=10s) false
            # DONATE authorization. GS already accepts an empty candidate list.
            candidates = self.collect_idle_candidates()
            signature = tuple(
                (key.task_session, key.replica_id, key.runtime_epoch, kind.value)
                for key, kind in candidates
            )
            if not signature:
                if self._idle_report_signature is not None:
                    try:
                        await self.group_scheduler.submit_idle_report.remote({
                            "task_session": self.task_session,
                            "candidates": (),
                        })
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        # An unacknowledged retraction is not safe to forget.
                        # Retry next tick; GS also enforces bounded report age.
                        print(
                            "[MultiTaskRollouter] idle report retraction failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        continue
                self._idle_report_signature = None
                self._idle_report_last_sent = None
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

    async def _begin_drain_once(
        self,
        replica_key: ReplicaKey,
        operation_id: str,
    ):
        """Start R drain with one exact replay for lost-reply reconciliation."""
        lb = self.llm_server_manager.global_load_balancer
        try:
            return await lb.begin_drain.remote(replica_key, operation_id)
        except BaseException:
            return await lb.begin_drain.remote(replica_key, operation_id)

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

            server_id = await self._read_rpc(lb.server_for_replica, replica_key)
            if not isinstance(server_id, str) or not server_id:
                raise RuntimeError("FORCE target has no active route")
            active_servers = tuple(await self._read_rpc(lb.get_all_servers))
            if not any(candidate != server_id for candidate in active_servers):
                raise RuntimeError("FORCE REMOVE requires another active rollout server")

            runtime = manager.inspect_runtime(replica_key)
            if runtime is None:
                raise RuntimeError("FORCE target runtime is unavailable")

            if state is ReplicaState.ACTIVE:
                manager.transition_replica(replica_key, ReplicaState.DRAINING)
                self._pending_operation_targets[operation_id] = replica_key

            try:
                drained_server = await self._begin_drain_once(
                    replica_key, operation_id
                )
                if drained_server != server_id:
                    raise RuntimeError("FORCE drain bound a different server")

                recovery = self._force_exit_recovery.get(operation_id)
                if recovery is None:
                    admitted_list = []
                    for request_id in await self._read_rpc(lb.requests_for_server, server_id):
                        if (
                            await self._read_rpc(lb.query_attempt, request_id)
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
                        await self._read_rpc(
                            lb.continuation_handoff_requests,
                            operation_id,
                        )
                    )
                    unsettled = await self._read_rpc(
                        lb.has_unsettled_requests,
                        server_id,
                    )
                    confirmed = len(handoffs.intersection(admitted))
                    if aborted_count is None:
                        # The abort ACK itself was lost. Do not repeat abort.
                        # Only a continuation proof for every request that was
                        # ADMITTED at the boundary can recover this conservatively.
                        handoff_complete = confirmed == len(admitted)
                    else:
                        handoff_complete = confirmed >= aborted_count
                    if not unsettled and handoff_complete:
                        print(
                            "MULTITASK_FORCE_HANDOFF "
                            + json.dumps(
                                {
                                    "operation_id": operation_id,
                                    "admitted_count": len(admitted),
                                    "abort_ack_known": aborted_count is not None,
                                    "aborted_count": aborted_count,
                                    "confirmed_count": confirmed,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
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
            server_id = await self._begin_drain_once(replica_key, operation_id)

            loop = asyncio.get_running_loop()
            if self._natural_drain_timeout_s <= 0:
                raise ValueError("multitask.drain_timeout_s must be positive")
            deadline = loop.time() + self._natural_drain_timeout_s
            next_log = loop.time() + 10.0
            while await self._read_rpc(lb.has_unsettled_requests, server_id):
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

    def clear_operation_binding(self, operation_id: str) -> bool:
        """Drop terminal-operation replay state; UNKNOWN operations are never sent here."""
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id must be a nonempty string")
        removed = self._pending_operation_targets.pop(operation_id, None)
        self._force_exit_recovery.pop(operation_id, None)
        return removed is not None

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

    def _service_identity(self, target: ReplicaKey, label: str):
        runtime = self.llm_server_manager.inspect_runtime(target)
        if runtime is None:
            raise RuntimeError(f"{label} runtime is unavailable")
        server_id = getattr(runtime, "_server_address", None)
        server_handle = getattr(runtime, "_server_handle", None)
        if not isinstance(server_id, str) or not server_id or server_handle is None:
            raise RuntimeError(f"{label} runtime lacks routable server identity")
        return runtime, server_id, server_handle

    def _activate_service_target(self, target: ReplicaKey) -> None:
        manager = self.llm_server_manager
        manager.activate_service(target)
        try:
            self._update_max_concurrent_samples()
            manager.transition_replica(target, ReplicaState.ACTIVE)
        except BaseException:
            try:
                manager.deactivate_service(target)
                self._update_max_concurrent_samples()
            except BaseException:
                pass
            raise

    def _retract_service_target(self, target: ReplicaKey) -> None:
        manager = self.llm_server_manager
        manager.transition_replica(target, ReplicaState.DRAINING)
        manager.deactivate_service(target)
        self._update_max_concurrent_samples()

    async def _commit_ready_or_reconcile(
        self,
        target: ReplicaKey,
        server_id: str,
        server_handle,
        operation_id: str,
        label: str,
    ) -> OperationEvidence | None:
        lb = self.llm_server_manager.global_load_balancer
        try:
            evidence = await lb.commit_ready.remote(
                target,
                server_id,
                server_handle,
                operation_id,
            )
            return _require_evidence(
                evidence,
                operation_id,
                EvidenceType.SERVICE_COMMITTED,
                f"{label} routing commit",
            )
        except BaseException:
            try:
                reconciled = await self._read_rpc(lb.query_ready_operation, operation_id)
            except BaseException as exc:
                raise RuntimeError(
                    f"{label} routing commit outcome is unknown"
                ) from exc
            if reconciled is None:
                return None
            return _require_evidence(
                reconciled,
                operation_id,
                EvidenceType.SERVICE_COMMITTED,
                f"{label} routing ledger",
            )

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
                reconciled = await self._read_rpc(
                    lb.query_ready_operation,
                    operation.operation_id,
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
            _runtime, server_id, server_handle = self._service_identity(
                target, "borrowed ADD"
            )
            try:
                self._activate_service_target(target)
                evidence = await self._commit_ready_or_reconcile(
                    target,
                    server_id,
                    server_handle,
                    operation.operation_id,
                    "ADD",
                )
                if evidence is None:
                    # R authoritatively reports no publish. Trainer still owns G
                    # and removes E before the hidden runtime is destroyed.
                    self._retract_service_target(target)
                    return None
                self._pending_operation_targets.pop(operation.operation_id, None)
                return evidence
            except BaseException:
                if manager.replica_state.get(target) is ReplicaState.CREATING:
                    manager.transition_replica(target, ReplicaState.QUARANTINED)
                raise

        # RESTORE uses the same routing commit helper after Trainer proves Vpub.
        if state is ReplicaState.DORMANT:
            if kind is not ReplicaKind.NATIVE:
                raise ValueError("only NATIVE replicas may restore from DORMANT")
            try:
                runtime, server_id, server_handle = self._service_identity(
                    target, "native RESTORE"
                )
                await runtime.wake_up()
                self._activate_service_target(target)
                try:
                    evidence = await self._commit_ready_or_reconcile(
                        target,
                        server_id,
                        server_handle,
                        operation.operation_id,
                        "RESTORE",
                    )
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "RESTORE routing ledger conflicts with the operation"
                    ) from exc
                if evidence is None:
                    # R is definitely absent. Trainer removes E before
                    # finalize_release() re-enters level-2 sleep.
                    self._retract_service_target(target)
                    return None
                self._pending_operation_targets.pop(operation.operation_id, None)
                return evidence
            except BaseException:
                # Before ACTIVE publication, a mutated DORMANT projection cannot
                # safely be advertised as sleeping.
                if manager.replica_state.get(target) is ReplicaState.DORMANT:
                    manager.transition_replica(target, ReplicaState.QUARANTINED)
                raise

        if state is not ReplicaState.DRAINING:
            raise ValueError(
                f"service exit commit requires DRAINING replica, got {state.value}"
            )

        try:
            server_id = await self._read_rpc(lb.server_for_replica, target)
            if server_id is not None and await self._read_rpc(lb.has_unsettled_requests, server_id):
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
