"""Native Fully Async server manager plus the Manager-owned M view."""

import asyncio
import time

import ray

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerManager
from verl.workers.rollout.router import DEFAULT_ROUTING_CACHE_SIZE

from multi_task_scheduler.orchestration.contracts import (
    FIRST_RELEASE_MAX_COLOCATE_COUNT,
    EvidenceType,
    Lease,
    OperationEvidence,
    ReplicaKey,
    ReplicaKind,
    ReplicaState,
)
from multi_task_scheduler.rollout.load_balancer import MultiTaskGlobalRequestLoadBalancer
from multi_task_scheduler.rollout.replica import MultiTaskvLLMReplica

_ALLOWED = {
    ReplicaState.CREATING: {ReplicaState.ACTIVE, ReplicaState.RELEASED, ReplicaState.QUARANTINED},
    ReplicaState.ACTIVE: {
        ReplicaState.DRAINING,
        ReplicaState.QUARANTINED,
    },
    ReplicaState.DRAINING: {
        ReplicaState.ACTIVE,
        ReplicaState.DORMANT,
        ReplicaState.RELEASED,
        ReplicaState.QUARANTINED,
    },
    ReplicaState.DORMANT: {ReplicaState.ACTIVE, ReplicaState.QUARANTINED},
    ReplicaState.RELEASED: set(),
    ReplicaState.QUARANTINED: set(),
}


class MultiTaskLLMServerManager(FullyAsyncLLMServerManager):
    """Ordinary Rollouter-owned object; only this object writes M."""

    def __init__(
        self,
        config,
        worker_group=None,
        rollout_resource_pool=None,
        *,
        task_session=None,
    ):
        self.task_session = task_session
        self.rollout_replica_class = MultiTaskvLLMReplica
        super().__init__(config, worker_group, rollout_resource_pool)
        self._load_balancer_cls = MultiTaskGlobalRequestLoadBalancer
        self.replica_state: dict[ReplicaKey, ReplicaState] = {}
        self.replica_kind: dict[ReplicaKey, ReplicaKind] = {}
        self._runtime_inventory: dict[ReplicaKey, object] = {}
        self.next_replica_rank = 0
        self.borrowed_operations: dict[str, dict] = {}
        self._native_release_evidence: dict[
            str, tuple[ReplicaKey, OperationEvidence]
        ] = {}
        self.replica_operation_lock = asyncio.Lock()

    async def _initialize_llm_servers(self, start_rank: int = 0):
        await super()._initialize_llm_servers(start_rank=start_rank)
        if not self.task_session:
            raise RuntimeError("Manager requires task_session before replica initialization")
        for index, replica in enumerate(self.rollout_replicas):
            # Borrowed replicas validate in init_from_lease(); native replicas
            # must prove the same runtime API surface before entering M=ACTIVE.
            await replica.validate_server_runtime()
            rank = getattr(replica, "replica_rank", index)
            key = ReplicaKey(self.task_session, f"native-{rank}", 0)
            self.register_replica(
                key,
                ReplicaKind.NATIVE,
                state=ReplicaState.ACTIVE,
                runtime=replica,
            )
            if type(rank) is not int or rank < 0:
                raise ValueError("native replica_rank must be a nonnegative integer")
            self.next_replica_rank = max(self.next_replica_rank, rank + 1)

    async def _init_global_load_balancer(self) -> None:
        initial_routes = {
            key: runtime._server_address
            for key, runtime in self._runtime_inventory.items()
            if getattr(runtime, "_server_address", None)
            and self.replica_state.get(key) is ReplicaState.ACTIVE
        }
        self.global_load_balancer = ray.remote(self._load_balancer_cls).remote(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
            full_determinism=getattr(self.rollout_config, "full_determinism", False),
            initial_routes=initial_routes,
        )

    def register_replica(
        self,
        key: ReplicaKey,
        kind: ReplicaKind,
        *,
        state: ReplicaState = ReplicaState.CREATING,
        runtime=None,
    ) -> None:
        if not isinstance(key, ReplicaKey):
            raise TypeError("key must be ReplicaKey")
        kind = ReplicaKind(kind)
        state = ReplicaState(state)
        if key in self.replica_state:
            if self.replica_kind[key] is not kind or self.replica_state[key] is not state:
                raise ValueError("conflicting lifecycle registration")
            if runtime is not None:
                existing_runtime = self._runtime_inventory.get(key)
                if existing_runtime is not None and existing_runtime is not runtime:
                    raise ValueError("ReplicaKey is already bound to another runtime")
                if existing_runtime is None:
                    self._runtime_inventory[key] = runtime
            return
        self._validate_kind_state(kind, state)
        self.replica_kind[key] = kind
        self.replica_state[key] = state
        if runtime is not None:
            self._runtime_inventory[key] = runtime

    @staticmethod
    def _validate_kind_state(kind: ReplicaKind, state: ReplicaState) -> None:
        if kind is ReplicaKind.BORROWED and state is ReplicaState.DORMANT:
            raise ValueError("borrowed replica cannot enter DORMANT")
        if kind is ReplicaKind.NATIVE and state is ReplicaState.RELEASED:
            raise ValueError("native replica must sleep instead of entering RELEASED")

    def transition_replica(self, key: ReplicaKey, new_state: ReplicaState) -> ReplicaState:
        if key not in self.replica_state:
            raise KeyError(key)
        current = self.replica_state[key]
        new_state = ReplicaState(new_state)
        if new_state is current:
            return current
        if new_state not in _ALLOWED[current]:
            raise ValueError(f"illegal replica transition: {current.value} -> {new_state.value}")
        self._validate_kind_state(self.replica_kind[key], new_state)
        self.replica_state[key] = new_state
        if new_state is ReplicaState.RELEASED:
            self._runtime_inventory.pop(key, None)
        return new_state

    def replica_meta(self, key: ReplicaKey) -> tuple[ReplicaKind, ReplicaState]:
        return self.replica_kind[key], self.replica_state[key]

    def inspect_runtime(self, key: ReplicaKey):
        return self._runtime_inventory.get(key)

    async def runtime_loss_verified(self, key: ReplicaKey) -> bool:
        """Return True only for runtime-owned permanent server-loss proof."""
        if not isinstance(key, ReplicaKey):
            raise TypeError("runtime_loss_verified requires ReplicaKey")
        if self.replica_state.get(key) not in {
            ReplicaState.ACTIVE,
            ReplicaState.DRAINING,
        }:
            return False
        runtime = self._runtime_inventory.get(key)
        if runtime is None:
            # Missing owner-local inventory is inconsistent, but it does not
            # prove that the physical/runtime actors are dead.
            return False
        return bool(await runtime.runtime_loss_verified())

    def query_release_evidence(
        self,
        key: ReplicaKey,
        operation_id: str,
    ) -> OperationEvidence | None:
        """Return already-verified physical release evidence for exact-op replay."""
        if not isinstance(key, ReplicaKey):
            raise TypeError("query_release_evidence requires ReplicaKey")
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id must be a nonempty string")

        kind = self.replica_kind.get(key)
        if kind is ReplicaKind.NATIVE:
            entry = self._native_release_evidence.get(operation_id)
            if entry is None:
                return None
            previous_key, evidence = entry
            if previous_key != key:
                raise ValueError("native release operation belongs to another replica")
            return evidence
        if kind is ReplicaKind.BORROWED:
            try:
                record = self._borrowed_record_for_key(key)
            except KeyError:
                return None
            evidence = record.get("destroy_evidence")
            if evidence is None:
                return None
            if evidence.operation_id != operation_id:
                return None
            return evidence
        return None

    def deactivate_service(self, key: ReplicaKey):
        """Remove one runtime from native active-service lists without destroying it."""
        runtime = self._runtime_inventory.get(key)
        if runtime is None:
            raise KeyError(key)
        address = getattr(runtime, "_server_address", None)
        handle = getattr(runtime, "_server_handle", None)
        index = None
        if address in self.server_addresses:
            index = self.server_addresses.index(address)
            if index >= len(self.server_handles) or self.server_handles[index] != handle:
                raise RuntimeError("native service address/handle inventory is inconsistent")

        if runtime in self.rollout_replicas:
            self.rollout_replicas.remove(runtime)
        if index is not None:
            self.server_addresses.pop(index)
            self.server_handles.pop(index)
        return runtime

    def activate_service(self, key: ReplicaKey):
        """Restore one retained runtime to native active-service lists."""
        runtime = self._runtime_inventory.get(key)
        if runtime is None:
            raise KeyError(key)
        address = getattr(runtime, "_server_address", None)
        handle = getattr(runtime, "_server_handle", None)
        if not isinstance(address, str) or not address or handle is None:
            raise RuntimeError("native runtime lacks a routable server identity")
        existing_index = None
        if address in self.server_addresses:
            existing_index = self.server_addresses.index(address)
            if (
                existing_index >= len(self.server_handles)
                or self.server_handles[existing_index] != handle
            ):
                raise RuntimeError("native service address/handle inventory is inconsistent")

        if runtime not in self.rollout_replicas:
            self.rollout_replicas.append(runtime)
        if existing_index is None:
            self.server_addresses.append(address)
            self.server_handles.append(handle)
        return runtime

    def validate_borrowed_spec(self, spec: dict) -> dict:
        """Normalize and validate first-release borrowed placement before Ray side effects."""
        if not isinstance(spec, dict):
            raise TypeError("borrowed placement spec must be a dict")

        for field_name in (
            "operation_id",
            "lease_id",
            "borrower_task_id",
            "borrower_replica_id",
        ):
            value = spec.get(field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"borrowed placement requires nonempty {field_name}")

        borrower_task_id = spec["borrower_task_id"]
        if self.task_session and borrower_task_id != self.task_session:
            raise ValueError("borrowed placement targets another task_session")

        raw_claims = spec.get("claims")
        if not isinstance(raw_claims, (list, tuple)) or not raw_claims:
            raise ValueError("borrowed placement requires nonempty claims")

        if any(
            int(getattr(self.rollout_config, field)) != 1
            for field in (
                "tensor_model_parallel_size",
                "data_parallel_size",
                "pipeline_model_parallel_size",
            )
        ):
            raise ValueError("current borrowed runtime requires TP=DP=PP=1")
        if len(raw_claims) != 1:
            raise ValueError("current borrowed runtime requires one GPU claim")

        placement_epoch = spec.get("placement_epoch", 0)
        if type(placement_epoch) is not int or placement_epoch < 0:
            raise ValueError("placement_epoch must be a nonnegative integer")

        lease = Lease(
            spec["lease_id"],
            tuple(raw_claims),
            spec.get("expires_at", 0),
        )
        if lease.expires_at and time.time() >= lease.expires_at:
            raise ValueError("borrowed placement lease is expired")

        claim = dict(lease.claims[0])
        if (
            claim.get("rank", 0) != 0
            or claim.get("node_rank") != 0
            or claim.get("local_rank") != 0
        ):
            raise ValueError("current borrowed claim requires rank=node_rank=local_rank=0")
        claim["rank"] = 0
        claims = [claim]

        normalized = dict(spec)
        normalized.pop("world_size", None)
        normalized.pop("max_colocate_count", None)
        normalized.pop("replica_rank", None)
        normalized["claims"] = claims
        normalized["placement_epoch"] = placement_epoch
        normalized["expires_at"] = lease.expires_at
        return normalized

    def _resolve_placement_groups(self, claims) -> dict[str, object]:
        """Resolve the verified single-claim donor PG without private Ray IDs."""
        claims = tuple(claims)
        if len(claims) != 1:
            raise ValueError("current borrowed runtime requires exactly one claim")
        claim = claims[0]
        pg_id = claim["pg_id"]
        namespace = ray.get_runtime_context().namespace
        expected_namespace = claim.get("pg_namespace")
        if expected_namespace is not None:
            if not isinstance(expected_namespace, str) or not expected_namespace:
                raise ValueError("pg_namespace must be a nonempty string when provided")
            if expected_namespace != namespace:
                raise ValueError("claim placement group belongs to another Ray namespace")

        info = ray.util.placement_group_table().get(pg_id)
        if info is None:
            raise ValueError(f"placement group {pg_id!r} is not present in Ray")
        if info.get("state") != "CREATED":
            raise ValueError(
                f"placement group {pg_id!r} is not CREATED: {info.get('state')!r}"
            )

        bundle_index = claim["bundle_index"]
        bundles = info.get("bundles") or {}
        if bundles.get(bundle_index, bundles.get(str(bundle_index))) is None:
            raise ValueError(f"placement group {pg_id!r} has no bundle {bundle_index}")

        bundle_nodes = info.get("bundles_to_node_id") or {}
        actual_node_id = bundle_nodes.get(bundle_index, bundle_nodes.get(str(bundle_index)))
        if not actual_node_id:
            raise ValueError(
                f"placement group {pg_id!r} bundle {bundle_index} has no node assignment"
            )
        if actual_node_id != claim["node_id"]:
            raise ValueError(
                f"claim node_id does not match Ray placement for {pg_id!r}/{bundle_index}"
            )

        name = info.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"placement group {pg_id!r} must be named for safe handle recovery"
            )
        pg = ray.util.get_placement_group(name)
        actual_pg_id = pg.id.hex()
        if actual_pg_id != pg_id:
            raise ValueError(
                f"resolved placement group id mismatch: expected {pg_id!r}, got {actual_pg_id!r}"
            )
        if bundle_index >= pg.bundle_count:
            raise ValueError(
                f"resolved placement group {pg_id!r} lacks bundle {bundle_index}"
            )
        return {pg_id: pg}


    def _borrowed_record_for_key(self, key: ReplicaKey) -> dict:
        matches = [
            record
            for record in self.borrowed_operations.values()
            if ReplicaKey(
                record["resolved_spec"]["borrower_task_id"],
                record["resolved_spec"]["borrower_replica_id"],
                record["resolved_spec"]["placement_epoch"],
            ) == key
        ]
        if len(matches) != 1:
            raise KeyError(f"expected one borrowed operation for {key!r}")
        return matches[0]

    async def create_borrowed_replica(self, spec: dict) -> dict:
        """Idempotently create one hidden borrowed runtime on verified lease placement.

        The in-process identity/rank fence prevents a retry of the same borrower
        lease from allocating a second rank.  R/C/E publication remains a separate
        boundary; RUNTIME_READY alone is not end-to-end ADD success.
        """
        normalized = self.validate_borrowed_spec(spec)
        lease_id = normalized["lease_id"]

        async with self.replica_operation_lock:
            existing = self.borrowed_operations.get(lease_id)
            if existing is not None:
                replay_spec = dict(existing["resolved_spec"])
                replay_spec.pop("replica_rank", None)
                if replay_spec != normalized:
                    raise ValueError("conflicting borrowed create replay")
                if existing.get("error") is not None:
                    error = existing["error"]
                    if error.get("type") == "NotImplementedError":
                        raise NotImplementedError(error.get("message", "borrowed create failed"))
                    raise RuntimeError(error.get("message", "borrowed create failed"))
                if existing.get("result") is not None:
                    return dict(existing["result"])
                raise RuntimeError("borrowed create is already in progress")

            replica_key = ReplicaKey(
                normalized["borrower_task_id"],
                normalized["borrower_replica_id"],
                normalized["placement_epoch"],
            )
            if replica_key in self.replica_state:
                raise ValueError(
                    "borrowed ReplicaKey is already registered; use a new runtime_epoch"
                )
            rank = self.next_replica_rank
            self.next_replica_rank += 1
            resolved_spec = dict(normalized)
            resolved_spec["claims"] = [dict(claim) for claim in normalized["claims"]]
            resolved_spec["replica_rank"] = rank
            self.register_replica(
                replica_key,
                ReplicaKind.BORROWED,
                state=ReplicaState.CREATING,
            )
            record = {
                # Manager-local replay fence and resolved placement. Retries
                # normalize replica_rank=None to this already allocated rank.
                "resolved_spec": resolved_spec,
            }
            self.borrowed_operations[lease_id] = record

        runtime = None
        try:
            pg_by_id = self._resolve_placement_groups(resolved_spec["claims"])
            if set(pg_by_id) != {claim["pg_id"] for claim in resolved_spec["claims"]}:
                raise RuntimeError("placement-group resolution returned incomplete coverage")

            # Keep ownership of the runtime before any awaited initialization so
            # a failing init can still report/perform verified cleanup.
            runtime = self.rollout_replica_class(
                replica_rank=resolved_spec["replica_rank"],
                config=self.rollout_config,
                model_config=self.model_config,
                gpus_per_node=self.rollout_config.n_gpus_per_node,
                replica_kind=ReplicaKind.BORROWED,
                placement_claims=resolved_spec["claims"],
                runtime_epoch=resolved_spec["placement_epoch"],
                max_colocate_count=FIRST_RELEASE_MAX_COLOCATE_COUNT,
            )
            runtime_receipt = await runtime.init_from_lease(
                resolved_spec,
                pg_by_id,
            )
            if (
                not isinstance(runtime_receipt, dict)
                or runtime_receipt.get("state") != "RUNTIME_READY"
            ):
                raise RuntimeError("borrowed runtime did not reach RUNTIME_READY")
            if runtime_receipt.get("replica_rank") != resolved_spec["replica_rank"]:
                raise RuntimeError("borrowed runtime returned a conflicting replica_rank")

            async with self.replica_operation_lock:
                current = self.borrowed_operations[lease_id]
                current["result"] = {
                    "operation_id": current["resolved_spec"]["operation_id"],
                    "lease_id": lease_id,
                    "replica_rank": current["resolved_spec"]["replica_rank"],
                    "state": "RUNTIME_READY",
                    "released": False,
                    "server_address": runtime_receipt.get("server_address"),
                }
                self._runtime_inventory[replica_key] = runtime
                return dict(current["result"])
        except BaseException as exc:
            if runtime is not None and not getattr(
                runtime,
                "borrowed_cleanup_verified",
                False,
            ):
                try:
                    await runtime.cleanup_borrowed_runtime()
                except BaseException:
                    # The original create failure remains authoritative; M below
                    # records QUARANTINED when cleanup cannot be proved.
                    pass

            async with self.replica_operation_lock:
                current = self.borrowed_operations[lease_id]
                # Before a runtime object exists, create_borrowed_replica has not
                # entered init_from_lease(), so no borrower actor/server side
                # effect is owned by this operation. That is a known-safe landing,
                # equivalent to verified cleanup for lifecycle compensation.
                cleanup_verified = bool(
                    runtime is None
                    or getattr(runtime, "borrowed_cleanup_verified", False)
                )
                if self.replica_state.get(replica_key) is ReplicaState.CREATING:
                    self.transition_replica(
                        replica_key,
                        (
                            ReplicaState.RELEASED
                            if cleanup_verified
                            else ReplicaState.QUARANTINED
                        ),
                    )
                # Runtime cleanup success is a resource fact, not create
                # success. Keep the operation FAILED so exact replay raises the
                # original failure instead of synthesizing RUNTIME_READY.
                current["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                current["result"] = None
            raise

    async def sleep(
        self,
        key: ReplicaKey,
        *,
        operation_id: str,
    ) -> OperationEvidence:
        """Deep-sleep one retained native runtime and bind release to real GPUs."""
        if not isinstance(key, ReplicaKey):
            raise TypeError("sleep requires ReplicaKey")
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("sleep requires operation_id")
        if self.replica_kind.get(key) is not ReplicaKind.NATIVE:
            raise ValueError("sleep is valid only for NATIVE replicas")

        previous = self._native_release_evidence.get(operation_id)
        if previous is not None:
            previous_key, evidence = previous
            if previous_key != key:
                raise ValueError("native release operation is bound to another replica")
            return evidence

        if self.replica_state.get(key) is not ReplicaState.DRAINING:
            raise ValueError("sleep requires a DRAINING native replica")

        runtime = self._runtime_inventory.get(key)
        if runtime is None:
            raise RuntimeError("native runtime handle is unavailable for verified sleep")
        (placement,) = await runtime.worker_placements()
        gpu_uuid = placement.get("gpu_uuid")
        if not isinstance(gpu_uuid, str) or not gpu_uuid:
            raise RuntimeError("native runtime placement is missing physical GPU UUID")

        await runtime.sleep()
        evidence = OperationEvidence.now(
            operation_id,
            EvidenceType.RELEASED,
            released_gpu_uuids=(gpu_uuid,),
        )
        self._native_release_evidence[operation_id] = (key, evidence)
        return evidence

    async def destroy(
        self,
        key: ReplicaKey,
        *,
        operation_id: str,
    ) -> OperationEvidence:
        """Destroy a borrowed runtime and return only verified RELEASED evidence."""
        if not isinstance(key, ReplicaKey):
            raise TypeError("destroy requires ReplicaKey")
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("destroy requires operation_id")
        if self.replica_kind.get(key) is not ReplicaKind.BORROWED:
            raise ValueError("destroy is valid only for BORROWED replicas")

        record = self._borrowed_record_for_key(key)
        previous = record.get("destroy_evidence")
        if previous is not None:
            if previous.operation_id != operation_id:
                raise ValueError("borrowed runtime was destroyed by another operation")
            return previous

        if self.replica_state.get(key) not in {
            ReplicaState.CREATING,
            ReplicaState.DRAINING,
        }:
            raise ValueError("destroy requires CREATING or DRAINING borrowed replica")

        runtime = self._runtime_inventory.get(key)
        if runtime is None:
            raise RuntimeError("borrowed runtime handle is unavailable for verified destroy")

        await runtime.cleanup_borrowed_runtime()
        gpu_uuids = tuple(
            claim["gpu_uuid"]
            for claim in record["resolved_spec"]["claims"]
        )
        evidence = OperationEvidence.now(
            operation_id,
            EvidenceType.RELEASED,
            released_gpu_uuids=gpu_uuids,
        )
        record["destroy_evidence"] = evidence
        return evidence
