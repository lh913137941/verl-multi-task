"""Native Fully Async server manager plus the Manager-owned M view."""

import asyncio
import time

import ray

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerManager
from verl.workers.rollout.router import DEFAULT_ROUTING_CACHE_SIZE

from multi_task_scheduler.orchestration.contracts import (
    FIRST_RELEASE_MAX_COLOCATE_COUNT,
    FIRST_RELEASE_RAY_GPU_FRACTION,
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
    ReplicaState.ACTIVE: {ReplicaState.DRAINING},
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
        self.retired_replica_ranks: set[int] = set()
        self._allocated_replica_ranks: set[int] = set()
        self.borrowed_operations: dict[str, dict] = {}
        self.replica_operation_lock = asyncio.Lock()

    async def _initialize_llm_servers(self, start_rank: int = 0):
        await super()._initialize_llm_servers(start_rank=start_rank)
        if not self.task_session:
            raise RuntimeError("Manager requires task_session before replica initialization")
        for index, replica in enumerate(self.rollout_replicas):
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
            self._allocated_replica_ranks.add(rank)
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
                self._runtime_inventory.setdefault(key, runtime)
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

    def deactivate_service(self, key: ReplicaKey):
        """Remove one runtime from native active-service lists without destroying it."""
        runtime = self._runtime_inventory.get(key)
        if runtime is None:
            raise KeyError(key)

        if runtime in self.rollout_replicas:
            self.rollout_replicas.remove(runtime)

        address = getattr(runtime, "_server_address", None)
        if address in self.server_addresses:
            index = self.server_addresses.index(address)
            self.server_addresses.pop(index)
            self.server_handles.pop(index)
        return runtime

    def activate_service(self, key: ReplicaKey):
        """Restore one retained runtime to native active-service lists."""
        runtime = self._runtime_inventory.get(key)
        if runtime is None:
            raise KeyError(key)
        if runtime not in self.rollout_replicas:
            self.rollout_replicas.append(runtime)
        address = getattr(runtime, "_server_address", None)
        handle = getattr(runtime, "_server_handle", None)
        if address and address not in self.server_addresses:
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

        has_claims = spec.get("claims") is not None
        has_selected_slots = spec.get("selected_slots") is not None
        if has_claims == has_selected_slots:
            raise ValueError(
                "borrowed placement requires exactly one of claims or selected_slots"
            )
        raw_claims = spec["claims"] if has_claims else spec["selected_slots"]
        if not isinstance(raw_claims, (list, tuple)) or not raw_claims:
            raise ValueError("borrowed placement requires nonempty claims")

        world_size = spec.get("world_size", spec.get("borrower_world_size"))
        if type(world_size) is not int or world_size <= 0:
            raise ValueError("borrowed placement world_size must be a positive integer")
        if world_size != len(raw_claims):
            raise ValueError("borrowed placement world_size must equal len(claims)")

        configured_world_size = (
            int(self.rollout_config.tensor_model_parallel_size)
            * int(self.rollout_config.data_parallel_size)
            * int(self.rollout_config.pipeline_model_parallel_size)
        )
        if world_size != configured_world_size:
            raise ValueError(
                "first release borrowed world_size must match the borrower task "
                f"parallel topology ({configured_world_size})"
            )

        max_colocate_count = spec.get(
            "max_colocate_count",
            FIRST_RELEASE_MAX_COLOCATE_COUNT,
        )
        if max_colocate_count != FIRST_RELEASE_MAX_COLOCATE_COUNT:
            raise ValueError(
                "first release max_colocate_count must match the native "
                f"Ray accounting layout ({FIRST_RELEASE_MAX_COLOCATE_COUNT})"
            )

        placement_epoch = spec.get("placement_epoch", 0)
        if type(placement_epoch) is not int or placement_epoch < 0:
            raise ValueError("placement_epoch must be a nonnegative integer")

        replica_rank = spec.get("replica_rank")
        if replica_rank is not None and (
            type(replica_rank) is not int or replica_rank < 0
        ):
            raise ValueError("replica_rank must be a nonnegative integer or None")

        lease = Lease(
            spec["lease_id"],
            tuple(raw_claims),
            spec.get("expires_at", 0),
        )
        if lease.expires_at and time.time() >= lease.expires_at:
            raise ValueError("borrowed placement lease is expired")

        claims = [dict(claim) for claim in lease.claims]
        for rank, claim in enumerate(claims):
            if claim["gpu_fraction"] != FIRST_RELEASE_RAY_GPU_FRACTION:
                raise ValueError(
                    "borrowed claim Ray GPU share does not match "
                    "max_colocate_count"
                )
            supplied_rank = claim.get("rank", rank)
            if type(supplied_rank) is not int or supplied_rank != rank:
                raise ValueError(
                    "borrowed claim ranks must cover 0..world_size-1 in list order"
                )
            claim["rank"] = rank

            node_rank = claim.get("node_rank")
            local_rank = claim.get("local_rank")
            if type(node_rank) is not int or node_rank != 0:
                raise ValueError("first release requires single-node node_rank=0")
            if type(local_rank) is not int or local_rank != rank:
                raise ValueError(
                    "first release requires local_rank to match rank on one node"
                )

        if len({claim["node_id"] for claim in claims}) != 1:
            raise ValueError("first release borrowed placement must be single-node")

        requested_source_leases = spec.get("lease_ids")
        if requested_source_leases is not None:
            if not isinstance(requested_source_leases, (list, tuple)):
                raise TypeError("lease_ids must be a list/tuple when provided")
            if set(requested_source_leases) != set(lease.source_lease_ids):
                raise ValueError("lease_ids do not match claim source leases")

        normalized = dict(spec)
        normalized.pop("selected_slots", None)
        normalized["claims"] = claims
        normalized["lease_ids"] = list(lease.source_lease_ids)
        normalized["world_size"] = world_size
        normalized["max_colocate_count"] = max_colocate_count
        normalized["placement_epoch"] = placement_epoch
        normalized["expires_at"] = lease.expires_at
        return normalized

    def _allocate_replica_rank_locked(self, requested_rank: int | None) -> int:
        """Allocate a task-local rank exactly once while replica_operation_lock is held."""
        if requested_rank is not None:
            if requested_rank in self.retired_replica_ranks:
                raise ValueError("retired replica_rank cannot be reused")
            if requested_rank in self._allocated_replica_ranks:
                raise ValueError("replica_rank is already allocated")
            rank = requested_rank
            self.next_replica_rank = max(self.next_replica_rank, rank + 1)
        else:
            rank = self.next_replica_rank
            while rank in self._allocated_replica_ranks or rank in self.retired_replica_ranks:
                rank += 1
            self.next_replica_rank = rank + 1
        self._allocated_replica_ranks.add(rank)
        return rank

    @staticmethod
    def _same_borrowed_create_request(record: dict, incoming: dict) -> bool:
        expected = dict(record["request_spec"])
        current = dict(incoming)
        expected_rank = expected.get("replica_rank")
        current_rank = current.get("replica_rank")
        resolved_rank = record["replica_rank"]
        if expected_rank is None and current_rank == resolved_rank:
            current["replica_rank"] = None
        elif current_rank is None and expected_rank == resolved_rank:
            expected["replica_rank"] = None
        return expected == current

    @staticmethod
    def _borrowed_receipt(record: dict) -> dict:
        result = record.get("result")
        if result is not None:
            return dict(result)
        return {
            "operation_id": record["operation_id"],
            "lease_id": record["lease_id"],
            "replica_rank": record["replica_rank"],
            "state": record["state"],
            "released": False,
            "error": record.get("error"),
        }

    def _resolve_placement_groups(self, claims) -> dict[str, object]:
        """Resolve verified PG handles from serialized claim metadata.

        The wire contract carries IDs, not Ray handles. The Ray PG table is
        keyed by pg.id.hex() and records the stable PG name; resolving by name
        avoids constructing private PlacementGroupID objects from strings.
        """
        claims = tuple(claims)
        if not claims:
            raise ValueError("placement resolution requires nonempty claims")

        table = ray.util.placement_group_table()
        namespace = ray.get_runtime_context().namespace
        resolved = {}

        for claim in claims:
            pg_id = claim["pg_id"]
            expected_namespace = claim.get("pg_namespace")
            if expected_namespace is not None:
                if not isinstance(expected_namespace, str) or not expected_namespace:
                    raise ValueError("pg_namespace must be a nonempty string when provided")
                if expected_namespace != namespace:
                    raise ValueError("claim placement group belongs to another Ray namespace")

            info = table.get(pg_id)
            if info is None:
                raise ValueError(f"placement group {pg_id!r} is not present in Ray")
            if info.get("state") != "CREATED":
                raise ValueError(
                    f"placement group {pg_id!r} is not CREATED: {info.get('state')!r}"
                )

            bundle_index = claim["bundle_index"]
            bundles = info.get("bundles") or {}
            bundle = bundles.get(bundle_index, bundles.get(str(bundle_index)))
            if bundle is None:
                raise ValueError(
                    f"placement group {pg_id!r} has no bundle {bundle_index}"
                )

            bundle_nodes = info.get("bundles_to_node_id") or {}
            actual_node_id = bundle_nodes.get(
                bundle_index,
                bundle_nodes.get(str(bundle_index)),
            )
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
            resolved[pg_id] = pg

        return resolved

    def _new_borrowed_runtime(self, resolved_spec: dict):
        return self.rollout_replica_class(
            replica_rank=resolved_spec["replica_rank"],
            config=self.rollout_config,
            model_config=self.model_config,
            gpus_per_node=self.rollout_config.n_gpus_per_node,
            replica_kind=ReplicaKind.BORROWED,
            placement_claims=resolved_spec["claims"],
            runtime_epoch=resolved_spec["placement_epoch"],
            max_colocate_count=resolved_spec["max_colocate_count"],
        )

    async def create_hidden(
        self,
        resolved_spec: dict,
        pg_by_id: dict[str, object],
    ) -> tuple[object, dict]:
        """Create one hidden borrower runtime; R/C/E remain untouched."""
        runtime = self._new_borrowed_runtime(resolved_spec)
        receipt = await runtime.init_from_lease(resolved_spec, pg_by_id)
        if not isinstance(receipt, dict) or receipt.get("state") != "RUNTIME_READY":
            raise RuntimeError("borrowed runtime did not reach RUNTIME_READY")
        if receipt.get("replica_rank") != resolved_spec["replica_rank"]:
            raise RuntimeError("borrowed runtime returned a conflicting replica_rank")
        return runtime, dict(receipt)

    def _borrowed_record_for_key(self, key: ReplicaKey) -> dict:
        matches = [
            record
            for record in self.borrowed_operations.values()
            if record.get("replica_key") == key
        ]
        if len(matches) != 1:
            raise KeyError(f"expected one borrowed operation for {key!r}")
        return matches[0]

    async def create_borrowed_replica(self, spec: dict) -> dict:
        """Idempotently register create intent, then stop at the unverified GPU boundary.

        The durable-in-process identity/rank fence is useful before the actor backend
        exists: a retry of the same borrower lease cannot allocate a second rank, and
        a conflicting replay cannot reach future Ray side effects.
        """
        normalized = self.validate_borrowed_spec(spec)
        lease_id = normalized["lease_id"]

        async with self.replica_operation_lock:
            existing = self.borrowed_operations.get(lease_id)
            if existing is not None:
                if (
                    existing["operation_id"] != normalized["operation_id"]
                    or not self._same_borrowed_create_request(existing, normalized)
                ):
                    raise ValueError("conflicting borrowed create replay")
                if existing["state"] == "FAILED":
                    error = existing.get("error") or {}
                    if error.get("type") == "NotImplementedError":
                        raise NotImplementedError(error.get("message", "borrowed create failed"))
                    raise RuntimeError(error.get("message", "borrowed create failed"))
                if existing.get("result") is not None:
                    return self._borrowed_receipt(existing)
                raise RuntimeError("borrowed create is already in progress")

            request_spec = dict(normalized)
            request_spec["claims"] = [dict(claim) for claim in normalized["claims"]]
            request_spec["lease_ids"] = list(normalized["lease_ids"])
            rank = self._allocate_replica_rank_locked(normalized.get("replica_rank"))
            resolved_spec = dict(normalized)
            resolved_spec["claims"] = [dict(claim) for claim in normalized["claims"]]
            resolved_spec["lease_ids"] = list(normalized["lease_ids"])
            resolved_spec["replica_rank"] = rank
            replica_key = ReplicaKey(
                normalized["borrower_task_id"],
                normalized["borrower_replica_id"],
                normalized["placement_epoch"],
            )
            self.register_replica(
                replica_key,
                ReplicaKind.BORROWED,
                state=ReplicaState.CREATING,
            )
            record = {
                "operation_id": normalized["operation_id"],
                "lease_id": lease_id,
                "borrower_task_id": normalized["borrower_task_id"],
                "replica_key": replica_key,
                "replica_rank": rank,
                "claim_ids": [claim["claim_id"] for claim in normalized["claims"]],
                "source_lease_ids": list(normalized["lease_ids"]),
                "state": "CREATING",
                "cancel_requested": False,
                "replica": None,
                "worker_handles": [],
                "server_handles": [],
                "created_actor_names": [],
                "result": None,
                "error": None,
                # Manager-local replay fence and resolved placement; neither is
                # serialized to GS. Keeping them separate lets replica_rank=None
                # retries recover the first allocated rank without false conflict.
                "request_spec": request_spec,
                "resolved_spec": resolved_spec,
            }
            self.borrowed_operations[lease_id] = record

        runtime = None
        try:
            pg_by_id = self._resolve_placement_groups(resolved_spec["claims"])
            if set(pg_by_id) != {claim["pg_id"] for claim in resolved_spec["claims"]}:
                raise RuntimeError("placement-group resolution returned incomplete coverage")

            runtime, runtime_receipt = await self.create_hidden(
                resolved_spec,
                pg_by_id,
            )
            async with self.replica_operation_lock:
                current = self.borrowed_operations[lease_id]
                replica_key = current["replica_key"]
                current["replica"] = runtime
                current["worker_handles"] = list(getattr(runtime, "workers", ()) or ())
                current["server_handles"] = list(getattr(runtime, "servers", ()) or ())
                current["created_actor_names"] = list(
                    tuple(getattr(runtime, "borrowed_worker_names", ()) or ())
                    + tuple(getattr(runtime, "borrowed_server_names", ()) or ())
                )
                current["state"] = "RUNTIME_READY"
                current["error"] = None
                current["result"] = {
                    "operation_id": current["operation_id"],
                    "lease_id": current["lease_id"],
                    "replica_rank": current["replica_rank"],
                    "state": "RUNTIME_READY",
                    "released": False,
                    "server_address": runtime_receipt.get("server_address"),
                }
                self._runtime_inventory[replica_key] = runtime
                return self._borrowed_receipt(current)
        except BaseException as exc:
            async with self.replica_operation_lock:
                current = self.borrowed_operations[lease_id]
                replica_key = current["replica_key"]
                cleanup_verified = bool(
                    runtime is not None
                    and getattr(runtime, "borrowed_cleanup_verified", False)
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
                if cleanup_verified:
                    self.retired_replica_ranks.add(current["replica_rank"])
                current["state"] = (
                    "RELEASED" if cleanup_verified else "FAILED"
                )
                current["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                current["result"] = self._borrowed_receipt(current)
            raise

    def sleep(self, *args, **kwargs):
        raise NotImplementedError("RuntimeBackend.sleep requires verified native backend")

    def wake_weights(self, *args, **kwargs):
        raise NotImplementedError("RuntimeBackend.wake_weights requires verified native backend")

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
        if self.replica_state.get(key) not in {
            ReplicaState.CREATING,
            ReplicaState.DRAINING,
        }:
            raise ValueError("destroy requires CREATING or DRAINING borrowed replica")

        record = self._borrowed_record_for_key(key)
        previous = record.get("destroy_evidence")
        if previous is not None:
            if previous.operation_id != operation_id:
                raise ValueError("borrowed runtime was destroyed by another operation")
            return previous

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
        record["state"] = "RELEASED"
        record["released"] = True
        self.retired_replica_ranks.add(record["replica_rank"])
        return evidence

    def query_runtime(self, key: ReplicaKey):
        return self.inspect_runtime(key)
