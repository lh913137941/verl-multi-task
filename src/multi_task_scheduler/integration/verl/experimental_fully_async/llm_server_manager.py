"""Native Fully Async server manager plus the Manager-owned M view."""

import asyncio
import time

import ray

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerManager
from verl.workers.rollout.router import DEFAULT_ROUTING_CACHE_SIZE

from multi_task_scheduler.orchestration.contracts import Lease, ReplicaKey, ReplicaKind, ReplicaState
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

        max_colocate_count = spec.get("max_colocate_count", 1)
        if type(max_colocate_count) is not int or max_colocate_count <= 0:
            raise ValueError("max_colocate_count must be a positive integer")

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
            record = {
                "operation_id": normalized["operation_id"],
                "lease_id": lease_id,
                "borrower_task_id": normalized["borrower_task_id"],
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

        try:
            raise NotImplementedError(
                "borrowed runtime creation requires verified PG/bundle actor backend"
            )
        except BaseException as exc:
            async with self.replica_operation_lock:
                current = self.borrowed_operations[lease_id]
                current["state"] = "FAILED"
                current["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                current["result"] = self._borrowed_receipt(current)
            raise

    def create_hidden(self, *args, **kwargs):
        raise NotImplementedError("RuntimeBackend.create_hidden requires verified native backend")

    def sleep(self, *args, **kwargs):
        raise NotImplementedError("RuntimeBackend.sleep requires verified native backend")

    def wake_weights(self, *args, **kwargs):
        raise NotImplementedError("RuntimeBackend.wake_weights requires verified native backend")

    def destroy(self, *args, **kwargs):
        raise NotImplementedError("RuntimeBackend.destroy requires verified native backend")

    def query_runtime(self, key: ReplicaKey):
        return self.inspect_runtime(key)
