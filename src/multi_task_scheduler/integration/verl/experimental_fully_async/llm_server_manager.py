"""Native Fully Async server manager plus the Manager-owned M view."""

import ray

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerManager
from verl.workers.rollout.router import DEFAULT_ROUTING_CACHE_SIZE

from multi_task_scheduler.orchestration.contracts import ReplicaKey, ReplicaKind, ReplicaState
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
        group_scheduler=None,
        task_session=None,
    ):
        self.group_scheduler = group_scheduler
        self.task_session = task_session
        self.rollout_replica_class = MultiTaskvLLMReplica
        super().__init__(config, worker_group, rollout_resource_pool)
        self._load_balancer_cls = MultiTaskGlobalRequestLoadBalancer
        self.replica_state: dict[ReplicaKey, ReplicaState] = {}
        self.replica_kind: dict[ReplicaKey, ReplicaKind] = {}
        self._runtime_inventory: dict[ReplicaKey, object] = {}

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

    async def _init_global_load_balancer(self) -> None:
        initial_routes = {
            key: runtime._server_address
            for key, runtime in self._runtime_inventory.items()
            if getattr(runtime, "_server_address", None)
        }
        self.global_load_balancer = ray.remote(self._load_balancer_cls).remote(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
            full_determinism=getattr(self.rollout_config, "full_determinism", False),
            group_scheduler=self.group_scheduler,
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
