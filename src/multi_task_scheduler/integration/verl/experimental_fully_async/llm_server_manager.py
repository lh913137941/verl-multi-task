"""Select rollout subclasses and own the Manager lifecycle view (M).

Native server creation/routing stays inherited. RuntimeBackend primitives are
kept on this task-local manager boundary for the first release and fail
explicitly until their real CUDA/NCCL/vLLM implementation is verified.
"""

import ray

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerManager
from verl.workers.rollout.llm_server import DEFAULT_ROUTING_CACHE_SIZE

from multi_task_scheduler.orchestration.replica_record import ReplicaRecord
from multi_task_scheduler.rollout.load_balancer import MultiTaskGlobalRequestLoadBalancer
from multi_task_scheduler.rollout.replica import MultiTaskvLLMReplica


class MultiTaskLLMServerManager(FullyAsyncLLMServerManager):
    """Ordinary object owned by Rollouter; Manager is the M-view owner."""

    def __init__(self, config, worker_group=None, rollout_resource_pool=None, *, group_scheduler=None):
        self.group_scheduler = group_scheduler
        self.rollout_replica_class = MultiTaskvLLMReplica
        super().__init__(config, worker_group, rollout_resource_pool)
        self._load_balancer_cls = MultiTaskGlobalRequestLoadBalancer
        self._lifecycle = {}
        # RuntimeBackend implementation details stay private and are keyed by
        # full ReplicaKey. A verified backend may store process/actor/port/IPC
        # identities here; none of those are public protocol records.
        self._runtime_inventory = {}

    async def _init_global_load_balancer(self) -> None:
        self.global_load_balancer = ray.remote(self._load_balancer_cls).remote(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
            full_determinism=getattr(self.rollout_config, "full_determinism", False),
            group_scheduler=self.group_scheduler,
        )

    def record_lifecycle(self, record: ReplicaRecord) -> ReplicaRecord:
        """Store one Manager lifecycle record by full runtime identity."""
        self._lifecycle[record.key] = record
        return record

    def inspect_runtime(self, ctx, key):
        """Read-only lifecycle lookup by ReplicaKey; never returns a donor handle."""
        return self._lifecycle.get(key)

    def create_hidden(self, ctx, key, placement):
        raise NotImplementedError(
            "RuntimeBackend.create_hidden requires verified native backend"
        )

    def sleep(self, ctx, key, proof):
        raise NotImplementedError(
            "RuntimeBackend.sleep requires verified native backend"
        )

    def wake_weights(self, ctx, key):
        raise NotImplementedError(
            "RuntimeBackend.wake_weights requires verified native backend"
        )

    def wake_kv_and_validate(self, ctx, key, weight):
        raise NotImplementedError(
            "RuntimeBackend.wake_kv_and_validate requires verified native backend"
        )

    def destroy(self, ctx, key, proof):
        raise NotImplementedError(
            "RuntimeBackend.destroy requires verified native backend"
        )

    def query_phase(self, ctx, phase):
        raise NotImplementedError(
            "RuntimeBackend.query_phase requires owner-side journal wiring"
        )
