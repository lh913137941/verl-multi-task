"""Select rollout subclasses and own the Manager lifecycle view (M).

Native server creation/routing stays inherited. Real create/sleep/destroy GPU
primitives remain explicit failures until their backend is verified.
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

    def materialize_hidden(self, ctx, key, placement):
        raise NotImplementedError(
            "RuntimeBackend.create_hidden requires verified native backend"
        )

    def sleep_runtime(self, ctx, key, proof):
        raise NotImplementedError(
            "RuntimeBackend.sleep requires verified native backend"
        )

    def destroy_runtime(self, ctx, key, proof):
        raise NotImplementedError(
            "RuntimeBackend.destroy requires verified native backend"
        )
