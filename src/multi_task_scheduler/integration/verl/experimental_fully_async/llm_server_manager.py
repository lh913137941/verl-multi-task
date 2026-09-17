"""Select rollout subclasses and add the runtime lifecycle overlay (M view).

The native ``_initialize_llm_servers`` / ``get_client`` / ``get_replicas``
stay inherited. The M view records each replica's lifecycle independently of
the donor runtime handles; ``materialize_hidden`` / ``sleep_runtime`` /
``destroy_runtime`` are GPU primitives and stay explicit failures until a
native backend is verified, never faked (section 4.2 / AGENTS.md).
"""

import ray

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerManager
from verl.workers.rollout.llm_server import DEFAULT_ROUTING_CACHE_SIZE

from multi_task_scheduler.orchestration.replica_record import ReplicaRecord
from multi_task_scheduler.rollout.load_balancer import MultiTaskGlobalRequestLoadBalancer
from multi_task_scheduler.rollout.replica import MultiTaskvLLMReplica


class MultiTaskLLMServerManager(FullyAsyncLLMServerManager):
    """Ordinary object owned by Rollouter; native replica lists remain authoritative."""

    def __init__(self, config, worker_group=None, rollout_resource_pool=None, *, group_scheduler=None):
        self.group_scheduler = group_scheduler
        # LLMServerManager explicitly preserves a preselected replica class.
        self.rollout_replica_class = MultiTaskvLLMReplica
        super().__init__(config, worker_group, rollout_resource_pool)
        self._load_balancer_cls = MultiTaskGlobalRequestLoadBalancer
        self._lifecycle = {}

    async def _init_global_load_balancer(self) -> None:
        # Native code forwards full_determinism only to its exact default class.
        # Our subclass keeps native routing, so it must receive the same flag.
        self.global_load_balancer = ray.remote(self._load_balancer_cls).remote(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
            full_determinism=getattr(self.rollout_config, "full_determinism", False),
            group_scheduler=self.group_scheduler,
        )

    # -- runtime lifecycle overlay (M view, section 4.2) ------------------- #

    def record_lifecycle(self, record: ReplicaRecord) -> ReplicaRecord:
        """Store a replica's lifecycle record without touching the runtime."""
        self._lifecycle[record.replica_id] = record
        return record

    def inspect_runtime(self, ctx, replica_id: str):
        """Read-only lifecycle lookup; never a GPU/runtime handle."""
        return self._lifecycle.get(replica_id)

    def materialize_hidden(self, ctx, placement, model_config):
        """Create a hidden runtime on a lease's GPUs; never published here."""
        raise NotImplementedError(
            "hidden materialization requires verified native backend"
        )

    def sleep_runtime(self, ctx, replica_id: str):
        """Real sleep after routing drain + CE exclusion (donor path)."""
        raise NotImplementedError(
            "real sleep requires verified native backend"
        )

    def destroy_runtime(self, ctx, replica_id: str, purpose: str = "recall"):
        """Borrowed reclaim / failed cleanup; native only for explicit exit."""
        raise NotImplementedError(
            "runtime destroy requires verified native backend"
        )
