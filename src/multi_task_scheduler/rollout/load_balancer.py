"""Native routing subclass with an orchestration overlay and no scheduling side effects.

The native ``acquire_server`` / ``release_server`` / ``add_servers`` routing stays
inherited; this overlay tracks a separate orchestration view: the monotonically
increasing ``routing_epoch``, the ``routable_ids`` set, the ``draining_ids`` set,
and per-replica attempt bookkeeping. ``begin_drain`` atomically advances the
epoch and drops the target from the routable set (section 4.1).

The async ``RoutingProtocol`` adapter used by ``ScaleTransaction`` maps the
Trainer/Rollouter's head-server descriptor to a ``replica_id`` before calling
these concrete methods.
"""

from verl.workers.rollout.llm_server import DEFAULT_ROUTING_CACHE_SIZE, GlobalRequestLoadBalancer


class MultiTaskGlobalRequestLoadBalancer(GlobalRequestLoadBalancer):
    """Manager wraps this ordinary class in Ray; native routing stays inherited."""

    def __init__(
        self,
        servers,
        max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
        full_determinism=False,
        *,
        group_scheduler=None,
    ):
        self.group_scheduler = group_scheduler
        super().__init__(servers, max_cache_size=max_cache_size, full_determinism=full_determinism)
        self.routing_epoch = 0
        self.routable_ids = set(servers)
        self.draining_ids = set()
        self.attempts = {}

    def begin_drain(self, replica_id: str) -> int:
        """Atomically advance routing_epoch and remove the target from routable."""
        self.draining_ids.add(replica_id)
        self.routable_ids.discard(replica_id)
        self.routing_epoch += 1
        return self.routing_epoch

    def commit_routable(self, replica_id: str) -> int:
        """Publish a target as routable after bootstrap/CE evidence is verified."""
        self.routable_ids.add(replica_id)
        self.draining_ids.discard(replica_id)
        self.routing_epoch += 1
        return self.routing_epoch

    def finish_remove(self, replica_id: str) -> bool:
        """Drop the target once its attempts are drained and CE excluded."""
        self.draining_ids.discard(replica_id)
        self.routable_ids.discard(replica_id)
        self.attempts.pop(replica_id, None)
        return True

    def query_routing_operation(self, operation_id: str) -> str:
        """Reconciliation hook after a lost routing receipt (section 4.1)."""
        return "unknown"
