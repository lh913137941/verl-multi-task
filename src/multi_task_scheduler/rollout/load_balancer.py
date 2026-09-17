"""Native routing subclass with a conservative orchestration R view.

Native request selection remains inherited.  The overlay owns only routing
eligibility, a monotonic route revision and attempt facts used by lifecycle
operations.  It never discards unresolved attempts in order to manufacture a
successful remove.
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
        self.route_metadata = {
            replica_id: {"serving_version": None, "ce_revision": None, "lease_valid": True}
            for replica_id in servers
        }

    def begin_drain(self, replica_id: str) -> int:
        """Atomically close new routing and advance the per-task route fence."""
        self.draining_ids.add(replica_id)
        self.routable_ids.discard(replica_id)
        self.routing_epoch += 1
        return self.routing_epoch

    def commit_routable(
        self,
        replica_id: str,
        *,
        serving_version: int,
        ce_revision: int,
        lease_valid: bool,
    ) -> int:
        """Open routing only with explicit weight/member/lease commit evidence."""
        if serving_version < 0 or ce_revision < 0:
            raise ValueError("serving_version and ce_revision must be nonnegative")
        if not lease_valid:
            raise ValueError("cannot route a replica without a valid lease/identity fence")
        self.route_metadata[replica_id] = {
            "serving_version": serving_version,
            "ce_revision": ce_revision,
            "lease_valid": True,
        }
        self.routable_ids.add(replica_id)
        self.draining_ids.discard(replica_id)
        self.routing_epoch += 1
        return self.routing_epoch

    def _has_unsettled_attempts(self, replica_id: str) -> bool:
        attempts = self.attempts.get(replica_id)
        if attempts is None:
            return False
        if isinstance(attempts, dict):
            terminal = {"TERMINAL", "RELEASED"}
            for value in attempts.values():
                state = getattr(value, "state", value)
                state = getattr(state, "value", state)
                if state not in terminal:
                    return True
            return False
        return bool(attempts)

    def finish_remove(self, replica_id: str) -> bool:
        """Commit R=REMOVED only when no unresolved attempt remains."""
        if self._has_unsettled_attempts(replica_id):
            return False
        self.draining_ids.discard(replica_id)
        self.routable_ids.discard(replica_id)
        self.route_metadata.pop(replica_id, None)
        self.attempts.pop(replica_id, None)
        self.routing_epoch += 1
        return True

    def query_routing_operation(self, operation_id: str) -> str:
        """Reconciliation hook; unknown is not evidence that nothing executed."""
        return "unknown"
