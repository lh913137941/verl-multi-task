"""VERL router extension owning the simplified R view."""

from __future__ import annotations

from verl.workers.rollout.router import DEFAULT_ROUTING_CACHE_SIZE, GlobalRequestLoadBalancer

from multi_task_scheduler.orchestration.contracts import AttemptState, ReplicaKey


class MultiTaskGlobalRequestLoadBalancer(GlobalRequestLoadBalancer):
    """Reuse native routing/counters and add only exact per-request exit facts."""

    def __init__(
        self,
        servers,
        max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
        full_determinism=False,
        *,
        group_scheduler=None,
    ):
        self.group_scheduler = group_scheduler
        super().__init__(
            servers,
            max_cache_size=max_cache_size,
            full_determinism=full_determinism,
        )
        self.routes: dict[ReplicaKey, str] = {}
        self.active_request_server: dict[str, str] = {}
        self.attempt_state: dict[str, AttemptState] = {}

    def require_release_fields(self) -> list[str]:
        return ["request_id"]

    def acquire_server(self, request_id: str, **extra):
        state = self.attempt_state.get(request_id)
        if state in {AttemptState.ADMITTED, AttemptState.TERMINATED}:
            raise RuntimeError(
                "first release allows at most one unsettled generation per request_id"
            )
        server_id, handle = super().acquire_server(request_id, **extra)
        self.active_request_server[request_id] = server_id
        self.attempt_state[request_id] = AttemptState.ADMITTED
        return server_id, handle

    def release_server(self, server_id: str, request_id: str | None = None) -> None:
        super().release_server(server_id, request_id=request_id)
        if request_id is None:
            return
        owner = self.active_request_server.get(request_id)
        if owner is not None and owner != server_id:
            raise ValueError("request_id release belongs to another server")
        state = self.attempt_state.get(request_id)
        if state is None:
            return
        # Natural completion settles the request. A FORCE path may already have
        # recorded TERMINATED from verified continuation; a late native release
        # must not erase that stronger terminal fact.
        if state is AttemptState.ADMITTED:
            self.attempt_state[request_id] = AttemptState.SETTLED
        self.active_request_server.pop(request_id, None)

    def query_attempt(self, request_id: str) -> AttemptState | None:
        return self.attempt_state.get(request_id)

    def confirm_continuation(
        self,
        request_id: str,
        client_id: str,
        prefix_digest: str,
    ) -> AttemptState:
        if not request_id or not client_id or not prefix_digest:
            raise ValueError("request_id, client_id and prefix_digest must be nonempty")
        state = self.attempt_state.get(request_id)
        if state is None:
            raise KeyError(f"unknown request_id {request_id!r}")
        if state in {AttemptState.SETTLED, AttemptState.TERMINATED}:
            return state
        if state is not AttemptState.ADMITTED:
            raise ValueError(
                f"request {request_id!r} is not eligible for continuation handoff"
            )
        self.attempt_state[request_id] = AttemptState.TERMINATED
        return AttemptState.TERMINATED

    def requests_for_server(self, server_id: str) -> tuple[str, ...]:
        return tuple(
            request_id
            for request_id, current_server in self.active_request_server.items()
            if current_server == server_id
        )

    def has_unsettled_requests(self, server_id: str) -> bool:
        return any(
            self.attempt_state.get(request_id) is not AttemptState.SETTLED
            for request_id in self.requests_for_server(server_id)
        )

    def commit_routable(self, key: ReplicaKey, server_id: str, handle) -> None:
        if not isinstance(key, ReplicaKey):
            raise TypeError("key must be ReplicaKey")
        current = self.routes.get(key)
        if current is not None and current != server_id:
            raise ValueError("ReplicaKey already maps to another server")
        if server_id not in self._servers:
            self.add_servers({server_id: handle})
        self.routes[key] = server_id

    def finish_remove(self, key: ReplicaKey) -> None:
        server_id = self.routes.get(key)
        if server_id is None:
            return
        if self.has_unsettled_requests(server_id):
            raise ValueError("cannot remove route while requests remain unsettled")
        self.remove_servers([server_id])
        self.routes.pop(key, None)

    def gc_settled_requests(self, request_ids) -> None:
        for request_id in request_ids:
            if self.attempt_state.get(request_id) is AttemptState.SETTLED:
                self.attempt_state.pop(request_id, None)
                self.active_request_server.pop(request_id, None)
