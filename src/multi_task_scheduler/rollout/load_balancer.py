"""VERL router extension owning the simplified R view."""

from __future__ import annotations

from verl.workers.rollout.router import DEFAULT_ROUTING_CACHE_SIZE, GlobalRequestLoadBalancer

from multi_task_scheduler.orchestration.contracts import (
    AttemptState,
    EvidenceType,
    OperationEvidence,
    ReplicaKey,
)


class MultiTaskGlobalRequestLoadBalancer(GlobalRequestLoadBalancer):
    """Reuse native routing/counters and add exact request lifecycle facts."""

    def __init__(self, servers, max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
                 full_determinism=False, *, initial_routes=None):
        super().__init__(servers, max_cache_size=max_cache_size,
                         full_determinism=full_determinism)
        self.routes: dict[ReplicaKey, str] = {}
        self.active_request_server: dict[str, str] = {}
        self.attempt_state: dict[str, AttemptState] = {}
        self.draining_servers: set[str] = set()
        self.draining_operations: dict[str, str] = {}
        for key, server_id in dict(initial_routes or {}).items():
            if not isinstance(key, ReplicaKey):
                raise TypeError("initial route key must be ReplicaKey")
            if server_id not in self._servers:
                raise ValueError("initial route references an unknown server")
            self.routes[key] = server_id

    def require_release_fields(self) -> list[str]:
        return ["request_id"]

    def _is_draining_server(self, server_id: str) -> bool:
        return server_id in self.draining_servers

    def acquire_server(self, request_id: str, **extra):
        state = self.attempt_state.get(request_id)
        if state in {AttemptState.ADMITTED, AttemptState.TERMINATED}:
            raise RuntimeError("request already has an unsettled generation")
        server_id, handle = super().acquire_server(request_id, **extra)
        if self._is_draining_server(server_id):
            super().release_server(server_id, request_id=request_id)
            raise RuntimeError("draining server cannot accept new requests")
        self.active_request_server[request_id] = server_id
        self.attempt_state[request_id] = AttemptState.ADMITTED
        return server_id, handle

    def release_server(self, server_id: str, request_id: str | None = None):
        state = self.attempt_state.get(request_id) if request_id else None
        if request_id:
            owner = self.active_request_server.get(request_id)
            if owner is not None and owner != server_id:
                raise ValueError("request release belongs to another server")
            if state is AttemptState.SETTLED:
                return
        super().release_server(server_id, request_id=request_id)
        if request_id and state in {AttemptState.ADMITTED, AttemptState.TERMINATED}:
            self.attempt_state[request_id] = AttemptState.SETTLED
            self.active_request_server.pop(request_id, None)

    def query_attempt(self, request_id: str):
        return self.attempt_state.get(request_id)

    def confirm_continuation(self, request_id: str, client_id: str,
                             prefix_digest: str) -> OperationEvidence:
        if not request_id or not client_id or not prefix_digest:
            raise ValueError("continuation fields must be nonempty")
        if self.attempt_state.get(request_id) is not AttemptState.ADMITTED:
            raise ValueError("request is not eligible for continuation")
        server_id = self.active_request_server.get(request_id)
        operation_id = self.draining_operations.get(server_id)
        if operation_id is None:
            raise ValueError("request is not part of an active drain operation")
        self.attempt_state[request_id] = AttemptState.TERMINATED
        return OperationEvidence.now(operation_id, EvidenceType.EXIT_READY)

    def requests_for_server(self, server_id: str) -> tuple[str, ...]:
        return tuple(r for r, s in self.active_request_server.items() if s == server_id)

    def has_unsettled_requests(self, server_id: str) -> bool:
        return any(self.attempt_state.get(r) is AttemptState.ADMITTED
                   for r in self.requests_for_server(server_id))

    def server_for_replica(self, key: ReplicaKey):
        if not isinstance(key, ReplicaKey):
            raise TypeError("key must be ReplicaKey")
        return self.routes.get(key)

    def begin_drain(self, key: ReplicaKey, operation_id: str):
        """Close admission and bind the drain to one lifecycle operation.

        The server is dropped from the native pool so least-loaded selection and
        the sticky cache stop choosing it: native ``acquire_server`` honours a
        removed server by clearing the stale sticky entry and re-selecting a
        healthy replica. Request facts stay in ``routes``/``active_request_server``,
        which the drain loop reads instead of the native sticky cache.
        """
        if not isinstance(key, ReplicaKey):
            raise TypeError("key must be ReplicaKey")
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("begin_drain requires operation_id")
        server_id = self.routes.get(key)
        if server_id is None:
            raise KeyError(key)
        existing = self.draining_operations.get(server_id)
        if existing is not None and existing != operation_id:
            raise ValueError("server is already draining under another operation")
        self.draining_servers.add(server_id)
        self.draining_operations[server_id] = operation_id
        if server_id in self._servers:
            self.remove_servers([server_id])
        return server_id

    def finish_remove(self, key: ReplicaKey):
        server_id = self.routes.get(key)
        if server_id is None:
            return
        if self.has_unsettled_requests(server_id):
            raise ValueError("cannot remove route while requests remain admitted")

        # TERMINATED is safe to stop serving; SETTLED is the point where the old
        # server no longer owns the request in R. Keep SETTLED for ACK-loss query.
        for request_id in self.requests_for_server(server_id):
            if self.attempt_state.get(request_id) is AttemptState.TERMINATED:
                self.attempt_state[request_id] = AttemptState.SETTLED
                self.active_request_server.pop(request_id, None)

        if server_id in self._servers:
            self.remove_servers([server_id])
        self.draining_servers.discard(server_id)
        self.draining_operations.pop(server_id, None)
        self.routes.pop(key, None)

    def gc_settled_requests(self, request_ids):
        for request_id in request_ids:
            if self.attempt_state.get(request_id) is AttemptState.SETTLED:
                self.attempt_state.pop(request_id, None)
                self.active_request_server.pop(request_id, None)
