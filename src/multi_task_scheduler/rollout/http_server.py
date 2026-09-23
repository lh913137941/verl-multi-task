"""Native vLLM HTTP server extension; generation behavior remains inherited."""

import asyncio

import ray
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


class MultiTaskvLLMHttpServer(vLLMHttpServer):
    async def runtime_health(self) -> dict:
        """Return first-release server/engine health facts or raise if unhealthy."""
        if self.nnodes != 1 or self.node_rank != 0:
            raise NotImplementedError(
                "first release runtime health supports one-node servers only"
            )
        engine = getattr(self, "engine", None)
        if engine is None:
            raise RuntimeError("vLLM engine is not initialized")
        server_task = getattr(self, "_server_task", None)
        if server_task is None or server_task.done():
            raise RuntimeError("uvicorn server task is not running")
        if self._server_port is None:
            raise RuntimeError("HTTP server port is not initialized")

        await engine.check_health()
        return {
            "node_id": ray.get_runtime_context().get_node_id(),
            "replica_rank": self.replica_rank,
            "node_rank": self.node_rank,
            "nnodes": self.nnodes,
            "server_address": self._server_address,
            "server_port": self._server_port,
            "engine_ready": True,
            "global_steps": self.global_steps,
        }

    async def shutdown_runtime(self) -> dict:
        """Gracefully stop a first-release borrowed server before actor kill.

        This proves only that server-local shutdown completed. The replica owner
        still kills the Ray actor and verifies its State API entry is DEAD before
        treating cleanup as complete.
        """
        if self.nnodes != 1 or self.node_rank != 0:
            raise NotImplementedError(
                "first release runtime shutdown supports one-node servers only"
            )

        self._submission_paused = True
        self._resume_event.clear()

        engine = getattr(self, "engine", None)
        if engine is not None:
            await engine.wait_for_requests_to_drain()

        server_task = getattr(self, "_server_task", None)
        if server_task is not None and not server_task.done():
            server_task.cancel()
            await asyncio.gather(server_task, return_exceptions=True)

        if engine is not None:
            await asyncio.to_thread(engine.shutdown)
            self.engine = None

        address = self._server_address
        port = self._server_port
        self._server_port = None
        return {
            "node_id": ray.get_runtime_context().get_node_id(),
            "replica_rank": self.replica_rank,
            "server_address": address,
            "server_port": port,
            "shutdown": True,
        }

    def wake_weights(self) -> None:
        raise NotImplementedError("native wake requires verified vLLM sleep backend")

    def abort_target(self, request_ids):
        raise NotImplementedError("targeted abort requires verified FORCE_VERIFIED backend")
