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

    def _require_sleep_engine(self):
        if self.nnodes != 1 or self.node_rank != 0:
            raise NotImplementedError(
                "first release native sleep/wake supports one-node servers only"
            )
        if not getattr(self.config, "enable_sleep_mode", False):
            raise RuntimeError("native sleep/wake requires rollout.enable_sleep_mode=true")
        if not getattr(self.config, "free_cache_engine", False):
            raise RuntimeError("native sleep/wake requires rollout.free_cache_engine=true")
        engine = getattr(self, "engine", None)
        if engine is None:
            raise RuntimeError("vLLM engine is not initialized")
        return engine

    async def sleep(self) -> dict:
        """Deep-sleep one retained native STANDALONE server.

        Level 2 intentionally discards weights and KV cache: a later RESTORE
        must load the current published parameters instead of reviving stale
        pre-DONATE weights.  Admission remains closed until a full wake succeeds.
        """
        engine = self._require_sleep_engine()
        self._submission_paused = True
        self._resume_event.clear()

        # Normal DONATE has already drained at R, but keep the runtime boundary
        # independently safe: level-2 sleep must never abort a hidden late request.
        await engine.wait_for_requests_to_drain()
        await engine.sleep(level=2, mode="wait")
        if not await engine.is_sleeping():
            raise RuntimeError("vLLM engine did not enter level-2 sleep")

        return {
            "replica_rank": self.replica_rank,
            "node_rank": self.node_rank,
            "sleep_level": 2,
            "sleeping": True,
            "global_steps": self.global_steps,
        }

    async def wake_up(self, tags: list[str] | None = None) -> dict:
        """Wake selected vLLM allocations while keeping admission fenced.

        Partial wakes (notably weights-only RESTORE preparation) deliberately
        keep the server gate closed.  Only a fully resident, healthy engine can
        reopen admission.
        """
        engine = self._require_sleep_engine()
        self._submission_paused = True
        self._resume_event.clear()
        if not await engine.is_sleeping():
            raise RuntimeError("native wake requires a sleeping vLLM engine")

        await engine.wake_up(tags=tags)
        sleeping = bool(await engine.is_sleeping())
        if sleeping:
            return {
                "replica_rank": self.replica_rank,
                "node_rank": self.node_rank,
                "sleeping": True,
                "fully_awake": False,
                "global_steps": self.global_steps,
            }

        # No request can cross the local gate while stale cache state is cleared
        # and the fully resident engine is health-checked.
        await engine.reset_prefix_cache(reset_connector=True)
        await engine.check_health()
        self._submission_paused = False
        self._resume_event.set()
        return {
            "replica_rank": self.replica_rank,
            "node_rank": self.node_rank,
            "sleeping": False,
            "fully_awake": True,
            "global_steps": self.global_steps,
        }

    async def wake_weights(self) -> dict:
        """Allocate weight memory only; parameters must be refreshed before full wake."""
        receipt = await self.wake_up(tags=["weights"])
        if receipt["fully_awake"] or not receipt["sleeping"]:
            raise RuntimeError(
                "weights-only wake unexpectedly made the vLLM engine fully awake"
            )
        return receipt

