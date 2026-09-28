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

        # Reuse VERL's own compatibility decision (MTP/LoRA/NPU may only
        # support level 1). Whole-GPU lending requires level 2, so do not emit
        # RELEASED evidence when the underlying rollout cannot safely discard
        # its weights.
        sleep_level = self._resolve_sleep_level()
        if sleep_level != 2:
            raise NotImplementedError(
                "whole-GPU DONATE requires a vLLM configuration safe for level-2 sleep"
            )

        # Normal DONATE has already drained at R. Close the local gate and wait
        # again at the runtime boundary; with no requests left, vLLM's portable
        # abort mode cannot abort user work and is compatible with both async-MP
        # and in-process engine clients.
        await engine.wait_for_requests_to_drain()
        await engine.sleep(level=2, mode="abort")
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
        sleeping_before = bool(await engine.is_sleeping())
        if not sleeping_before:
            # CE target bootstrap wakes KV cache itself.  The final no-tag wake
            # is therefore allowed to act as an admission/health commit after
            # parameters and KV are already resident.  Tagged wakes on an awake
            # engine remain an error so stale sequencing cannot be hidden.
            if tags is not None:
                raise RuntimeError("tagged native wake requires a sleeping vLLM engine")
        else:
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

    async def release_kv_cache(self):
        """Preserve VERL sync semantics, but make RESTORE's partial wake idempotent."""
        if self.node_rank != 0 or not getattr(self.config, "free_cache_engine", False):
            return None
        engine = self._require_sleep_engine()
        if await engine.is_sleeping():
            # After level-2 -> wake_weights, KV memory is already absent.  CE
            # bootstrap should transfer current weights directly rather than
            # issuing another sleep/wake cycle.
            return {"kv_cache_released": True, "already_sleeping": True}
        return await super().release_kv_cache()

    async def wake_weights(self) -> dict:
        """Allocate weight memory only; parameters must be refreshed before full wake."""
        receipt = await self.wake_up(tags=["weights"])
        if receipt["fully_awake"] or not receipt["sleeping"]:
            raise RuntimeError(
                "weights-only wake unexpectedly made the vLLM engine fully awake"
            )
        return receipt

