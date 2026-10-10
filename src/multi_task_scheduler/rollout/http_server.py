"""Native vLLM HTTP server extension; generation behavior remains inherited."""

import asyncio
import inspect

import ray
from verl.utils.device import get_resource_name
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


class MultiTaskvLLMHttpServer(vLLMHttpServer):
    async def _wait_admission_barrier(self, *, timeout_s: float = 10.0) -> None:
        """Wait until requests that already crossed the local gate reach vLLM."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while getattr(self, "_admitting", 0) > 0:
            if loop.time() >= deadline:
                raise RuntimeError(
                    f"local admission barrier timed out with {self._admitting} request(s)"
                )
            await asyncio.sleep(0.01)

    @staticmethod
    def _validate_multitask_engine_capabilities(engine) -> None:
        required = (
            "check_health",
            "is_sleeping",
            "sleep",
            "wake_up",
            "wait_for_requests_to_drain",
            "reset_prefix_cache",
        )
        missing = tuple(
            name for name in required
            if not callable(getattr(engine, name, None))
        )
        if missing:
            raise RuntimeError(
                "vLLM engine lacks MultiTask-required API(s): "
                + ", ".join(missing)
            )
        try:
            sleep_signature = inspect.signature(engine.sleep)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "cannot inspect vLLM engine.sleep signature"
            ) from exc
        parameters = sleep_signature.parameters
        if "level" not in parameters:
            raise RuntimeError(
                "MultiTask requires vLLM engine.sleep(level=...)"
            )
        # mode="abort" is used when the backend exposes it. vLLM-Ascend
        # follows VERL's portable sleep(level=...) API without requiring mode;
        # admission fencing plus request drain remain the safety boundary.

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

        self._validate_multitask_engine_capabilities(engine)
        await engine.check_health()
        return {
            "node_id": ray.get_runtime_context().get_node_id(),
            "replica_rank": self.replica_rank,
            "server_address": self._server_address,
            "server_port": self._server_port,
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
        await self._wait_admission_barrier()

        engine = getattr(self, "engine", None)
        if engine is not None:
            await engine.wait_for_requests_to_drain()
            # vLLM V1 AsyncLLM.shutdown() tears down EngineCore before it
            # cancels its background output_handler. The handler may observe
            # that planned teardown as EngineDeadError and report a spurious
            # ERROR even when every request drained successfully. Cancel it
            # only at this final BORROWED shutdown boundary, while EngineCore
            # is still alive; do not interfere with normal request processing
            # or native sleep/RESTORE.
            output_handler = getattr(engine, "output_handler", None)
            if isinstance(output_handler, asyncio.Task) and not output_handler.done():
                output_handler.cancel()
                await asyncio.gather(output_handler, return_exceptions=True)

        server_task = getattr(self, "_server_task", None)
        if server_task is not None and not server_task.done():
            server_task.cancel()
            await asyncio.gather(server_task, return_exceptions=True)

        if engine is not None:
            await asyncio.to_thread(engine.shutdown)
            self.engine = None

        self._server_port = None

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

    def _receipt(self, **state) -> dict:
        return {
            "replica_rank": self.replica_rank,
            "node_rank": self.node_rank,
            "global_steps": self.global_steps,
            **state,
        }

    def _multitask_sleep_stage(self) -> str:
        return getattr(self, "_multitask_sleep_stage_value", "awake")

    def _set_multitask_sleep_stage(self, stage: str) -> None:
        if stage not in {"awake", "level1", "level2", "weights"}:
            raise ValueError(f"invalid multitask sleep stage: {stage!r}")
        self._multitask_sleep_stage_value = stage

    async def sleep(self) -> dict:
        """Deep-sleep one retained native STANDALONE server.

        Level 2 intentionally discards weights and KV cache: a later RESTORE
        must load the current published parameters instead of reviving stale
        pre-DONATE weights.  Admission remains closed until a full wake succeeds.
        """
        engine = self._require_sleep_engine()
        stage = self._multitask_sleep_stage()
        if stage in {"level1", "level2"}:
            if not await engine.is_sleeping():
                raise RuntimeError("sleep ledger disagrees with vLLM engine")
            self._submission_paused = True
            self._resume_event.clear()
            return self._receipt(
                sleep_level=int(stage.removeprefix("level")),
                sleeping=True,
            )
        if stage == "weights":
            self._submission_paused = True
            self._resume_event.clear()
            if await engine.is_sleeping():
                await super().resume_kv_cache()
                if await engine.is_sleeping():
                    raise RuntimeError(
                        "RESTORE rollback could not restore KV before sleep"
                    )
        elif stage != "awake":
            raise RuntimeError(
                f"cannot enter sleep from partial stage {stage!r}"
            )

        # Reuse VERL's backend compatibility decision. CUDA whole-GPU lending
        # requires level 2. vLLM-Ascend currently exposes level 1 as its
        # deepest supported NPU release primitive.
        sleep_level = self._resolve_sleep_level()
        resource_name = get_resource_name()
        expected_level = (
            2 if resource_name == "GPU"
            else 1 if resource_name == "NPU"
            else None
        )
        if expected_level is None:
            raise NotImplementedError(
                f"native sleep is unsupported for Ray resource {resource_name!r}"
            )
        if sleep_level != expected_level:
            raise NotImplementedError(
                f"{resource_name} DONATE requires vLLM sleep level {expected_level}, "
                f"got {sleep_level}"
            )

        self._submission_paused = True
        self._resume_event.clear()
        await self._wait_admission_barrier()

        # Normal DONATE has already drained at R. Close the local gate and wait
        # again at the runtime boundary; with no requests left, vLLM's portable
        # abort mode cannot abort user work and is compatible with both async-MP
        # and in-process engine clients.
        await engine.wait_for_requests_to_drain()
        sleep_parameters = inspect.signature(engine.sleep).parameters
        if "mode" in sleep_parameters:
            await engine.sleep(level=sleep_level, mode="abort")
        else:
            await engine.sleep(level=sleep_level)
        if not await engine.is_sleeping():
            raise RuntimeError(
                f"vLLM engine did not enter level-{sleep_level} sleep"
            )
        self._set_multitask_sleep_stage(f"level{sleep_level}")

        return self._receipt(sleep_level=sleep_level, sleeping=True)

    async def wake_up(self, tags: list[str] | None = None) -> dict:
        """Wake selected vLLM allocations while keeping admission fenced.

        Partial wakes (notably weights-only RESTORE preparation) deliberately
        keep the server gate closed.  Only a fully resident, healthy engine can
        reopen admission.
        """
        engine = self._require_sleep_engine()
        stage = self._multitask_sleep_stage()
        self._submission_paused = True
        self._resume_event.clear()
        sleeping_before = bool(await engine.is_sleeping())
        kv_already_reset = tags is None and stage == "weights" and not sleeping_before
        if tags == ["weights"] and stage == "weights":
            if not sleeping_before:
                raise RuntimeError("weights-wake ledger disagrees with vLLM engine")
            return self._receipt(sleeping=True, fully_awake=False)
        if tags is None and stage in {"level1", "level2"}:
            raise RuntimeError(
                "full native wake requires weights-only RESTORE preparation"
            )
        if tags is None and stage == "weights" and sleeping_before:
            raise RuntimeError(
                "full native wake requires CE KV restore before admission commit"
            )
        if not sleeping_before:
            # CE target bootstrap wakes KV cache itself.  The final no-tag wake
            # is therefore allowed to act as an admission/health commit after
            # parameters and KV are already resident.  Tagged wakes on an awake
            # engine remain an error so stale sequencing cannot be hidden.
            if tags is not None:
                raise RuntimeError("tagged native wake requires a sleeping vLLM engine")
            if stage not in {"weights", "awake"}:
                raise RuntimeError("native wake ledger disagrees with vLLM engine")
        else:
            await engine.wake_up(tags=tags)
        sleeping = bool(await engine.is_sleeping())
        if tags == ["weights"] and not sleeping:
            # A weights-only RESTORE step must never pass through the full-wake
            # admission path, even if a backend reports an unexpected sleeping
            # state. Keep the local gate closed and fail for reconciliation.
            raise RuntimeError(
                "weights-only wake unexpectedly made the vLLM engine fully awake"
            )
        if sleeping:
            if tags == ["weights"]:
                self._set_multitask_sleep_stage("weights")
            return self._receipt(sleeping=True, fully_awake=False)

        # No request can cross the local gate while stale cache state is
        # cleared and the fully resident engine is health-checked. CE's staged
        # KV resume already reset the prefix cache, so do not repeat that work
        # at the final admission commit.
        if not kv_already_reset:
            await engine.reset_prefix_cache(reset_connector=True)
        await engine.check_health()
        self._set_multitask_sleep_stage("awake")
        self._submission_paused = False
        self._resume_event.set()
        return self._receipt(sleeping=False, fully_awake=True)
