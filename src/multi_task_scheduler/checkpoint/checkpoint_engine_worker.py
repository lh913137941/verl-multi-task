"""Native receiver extension with an opt-in per-parameter audit manifest."""

import hashlib
import json
import os

import torch
import ray

from verl.checkpoint_engine.base import CheckpointEngineWorker
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.rollout.utils import ensure_async_iterator
from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter


class _MultiTaskServerAdapter(ServerAdapter):
    """Resolve only the exact task-scoped native vLLM server name.

    VERL's ServerAdapter has no name_suffix argument and otherwise looks for
    vllm_server_0_0, while the MultiTask Replica creates
    vllm_server_0_0_mt_<task_session>. Never fall back to an unscoped actor:
    it could belong to another concurrently running VERL job.
    """

    def __init__(self, *args, server_name_suffix: str, **kwargs):
        if not isinstance(server_name_suffix, str) or not server_name_suffix.startswith("_"):
            raise ValueError("MultiTask vLLM server name_suffix must begin with '_'")
        super().__init__(*args, **kwargs)
        self._server_name_suffix = server_name_suffix

    def _ensure_server_handle(self) -> bool:
        if not self._has_server:
            return False
        if self.server_handle is None:
            # MultiTask's first-release runtime is standalone TP=DP=PP=1,
            # not PD-disaggregation; preserve a hard boundary for other layouts.
            if self._pd_role is not None:
                raise NotImplementedError(
                    "MultiTask task-scoped ServerAdapter does not support PD routing"
                )
            actor_name = (
                f"{self._get_server_name_prefix()}server_"
                f"{self.replica_rank}_{self.node_rank}{self._server_name_suffix}"
            )
            self.server_handle = ray.get_actor(actor_name)
        return True


class MultiTaskCheckpointEngineWorker(CheckpointEngineWorker):
    """Record receiver-side parameter fingerprints while reusing native transport."""

    def __init__(self, *args, server_name_suffix: str = "", **kwargs):
        if server_name_suffix:
            if kwargs.get("server_adapter") is not None or len(args) > 2:
                raise ValueError("MultiTask CE requires one explicit task-scoped vLLM adapter")
            rollout_config = kwargs.get("rollout_config", args[0] if args else None)
            model_config = kwargs.get("model_config", args[1] if len(args) > 1 else None)
            if rollout_config is None or model_config is None:
                raise ValueError("MultiTask CE requires rollout_config and model_config")
            kwargs["server_adapter"] = _MultiTaskServerAdapter(
                config=rollout_config,
                model_config=model_config,
                device_mesh=None,
                replica_rank=kwargs.get("replica_rank", -1),
                server_name_suffix=server_name_suffix,
            )
        try:
            super().__init__(*args, **kwargs)
        except Exception as error:
            if "EADDRINUSE" in str(error) or "address already in use" in str(error).lower():
                # Observe the rendezvous actually selected by Ray/native verl.
                # Changing just rank 0's port would strand every other rank.
                diagnostic = {key: os.environ.get(key) for key in
                              ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "DIST_INIT_METHOD", "WG_PREFIX")}
                diagnostic.update(pid=os.getpid(), replica_rank=kwargs.get("replica_rank"), error=str(error))
                print("CE_RENDEZVOUS_CONFLICT " + json.dumps(diagnostic, sort_keys=True), flush=True)
            raise
        self.parameter_validation_enabled = (
            os.environ.get("MULTITASK_PARAMETER_VALIDATION", "0") == "1"
            or os.environ.get("MULTITASK_SOURCE_VALIDATION", "0") == "1"
        )
        self._last_parameter_manifest = {
            "complete": False,
            "global_steps": None,
            "wire_format": None,
            "parameters": [],
            "parameter_count": 0,
            "total_numel": 0,
        }

    @staticmethod
    def _tensor_sha256(tensor: torch.Tensor) -> str:
        # Hash the exact contiguous byte representation without moving the
        # tensor payload back to the driver or retaining a second model copy.
        raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        """Reuse native receive/load flow and retain one entry per parameter."""
        if not self.parameter_validation_enabled:
            return await super().update_weights(global_steps=global_steps)
        wire_format = getattr(self.checkpoint_engine, "wire_format", "named_tensors")
        manifest = {
            "complete": False,
            "global_steps": global_steps,
            "wire_format": wire_format,
            "parameters": [],
            "parameter_count": 0,
            "total_numel": 0,
        }
        # Publish this attempt before any await: cancellation or a failed receive
        # must not leave the previous transfer advertised as complete.
        self._last_parameter_manifest = manifest
        if wire_format != "named_tensors":
            raise NotImplementedError(
                f"per-parameter validation currently requires wire_format='named_tensors', got {wire_format!r}"
            )

        received = self.checkpoint_engine.receive_weights(global_steps=global_steps)
        stream_consumed = False

        async def audited_weights():
            nonlocal stream_consumed
            async for name, tensor in ensure_async_iterator(received):
                manifest["parameters"].append(
                    {
                        "name": str(name),
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "numel": int(tensor.numel()),
                        "sha256": self._tensor_sha256(tensor),
                    }
                )
                manifest["parameter_count"] += 1
                manifest["total_numel"] += int(tensor.numel())
                yield name, tensor
            stream_consumed = True

        await self.server_adapter.update_weights(
            audited_weights(),
            global_steps=global_steps,
            wire_format=wire_format,
        )
        if not stream_consumed or not manifest["parameters"]:
            raise RuntimeError("parameter manifest requires a fully consumed nonempty weight stream")
        manifest["complete"] = True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_parameter_manifest(self) -> dict:
        """Return only the last receiver-side audit metadata."""
        return self._last_parameter_manifest
