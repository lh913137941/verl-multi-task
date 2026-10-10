"""Opt-in HCCL backend for teardown compatibility and source manifests.

Load through verl's ``checkpoint_engine.custom_backend_module`` on both the
training and receiving workers.  The native ``nccl`` registry entry is unchanged.
"""

import hashlib
import os

import torch

from verl.checkpoint_engine.base import CheckpointEngineRegistry
from verl.checkpoint_engine.hccl_checkpoint_engine import HCCLCheckpointEngine
from verl.workers.rollout.utils import ensure_async_iterator


@CheckpointEngineRegistry.register("multitask_hccl")
class MultiTaskHCCLCheckpointEngine(HCCLCheckpointEngine):
    """Inherit native HCCL transfer, teardown, and optional source auditing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_validation_enabled = os.environ.get("MULTITASK_SOURCE_VALIDATION", "0") == "1"
        self._source_manifest = {
            "complete": False,
            "global_steps": None,
            "wire_format": "named_tensors",
            "parameters": [],
            "parameter_count": 0,
            "total_numel": 0,
        }

    @staticmethod
    def _tensor_sha256(tensor: torch.Tensor) -> str:
        raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    async def send_weights(self, weights, global_steps: int | None = None):
        """Reuse native HCCL transfer and audit the rank-0 source stream."""
        if not self.source_validation_enabled:
            return await super().send_weights(weights, global_steps=global_steps)
        manifest = {
            "complete": False,
            "global_steps": global_steps,
            "wire_format": "named_tensors",
            "parameters": [],
            "parameter_count": 0,
            "total_numel": 0,
        }
        # Invalidate the previous audit even if this transfer is cancelled or
        # this rank no longer owns the canonical source after a group rebuild.
        self._source_manifest = manifest
        # HCCL trainer ranks other than rank 0 only consume the stream to stay
        # synchronized. The rank-0 stream is the canonical actor source.
        if self.rank != 0:
            return await super().send_weights(weights, global_steps=global_steps)
        stream_consumed = False

        async def audited_weights():
            nonlocal stream_consumed
            async for name, tensor in ensure_async_iterator(weights):
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

        result = await super().send_weights(audited_weights(), global_steps=global_steps)
        if not stream_consumed or not manifest["parameters"]:
            raise RuntimeError("source manifest requires a fully consumed nonempty weight stream")
        manifest["complete"] = True
        return result

    def get_source_manifest(self) -> dict:
        """Return source metadata from the most recent rank-0 send."""
        return self._source_manifest

    def finalize(self) -> None:
        """Destroy the owned communicator once, then release transfer buffers.

        vLLM-Ascend exposes destruction on the HCCL library wrapper; some older
        integrations expose ``destroyComm`` on the communicator itself.  A
        failed destroy must propagate and retain the handle for diagnosis/retry.
        """
        if self.rebuild_group:
            communicator = self.pyhccl
            if communicator is not None:
                # HCCL operations are asynchronous.  Wait on the owning device
                # before destroying its communicator or freeing weight buckets.
                with torch.npu.device(self.device):
                    torch.npu.synchronize()
                    # Modern vLLM-Ascend exposes close(), which handles
                    # device synchronization and marks the communicator closed.
                    # Older releases have only the underlying HCCL library.
                    close = getattr(communicator, "close", None)
                    destroy = getattr(communicator, "destroyComm", None)
                    if callable(close):
                        close()
                    elif callable(destroy):
                        destroy(communicator.comm)
                    else:
                        library_destroy = getattr(
                            getattr(communicator, "hccl", None),
                            "hcclCommDestroy",
                            None,
                        )
                        if not callable(library_destroy):
                            raise RuntimeError(
                                "PyHcclCommunicator exposes no supported HCCL destruction API"
                            )
                        library_destroy(communicator.comm)
                self.pyhccl = None
            # Non-sending training ranks have no communicator.  Checking the
            # handle also makes repeated finalize safe after rank becomes None.
            self.rank = None
            self.world_size = None

        self.send_buf = None
        self.recv_buf = None
        torch.npu.empty_cache()
