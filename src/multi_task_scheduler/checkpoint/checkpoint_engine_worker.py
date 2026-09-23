"""Native CheckpointEngine receiver extension for target-only bootstrap."""

from __future__ import annotations

import os
import subprocess

import ray
from verl.checkpoint_engine.base import CheckpointEngineWorker
from verl.utils.device import get_resource_name, get_visible_devices_keyword


class MultiTaskCheckpointEngineWorker(CheckpointEngineWorker):
    """Keep transfer replay facts local to the checkpoint owner.

    Replay readiness is reported only once a native backend can prove it moved
    the weights; until then both hooks fail explicitly rather than synthesising
    a READY record.
    """

    @staticmethod
    def _resolve_nvidia_gpu_uuid(accelerator_id: str) -> str:
        """Map Ray's GPU resource id to a physical UUID without guessing."""
        accelerator_id = str(accelerator_id)
        if accelerator_id.startswith("GPU-"):
            return accelerator_id
        if accelerator_id.startswith("MIG-"):
            raise NotImplementedError("first release does not support MIG placement")
        if not accelerator_id.isdigit():
            raise RuntimeError(
                f"cannot map Ray GPU accelerator id {accelerator_id!r} to a UUID"
            )

        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        mapping = {}
        for line in output.splitlines():
            if not line.strip():
                continue
            index, uuid = [part.strip() for part in line.split(",", 1)]
            mapping[index] = uuid
        try:
            return mapping[accelerator_id]
        except KeyError as exc:
            raise RuntimeError(
                f"nvidia-smi did not report Ray GPU id {accelerator_id!r}"
            ) from exc

    def runtime_placement(self) -> dict:
        """Return read-only physical placement evidence for borrower validation."""
        resource_name = get_resource_name()
        if resource_name != "GPU":
            raise NotImplementedError(
                f"first release placement probe supports GPU only, got {resource_name!r}"
            )
        ids = ray.get_runtime_context().get_accelerator_ids().get(resource_name, [])
        if len(ids) != 1:
            raise RuntimeError(
                f"expected exactly one Ray GPU id for CE actor, got {ids!r}"
            )
        accelerator_id = str(ids[0])
        return {
            "node_id": ray.get_runtime_context().get_node_id(),
            "accelerator_id": accelerator_id,
            "gpu_uuid": self._resolve_nvidia_gpu_uuid(accelerator_id),
            "visible_devices": os.environ.get(
                get_visible_devices_keyword().upper(),
                "",
            ),
        }

    def replay_current_weights(self, transfer_id: str):
        if not isinstance(transfer_id, str) or not transfer_id:
            raise ValueError("transfer_id must be a nonempty string")
        raise NotImplementedError(
            "weight replay requires a verified native checkpoint transfer backend"
        )

    def replay_status(self, transfer_id: str):
        raise NotImplementedError(
            "replay status requires a verified native checkpoint transfer backend"
        )
