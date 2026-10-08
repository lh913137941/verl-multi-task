"""Importable Ray training sender for the opt-in real NPU RESTORE acceptance.

Ray must be able to import this actor from the same installed package on every
worker. A nested class defined inside a pytest test can force cloudpickle to
deserialize the test module (and its unpicklable runtime dependencies).
This module is test support only; it does not change the production Worker API.
"""

import torch

from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.import_utils import import_external_libs
from verl.workers.engine_workers import TrainingWorker


def zero_and_verify_restore_output_weights(engine) -> dict:
    """Mutate and inspect the exact FSDP-exported output projection.

    Tied input/output embeddings may be represented by a single canonical
    parameter in the weight stream. Verify before the HCCL collective starts:
    raising an exception halfway through send_weights can strand the receiver.
    """
    module = getattr(engine.module, "_fsdp_wrapped_module", engine.module)
    get_output = getattr(module, "get_output_embeddings", None)
    if not callable(get_output):
        raise RuntimeError("RESTORE acceptance model has no output embedding accessor")
    output = get_output()
    output_weight = getattr(output, "weight", None)
    if output_weight is None:
        raise RuntimeError("RESTORE acceptance model has no output embedding weight")

    hf_config = getattr(getattr(engine, "model_config", None), "hf_config", None)
    if hf_config is None:
        raise RuntimeError("RESTORE acceptance cannot determine weight tying")
    tied = bool(getattr(hf_config, "tie_word_embeddings", False))

    with torch.no_grad():
        output_weight.zero_()
        if tied:
            get_input = getattr(module, "get_input_embeddings", None)
            if not callable(get_input):
                raise RuntimeError("tied RESTORE model has no input embedding accessor")
            input_layer = get_input()
            input_weight = getattr(input_layer, "weight", None)
            if input_weight is None:
                raise RuntimeError("tied RESTORE model has no input embedding weight")
            input_weight.zero_()

    exported, _ = engine.get_per_tensor_param()
    seen = {}
    for name, tensor in exported:
        is_head = name.endswith("lm_head.weight")
        is_input = name.endswith("embed_tokens.weight")
        if is_head or (tied and is_input):
            seen[name] = int(torch.count_nonzero(tensor).item())

    if tied and not any(name.endswith("embed_tokens.weight") for name in seen):
        raise RuntimeError("tied RESTORE model export omitted canonical input embeddings")
    if not seen:
        raise RuntimeError("RESTORE acceptance cannot find output projection in Vpub export")
    if any(seen.values()):
        raise RuntimeError(
            "RESTORE mutation did not reach the actual Vpub export: " + repr(seen)
        )
    return {"tied": tied, "export_nonzero": seen}


class RestoreTrainingWorker(TrainingWorker):
    """Stable-module Ray actor for NPU RESTORE's source weight mutation."""

    def __init__(self, config, checkpoint_engine_config):
        super().__init__(config)
        backend = checkpoint_engine_config.backend
        bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
        engine_kwargs = dict(checkpoint_engine_config.engine_kwargs.get(backend, {}))
        if torch.distributed.get_rank() == 0:
            engine_kwargs["is_master"] = True
        # Import the opt-in HCCL plugin independently on every remote worker.
        import_external_libs(checkpoint_engine_config.custom_backend_module or None)
        self.checkpoint_engine = CheckpointEngineRegistry.new(
            backend, bucket_size=bucket_size, **engine_kwargs
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def zero_output_weights_for_restore_acceptance(self):
        return zero_and_verify_restore_output_weights(self.engine)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        print(
            "NPU_RESTORE_STAGE",
            {"stage": "trainer-export-start", "global_steps": global_steps},
            flush=True,
        )
        weights, _ = self.engine.get_per_tensor_param()

        async def traced_weights():
            first = True
            for name, tensor in weights:
                if first:
                    print(
                        "NPU_RESTORE_STAGE",
                        {
                            "stage": "trainer-first-weight",
                            "name": str(name),
                            "shape": tuple(tensor.shape),
                        },
                        flush=True,
                    )
                    first = False
                yield name, tensor

        await self.checkpoint_engine.send_weights(
            traced_weights(), global_steps=global_steps
        )
        print(
            "NPU_RESTORE_STAGE",
            {"stage": "trainer-send-done", "global_steps": global_steps},
            flush=True,
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)
