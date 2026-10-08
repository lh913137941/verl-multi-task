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
    """Mutate *live FSDP1 parameters*, then verify VERL's export.

    FSDP1 may replace/flatten its original Parameter objects. In particular,
    get_output_embeddings().weight on the wrapped module can be a stale view:
    zero_() succeeds locally but FSDP's get_per_tensor_param() still exports
    the unchanged flattened parameter. summon_full_params(writeback=True)
    exposes the authoritative unflattened parameters and commits the edits
    back to their shards on context exit.

    This acceptance uses strategy='fsdp', fsdp_size=1. Other strategies must
    fail closed rather than silently treating accessor writes as source Vpub.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if not isinstance(engine.module, FSDP):
        raise NotImplementedError(
            "NPU RESTORE Vpub mutation requires the FSDP1 training module"
        )
    hf_config = getattr(getattr(engine, "model_config", None), "hf_config", None)
    if hf_config is None:
        raise RuntimeError("RESTORE acceptance cannot determine weight tying")
    tied = bool(getattr(hf_config, "tie_word_embeddings", False))

    # Never mutate accessor objects outside the FSDP full-param context:
    # FSDP1 may hold the actual values in flat_param instead.
    with torch.no_grad(), FSDP.summon_full_params(
        engine.module, recurse=True, writeback=True
    ):
        modified = []
        # named_parameters() resolves the actual model storage while summoned;
        # remove_duplicate=False is unnecessary: tied aliases share storage.
        for name, parameter in engine.module.named_parameters():
            is_head = name.endswith("lm_head.weight")
            is_input = tied and name.endswith("embed_tokens.weight")
            if is_head or is_input:
                parameter.zero_()
                modified.append(name)

    if tied and not any(name.endswith("embed_tokens.weight") for name in modified):
        raise RuntimeError("tied RESTORE model lacks live input embedding parameter")
    if not modified:
        raise RuntimeError("RESTORE acceptance cannot locate live output projection")

    # The sender exports this stream, not a HF accessor. Perform the check
    # *before* HCCL group construction so failed mutation cannot strand ranks.
    exported, _ = engine.get_per_tensor_param()
    seen = {}
    for name, tensor in exported:
        is_head = name.endswith("lm_head.weight")
        is_input = tied and name.endswith("embed_tokens.weight")
        if is_head or is_input:
            seen[name] = int(torch.count_nonzero(tensor).item())

    if tied and not any(name.endswith("embed_tokens.weight") for name in seen):
        raise RuntimeError("tied RESTORE model export omitted canonical input embeddings")
    if not seen:
        raise RuntimeError("RESTORE acceptance cannot find output projection in Vpub export")
    if any(seen.values()):
        raise RuntimeError(
            "RESTORE mutation did not reach the actual Vpub export: " + repr(seen)
        )
    return {"tied": tied, "modified_parameters": modified, "export_nonzero": seen}


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
