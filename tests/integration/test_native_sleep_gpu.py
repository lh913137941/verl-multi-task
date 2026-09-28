"""Real CUDA/vLLM acceptance for the retained native sleep primitive.

This file is deliberately opt-in. It is not part of the default unit suite and
must never be used as evidence for RESTORE/current-Vpub correctness: no Trainer
or checkpoint-engine sender participates here.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.native, pytest.mark.gpu_integration]

MODEL_ENV = "VERL_MULTITASK_GPU_MODEL_PATH"
MIN_RELEASE_ENV = "VERL_MULTITASK_MIN_RELEASE_MIB"


def _require_model_path() -> str:
    value = os.environ.get(MODEL_ENV)
    if not value:
        pytest.skip(f"set {MODEL_ENV} to a local model path for real GPU acceptance")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        pytest.skip(f"{MODEL_ENV} does not exist: {path}")
    return str(path)


def _gpu_memory_used_mib(gpu_uuid: str) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu_uuid}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()
    return int(output.splitlines()[0].strip())


def _config(model_path: str):
    from hydra import compose, initialize_config_dir

    import verl

    config_dir = Path(verl.__file__).resolve().parent / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer")

    config.trainer.nnodes = 1
    config.trainer.n_gpus_per_node = 1
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = "vllm"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.nnodes = 1
    config.actor_rollout_ref.rollout.n_gpus_per_node = 1
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
    config.actor_rollout_ref.rollout.data_parallel_size = 1
    config.actor_rollout_ref.rollout.pipeline_model_parallel_size = 1
    config.actor_rollout_ref.rollout.enable_sleep_mode = True
    config.actor_rollout_ref.rollout.free_cache_engine = True
    config.actor_rollout_ref.rollout.load_format = "auto"
    config.actor_rollout_ref.rollout.skip_tokenizer_init = False
    return config


async def _wake_weights_servers(replica):
    receipts = await asyncio.gather(
        *[server.wake_weights.remote() for server in replica.servers]
    )
    return tuple(receipts)


def test_real_standalone_level2_sleep_releases_device_memory_and_weights_wake_stays_fenced():
    """Prove the DONATE sleep primitive on a real physical GPU.

    Acceptance:
      * native standalone generation succeeds;
      * a CE worker reports the physical GPU UUID;
      * level-2 sleep is confirmed by the real vLLM engine;
      * physical GPU memory drops by the configured threshold;
      * weights-only wake remains partial/fenced.

    This does NOT prove RESTORE. A real Trainer/CheckpointEngine sender must
    reinstall current Vpub before final wake in the full lifecycle acceptance.
    """

    model_path = _require_model_path()

    pytest.importorskip("ray")
    pytest.importorskip("vllm")
    pytest.importorskip("torch")

    import ray
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("CUDA GPU is unavailable")

    from transformers import AutoTokenizer

    from multi_task_scheduler.orchestration.contracts import ReplicaKind
    from multi_task_scheduler.rollout.replica import MultiTaskvLLMReplica
    from verl.utils.tokenizer import normalize_token_ids

    config = _config(model_path)
    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model

    ray.shutdown()
    ray.init(
        num_gpus=1,
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        },
        ignore_reinit_error=True,
    )

    try:
        replica = MultiTaskvLLMReplica(
            replica_rank=0,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=1,
            replica_kind=ReplicaKind.NATIVE,
        )
        asyncio.run(replica.init_standalone())
        assert len(replica.workers) == 1
        assert len(replica.servers) == 1

        placement = ray.get(replica.workers[0].runtime_placement.remote())
        gpu_uuid = placement["gpu_uuid"]
        assert gpu_uuid.startswith("GPU-")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        prompt_ids = normalize_token_ids(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "Say hello in one short sentence."}],
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        output = ray.get(
            replica._server_handle.generate.remote(
                request_id=f"multitask-sleep-{uuid4().hex}",
                prompt_ids=prompt_ids,
                sampling_params={
                    "temperature": 0.0,
                    "max_tokens": 16,
                },
                image_data=None,
            ),
            timeout=120,
        )
        assert getattr(output, "token_ids", None)

        before_mib = _gpu_memory_used_mib(gpu_uuid)
        receipts = asyncio.run(replica.sleep())
        assert all(
            receipt["sleep_level"] == 2 and receipt["sleeping"] is True
            for receipt in receipts
        )
        after_mib = _gpu_memory_used_mib(gpu_uuid)

        min_release_mib = int(os.environ.get(MIN_RELEASE_ENV, "128"))
        assert before_mib - after_mib >= min_release_mib, (
            f"level-2 sleep released only {before_mib - after_mib} MiB "
            f"on {gpu_uuid}; expected at least {min_release_mib} MiB"
        )

        weights_receipts = asyncio.run(_wake_weights_servers(replica))
        assert all(
            receipt["sleeping"] is True
            and receipt["fully_awake"] is False
            for receipt in weights_receipts
        )
    finally:
        ray.shutdown()
