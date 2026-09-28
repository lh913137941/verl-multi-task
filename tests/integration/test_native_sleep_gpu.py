"""Real CUDA/vLLM acceptance for the retained native sleep primitive.

This file is deliberately opt-in. It is not part of the default unit suite and
must never be used as evidence for RESTORE/current-Vpub correctness: no Trainer
or checkpoint-engine sender participates here.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
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


def _wait_for_gpu_memory_drop(
    gpu_uuid: str,
    before_mib: int,
    min_release_mib: int,
    *,
    timeout_s: float = 15.0,
) -> int:
    deadline = time.monotonic() + timeout_s
    last_mib = _gpu_memory_used_mib(gpu_uuid)
    while before_mib - last_mib < min_release_mib:
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"level-2 sleep released only {before_mib - last_mib} MiB "
                f"on {gpu_uuid}; expected at least {min_release_mib} MiB"
            )
        time.sleep(0.25)
        last_mib = _gpu_memory_used_mib(gpu_uuid)
    return last_mib


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


def _exercise_same_gpu_borrower(
    *,
    donor_replica,
    donor_key,
    placement: dict,
    rollout_config,
    model_config,
    prompt_ids,
    token: str,
):
    import ray

    from multi_task_scheduler.integration.verl.experimental_fully_async.llm_server_manager import (
        MultiTaskLLMServerManager,
    )
    from multi_task_scheduler.orchestration.contracts import (
        EvidenceType,
        FIRST_RELEASE_MAX_COLOCATE_COUNT,
        FIRST_RELEASE_RAY_GPU_FRACTION,
        ReplicaKey,
        ReplicaState,
    )
    from multi_task_scheduler.rollout.replica import MultiTaskvLLMReplica

    gpu_uuid = placement["gpu_uuid"]
    pg = donor_replica.resource_pool.get_placement_groups()[0]

    borrower_manager = MultiTaskLLMServerManager.__new__(
        MultiTaskLLMServerManager
    )
    borrower_task = f"{token}-borrower"
    borrower_manager.task_session = borrower_task
    borrower_manager.rollout_replica_class = MultiTaskvLLMReplica
    borrower_manager.rollout_config = rollout_config
    borrower_manager.model_config = model_config
    borrower_manager.replica_state = {}
    borrower_manager.replica_kind = {}
    borrower_manager._runtime_inventory = {}
    borrower_manager.next_replica_rank = 0
    borrower_manager.retired_replica_ranks = set()
    borrower_manager._allocated_replica_ranks = set()
    borrower_manager.borrowed_operations = {}
    borrower_manager.replica_operation_lock = asyncio.Lock()

    borrowed_key = ReplicaKey(borrower_task, "borrowed-0")
    borrowed_spec = {
        "operation_id": f"{token}-add",
        "lease_id": f"{token}-lease",
        "borrower_task_id": borrowed_key.task_session,
        "borrower_replica_id": borrowed_key.replica_id,
        "replica_rank": None,
        "claims": [
            {
                "claim_id": f"{token}-claim",
                "source_lease_id": f"{token}-source-lease",
                "donor_task_id": donor_key.task_session,
                "donor_replica_rank": 0,
                "pg_id": pg.id.hex(),
                "bundle_index": 0,
                "node_id": placement["node_id"],
                "gpu_uuid": gpu_uuid,
                "rank": 0,
                "node_rank": 0,
                "local_rank": 0,
                "gpu_fraction": FIRST_RELEASE_RAY_GPU_FRACTION,
                "cpu_request": 1.0,
            }
        ],
        "world_size": 1,
        "max_colocate_count": FIRST_RELEASE_MAX_COLOCATE_COUNT,
        "expires_at": 0,
        "placement_epoch": 0,
    }
    create_receipt = asyncio.run(
        borrower_manager.create_borrowed_replica(borrowed_spec)
    )
    assert create_receipt["state"] == "RUNTIME_READY"

    borrowed_runtime = borrower_manager.inspect_runtime(borrowed_key)
    assert borrowed_runtime is not None
    borrowed_placement = asyncio.run(
        borrowed_runtime.validate_worker_placement()
    )
    assert borrowed_placement[0]["gpu_uuid"] == gpu_uuid
    assert borrowed_placement[0]["node_id"] == placement["node_id"]

    borrowed_output = ray.get(
        borrowed_runtime._server_handle.generate.remote(
            request_id=f"{token}-generate-{uuid4().hex}",
            prompt_ids=prompt_ids,
            sampling_params={
                "temperature": 0.0,
                "max_tokens": 16,
            },
            image_data=None,
        ),
        timeout=120,
    )
    assert getattr(borrowed_output, "token_ids", None)

    destroy_evidence = asyncio.run(
        borrower_manager.destroy(
            borrowed_key,
            operation_id=f"{token}-remove",
        )
    )
    assert destroy_evidence.type is EvidenceType.RELEASED
    assert destroy_evidence.released_gpu_uuids == (gpu_uuid,)
    borrower_manager.transition_replica(
        borrowed_key,
        ReplicaState.RELEASED,
    )
    return destroy_evidence


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

    from multi_task_scheduler.integration.verl.experimental_fully_async.llm_server_manager import (
        MultiTaskLLMServerManager,
    )
    from multi_task_scheduler.orchestration.contracts import (
        EvidenceType,
        ReplicaKey,
        ReplicaKind,
        ReplicaState,
    )
    from multi_task_scheduler.rollout.replica import MultiTaskvLLMReplica
    from verl.utils.tokenizer import normalize_token_ids

    config = _config(model_path)
    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model

    ray.shutdown()
    ray.init(
        num_cpus=4,
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
            tokenizer.encode(
                "Hello from the real GPU sleep acceptance test.",
                add_special_tokens=True,
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

        key = ReplicaKey("gpu-acceptance", "native-0")
        manager = MultiTaskLLMServerManager.__new__(MultiTaskLLMServerManager)
        manager.replica_kind = {key: ReplicaKind.NATIVE}
        manager.replica_state = {key: ReplicaState.DRAINING}
        manager._runtime_inventory = {key: replica}

        before_mib = _gpu_memory_used_mib(gpu_uuid)
        release = asyncio.run(
            manager.sleep(key, operation_id="gpu-donate")
        )
        assert release.type is EvidenceType.RELEASED
        assert release.released_gpu_uuids == (gpu_uuid,)
        min_release_mib = int(os.environ.get(MIN_RELEASE_ENV, "128"))
        after_mib = _wait_for_gpu_memory_drop(
            gpu_uuid,
            before_mib,
            min_release_mib,
        )
        assert after_mib < before_mib

        # Exact replay must return byte-for-byte equivalent evidence without
        # invoking a second vLLM sleep.
        manager.replica_state[key] = ReplicaState.DORMANT
        replay = asyncio.run(
            manager.sleep(key, operation_id="gpu-donate")
        )
        assert replay == release

        destroy_evidence = _exercise_same_gpu_borrower(
            donor_replica=replica,
            donor_key=key,
            placement=placement,
            rollout_config=rollout_config,
            model_config=model_config,
            prompt_ids=prompt_ids,
            token="gpu-cycle",
        )
        assert destroy_evidence.released_gpu_uuids == (gpu_uuid,)

        weights_receipts = asyncio.run(manager.wake_weights(key))
        assert all(
            receipt["sleeping"] is True
            and receipt["fully_awake"] is False
            for receipt in weights_receipts
        )
    finally:
        ray.shutdown()



def _training_sender_class():
    import torch

    from verl.checkpoint_engine import CheckpointEngineRegistry
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.engine_workers import TrainingWorker

    class RestoreTrainingWorker(TrainingWorker):
        def __init__(self, config, checkpoint_engine_config):
            super().__init__(config)
            backend = checkpoint_engine_config.backend
            bucket_size = (
                checkpoint_engine_config.update_weights_bucket_megabytes << 20
            )
            engine_kwargs = dict(
                checkpoint_engine_config.engine_kwargs.get(backend, {})
            )
            if torch.distributed.get_rank() == 0:
                engine_kwargs["is_master"] = True
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                backend,
                bucket_size=bucket_size,
                **engine_kwargs,
            )

        @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
        async def update_weights(self, global_steps: int = None, mode: str = "auto"):
            weights, _ = self.engine.get_per_tensor_param()
            await self.checkpoint_engine.send_weights(
                weights,
                global_steps=global_steps,
            )

        @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
        def execute_checkpoint_engine(self, method: str, *args, **kwargs):
            return getattr(self.checkpoint_engine, method)(*args, **kwargs)

    return RestoreTrainingWorker


def test_real_level2_restore_reinstalls_current_vpub_and_generates_again():
    """Two-GPU acceptance for the current-Vpub RESTORE data path.

    One GPU hosts a real FSDP TrainingWorker/NCCL sender and another hosts the
    retained STANDALONE vLLM replica. The receiver is level-2 slept first, so its weights
    must be reconstructed by the checkpoint-engine transfer before final wake.
    """

    model_path = _require_model_path()

    pytest.importorskip("ray")
    pytest.importorskip("vllm")
    pytest.importorskip("torch")

    import ray
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("current-Vpub RESTORE acceptance requires at least 2 CUDA GPUs")

    from multi_task_scheduler.checkpoint.checkpoint_engine_manager import (
        MultiTaskCheckpointEngineManager,
    )
    from multi_task_scheduler.integration.verl.experimental_fully_async.llm_server_manager import (
        MultiTaskLLMServerManager,
    )
    from multi_task_scheduler.orchestration.contracts import (
        EvidenceType,
        ReplicaKey,
        ReplicaKind,
        ReplicaState,
    )
    from multi_task_scheduler.rollout.replica import MultiTaskvLLMReplica
    from verl.single_controller.ray import (
        RayClassWithInitArgs,
        RayResourcePool,
        RayWorkerGroup,
    )
    from verl.utils.device import get_device_name
    from verl.utils.tokenizer import normalize_token_ids
    from verl.workers.config import (
        CheckpointEngineConfig,
        FSDPEngineConfig,
        HFModelConfig,
        RolloutConfig,
        TrainingWorkerConfig,
    )

    checkpoint_config = CheckpointEngineConfig(
        backend="nccl",
        update_weights_bucket_megabytes=64,
        engine_kwargs={"nccl": {"rebuild_group": True}},
    )
    model_config = HFModelConfig(
        path=model_path,
        trust_remote_code=True,
        use_remove_padding=True,
    )
    rollout_config = RolloutConfig(
        name="vllm",
        mode="async",
        nnodes=1,
        n_gpus_per_node=1,
        tensor_model_parallel_size=1,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        gpu_memory_utilization=0.5,
        max_num_seqs=16,
        response_length=32,
        load_format="auto",
        skip_tokenizer_init=False,
        enable_sleep_mode=True,
        free_cache_engine=True,
        checkpoint_engine=checkpoint_config,
    )

    ray.shutdown()
    ray.init(
        num_cpus=8,
        num_gpus=2,
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
        trainer_pool = RayResourcePool(
            process_on_nodes=[1],
            use_gpu=True,
            name_prefix="multitask_restore_sender_",
            max_colocate_count=1,
        )
        engine_config = FSDPEngineConfig(
            forward_only=True,
            fsdp_size=1,
            strategy="fsdp",
            use_torch_compile=False,
        )
        trainer_config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=model_config,
            engine_config=engine_config,
        )
        sender_cls = _training_sender_class()
        sender_init = RayClassWithInitArgs(
            cls=ray.remote(sender_cls),
            config=trainer_config,
            checkpoint_engine_config=checkpoint_config,
        )
        actor_wg = RayWorkerGroup(
            resource_pool=trainer_pool,
            ray_cls_with_init=sender_init,
            device_name=get_device_name(),
        )
        actor_wg.reset()

        replica = MultiTaskvLLMReplica(
            replica_rank=0,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=1,
            replica_kind=ReplicaKind.NATIVE,
        )
        asyncio.run(replica.init_standalone())
        placement = ray.get(replica.workers[0].runtime_placement.remote())

        prompt_ids = normalize_token_ids(
            model_config.tokenizer.encode(
                "Hello from the current-Vpub RESTORE acceptance test.",
                add_special_tokens=True,
            )
        )

        def generate_once(label: str):
            return ray.get(
                replica._server_handle.generate.remote(
                    request_id=f"{label}-{uuid4().hex}",
                    prompt_ids=prompt_ids,
                    sampling_params={
                        "temperature": 0.0,
                        "max_tokens": 16,
                    },
                    image_data=None,
                ),
                timeout=120,
            )

        baseline = generate_once("before-restore")
        assert getattr(baseline, "token_ids", None)

        key = ReplicaKey("gpu-acceptance", "native-0")
        manager = MultiTaskLLMServerManager.__new__(MultiTaskLLMServerManager)
        manager.replica_kind = {key: ReplicaKind.NATIVE}
        manager.replica_state = {key: ReplicaState.DRAINING}
        manager._runtime_inventory = {key: replica}

        release = asyncio.run(
            manager.sleep(key, operation_id="gpu-restore-donate")
        )
        assert release.type is EvidenceType.RELEASED
        assert release.released_gpu_uuids == (placement["gpu_uuid"],)
        manager.replica_state[key] = ReplicaState.DORMANT

        destroy_evidence = _exercise_same_gpu_borrower(
            donor_replica=replica,
            donor_key=key,
            placement=placement,
            rollout_config=rollout_config,
            model_config=model_config,
            prompt_ids=prompt_ids,
            token="gpu-restore-cycle",
        )
        assert destroy_evidence.released_gpu_uuids == release.released_gpu_uuids

        checkpoint_manager = MultiTaskCheckpointEngineManager(
            config=checkpoint_config,
            actor_wg=actor_wg,
            replicas=[],
        )
        checkpoint_manager.register_pending(
            key,
            [replica],
            operation_id="gpu-restore",
        )
        # CE owns the weights-only wake and current-Vpub transfer as one
        # serialized target-bootstrap boundary. No GPU mutation occurs before it.
        weight_ready = asyncio.run(
            checkpoint_manager.bootstrap_target(
                key,
                operation_id="gpu-restore",
                loaded_version=17,
            )
        )
        assert weight_ready.type is EvidenceType.WEIGHT_READY
        assert key not in checkpoint_manager.effective_replicas

        # bootstrap_target has restored KV memory, but the local admission gate
        # must remain closed until the explicit final wake commits service.
        parked_ref = replica._server_handle.generate.remote(
            request_id=f"parked-before-final-wake-{uuid4().hex}",
            prompt_ids=prompt_ids,
            sampling_params={
                "temperature": 0.0,
                "max_tokens": 16,
            },
            image_data=None,
        )
        ready, _pending = ray.wait([parked_ref], timeout=1.0)
        assert ready == []

        wake_receipts = asyncio.run(replica.wake_up())
        assert all(
            receipt["fully_awake"] is True
            and receipt["sleeping"] is False
            for receipt in wake_receipts
        )
        health = asyncio.run(replica.validate_server_runtime())
        assert health["global_steps"] == 17

        checkpoint_manager.commit_pending(
            key,
            weight_ready,
            loaded_version=17,
        )
        assert checkpoint_manager.effective_replicas[key][1] == 17

        restored = ray.get(parked_ref, timeout=120)
        assert getattr(restored, "token_ids", None)
        assert restored.token_ids == baseline.token_ids
    finally:
        ray.shutdown()
