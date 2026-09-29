"""Real Ascend NPU acceptance for MultiTask lifecycle primitives.

This suite is deliberately opt-in.  It uses vLLM-Ascend/torch_npu and real
devices; ordinary unit/native/ray integration suites must not depend on it.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.native, pytest.mark.npu_integration]


@pytest.fixture(autouse=True)
def _pin_verl_npu_platform(monkeypatch):
    """Pin only executing NPU tests; collection must not poison CUDA tests."""
    monkeypatch.setenv("VERL_PLATFORM", "huawei")

MODEL_ENV = "VERL_MULTITASK_NPU_MODEL_PATH"


def _require_model_path() -> str:
    value = os.environ.get(MODEL_ENV)
    if not value:
        pytest.skip(f"set {MODEL_ENV} to a local model path for real NPU acceptance")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        pytest.skip(f"{MODEL_ENV} does not exist: {path}")
    return str(path)


def _require_npu(min_devices: int) -> None:
    pytest.importorskip("torch_npu")
    pytest.importorskip("vllm")
    pytest.importorskip("vllm_ascend")

    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    if torch.npu.device_count() < min_devices:
        pytest.skip(
            f"NPU acceptance requires at least {min_devices} NPU device(s)"
        )

    from verl.utils.device import get_device_name, get_resource_name

    if get_device_name() != "npu" or get_resource_name() != "NPU":
        pytest.skip("VERL platform did not resolve to the Ascend NPU backend")


def _runtime_env() -> dict:
    return {
        "env_vars": {
            "VERL_PLATFORM": "huawei",
            "TOKENIZERS_PARALLELISM": "true",
            "HCCL_CONNECT_TIMEOUT": "1500",
            "HCCL_HOST_SOCKET_PORT_RANGE": "60000-60050",
            "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61050",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "VLLM_ASCEND_AUTO_DETECT_QUANTIZATION": "0",
            "VLLM_LOGGING_LEVEL": "INFO",
            "VLLM_USE_V1": "1",
        }
    }


def _require_ray_npus(ray, min_devices: int) -> None:
    resources = ray.cluster_resources()
    count = float(resources.get("NPU", 0.0))
    if count < min_devices:
        pytest.skip(
            f"Ray exposes only {count:g} NPU resource(s); "
            f"{min_devices} required"
        )


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


def _exercise_same_npu_borrower(
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

    physical_id = placement["gpu_uuid"]
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
    borrower_manager._allocated_replica_ranks = set()
    borrower_manager.borrowed_operations = {}
    borrower_manager._native_release_evidence = {}
    borrower_manager.replica_operation_lock = asyncio.Lock()

    borrowed_key = ReplicaKey(borrower_task, "borrowed-0")
    borrowed_spec = {
        "operation_id": f"{token}-add",
        "lease_id": f"{token}-lease",
        "borrower_task_id": borrowed_key.task_session,
        "borrower_replica_id": borrowed_key.replica_id,
        "claims": [
            {
                "claim_id": f"{token}-claim",
                "source_lease_id": f"{token}-source-lease",
                "donor_task_id": donor_key.task_session,
                "donor_replica_rank": 0,
                "pg_id": pg.id.hex(),
                "bundle_index": 0,
                "node_id": placement["node_id"],
                "gpu_uuid": physical_id,
                "rank": 0,
                "node_rank": 0,
                "local_rank": 0,
                "gpu_fraction": FIRST_RELEASE_RAY_GPU_FRACTION,
                "cpu_request": 1.0,
            }
        ],
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
    assert borrowed_placement[0]["gpu_uuid"] == physical_id
    assert borrowed_placement[0]["node_id"] == placement["node_id"]
    assert borrowed_placement[0]["resource_name"] == "NPU"

    output = ray.get(
        borrowed_runtime._server_handle.generate.remote(
            request_id=f"{token}-generate-{uuid4().hex}",
            prompt_ids=prompt_ids,
            sampling_params={"temperature": 0.0, "max_tokens": 16},
            image_data=None,
        ),
        timeout=120,
    )
    assert getattr(output, "token_ids", None)

    destroy_evidence = asyncio.run(
        borrower_manager.destroy(
            borrowed_key,
            operation_id=f"{token}-remove",
        )
    )
    assert destroy_evidence.type is EvidenceType.RELEASED
    assert destroy_evidence.released_gpu_uuids == (physical_id,)
    borrower_manager.transition_replica(
        borrowed_key,
        ReplicaState.RELEASED,
    )
    return destroy_evidence


def test_real_npu_sleep_releases_same_slot_to_borrower():
    """One-NPU DONATE acceptance using vLLM-Ascend level-1 sleep.

    Acceptance is intentionally stronger than a memory counter: after the
    donor sleeps, a borrowed vLLM replica must start on the exact same Ray NPU
    placement and complete generation.
    """

    model_path = _require_model_path()
    _require_npu(1)

    import ray
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
        runtime_env=_runtime_env(),
        ignore_reinit_error=True,
    )

    try:
        _require_ray_npus(ray, 1)

        replica = MultiTaskvLLMReplica(
            replica_rank=0,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=1,
            replica_kind=ReplicaKind.NATIVE,
        )
        asyncio.run(replica.init_standalone())
        placement = asyncio.run(replica.worker_placements())[0]
        assert placement["resource_name"] == "NPU"
        assert placement["gpu_uuid"].startswith("NPU:")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        prompt_ids = normalize_token_ids(
            tokenizer.encode(
                "Hello from the real NPU sleep acceptance test.",
                add_special_tokens=True,
            )
        )
        baseline = ray.get(
            replica._server_handle.generate.remote(
                request_id=f"npu-sleep-{uuid4().hex}",
                prompt_ids=prompt_ids,
                sampling_params={"temperature": 0.0, "max_tokens": 16},
                image_data=None,
            ),
            timeout=120,
        )
        assert getattr(baseline, "token_ids", None)

        key = ReplicaKey("npu-donor", "native-0")
        manager = MultiTaskLLMServerManager.__new__(MultiTaskLLMServerManager)
        manager.replica_kind = {key: ReplicaKind.NATIVE}
        manager.replica_state = {key: ReplicaState.DRAINING}
        manager._runtime_inventory = {key: replica}
        manager._native_release_evidence = {}

        release = asyncio.run(
            manager.sleep(key, operation_id="npu-donate")
        )
        assert release.type is EvidenceType.RELEASED
        assert release.released_gpu_uuids == (placement["gpu_uuid"],)

        # Exact replay must not execute a second runtime sleep.
        manager.replica_state[key] = ReplicaState.DORMANT
        assert asyncio.run(
            manager.sleep(key, operation_id="npu-donate")
        ) == release

        # The server receipt must prove the platform-selected NPU sleep level.
        sleep_receipt = ray.get(replica.servers[0].sleep.remote())
        assert sleep_receipt["sleeping"] is True
        assert sleep_receipt["sleep_level"] == 1

        destroy_evidence = _exercise_same_npu_borrower(
            donor_replica=replica,
            donor_key=key,
            placement=placement,
            rollout_config=rollout_config,
            model_config=model_config,
            prompt_ids=prompt_ids,
            token="npu-cycle",
        )
        assert destroy_evidence.released_gpu_uuids == release.released_gpu_uuids
    finally:
        ray.shutdown()


def test_real_npu_force_remove_continues_on_another_replica():
    """Two-NPU FORCE acceptance for targeted abort + logical continuation."""

    model_path = _require_model_path()
    _require_npu(2)

    import ray
    from omegaconf import OmegaConf, open_dict
    from transformers import AutoTokenizer

    from multi_task_scheduler.integration.verl.experimental_fully_async.rollouter import (
        MultiTaskFullyAsyncRollouter,
        _MultiTaskFullyAsyncLLMServerClient,
    )
    from multi_task_scheduler.orchestration.contracts import (
        AttemptState,
        EvidenceType,
        ReplicaKey,
        ReplicaKind,
        ReplicaState,
    )
    from multi_task_scheduler.rollout.load_balancer import (
        MultiTaskGlobalRequestLoadBalancer,
    )
    from multi_task_scheduler.rollout.replica import MultiTaskvLLMReplica
    from verl.single_controller.ray.base import _unwrap_ray_remote
    from verl.utils.tokenizer import normalize_token_ids

    config = _config(model_path)
    with open_dict(config):
        config.async_training = OmegaConf.create({"partial_rollout": True})

    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model

    ray.shutdown()
    ray.init(
        num_cpus=8,
        runtime_env=_runtime_env(),
        ignore_reinit_error=True,
    )

    try:
        _require_ray_npus(ray, 2)

        target = MultiTaskvLLMReplica(
            replica_rank=0,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=1,
            replica_kind=ReplicaKind.BORROWED,
        )
        alternate = MultiTaskvLLMReplica(
            replica_rank=1,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=1,
            replica_kind=ReplicaKind.NATIVE,
        )

        async def init_replicas():
            await asyncio.gather(
                target.init_standalone(),
                alternate.init_standalone(),
            )

        asyncio.run(init_replicas())

        target_key = ReplicaKey("npu-force", "borrowed-0")
        lb = ray.remote(MultiTaskGlobalRequestLoadBalancer).remote(
            {target._server_address: target._server_handle},
            full_determinism=False,
            initial_routes={target_key: target._server_address},
        )
        client = _MultiTaskFullyAsyncLLMServerClient(
            config=config,
            load_balancer_handle=lb,
            client_id="npu-force-client",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        prompt_ids = normalize_token_ids(
            tokenizer.encode(
                "Continue this response long enough to validate NPU FORCE handoff.",
                add_special_tokens=True,
            )
        )
        logical_request_id = f"npu-force-{uuid4().hex}"

        class Manager:
            global_load_balancer = lb

            def __init__(self):
                self.replica_state = {target_key: ReplicaState.ACTIVE}
                self.replica_kind = {target_key: ReplicaKind.BORROWED}

            def replica_meta(self, key):
                return self.replica_kind[key], self.replica_state[key]

            def inspect_runtime(self, key):
                assert key == target_key
                return target

            def transition_replica(self, key, state):
                self.replica_state[key] = state

        async def scenario():
            generation = asyncio.create_task(
                client.generate(
                    request_id=logical_request_id,
                    prompt_ids=prompt_ids,
                    sampling_params={
                        "temperature": 0.0,
                        "max_tokens": 128,
                        "min_tokens": 128,
                    },
                )
            )

            deadline = asyncio.get_running_loop().time() + 30.0
            while True:
                owned = await lb.requests_for_server.remote(
                    target._server_address
                )
                state = await lb.query_attempt.remote(logical_request_id)
                if logical_request_id in owned and state is AttemptState.ADMITTED:
                    break
                if generation.done():
                    raise AssertionError(
                        "generation completed before FORCE observed admission"
                    )
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError(
                        "request was not admitted to NPU FORCE target in time"
                    )
                await asyncio.sleep(0.01)

            await lb.add_servers.remote(
                {alternate._server_address: alternate._server_handle}
            )

            rollouter_cls = _unwrap_ray_remote(MultiTaskFullyAsyncRollouter)
            rollouter = rollouter_cls.__new__(rollouter_cls)
            rollouter.config = config
            rollouter.llm_server_manager = Manager()
            rollouter._pending_operation_targets = {}

            exit_ready = await rollouter.prepare_exit(
                target_key,
                operation_id="npu-force-remove",
                force=True,
            )
            assert exit_ready.type is EvidenceType.EXIT_READY
            assert (
                await lb.continuation_handoff_requests.remote(
                    "npu-force-remove"
                )
            ) == (logical_request_id,)
            assert not await lb.has_unsettled_requests.remote(
                target._server_address
            )

            output = await asyncio.wait_for(generation, timeout=120.0)
            assert output.stop_reason not in {"abort", "aborted"}
            assert len(output.token_ids) == 128

            await lb.finish_remove.remote(target_key)
            assert (
                await lb.continuation_handoff_requests.remote(
                    "npu-force-remove"
                )
            ) == ()
            return output

        output = asyncio.run(scenario())
        assert getattr(output, "token_ids", None)
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

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def zero_output_weights_for_restore_acceptance(self):
            module = getattr(
                self.engine.module,
                "_fsdp_wrapped_module",
                self.engine.module,
            )
            get_output_embeddings = getattr(
                module,
                "get_output_embeddings",
                None,
            )
            if get_output_embeddings is None:
                raise RuntimeError(
                    "acceptance model does not expose output embeddings"
                )
            output = get_output_embeddings()
            weight = getattr(output, "weight", None)
            if weight is None:
                raise RuntimeError(
                    "acceptance model output embedding has no weight"
                )
            with torch.no_grad():
                weight.zero_()
            local_weight = (
                weight.to_local() if hasattr(weight, "to_local") else weight
            )
            return {
                "shape": tuple(weight.shape),
                "abs_sum": float(
                    local_weight.detach().float().abs().sum().item()
                ),
            }

        @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
        async def update_weights(
            self,
            global_steps: int = None,
            mode: str = "auto",
        ):
            weights, _ = self.engine.get_per_tensor_param()
            await self.checkpoint_engine.send_weights(
                weights,
                global_steps=global_steps,
            )

        @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
        def execute_checkpoint_engine(self, method: str, *args, **kwargs):
            return getattr(self.checkpoint_engine, method)(*args, **kwargs)

    return RestoreTrainingWorker


def test_real_npu_restore_reinstalls_current_vpub_and_generates_again():
    """Two-NPU RESTORE acceptance using the HCCL implementation registered as nccl."""

    model_path = _require_model_path()
    _require_npu(2)

    import ray

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
        runtime_env=_runtime_env(),
        ignore_reinit_error=True,
    )

    try:
        _require_ray_npus(ray, 2)

        trainer_pool = RayResourcePool(
            process_on_nodes=[1],
            use_gpu=True,
            name_prefix="multitask_npu_restore_sender_",
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
        placement = asyncio.run(replica.worker_placements())[0]
        assert placement["resource_name"] == "NPU"

        prompt_ids = normalize_token_ids(
            model_config.tokenizer.encode(
                "Hello from the current-Vpub NPU RESTORE acceptance test.",
                add_special_tokens=True,
            )
        )

        def generate_once(label: str):
            return ray.get(
                replica._server_handle.generate.remote(
                    request_id=f"{label}-{uuid4().hex}",
                    prompt_ids=prompt_ids,
                    sampling_params={"temperature": 0.0, "max_tokens": 16},
                    image_data=None,
                ),
                timeout=120,
            )

        baseline = generate_once("npu-before-restore")
        assert getattr(baseline, "token_ids", None)

        key = ReplicaKey("npu-acceptance", "native-0")
        manager = MultiTaskLLMServerManager.__new__(MultiTaskLLMServerManager)
        manager.replica_kind = {key: ReplicaKind.NATIVE}
        manager.replica_state = {key: ReplicaState.DRAINING}
        manager._runtime_inventory = {key: replica}
        manager._native_release_evidence = {}

        release = asyncio.run(
            manager.sleep(key, operation_id="npu-restore-donate")
        )
        assert release.type is EvidenceType.RELEASED
        assert release.released_gpu_uuids == (placement["gpu_uuid"],)
        manager.replica_state[key] = ReplicaState.DORMANT

        destroy_evidence = _exercise_same_npu_borrower(
            donor_replica=replica,
            donor_key=key,
            placement=placement,
            rollout_config=rollout_config,
            model_config=model_config,
            prompt_ids=prompt_ids,
            token="npu-restore-cycle",
        )
        assert destroy_evidence.released_gpu_uuids == release.released_gpu_uuids

        mutation_receipts = ray.get(
            actor_wg.zero_output_weights_for_restore_acceptance()
        )
        assert all(
            receipt["abs_sum"] == 0.0 for receipt in mutation_receipts
        )

        checkpoint_manager = MultiTaskCheckpointEngineManager(
            config=checkpoint_config,
            actor_wg=actor_wg,
            replicas=[],
        )
        checkpoint_manager.register_pending(
            key,
            [replica],
            operation_id="npu-restore",
        )
        weight_ready = asyncio.run(
            checkpoint_manager.bootstrap_target(
                key,
                operation_id="npu-restore",
                loaded_version=17,
            )
        )
        assert weight_ready.type is EvidenceType.WEIGHT_READY
        assert key not in checkpoint_manager.effective_replicas

        parked_ref = replica._server_handle.generate.remote(
            request_id=f"npu-parked-before-final-wake-{uuid4().hex}",
            prompt_ids=prompt_ids,
            sampling_params={"temperature": 0.0, "max_tokens": 16},
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
        assert restored.token_ids != baseline.token_ids, (
            "NPU RESTORE output did not change after sender Vpub mutation; "
            "global_steps alone is not sufficient acceptance evidence"
        )
    finally:
        ray.shutdown()
