"""Native vLLM replica extension for the supported STANDALONE profile."""

import ray

from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.utils.device import get_device_name
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

from multi_task_scheduler.checkpoint.checkpoint_engine_worker import (
    MultiTaskCheckpointEngineWorker,
)
from multi_task_scheduler.orchestration.contracts import (
    FIRST_RELEASE_MAX_COLOCATE_COUNT,
    ReplicaKind,
)
from .http_server import MultiTaskvLLMHttpServer


class MultiTaskvLLMReplica(vLLMReplica):
    def __init__(
        self,
        *args,
        replica_kind=ReplicaKind.NATIVE,
        placement_claims=None,
        runtime_epoch=0,
        max_colocate_count=FIRST_RELEASE_MAX_COLOCATE_COUNT,
        **kwargs,
    ):
        self.replica_kind = ReplicaKind(replica_kind)
        self.placement_claims = placement_claims
        self.runtime_epoch = runtime_epoch
        if max_colocate_count != FIRST_RELEASE_MAX_COLOCATE_COUNT:
            raise ValueError(
                "first release requires max_colocate_count="
                f"{FIRST_RELEASE_MAX_COLOCATE_COUNT}"
            )
        self.max_colocate_count = max_colocate_count
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(MultiTaskvLLMHttpServer)

    async def init_standalone(self):
        """Create native PGs with the same Ray accounting M used by leases."""
        self.rollout_mode = RolloutMode.STANDALONE
        if self.is_reward_model:
            resource_pool_name = f"rollout_pool_reward_{self.replica_rank}{self.name_suffix}"
        elif self.is_teacher_model:
            resource_pool_name = f"rollout_pool_teacher_{self.replica_rank}{self.name_suffix}"
        else:
            resource_pool_name = f"rollout_pool_{self.replica_rank}{self.name_suffix}"

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec={
                resource_pool_name: [self.gpus_per_replica_node] * self.nnodes,
            },
            mapping=None,
            max_colocate_count=self.max_colocate_count,
        )
        resource_pool_manager.create_resource_pool()
        self.resource_pool = resource_pool_manager.resource_pool_dict[
            resource_pool_name
        ]

        if self.is_reward_model:
            name_prefix = (
                f"rollout_reward_standalone_{self.replica_rank}{self.name_suffix}"
            )
        elif self.is_teacher_model:
            name_prefix = (
                f"rollout_teacher_standalone_{self.replica_rank}{self.name_suffix}"
            )
        else:
            name_prefix = f"rollout_standalone_{self.replica_rank}{self.name_suffix}"

        worker_group = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=self.get_ray_class_with_init_args(),
            bin_pack=False,
            name_prefix=name_prefix,
            use_gpu=True,
            device_name=get_device_name(),
        )
        self.workers = worker_group.workers
        await self.launch_servers()

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        return RayClassWithInitArgs(
            cls=ray.remote(MultiTaskCheckpointEngineWorker),
            rollout_config=self.config,
            model_config=self.model_config,
            replica_rank=self.replica_rank,
        )

    def prepare_create(self) -> dict:
        raise NotImplementedError(
            "replica creation requires verified native runtime backend"
        )

    def mark_weight_ready(self, operation_id: str):
        raise NotImplementedError(
            "weight readiness requires verified replay/bootstrap backend"
        )

    def prepare_exit(self, operation_id: str):
        raise NotImplementedError(
            "exit readiness requires verified vLLM sleep/drain backend"
        )

    def release_gpu(self, operation_id: str, gpu_uuids):
        raise NotImplementedError(
            "GPU release evidence requires verified runtime destroy backend"
        )

    def _setup_env_cuda_visible_devices(self, *args, **kwargs):
        if self.replica_kind is ReplicaKind.BORROWED:
            raise NotImplementedError(
                "borrowed replica requires a verified lease-aware GPU binding backend"
            )
        return super()._setup_env_cuda_visible_devices(*args, **kwargs)

    def abort_target(self, request_ids):
        raise NotImplementedError(
            "targeted abort requires verified FORCE_VERIFIED backend"
        )
