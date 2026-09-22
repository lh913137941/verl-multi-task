"""Native vLLM replica extension for the supported STANDALONE profile."""
import ray
from verl.single_controller.ray import RayClassWithInitArgs
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica
from multi_task_scheduler.checkpoint.checkpoint_engine_worker import MultiTaskCheckpointEngineWorker
from multi_task_scheduler.orchestration.contracts import ReplicaKind
from .http_server import MultiTaskvLLMHttpServer

class MultiTaskvLLMReplica(vLLMReplica):
    def __init__(self, *args, replica_kind=ReplicaKind.NATIVE, placement_claims=None, runtime_epoch=0, **kwargs):
        self.replica_kind = ReplicaKind(replica_kind)
        self.placement_claims = placement_claims
        self.runtime_epoch = runtime_epoch
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(MultiTaskvLLMHttpServer)
    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        return RayClassWithInitArgs(cls=ray.remote(MultiTaskCheckpointEngineWorker), rollout_config=self.config, model_config=self.model_config, replica_rank=self.replica_rank)
    def _setup_env_cuda_visible_devices(self, *args, **kwargs):
        if self.replica_kind is ReplicaKind.BORROWED:
            raise NotImplementedError("borrowed CUDA_VISIBLE_DEVICES binding requires verified lease-aware native backend")
        return super()._setup_env_cuda_visible_devices(*args, **kwargs)
    def abort_target(self, request_ids):
        raise NotImplementedError("targeted abort/continuation requires verified native backend")
