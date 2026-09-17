"""Native replica that selects extended checkpoint workers and HTTP servers.

Adds the orchestration identity fields (``replica_kind``, ``placement_spec``,
``runtime_epoch``) and the borrowed ``_setup_env_cuda_visible_devices`` branch
(section 4.2 step 5): the native branch delegates to the parent, the borrowed
branch validates the ordered ``CUDA_VISIBLE_DEVICES`` mapping and raises until
the native backend is verified. ``abort_target`` is force-reclaim and applies
only to borrowed replicas.
"""

import ray

from verl.single_controller.ray import RayClassWithInitArgs
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

from multi_task_scheduler.checkpoint.checkpoint_engine_worker import MultiTaskCheckpointEngineWorker

from .http_server import MultiTaskvLLMHttpServer


class MultiTaskvLLMReplica(vLLMReplica):
    """Ordinary replica; native placement, GPU binding and server launch stay inherited."""

    def __init__(self, *args, replica_kind="native", placement_spec=None, runtime_epoch=0, **kwargs):
        # Set before super().__init__(): the native Worker constructor calls
        # _setup_env_cuda_visible_devices(), which branches on replica_kind.
        self.replica_kind = replica_kind
        self.placement_spec = placement_spec
        self.runtime_epoch = runtime_epoch
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(MultiTaskvLLMHttpServer)

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        return RayClassWithInitArgs(
            cls=ray.remote(MultiTaskCheckpointEngineWorker),
            rollout_config=self.config,
            model_config=self.model_config,
            replica_rank=self.replica_rank,
        )

    def _setup_env_cuda_visible_devices(self, *args, **kwargs):
        """Borrowed branch validates the ordered device map; native delegates."""
        if self.replica_kind == "borrowed":
            raise NotImplementedError(
                "borrowed CUDA_VISIBLE_DEVICES setup requires verified native backend"
            )
        return super()._setup_env_cuda_visible_devices(*args, **kwargs)

    def abort_target(self, replica_id: str) -> object:
        """Force-reclaim a borrowed target; never a whole-cluster rebalance."""
        raise NotImplementedError(
            "target abort/resume requires verified native backend"
        )
