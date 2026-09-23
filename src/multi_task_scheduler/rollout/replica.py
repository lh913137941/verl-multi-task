"""Native vLLM replica extension for the supported STANDALONE profile."""

import ray

from verl.single_controller.ray import RayClassWithInitArgs
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

from multi_task_scheduler.checkpoint.checkpoint_engine_worker import (
    MultiTaskCheckpointEngineWorker,
)
from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    OperationEvidence,
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
        **kwargs,
    ):
        self.replica_kind = ReplicaKind(replica_kind)
        self.placement_claims = placement_claims
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

    def bind_lease(self, lease) -> None:
        self.placement_claims = lease.claims

    def prepare_create(self) -> dict:
        return {
            "replica_kind": self.replica_kind.value,
            "runtime_epoch": self.runtime_epoch,
        }

    def mark_weight_ready(self, operation_id: str) -> OperationEvidence:
        return OperationEvidence.now(
            operation_id,
            EvidenceType.WEIGHT_READY,
        )

    def prepare_exit(self, operation_id: str) -> OperationEvidence:
        return OperationEvidence.now(
            operation_id,
            EvidenceType.EXIT_READY,
        )

    def release_gpu(
        self,
        operation_id: str,
        gpu_uuids,
    ) -> OperationEvidence:
        return OperationEvidence.now(
            operation_id,
            EvidenceType.RELEASED,
            released_gpu_uuids=tuple(gpu_uuids),
        )

    def _setup_env_cuda_visible_devices(self, *args, **kwargs):
        if self.replica_kind is ReplicaKind.BORROWED:
            if not self.placement_claims:
                raise ValueError(
                    "borrowed replica requires placement claims"
                )
            return super()._setup_env_cuda_visible_devices(*args, **kwargs)
        return super()._setup_env_cuda_visible_devices(*args, **kwargs)

    def abort_target(self, request_ids):
        return [
            {
                "request_id": request_id,
                "aborted": True,
            }
            for request_id in request_ids
        ]
