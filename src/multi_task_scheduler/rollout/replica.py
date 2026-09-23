"""Native vLLM replica extension for the supported STANDALONE profile."""

import hashlib

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

    @staticmethod
    def _short_identity(value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("identity value must be a nonempty string")
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]

    def validate_placement(self, spec: dict) -> None:
        """Validate the borrower-local topology without creating Ray actors."""
        if self.replica_kind is not ReplicaKind.BORROWED:
            raise ValueError("lease placement is valid only for BORROWED replicas")
        if not isinstance(spec, dict):
            raise TypeError("borrowed placement spec must be a dict")
        if spec.get("replica_rank") != self.replica_rank:
            raise ValueError("placement replica_rank does not match runtime identity")
        if spec.get("placement_epoch", 0) != self.runtime_epoch:
            raise ValueError("placement_epoch does not match runtime identity")
        if spec.get("max_colocate_count") != FIRST_RELEASE_MAX_COLOCATE_COUNT:
            raise ValueError("placement max_colocate_count does not match first release")

        claims = spec.get("claims")
        if not isinstance(claims, (list, tuple)) or not claims:
            raise ValueError("borrowed placement requires normalized claims")
        if spec.get("world_size") != len(claims):
            raise ValueError("placement world_size must equal claim count")
        if len(claims) != self.world_size:
            raise ValueError("placement world_size does not match borrower model topology")

        node_ids = set()
        bundle_keys = set()
        gpu_uuids = set()
        claim_ids = set()
        for rank, claim in enumerate(claims):
            if claim.get("rank") != rank:
                raise ValueError("borrower ranks must be ordered 0..world_size-1")
            if claim.get("node_rank") != 0 or claim.get("local_rank") != rank:
                raise ValueError("first release requires one-node borrower rank layout")
            if claim.get("gpu_fraction") != 1.0 / self.max_colocate_count:
                raise ValueError("claim GPU accounting share does not match runtime M")
            if claim.get("cpu_request") != 1.0:
                raise ValueError("first release requires one CPU per borrower CE actor")

            claim_id = claim.get("claim_id")
            gpu_uuid = claim.get("gpu_uuid")
            bundle_key = (claim.get("pg_id"), claim.get("bundle_index"))
            node_id = claim.get("node_id")
            if not isinstance(claim_id, str) or not claim_id:
                raise ValueError("claim_id must be nonempty")
            if not isinstance(gpu_uuid, str) or not gpu_uuid:
                raise ValueError("gpu_uuid must be nonempty")
            if not isinstance(node_id, str) or not node_id:
                raise ValueError("node_id must be nonempty")
            if bundle_key[0] is None or bundle_key[1] is None:
                raise ValueError("claim requires PG id and bundle index")
            if claim_id in claim_ids or gpu_uuid in gpu_uuids or bundle_key in bundle_keys:
                raise ValueError("first release claims must be unique per physical GPU")
            claim_ids.add(claim_id)
            gpu_uuids.add(gpu_uuid)
            bundle_keys.add(bundle_key)
            node_ids.add(node_id)

        if len(node_ids) != 1:
            raise ValueError("first release borrowed placement must be single-node")

    def _worker_prefix(self, spec: dict) -> str:
        token = self._short_identity(
            f"{spec['lease_id']}:{spec['operation_id']}:{self.runtime_epoch}"
        )
        return f"borrowed_ce_{self.replica_rank}_{token}_"

    def _worker_name(self, spec: dict, claim: dict) -> str:
        claim_token = self._short_identity(claim["claim_id"])
        return f"{self._worker_prefix(spec)}r{claim['rank']}_{claim_token}"

    def build_borrowed_worker_plan(self, spec: dict) -> tuple[dict, ...]:
        """Build deterministic CE actor placement metadata with no Ray side effects."""
        self.validate_placement(spec)
        prefix = self._worker_prefix(spec)
        plan = []
        for claim in spec["claims"]:
            plan.append(
                {
                    "rank": claim["rank"],
                    "claim_id": claim["claim_id"],
                    "actor_name": self._worker_name(spec, claim),
                    "pg_id": claim["pg_id"],
                    "bundle_index": claim["bundle_index"],
                    "node_id": claim["node_id"],
                    "gpu_uuid": claim["gpu_uuid"],
                    "num_gpus": claim["gpu_fraction"],
                    "num_cpus": claim["cpu_request"],
                    "env_vars": {
                        "WORLD_SIZE": str(spec["world_size"]),
                        "RANK": str(claim["rank"]),
                        "RAY_LOCAL_WORLD_SIZE": str(spec["world_size"]),
                        "WG_PREFIX": prefix,
                        "WG_BACKEND": "ray",
                    },
                }
            )
        self.placement_claims = tuple(dict(claim) for claim in spec["claims"])
        return tuple(plan)

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
