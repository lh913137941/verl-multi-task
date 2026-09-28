"""Native vLLM replica extension for the supported STANDALONE profile."""

import asyncio
import subprocess

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.state import list_actors

from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import get_master_addr_port
from verl.utils.device import get_device_name, get_resource_name
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

from multi_task_scheduler.orchestration.contracts import (
    FIRST_RELEASE_MAX_COLOCATE_COUNT,
    Lease,
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
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(MultiTaskvLLMHttpServer)

    @staticmethod
    def _runtime_placement_probe(_worker) -> dict:
        """Run inside VERL's native CheckpointEngineWorker via __ray_call__."""
        resource_name = get_resource_name()
        if resource_name != "GPU":
            raise NotImplementedError(
                f"first release placement probe supports GPU only, got {resource_name!r}"
            )
        context = ray.get_runtime_context()
        ids = context.get_accelerator_ids().get(resource_name, [])
        if len(ids) != 1:
            raise RuntimeError(
                f"expected exactly one Ray GPU id for CE actor, got {ids!r}"
            )
        accelerator_id = str(ids[0])
        if accelerator_id.startswith("GPU-"):
            gpu_uuid = accelerator_id
        elif accelerator_id.startswith("MIG-"):
            raise NotImplementedError("first release does not support MIG placement")
        elif accelerator_id.isdigit():
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            )
            mapping = {}
            for line in output.splitlines():
                if line.strip():
                    index, uuid = [part.strip() for part in line.split(",", 1)]
                    mapping[index] = uuid
            try:
                gpu_uuid = mapping[accelerator_id]
            except KeyError as exc:
                raise RuntimeError(
                    f"nvidia-smi did not report Ray GPU id {accelerator_id!r}"
                ) from exc
        else:
            raise RuntimeError(
                f"cannot map Ray GPU accelerator id {accelerator_id!r} to a UUID"
            )
        return {"node_id": context.get_node_id(), "gpu_uuid": gpu_uuid}

    def validate_placement(self, spec: dict) -> None:
        """Validate runtime identity/topology on a normalized Lease spec."""
        if self.replica_kind is not ReplicaKind.BORROWED:
            raise ValueError("lease placement is valid only for BORROWED replicas")
        if not isinstance(spec, dict):
            raise TypeError("borrowed placement spec must be a dict")
        if spec.get("replica_rank") != self.replica_rank:
            raise ValueError("placement replica_rank does not match runtime identity")
        if spec.get("placement_epoch", 0) != self.runtime_epoch:
            raise ValueError("placement_epoch does not match runtime identity")
        claims = Lease(
            spec["lease_id"],
            tuple(spec.get("claims") or ()),
            spec.get("expires_at", 0),
        ).claims
        if self.world_size != 1 or len(claims) != 1:
            raise ValueError("current borrowed runtime requires one TP=1 claim")
        claim = claims[0]
        if (
            claim.get("rank") != 0
            or claim.get("node_rank") != 0
            or claim.get("local_rank") != 0
        ):
            raise ValueError("current borrowed claim requires rank=node_rank=local_rank=0")

    def build_borrowed_worker_plan(self, spec: dict) -> tuple[dict, ...]:
        """Build deterministic CE actor placement metadata with no Ray side effects."""
        self.validate_placement(spec)
        prefix = f"borrowed_ce_{self.replica_rank}_"
        claim = spec["claims"][0]
        plan = ({
            "rank": 0,
            "claim_id": claim["claim_id"],
            "actor_name": f"{prefix}r0",
            "pg_id": claim["pg_id"],
            "bundle_index": claim["bundle_index"],
            "node_id": claim["node_id"],
            "gpu_uuid": claim["gpu_uuid"],
            "num_gpus": claim["gpu_fraction"],
            "num_cpus": claim["cpu_request"],
            "env_vars": {
                "WORLD_SIZE": "1",
                "RANK": "0",
                "RAY_LOCAL_WORLD_SIZE": "1",
                "WG_PREFIX": prefix,
                "WG_BACKEND": "ray",
            },
        },)
        self.placement_claims = (dict(claim),)
        self.borrowed_server_names = (
            f"{super()._get_server_name_prefix()}server_"
            f"{self.replica_rank}_0{self.name_suffix}",
        )
        return plan

    async def _get_master_addr_port_for_slot(self, pg, bundle_index: int):
        """Create a borrower communication root on the selected borrower bundle."""
        return await get_master_addr_port.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bundle_index,
            )
        ).remote()

    async def worker_placements(self) -> tuple[dict, ...]:
        placements = await asyncio.gather(
            *[
                worker.__ray_call__.remote(self._runtime_placement_probe)
                for worker in self.workers
            ]
        )
        if any(not isinstance(item, dict) for item in placements):
            raise TypeError("runtime placement probe returned a non-dict result")
        return tuple(dict(item) for item in placements)

    async def validate_worker_placement(self) -> tuple[dict, ...]:
        """Verify each borrower CE actor landed on the claimed node/GPU UUID."""
        claims = tuple(self.placement_claims or ())
        if len(self.workers) != len(claims):
            raise RuntimeError(
                "borrower worker count does not match normalized placement claims"
            )
        placements = await self.worker_placements()
        for claim, actual in zip(claims, placements, strict=True):
            if actual.get("node_id") != claim["node_id"]:
                raise RuntimeError(
                    f"borrower rank {claim['rank']} landed on unexpected node"
                )
            if actual.get("gpu_uuid") != claim["gpu_uuid"]:
                raise RuntimeError(
                    f"borrower rank {claim['rank']} landed on unexpected GPU UUID"
                )
        return placements

    @staticmethod
    def _non_dead_actor_names(
        names: tuple[str, ...],
        namespace: str,
    ) -> tuple[str, ...]:
        active = []
        for name in names:
            states = list_actors(
                filters=[
                    ("ray_namespace", "=", namespace),
                    ("name", "=", name),
                ]
            )
            for state in states:
                value = state.state if hasattr(state, "state") else state["state"]
                if value != "DEAD":
                    active.append(name)
                    break
        return tuple(active)

    async def _wait_actor_names_dead(
        self,
        names: tuple[str, ...],
        *,
        timeout_s: float = 10.0,
    ) -> None:
        if not names:
            return
        namespace = ray.get_runtime_context().namespace
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            active = await asyncio.to_thread(
                self._non_dead_actor_names,
                names,
                namespace,
            )
            if not active:
                return
            if loop.time() >= deadline:
                raise RuntimeError(
                    f"Ray actors did not reach DEAD before timeout: {active!r}"
                )
            await asyncio.sleep(0.1)

    async def _kill_workers_verified(
        self,
        workers,
        names: tuple[str, ...],
    ) -> None:
        for worker in workers:
            try:
                ray.kill(worker, no_restart=True)
            except BaseException:
                # Kill acknowledgement is not release proof. The state query
                # below remains authoritative for this cleanup attempt.
                pass
        await self._wait_actor_names_dead(names)

    async def _create_workers_from_claims(
        self,
        spec: dict,
        pg_by_id: dict[str, object],
    ) -> RayWorkerGroup:
        """Create the verified TP=1 borrower CE actor on its claimed PG bundle."""
        item = self.build_borrowed_worker_plan(spec)[0]
        if set(pg_by_id) != {item["pg_id"]}:
            raise ValueError("placement-group handles do not exactly cover claims")

        pg = pg_by_id[item["pg_id"]]
        master_addr, master_port = await self._get_master_addr_port_for_slot(
            pg, item["bundle_index"]
        )
        base = self.get_ray_class_with_init_args()
        actor_args = RayClassWithInitArgs(base.cls, *base.args, **base.kwargs)
        env_vars = dict(item["env_vars"], MASTER_ADDR=str(master_addr), MASTER_PORT=str(master_port))
        actor_args.update_options(
            {
                "name": item["actor_name"],
                "num_cpus": item["num_cpus"],
                "runtime_env": {"env_vars": env_vars},
            }
        )
        worker = None
        try:
            worker = actor_args(
                placement_group=pg,
                placement_group_bundle_idx=item["bundle_index"],
                use_gpu=True,
                num_gpus=item["num_gpus"],
                device_name=get_device_name(),
            )
            worker_group = RayWorkerGroup.from_detached(
                worker_handles=[worker],
                ray_cls_with_init=RayClassWithInitArgs(base.cls, *base.args, **base.kwargs),
                name_prefix=f"borrowed_ce_{self.replica_rank}_",
                use_gpu=True,
                device_name=get_device_name(),
            )
            self.workers = list(worker_group.workers)
            self.borrowed_worker_names = (item["actor_name"],)
            await self.validate_worker_placement()
            return worker_group
        except BaseException as exc:
            try:
                await self._kill_workers_verified(
                    [] if worker is None else [worker],
                    (item["actor_name"],) if worker is not None else (),
                )
            except BaseException as cleanup_exc:
                self.workers = []
                raise RuntimeError(
                    "borrowed CE creation failed and actor cleanup is unverified"
                ) from cleanup_exc
            self.workers = []
            raise exc

    async def validate_server_runtime(self) -> dict:
        """Verify one first-release server/engine for ADD or native RESTORE."""
        if self.nnodes != 1 or len(self.servers) != 1:
            raise RuntimeError("first release runtime validation requires exactly one server")

        health = await self.servers[0].runtime_health.remote()
        if not isinstance(health, dict):
            raise TypeError("server runtime health returned a non-dict result")
        if health.get("replica_rank") != self.replica_rank:
            raise RuntimeError("server replica_rank mismatch")
        if health.get("node_rank") != 0 or health.get("nnodes") != 1:
            raise RuntimeError("server topology mismatch")
        if not health.get("engine_ready"):
            raise RuntimeError("vLLM engine is not healthy")
        if not health.get("server_address") or not health.get("server_port"):
            raise RuntimeError("HTTP server address is incomplete")
        if self._server_handle is not self.servers[0]:
            raise RuntimeError("primary server handle is inconsistent")
        if not self._server_address:
            raise RuntimeError("primary server address is missing")

        if self.replica_kind is ReplicaKind.BORROWED:
            if not self.placement_claims:
                raise RuntimeError("borrowed runtime has no placement claims")
            expected_node = self.placement_claims[0]["node_id"]
            if health.get("node_id") != expected_node:
                raise RuntimeError("borrowed server landed on unexpected node")
            return dict(health)

        if self.replica_kind is not ReplicaKind.NATIVE:
            raise ValueError("unsupported replica kind for runtime validation")
        workers = tuple(getattr(self, "workers", ()) or ())
        if not workers:
            raise RuntimeError("native runtime has no CE workers for health validation")
        placements = await self.worker_placements()
        worker_nodes = {placement.get("node_id") for placement in placements}
        if len(worker_nodes) != 1 or health.get("node_id") not in worker_nodes:
            raise RuntimeError("native server/worker node placement mismatch")
        return dict(health)

    async def _shutdown_servers_verified(self) -> None:
        servers = list(getattr(self, "servers", []) or [])
        names = tuple(getattr(self, "borrowed_server_names", ()) or ())
        if servers:
            shutdown_refs = [server.shutdown_runtime.remote() for server in servers]
            try:
                await asyncio.wait_for(
                    asyncio.gather(*shutdown_refs, return_exceptions=True),
                    timeout=30.0,
                )
            except BaseException:
                # Graceful shutdown is best-effort. Actor death below is the
                # authoritative cleanup condition for the first release.
                pass
            for server in servers:
                try:
                    ray.kill(server, no_restart=True)
                except BaseException:
                    pass
        await self._wait_actor_names_dead(names)
        self.servers = []
        self._server_handle = None
        self._server_address = None

    async def cleanup_borrowed_runtime(self) -> None:
        """Destroy borrower-owned actors and own the cleanup proof flag."""
        if self.replica_kind is not ReplicaKind.BORROWED:
            raise ValueError("borrowed cleanup is BORROWED-only")
        self.borrowed_cleanup_verified = False
        errors = []
        try:
            await self._shutdown_servers_verified()
        except BaseException as exc:
            errors.append(exc)
        try:
            await self._kill_workers_verified(
                list(getattr(self, "workers", []) or []),
                tuple(getattr(self, "borrowed_worker_names", ()) or ()),
            )
        except BaseException as exc:
            errors.append(exc)
        self.workers = []
        if errors:
            raise RuntimeError("borrowed runtime cleanup is unverified") from errors[0]
        self.borrowed_cleanup_verified = True

    async def init_from_lease(
        self,
        spec: dict,
        pg_by_id: dict[str, object],
    ) -> dict:
        """Build and validate a hidden borrower runtime on existing donor PGs.

        Manager calls this low-level transaction for hidden creation only;
        target-only bootstrap and service publication remain separate gates.
        """
        # _create_workers_from_claims() validates the normalized Lease
        # before its first Ray side effect.
        self.rollout_mode = RolloutMode.STANDALONE
        self.nnodes = 1
        self.gpus_per_replica_node = self.world_size
        self.borrowed_cleanup_verified = False

        try:
            await self._create_workers_from_claims(spec, pg_by_id)
            await self.launch_servers()
            health = await self.validate_server_runtime()
            return {
                "state": "RUNTIME_READY",
                "replica_rank": self.replica_rank,
                "worker_names": tuple(self.borrowed_worker_names),
                "server_names": tuple(self.borrowed_server_names),
                "server_address": self._server_address,
                "health": dict(health),
            }
        except BaseException as exc:
            try:
                await self.cleanup_borrowed_runtime()
            except BaseException as cleanup_exc:
                raise RuntimeError(
                    "borrowed runtime creation failed and cleanup is unverified"
                ) from cleanup_exc
            raise exc

    async def sleep(self):
        """Deep-sleep the retained native runtime through verified server receipts."""
        if self.replica_kind is not ReplicaKind.NATIVE:
            raise ValueError("sleep is valid only for retained NATIVE replicas")
        servers = tuple(getattr(self, "servers", ()) or ())
        if not servers:
            raise RuntimeError("native replica has no rollout servers to sleep")
        receipts = await asyncio.gather(
            *[server.sleep.remote() for server in servers]
        )
        for receipt in receipts:
            if not isinstance(receipt, dict):
                raise TypeError("native server sleep returned a non-dict receipt")
            if receipt.get("sleep_level") != 2 or receipt.get("sleeping") is not True:
                raise RuntimeError("native server did not confirm level-2 sleep")
        return tuple(receipts)

    async def wake_up(self, tags: list[str] | None = None):
        """Wake native servers and verify partial/full residency receipts."""
        if self.replica_kind is not ReplicaKind.NATIVE:
            raise ValueError("wake_up is valid only for retained NATIVE replicas")
        servers = tuple(getattr(self, "servers", ()) or ())
        if not servers:
            raise RuntimeError("native replica has no rollout servers to wake")
        receipts = await asyncio.gather(
            *[server.wake_up.remote(tags=tags) for server in servers]
        )
        weights_only = tags == ["weights"]
        for receipt in receipts:
            if not isinstance(receipt, dict):
                raise TypeError("native server wake returned a non-dict receipt")
            if weights_only:
                if receipt.get("fully_awake") is not False or receipt.get("sleeping") is not True:
                    raise RuntimeError("native server did not confirm weights-only wake")
            elif receipt.get("fully_awake") is not True or receipt.get("sleeping") is not False:
                raise RuntimeError("native server did not confirm full wake")
        return tuple(receipts)


