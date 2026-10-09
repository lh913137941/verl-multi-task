"""REPLICA wiring regression scenarios; shared test fakes live in _wiring_support."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import pytest
import asyncio
import hashlib
import time
import pytest
from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    FIRST_RELEASE_MAX_COLOCATE_COUNT,
    FIRST_RELEASE_RAY_GPU_FRACTION,
    Lease,
    OperationEvidence,
    ReplicaKey,
    ReplicaKind,
    ReplicaState,
)
from _wiring_support import (
    INTEGRATION,
    isolated,
    AsyncRemoteMethod,
    replica_class,
    runtime_replica_class,
    native_manager_class,
    _borrowed_test_claim,
)


def test_manager_owns_state_kind_and_runtime_inventory_separately():
    class Parent:
        def __init__(self, *args):
            self.rollout_replicas = []
            self.server_addresses = []
            self.server_handles = []
            self.rollout_config = type(
                "RolloutConfig",
                (),
                {
                    "tensor_model_parallel_size": 1,
                    "data_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "n_gpus_per_node": 1,
                },
            )()
            self.model_config = object()

    allowed = {
        ReplicaState.CREATING: {
            ReplicaState.ACTIVE,
            ReplicaState.RELEASED,
            ReplicaState.QUARANTINED,
        },
        ReplicaState.ACTIVE: {ReplicaState.DRAINING},
        ReplicaState.DRAINING: {
            ReplicaState.ACTIVE,
            ReplicaState.DORMANT,
            ReplicaState.RELEASED,
            ReplicaState.QUARANTINED,
        },
        ReplicaState.DORMANT: {
            ReplicaState.ACTIVE,
            ReplicaState.QUARANTINED,
        },
        ReplicaState.RELEASED: set(),
        ReplicaState.QUARANTINED: set(),
    }
    class FakePGId:
        def __init__(self, value):
            self.value = value

        def hex(self):
            return self.value

    class FakePG:
        def __init__(self, value, bundle_count=1):
            self.id = FakePGId(value)
            self.bundle_count = bundle_count

    fake_pg = FakePG("pg")
    fake_ray = type(
        "ManagerRay",
        (),
        {
            "util": type(
                "Util",
                (),
                {
                    "placement_group_table": staticmethod(
                        lambda: {
                            "pg": {
                                "state": "CREATED",
                                "name": "verl-pg-0",
                                "bundles": {0: {"CPU": 1, "GPU": 1}},
                                "bundles_to_node_id": {0: "n0"},
                            }
                        }
                    ),
                    "get_placement_group": staticmethod(
                        lambda name: fake_pg
                        if name == "verl-pg-0"
                        else (_ for _ in ()).throw(ValueError(name))
                    ),
                },
            )(),
            "get_runtime_context": staticmethod(
                lambda: type("RuntimeContext", (), {"namespace": "verl-test"})()
            ),
        },
    )
    class FakeBorrowedRuntime:
        def __init__(self, **kwargs):
            self.replica_rank = kwargs["replica_rank"]
            self.name_suffix = kwargs.get("name_suffix", "")
            self.workers = ["worker-0"]
            self.servers = ["server-0"]
            self.borrowed_worker_names = ("worker-name",)
            self.borrowed_server_names = ("server-name",)
            self._server_address = "s-borrowed"
            self._server_handle = "h-borrowed"
            self.borrowed_cleanup_verified = False
            self.cleaned = False

        async def init_from_lease(self, spec, pg_by_id):
            assert pg_by_id == {"pg": fake_pg}
            return {
                "state": "RUNTIME_READY",
                "replica_rank": self.replica_rank,
                "server_address": self._server_address,
            }

        async def cleanup_borrowed_runtime(self):
            self.cleaned = True
            self.borrowed_cleanup_verified = True

    cls = isolated(
        f"{INTEGRATION}/llm_server_manager.py",
        "MultiTaskLLMServerManager",
        Parent,
        MultiTaskvLLMReplica=FakeBorrowedRuntime,
        MultiTaskGlobalRequestLoadBalancer=object(),
        ReplicaKey=ReplicaKey,
        ReplicaKind=ReplicaKind,
        ReplicaState=ReplicaState,
        Lease=Lease,
        EvidenceType=EvidenceType,
        OperationEvidence=OperationEvidence,
        FIRST_RELEASE_MAX_COLOCATE_COUNT=FIRST_RELEASE_MAX_COLOCATE_COUNT,
        FIRST_RELEASE_RAY_GPU_FRACTION=FIRST_RELEASE_RAY_GPU_FRACTION,
        asyncio=asyncio,
        hashlib=hashlib,
        ray=fake_ray,
        time=time,
        _ALLOWED=allowed,
    )
    manager = cls(object())
    key = ReplicaKey("task-a", "r0")
    runtime = type("Runtime", (), {"_server_address": "s0", "_server_handle": "h0"})()
    manager.register_replica(key, ReplicaKind.NATIVE, state=ReplicaState.ACTIVE, runtime=runtime)
    manager.register_replica(key, ReplicaKind.NATIVE, state=ReplicaState.ACTIVE, runtime=runtime)
    with pytest.raises(ValueError, match="another runtime"):
        manager.register_replica(
            key,
            ReplicaKind.NATIVE,
            state=ReplicaState.ACTIVE,
            runtime=object(),
        )
    manager.rollout_replicas.append(runtime)
    manager.server_addresses.append("s0")
    manager.server_handles.append("h0")

    manager.transition_replica(key, ReplicaState.DRAINING)
    assert manager.deactivate_service(key) is runtime
    assert manager.inspect_runtime(key) is runtime
    assert manager.rollout_replicas == []
    assert manager.server_addresses == []
    assert manager.server_handles == []

    manager.replica_state[key] = ReplicaState.ACTIVE
    manager.rollout_replicas = []
    manager.server_addresses = ["s0"]
    manager.server_handles = ["wrong-handle"]
    with pytest.raises(RuntimeError, match="address/handle inventory"):
        manager.activate_service(key)
    manager.server_addresses = []
    manager.server_handles = []

    manager.task_session = "task-a"
    valid_spec = {
        "operation_id": "op-add",
        "lease_id": "borrower-lease",
        "borrower_task_id": "task-a",
        "borrower_replica_id": "borrowed-0",
        "claims": [
            {
                "claim_id": "claim-0",
                "source_lease_id": "source-lease-0",
                "donor_task_id": "donor-task",
                "donor_replica_rank": 0,
                "pg_id": "pg",
                "bundle_index": 0,
                "node_id": "n0",
                "gpu_uuid": "u0",
                "node_rank": 0,
                "local_rank": 0,
                "gpu_fraction": FIRST_RELEASE_RAY_GPU_FRACTION,
                "cpu_request": 1.0,
            }
        ],
        "world_size": 1,
        "max_colocate_count": FIRST_RELEASE_MAX_COLOCATE_COUNT,
        "replica_rank": None,
        "expires_at": 0,
        "placement_epoch": 0,
    }
    normalized = manager.validate_borrowed_spec(valid_spec)
    assert "max_colocate_count" not in normalized
    assert normalized["claims"][0]["rank"] == 0

    wrong_task = dict(valid_spec, borrower_task_id="task-b")
    with pytest.raises(ValueError, match="another task_session"):
        manager.validate_borrowed_spec(wrong_task)

    manager.rollout_config.tensor_model_parallel_size = 2
    with pytest.raises(ValueError, match="TP=DP=PP=1"):
        manager.validate_borrowed_spec(valid_spec)
    manager.rollout_config.tensor_model_parallel_size = 1

    wrong_share = dict(valid_spec)
    wrong_share["claims"] = [
        dict(valid_spec["claims"][0], gpu_fraction=1.0)
    ]
    with pytest.raises(ValueError, match="Ray GPU accounting share"):
        manager.validate_borrowed_spec(wrong_share)

    expired = dict(valid_spec, expires_at=time.time() - 1)
    with pytest.raises(ValueError, match="expired"):
        manager.validate_borrowed_spec(expired)

    wrong_namespace = dict(valid_spec)
    wrong_namespace["claims"] = [
        dict(valid_spec["claims"][0], pg_namespace="other")
    ]
    with pytest.raises(ValueError, match="another Ray namespace"):
        manager._resolve_placement_groups(
            manager.validate_borrowed_spec(wrong_namespace)["claims"]
        )

    # Failure before a runtime object exists is a known zero-side-effect
    # landing: the Manager records RELEASED rather than quarantining the key.
    original_resolver = manager._resolve_placement_groups
    manager._resolve_placement_groups = lambda claims: (_ for _ in ()).throw(
        RuntimeError("placement lookup failed before runtime creation")
    )
    pre_runtime_failure = dict(
        valid_spec,
        operation_id="op-add-pre-runtime-failure",
        lease_id="borrower-lease-pre-runtime-failure",
        borrower_replica_id="borrowed-pre-runtime-failure",
        replica_rank=None,
    )
    with pytest.raises(RuntimeError, match="before runtime creation"):
        asyncio.run(manager.create_borrowed_replica(pre_runtime_failure))
    pre_runtime_key = ReplicaKey("task-a", "borrowed-pre-runtime-failure", 0)
    assert manager.replica_state[pre_runtime_key] is ReplicaState.RELEASED
    manager._resolve_placement_groups = original_resolver

    wrong_node = dict(valid_spec)
    wrong_node["claims"] = [
        dict(valid_spec["claims"][0], node_id="n9")
    ]
    with pytest.raises(ValueError, match="node_id does not match"):
        manager._resolve_placement_groups(
            manager.validate_borrowed_spec(wrong_node)["claims"]
        )

    resolved_pgs = manager._resolve_placement_groups(normalized["claims"])
    assert resolved_pgs == {"pg": fake_pg}

    receipt = asyncio.run(manager.create_borrowed_replica(valid_spec))

    record = manager.borrowed_operations["borrower-lease"]
    assert receipt["state"] == "RUNTIME_READY"
    assert receipt["server_address"] == "s-borrowed"
    # Rank 0 was consumed by the earlier failed allocation attempt. Runtime
    # actor ranks are monotonic and are not recycled after verified cleanup.
    assigned_rank = record["resolved_spec"]["replica_rank"]
    assert assigned_rank == 1
    borrowed_key = ReplicaKey("task-a", "borrowed-0", 0)
    assert ReplicaKey(
        record["resolved_spec"]["borrower_task_id"],
        record["resolved_spec"]["borrower_replica_id"],
        record["resolved_spec"]["placement_epoch"],
    ) == borrowed_key
    assert manager.replica_kind[borrowed_key] is ReplicaKind.BORROWED
    assert manager.replica_state[borrowed_key] is ReplicaState.CREATING
    borrowed_runtime = manager.inspect_runtime(borrowed_key)
    assert borrowed_runtime is not None
    assert borrowed_runtime.name_suffix.startswith("borrowed_")
    assert "borrower-lease" not in borrowed_runtime.name_suffix
    assert manager.next_replica_rank == assigned_rank + 1

    # Exact replay returns the same hidden runtime receipt without another rank.
    assert asyncio.run(manager.create_borrowed_replica(valid_spec)) == receipt
    assert manager.next_replica_rank == assigned_rank + 1

    # A retry may carry the rank recovered from the manager's first record.
    resolved_retry = dict(valid_spec, replica_rank=assigned_rank)
    assert asyncio.run(manager.create_borrowed_replica(resolved_retry)) == receipt
    assert manager.next_replica_rank == assigned_rank + 1

    release = asyncio.run(
        manager.destroy(borrowed_key, operation_id="op-remove")
    )
    assert release.type is EvidenceType.RELEASED
    assert release.released_gpu_uuids == ("u0",)
    assert borrowed_runtime.cleaned is True
    assert asyncio.run(
        manager.destroy(borrowed_key, operation_id="op-remove")
    ) == release
    manager.transition_replica(borrowed_key, ReplicaState.RELEASED)
    assert asyncio.run(
        manager.destroy(borrowed_key, operation_id="op-remove")
    ) == release

    duplicate_identity = dict(
        valid_spec,
        operation_id="op-add-duplicate-key",
        lease_id="borrower-lease-duplicate-key",
    )
    next_rank_before = manager.next_replica_rank
    with pytest.raises(ValueError, match="new runtime_epoch"):
        asyncio.run(manager.create_borrowed_replica(duplicate_identity))
    assert manager.next_replica_rank == next_rank_before

    # Caller-supplied replica_rank is compatibility metadata, not an owner
    # identity fact. Manager strips it before replay comparison and keeps the
    # already allocated rank from its own record.
    non_authoritative_rank = dict(valid_spec, replica_rank=7)
    assert (
        asyncio.run(manager.create_borrowed_replica(non_authoritative_rank))
        == receipt
    )
    assert manager.next_replica_rank == assigned_rank + 1

    conflicting = dict(valid_spec, operation_id="op-other")
    with pytest.raises(ValueError, match="conflicting borrowed create replay"):
        asyncio.run(manager.create_borrowed_replica(conflicting))
    assert manager.next_replica_rank == assigned_rank + 1

    class InvalidReceiptRuntime(FakeBorrowedRuntime):
        async def init_from_lease(self, spec, pg_by_id):
            return {
                "state": "BROKEN",
                "replica_rank": self.replica_rank,
            }

    manager.rollout_replica_class = InvalidReceiptRuntime
    invalid_receipt = dict(
        valid_spec,
        operation_id="op-add-invalid-receipt",
        lease_id="borrower-lease-invalid-receipt",
        borrower_replica_id="borrowed-invalid-receipt",
        replica_rank=None,
    )
    with pytest.raises(RuntimeError, match="RUNTIME_READY"):
        asyncio.run(manager.create_borrowed_replica(invalid_receipt))
    invalid_key = ReplicaKey("task-a", "borrowed-invalid-receipt", 0)
    invalid_record = manager.borrowed_operations["borrower-lease-invalid-receipt"]
    assert manager.replica_state[invalid_key] is ReplicaState.RELEASED
    assert invalid_record["error"] is not None

    class CleanedFailureRuntime(FakeBorrowedRuntime):
        async def init_from_lease(self, spec, pg_by_id):
            self.borrowed_cleanup_verified = True
            raise RuntimeError("borrowed init failed after verified cleanup")

    manager.rollout_replica_class = CleanedFailureRuntime
    cleaned_failure = dict(
        valid_spec,
        operation_id="op-add-cleaned-failure",
        lease_id="borrower-lease-cleaned-failure",
        borrower_replica_id="borrowed-cleaned-failure",
        replica_rank=None,
    )
    with pytest.raises(RuntimeError, match="verified cleanup"):
        asyncio.run(manager.create_borrowed_replica(cleaned_failure))
    cleaned_key = ReplicaKey("task-a", "borrowed-cleaned-failure", 0)
    cleaned_record = manager.borrowed_operations["borrower-lease-cleaned-failure"]
    assert manager.replica_state[cleaned_key] is ReplicaState.RELEASED
    assert cleaned_record["error"] is not None
    with pytest.raises(RuntimeError, match="verified cleanup"):
        asyncio.run(manager.create_borrowed_replica(cleaned_failure))

    class UnverifiedFailureRuntime(FakeBorrowedRuntime):
        async def init_from_lease(self, spec, pg_by_id):
            raise RuntimeError("borrowed init failed before cleanup")

        async def cleanup_borrowed_runtime(self):
            self.borrowed_cleanup_verified = False
            raise RuntimeError("cleanup failed")

    manager.rollout_replica_class = UnverifiedFailureRuntime
    unverified_failure = dict(
        valid_spec,
        operation_id="op-add-unverified-failure",
        lease_id="borrower-lease-unverified-failure",
        borrower_replica_id="borrowed-unverified-failure",
        replica_rank=None,
    )
    with pytest.raises(RuntimeError, match="before cleanup"):
        asyncio.run(manager.create_borrowed_replica(unverified_failure))
    quarantined_key = ReplicaKey("task-a", "borrowed-unverified-failure", 0)
    quarantined_record = manager.borrowed_operations["borrower-lease-unverified-failure"]
    assert manager.replica_state[quarantined_key] is ReplicaState.QUARANTINED
    assert quarantined_record["error"] is not None


def test_borrowed_worker_plan_is_deterministic_and_side_effect_free():
    config = type(
        "Config",
        (),
        {
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
        },
    )()
    cls = replica_class()
    replica = cls(
        replica_rank=7,
        config=config,
        model_config=object(),
        replica_kind=ReplicaKind.BORROWED,
        runtime_epoch=3,
        name_suffix="borrowed_deadbeef_3",
    )
    spec = {
        "operation_id": "op-add",
        "lease_id": "lease-1",
        "replica_rank": 7,
        "placement_epoch": 3,
        "world_size": 1,
        "max_colocate_count": FIRST_RELEASE_MAX_COLOCATE_COUNT,
        "claims": [_borrowed_test_claim(bundle_index=4, node_id="node-a", gpu_uuid="GPU-0")],
    }

    first = replica.build_borrowed_worker_plan(spec)
    second = replica.build_borrowed_worker_plan(spec)

    assert first == second
    assert first["actor_name"].startswith(
        "borrowed_ce_7borrowed_deadbeef_3_"
    )
    assert replica.borrowed_server_names == (
        "vllm_server_7_0borrowed_deadbeef_3",
    )
    assert first["pg_id"] == "pg"
    assert first["bundle_index"] == 4
    assert first["num_gpus"] == FIRST_RELEASE_RAY_GPU_FRACTION
    assert first["env_vars"] == {
        "WORLD_SIZE": "1",
        "RANK": "0",
        "RAY_LOCAL_WORLD_SIZE": "1",
        "WG_PREFIX": first["env_vars"]["WG_PREFIX"],
        "WG_BACKEND": "ray",
    }
    assert replica.placement_claims[0]["claim_id"] == "claim-0"


def test_borrowed_worker_plan_rejects_donor_topology_or_wrong_identity():
    config = type(
        "Config",
        (),
        {
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
        },
    )()
    replica = replica_class()(
        replica_rank=2,
        config=config,
        model_config=object(),
        replica_kind=ReplicaKind.BORROWED,
        runtime_epoch=1,
    )
    base = {
        "operation_id": "op-add",
        "lease_id": "lease-1",
        "replica_rank": 2,
        "placement_epoch": 1,
        "world_size": 1,
        "max_colocate_count": FIRST_RELEASE_MAX_COLOCATE_COUNT,
        "claims": [_borrowed_test_claim(node_id="node-a", gpu_uuid="GPU-0")],
    }

    wrong_rank = dict(base, replica_rank=3)
    with pytest.raises(ValueError, match="replica_rank"):
        replica.build_borrowed_worker_plan(wrong_rank)

    donor_layout = dict(base)
    donor_layout["claims"] = [dict(base["claims"][0], node_rank=9)]
    with pytest.raises(ValueError, match="rank=node_rank=local_rank=0"):
        replica.build_borrowed_worker_plan(donor_layout)


def test_worker_gpu_uuid_probe_uses_native_worker_ray_call_context():
    accelerator = {"value": "1"}

    class Context:
        def get_accelerator_ids(self):
            return {"GPU": [accelerator["value"]]}

        def get_node_id(self):
            return "node-a"

        def get_placement_group_id(self):
            return "pg-native-a"

    fake_ray = type(
        "ProbeRay",
        (),
        {"get_runtime_context": staticmethod(lambda: Context())},
    )
    fake_subprocess = type(
        "Subprocess",
        (),
        {
            "check_output": staticmethod(
                lambda *args, **kwargs: "0, GPU-a\n1, GPU-b\n"
            )
        },
    )
    cls = isolated(
        "rollout/replica.py",
        "MultiTaskvLLMReplica",
        object,
        ReplicaKind=ReplicaKind,
        Lease=Lease,
        FIRST_RELEASE_MAX_COLOCATE_COUNT=FIRST_RELEASE_MAX_COLOCATE_COUNT,
        MultiTaskvLLMHttpServer=object,
        MultiTaskCheckpointEngineWorker=object,
        os=__import__("os"),
        subprocess=fake_subprocess,
        ray=fake_ray,
        get_resource_name=lambda: "GPU",
        get_visible_devices_keyword=lambda: "CUDA_VISIBLE_DEVICES",
    )

    placement = cls._runtime_placement_probe(None)
    assert placement["gpu_uuid"] == "GPU-b"
    assert placement["pg_id"] == "pg-native-a"
    accelerator["value"] = "GPU-direct"
    assert cls._runtime_placement_probe(None)["gpu_uuid"] == "GPU-direct"
    accelerator["value"] = "MIG-abc"
    with pytest.raises(NotImplementedError, match="MIG"):
        cls._runtime_placement_probe(None)
    accelerator["value"] = "opaque-id"
    with pytest.raises(RuntimeError, match="cannot map"):
        cls._runtime_placement_probe(None)


def test_create_workers_from_claims_clones_actor_options_per_rank():
    class Parent:
        def __init__(
            self,
            replica_rank,
            config,
            model_config,
            gpus_per_node=8,
            is_reward_model=False,
            is_teacher_model=False,
            name_suffix="",
        ):
            self.replica_rank = replica_rank
            self.config = config
            self.model_config = model_config
            self.world_size = 1
            self.gpus_per_node = gpus_per_node
            self.gpus_per_replica_node = 1
            self.nnodes = 1
            self.is_reward_model = is_reward_model
            self.is_teacher_model = is_teacher_model
            self.name_suffix = name_suffix
            self.workers = []
            self.servers = []

        def _get_server_name_prefix(self):
            return "vllm_"

        def get_ray_class_with_init_args(self):
            return FakeCIA(
                object,
                rollout_config=self.config,
                model_config=self.model_config,
                replica_rank=self.replica_rank,
            )

    created = []
    killed = []

    class FakeHandle:
        def __init__(self, record, *, gpu_uuid="GPU-x"):
            self.record = record
            self.__ray_call__ = AsyncRemoteMethod(
                lambda _probe: {
                    "node_id": "node",
                    "accelerator_id": "0",
                    "gpu_uuid": gpu_uuid,
                    "visible_devices": "0",
                }
            )

    class FakeCIA:
        next_gpu_uuid = "GPU-x"

        def __init__(self, cls, *args, **kwargs):
            self.cls = cls
            self.args = args
            self.kwargs = kwargs
            self.options = {}

        def update_options(self, options):
            self.options.update(options)

        def __call__(self, **kwargs):
            handle = FakeHandle(
                {
                    "options": dict(self.options),
                    "placement": dict(kwargs),
                },
                gpu_uuid=type(self).next_gpu_uuid,
            )
            created.append(handle)
            return handle

    class FakeWG:
        def __init__(self, workers):
            self.workers = list(workers)

        @classmethod
        def from_detached(cls, *, worker_handles, **kwargs):
            return cls(worker_handles)

    fake_ray = type(
        "ReplicaRay",
        (),
        {
            "remote": staticmethod(lambda cls: cls),
            "kill": staticmethod(lambda worker, **kwargs: killed.append(worker)),
            "get_runtime_context": staticmethod(
                lambda: type("Context", (), {"namespace": "test"})()
            ),
        },
    )
    cls = isolated(
        "rollout/replica.py",
        "MultiTaskvLLMReplica",
        Parent,
        ReplicaKind=ReplicaKind,
        Lease=Lease,
        FIRST_RELEASE_MAX_COLOCATE_COUNT=FIRST_RELEASE_MAX_COLOCATE_COUNT,
        RayClassWithInitArgs=FakeCIA,
        RayWorkerGroup=FakeWG,
        ResourcePoolManager=object,
        RolloutMode=object,
        PlacementGroupSchedulingStrategy=object,
        get_master_addr_port=object,
        get_device_name=lambda: "cuda",
        MultiTaskvLLMHttpServer=object,
        MultiTaskCheckpointEngineWorker=object,
        asyncio=asyncio,
        # A killed worker must have a visible Ray DEAD record before cleanup
        # is considered verified; an empty State API result is not proof.
        list_actors=lambda **kwargs: [{"state": "DEAD"}] if killed else [{"state": "ALIVE"}],
        ray=fake_ray,
    )
    config = type(
        "Config",
        (),
        {
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
        },
    )()
    replica = cls(
        replica_rank=4,
        config=config,
        model_config=object(),
        replica_kind=ReplicaKind.BORROWED,
        runtime_epoch=0,
    )
    async def fake_master(pg, bundle_index):
        assert pg == "PG"
        assert bundle_index == 2
        return "127.0.0.1", "23456"

    replica._get_master_addr_port_for_slot = fake_master
    spec = {
        "operation_id": "op",
        "lease_id": "lease",
        "replica_rank": 4,
        "placement_epoch": 0,
        "world_size": 1,
        "max_colocate_count": FIRST_RELEASE_MAX_COLOCATE_COUNT,
        "claims": [_borrowed_test_claim(bundle_index=2)],
    }

    assert asyncio.run(replica._create_workers_from_claims(spec, {"pg": "PG"})) is None

    assert replica.workers == created
    assert len(created) == 1
    record = created[0].record
    assert record["placement"]["placement_group"] == "PG"
    assert record["placement"]["placement_group_bundle_idx"] == 2
    assert record["placement"]["num_gpus"] == FIRST_RELEASE_RAY_GPU_FRACTION
    env = record["options"]["runtime_env"]["env_vars"]
    assert env["MASTER_ADDR"] == "127.0.0.1"
    assert env["MASTER_PORT"] == "23456"
    assert env["WORLD_SIZE"] == "1"
    assert env["RANK"] == "0"
    assert replica.borrowed_worker_names == (
        record["options"]["name"],
    )
    assert asyncio.run(replica.worker_placements())[0]["gpu_uuid"] == "GPU-x"
    assert killed == []

    FakeCIA.next_gpu_uuid = "GPU-wrong"
    failed = cls(
        replica_rank=4,
        config=config,
        model_config=object(),
        replica_kind=ReplicaKind.BORROWED,
        runtime_epoch=0,
    )
    failed._get_master_addr_port_for_slot = fake_master
    with pytest.raises(RuntimeError, match="unexpected physical accelerator"):
        asyncio.run(failed._create_workers_from_claims(spec, {"pg": "PG"}))
    assert killed
    assert failed.workers == []


def test_init_from_lease_reaches_runtime_ready_only_after_server_health():
    class Parent:
        def __init__(
            self,
            replica_rank,
            config,
            model_config,
            gpus_per_node=8,
            is_reward_model=False,
            is_teacher_model=False,
            name_suffix="",
        ):
            self.replica_rank = replica_rank
            self.config = config
            self.model_config = model_config
            self.world_size = 1
            self.gpus_per_node = gpus_per_node
            self.gpus_per_replica_node = 1
            self.nnodes = 1
            self.is_reward_model = is_reward_model
            self.is_teacher_model = is_teacher_model
            self.name_suffix = name_suffix
            self.workers = []
            self.servers = []

        def _get_server_name_prefix(self):
            return "vllm_"
            self._server_handle = None
            self._server_address = None

        async def launch_servers(self):
            health = {
                "node_id": "node",
                "replica_rank": self.replica_rank,
                "server_address": "127.0.0.1",
                "server_port": 8000,
                "global_steps": None,
            }
            server = type(
                "Server",
                (),
                {
                    "runtime_health": AsyncRemoteMethod(lambda: health),
                    "shutdown_runtime": AsyncRemoteMethod(lambda: None),
                },
            )()
            self.servers = [server]
            self._server_handle = server
            self._server_address = "127.0.0.1:8000"

    fake_ray = type("ReplicaRay", (), {"remote": staticmethod(lambda cls: cls)})
    cls = isolated(
        "rollout/replica.py",
        "MultiTaskvLLMReplica",
        Parent,
        ReplicaKind=ReplicaKind,
        Lease=Lease,
        FIRST_RELEASE_MAX_COLOCATE_COUNT=FIRST_RELEASE_MAX_COLOCATE_COUNT,
        RayClassWithInitArgs=object,
        RayWorkerGroup=object,
        ResourcePoolManager=object,
        RolloutMode=type("RolloutMode", (), {"STANDALONE": "standalone"}),
        PlacementGroupSchedulingStrategy=object,
        get_master_addr_port=object,
        get_device_name=lambda: "cuda",
        MultiTaskvLLMHttpServer=object,
        MultiTaskCheckpointEngineWorker=object,
        asyncio=asyncio,
        list_actors=lambda **kwargs: [],
        ray=fake_ray,
    )
    config = type(
        "Config",
        (),
        {
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
        },
    )()
    replica = cls(
        replica_rank=6,
        config=config,
        model_config=object(),
        replica_kind=ReplicaKind.BORROWED,
        runtime_epoch=2,
    )
    worker = type(
        "Worker",
        (),
        {
            "__ray_call__": AsyncRemoteMethod(
                lambda _probe: {
                    "node_id": "node",
                    "accelerator_id": "0",
                    "gpu_uuid": "GPU-x",
                    "visible_devices": "0",
                }
            )
        },
    )()

    async def fake_create(spec, pg_by_id):
        replica.build_borrowed_worker_plan(spec)
        replica.workers = [worker]
        replica.borrowed_worker_names = ("worker-0",)
        await replica.validate_worker_placement()
        return object()

    replica._create_workers_from_claims = fake_create
    spec = {
        "operation_id": "op",
        "lease_id": "lease",
        "replica_rank": 6,
        "placement_epoch": 2,
        "world_size": 1,
        "max_colocate_count": FIRST_RELEASE_MAX_COLOCATE_COUNT,
        "claims": [_borrowed_test_claim()],
    }

    receipt = asyncio.run(replica.init_from_lease(spec, {"pg": "PG"}))

    assert receipt["state"] == "RUNTIME_READY"
    assert receipt["server_address"] == "127.0.0.1:8000"
    assert receipt["health"]["server_port"] == 8000


def test_native_runtime_health_probe_confirms_server_and_ce_worker_node():
    health = {
        "node_id": "node-0",
        "replica_rank": 2,
        "server_address": "127.0.0.1",
        "server_port": 8000,
        "global_steps": 12,
    }

    class Server:
        runtime_health = AsyncRemoteMethod(lambda: dict(health))

    cls = runtime_replica_class()
    replica = cls.__new__(cls)
    replica.replica_kind = ReplicaKind.NATIVE
    replica.replica_rank = 2
    replica.nnodes = 1
    replica.servers = [Server()]
    replica.workers = [object()]

    async def worker_placements():
        return ({"node_id": "node-0", "gpu_uuid": "u0"},)

    replica.worker_placements = worker_placements
    replica._server_handle = replica.servers[0]
    replica._server_address = "127.0.0.1:8000"

    result = asyncio.run(replica.validate_server_runtime())
    assert result["global_steps"] == 12
    assert result["node_id"] == "node-0"


def test_native_replica_requires_verified_server_receipts_for_sleep_and_wake():
    class Server:
        sleep = AsyncRemoteMethod(
            lambda: {
                "sleep_level": 2,
                "sleeping": True,
            }
        )
        wake_up = AsyncRemoteMethod(
            lambda *args, **kwargs: {
                "sleeping": False,
                "fully_awake": True,
            }
        )

    cls = runtime_replica_class()
    replica = cls.__new__(cls)
    replica.replica_kind = ReplicaKind.NATIVE
    replica.servers = [Server()]

    assert asyncio.run(replica.sleep())[0]["sleep_level"] == 2
    assert asyncio.run(replica.wake_up())[0]["fully_awake"] is True


def test_manager_native_sleep_binds_released_evidence_to_runtime_gpu_uuids():
    key = ReplicaKey("task-a", "native-0")

    class Runtime:
        workers = [object()]

        async def worker_placements(self):
            return ({"node_id": "n0", "gpu_uuid": "u0"},)

        async def sleep(self):
            return (
                {"sleep_level": 2, "sleeping": True},
            )

    cls = native_manager_class()
    manager = cls.__new__(cls)
    manager.replica_kind = {key: ReplicaKind.NATIVE}
    manager.replica_state = {key: ReplicaState.DRAINING}
    manager._runtime_inventory = {key: Runtime()}
    manager._native_release_evidence = {}

    evidence = asyncio.run(manager.sleep(key, operation_id="op-donate"))
    assert evidence.type is EvidenceType.RELEASED
    assert evidence.operation_id == "op-donate"
    assert evidence.released_gpu_uuids == ("u0",)


def test_manager_release_evidence_query_reads_native_owner_ledger():
    key = ReplicaKey("task-a", "native-0")
    evidence = OperationEvidence(
        "op-donate",
        EvidenceType.RELEASED,
        3,
        ("u0",),
    )
    cls = native_manager_class()
    manager = cls.__new__(cls)
    manager.replica_kind = {key: ReplicaKind.NATIVE}
    manager._native_release_evidence = {"op-donate": (key, evidence)}

    assert manager.query_release_evidence(key, "op-donate") == evidence
    assert manager.query_release_evidence(key, "other") is None


def test_manager_native_release_replay_returns_identical_evidence():
    key = ReplicaKey("task-a", "native-0")
    sleep_calls = []

    class Runtime:
        workers = [object()]

        async def worker_placements(self):
            return ({"node_id": "n0", "gpu_uuid": "u0"},)

        async def sleep(self):
            sleep_calls.append("sleep")
            return ({"sleep_level": 2, "sleeping": True},)

    cls = native_manager_class()
    manager = cls.__new__(cls)
    manager.replica_kind = {key: ReplicaKind.NATIVE}
    manager.replica_state = {key: ReplicaState.DRAINING}
    manager._runtime_inventory = {key: Runtime()}
    manager._native_release_evidence = {}

    first = asyncio.run(manager.sleep(key, operation_id="op-donate"))
    manager.replica_state[key] = ReplicaState.DORMANT
    second = asyncio.run(manager.sleep(key, operation_id="op-donate"))

    assert second == first
    assert sleep_calls == ["sleep"]



# --- test_borrowed_cleanup_proof.py (consolidated boundary scenarios) ---

_CLEANUP_REPLICA_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/multi_task_scheduler/rollout/replica.py"
)


def _cleanup_replica_class(*, list_actors, kill):
    """Compile just the adapter class so these tests do not need real GPUs."""
    tree = ast.parse(_CLEANUP_REPLICA_SOURCE.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MultiTaskvLLMReplica"
    )
    node.bases = [ast.Name(id="object", ctx=ast.Load())]
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            node,
        ],
        type_ignores=[],
    )
    ray = SimpleNamespace(
        kill=kill,
        get_runtime_context=lambda: SimpleNamespace(namespace="test-namespace"),
    )
    scope = {
        "asyncio": asyncio,
        "ray": ray,
        "list_actors": list_actors,
        "ReplicaKind": SimpleNamespace(NATIVE="native", BORROWED="borrowed"),
        "FIRST_RELEASE_MAX_COLOCATE_COUNT": 2,
    }
    exec(compile(ast.fix_missing_locations(module), str(_CLEANUP_REPLICA_SOURCE), "exec"), scope)
    return scope["MultiTaskvLLMReplica"]


def test_missing_ray_actor_state_is_not_dead_proof():
    observed_filters = []
    records = []
    def list_actors(*, filters):
        observed_filters.append(filters)
        return list(records)

    cls = _cleanup_replica_class(list_actors=list_actors, kill=lambda *a, **k: None)
    replica = cls.__new__(cls)

    assert cls._non_dead_actor_names(("borrowed-server",), "test-namespace") == (
        "borrowed-server",
    )
    with pytest.raises(RuntimeError, match="did not reach DEAD"):
        asyncio.run(replica._wait_actor_names_dead(("borrowed-server",), timeout_s=0))
    assert observed_filters[0] == [
        ("ray_namespace", "=", "test-namespace"),
        ("name", "=", "borrowed-server"),
    ]

    records[:] = [{"state": "DEAD"}]
    asyncio.run(replica._wait_actor_names_dead(("borrowed-server",), timeout_s=0))

    records[:] = [{"state": "DEAD"}, {"state": "ALIVE"}]
    with pytest.raises(RuntimeError, match="did not reach DEAD"):
        asyncio.run(replica._wait_actor_names_dead(("borrowed-server",), timeout_s=0))


def test_unverified_worker_cleanup_retains_handle_for_exact_retry():
    states = []
    killed = []
    worker = object()
    cls = _cleanup_replica_class(
        list_actors=lambda **k: list(states),
        kill=lambda actor, **k: killed.append(actor),
    )
    replica = cls.__new__(cls)
    replica.replica_kind = "borrowed"
    replica.servers = []
    replica.borrowed_server_names = ()
    replica.workers = [worker]
    replica.borrowed_worker_names = ("worker-name",)
    replica.borrowed_cleanup_verified = False

    # Avoid a ten-second poll: Ray has no DEAD record on the first attempt.
    original_wait = replica._wait_actor_names_dead
    async def short_wait(names):
        return await original_wait(names, timeout_s=0)
    replica._wait_actor_names_dead = short_wait

    with pytest.raises(RuntimeError, match="cleanup is unverified"):
        asyncio.run(replica.cleanup_borrowed_runtime())
    assert replica.borrowed_cleanup_verified is False
    assert replica.workers == [worker]
    assert killed == [worker]

    states[:] = [{"state": "DEAD"}]
    asyncio.run(replica.cleanup_borrowed_runtime())
    assert replica.borrowed_cleanup_verified is True
    assert replica.workers == []
    assert killed == [worker, worker]


def test_shutdown_submission_failure_still_kills_server_and_verifies_dead():
    killed = []
    server = SimpleNamespace(
        shutdown_runtime=SimpleNamespace(
            remote=lambda: (_ for _ in ()).throw(RuntimeError("actor already stopping"))
        ),
    )
    cls = _cleanup_replica_class(
        list_actors=lambda **k: [{"state": "DEAD"}] if killed else [],
        kill=lambda actor, **k: killed.append(actor),
    )
    replica = cls.__new__(cls)
    replica.servers = [server]
    replica.borrowed_server_names = ("borrowed-server",)
    replica._server_handle = server
    replica._server_address = "127.0.0.1:8000"

    asyncio.run(replica._shutdown_servers_verified())
    assert killed == [server]
    assert replica.servers == []
    assert replica._server_handle is None
    assert replica._server_address is None


def test_missing_actor_names_cannot_prove_live_handle_release():
    killed = []
    cls = _cleanup_replica_class(
        list_actors=lambda **k: [],
        kill=lambda actor, **k: killed.append(actor),
    )
    replica = cls.__new__(cls)
    replica.workers = [object()]
    with pytest.raises(RuntimeError, match="no actor names"):
        asyncio.run(replica._kill_workers_verified(replica.workers, ()))
    assert killed == replica.workers
