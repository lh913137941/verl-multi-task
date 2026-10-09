"""Dependency-light fakes and isolated VERL component builders for test_wiring.

These helpers deliberately load only the requested class AST, not the native
Ray/VERL actor module, so boundary tests run without GPU/Ray/VERL imports.
Scenario-specific behavior and assertions remain in test_wiring.py.
"""

import ast
import asyncio
import threading
import time
from pathlib import Path

from multi_task_scheduler.orchestration.contracts import (
    CONTROL_RPC_TIMEOUT_S,
    EvidenceType,
    AttemptState,
    FIRST_RELEASE_MAX_COLOCATE_COUNT,
    FIRST_RELEASE_RAY_GPU_FRACTION,
    Lease,
    OperationCommand,
    OperationEvidence,
    OperationKind,
    OperationRecord,
    OperationStatus,
    ReplicaKey,
    ReplicaKind,
    ReplicaState,
    native_replica_key,
    require_operation_evidence,
)
from multi_task_scheduler.orchestration.operation_journal import OperationJournal
from multi_task_scheduler.orchestration.replica_sync_gate import GateKind, ReplicaSyncGate

SOURCE = Path(__file__).resolve().parents[2] / "src/multi_task_scheduler"
INTEGRATION = "integration/verl/experimental_fully_async"


def isolated(relative, name, parent, **scope):
    path = SOURCE / relative
    tree = ast.parse(path.read_text())
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == name
    )
    node.bases = [ast.Name(id="Parent", ctx=ast.Load())]
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
    env = {"Parent": parent, **scope}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env[name]


class RemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class AsyncRemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    async def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def _test_require_evidence(value, operation_id, expected, label):
    if not isinstance(value, OperationEvidence):
        raise TypeError(f"{label} did not return OperationEvidence")
    if value.operation_id != operation_id:
        raise ValueError(f"{label} evidence belongs to another operation")
    if value.type is not expected:
        raise ValueError(f"expected {expected.value}, got {value.type.value}")
    return value


class FakeRay:
    class exceptions:
        class GetTimeoutError(Exception):
            pass

        class RayActorError(Exception):
            pass

    @staticmethod
    def get(value, timeout=None):
        return value


def taskrunner_class():
    class Parent:
        def __init__(self):
            self.components = {}

    return isolated(
        f"{INTEGRATION}/task_runner.py",
        "MultiTaskFullyAsyncTaskRunner",
        Parent,
        OperationJournal=OperationJournal,
        Lease=Lease,
        OperationCommand=OperationCommand,
        OperationRecord=OperationRecord,
        OperationStatus=OperationStatus,
        OperationKind=OperationKind,
        OperationEvidence=OperationEvidence,
        EvidenceType=EvidenceType,
        threading=threading,
        ray=FakeRay,
        FIRST_RELEASE_MAX_COLOCATE_COUNT=FIRST_RELEASE_MAX_COLOCATE_COUNT,
        CONTROL_RPC_TIMEOUT_S=CONTROL_RPC_TIMEOUT_S,
        _require_evidence=require_operation_evidence,
        logger=type(
            "Logger",
            (),
            {
                "exception": lambda *args, **kwargs: None,
                "warning": lambda *args, **kwargs: None,
            },
        )(),
    )


def trainer_class():
    class Parent:
        def __init__(self, *args, **kwargs):
            self.rollouter = None
            self.checkpoint_manager = None
            self.current_param_version = 0

    return isolated(
        f"{INTEGRATION}/trainer.py",
        "MultiTaskFullyAsyncTrainer",
        Parent,
        OperationRecord=OperationRecord,
        OperationEvidence=OperationEvidence,
        EvidenceType=EvidenceType,
        ReplicaKey=ReplicaKey,
        ReplicaKind=ReplicaKind,
        native_replica_key=native_replica_key,
        GateKind=GateKind,
        ReplicaSyncGate=ReplicaSyncGate,
        _require_evidence=_test_require_evidence,
    )


def load_balancer_class():
    class Parent:
        def __init__(self, servers, **kwargs):
            self._servers = dict(servers)
            self._inflight_requests = {server_id: 0 for server_id in servers}

        def acquire_server(self, request_id, **extra):
            if not self._servers:
                raise RuntimeError("No available servers")
            server_id = next(iter(self._servers))
            self._inflight_requests[server_id] += 1
            return server_id, self._servers[server_id]

        def release_server(self, server_id, request_id=None):
            if server_id in self._inflight_requests:
                self._inflight_requests[server_id] -= 1

        def remove_servers(self, server_ids):
            for server_id in server_ids:
                self._servers.pop(server_id, None)
                self._inflight_requests.pop(server_id, None)

        def add_servers(self, servers):
            for server_id, handle in dict(servers).items():
                self._servers[server_id] = handle
                self._inflight_requests.setdefault(server_id, 0)

    return isolated(
        "rollout/load_balancer.py",
        "MultiTaskGlobalRequestLoadBalancer",
        Parent,
        DEFAULT_ROUTING_CACHE_SIZE=128,
        ReplicaKey=ReplicaKey,
        AttemptState=AttemptState,
        OperationEvidence=OperationEvidence,
        EvidenceType=EvidenceType,
    )



def taskrunner_lease():
    return Lease(
        "l1",
        (
            {
                "claim_id": "claim-0",
                "source_lease_id": "source-lease-0",
                "donor_task_id": "donor-task",
                "donor_replica_rank": 0,
                "pg_id": "pg",
                "bundle_index": 0,
                "node_id": "n0",
                "gpu_uuid": "u0",
                "gpu_fraction": FIRST_RELEASE_RAY_GPU_FRACTION,
                "cpu_request": 1.0,
                "node_rank": 0,
                "local_rank": 0,
            },
        ),
        0,
    )


def replica_class():
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
            self.world_size = (
                config.tensor_model_parallel_size
                * config.data_parallel_size
                * config.pipeline_model_parallel_size
            )
            self.gpus_per_node = gpus_per_node
            self.gpus_per_replica_node = min(gpus_per_node, self.world_size)
            self.nnodes = self.world_size // self.gpus_per_replica_node
            self.is_reward_model = is_reward_model
            self.is_teacher_model = is_teacher_model
            self.name_suffix = name_suffix
            self.workers = []
            self.servers = []

        def _get_server_name_prefix(self):
            return "vllm_"

    fake_ray = type("ReplicaRay", (), {"remote": staticmethod(lambda cls: cls)})
    return isolated(
        "rollout/replica.py",
        "MultiTaskvLLMReplica",
        Parent,
        ReplicaKind=ReplicaKind,
        Lease=Lease,
        FIRST_RELEASE_MAX_COLOCATE_COUNT=FIRST_RELEASE_MAX_COLOCATE_COUNT,
        RayClassWithInitArgs=object,
        RayWorkerGroup=object,
        ResourcePoolManager=object,
        RolloutMode=object,
        get_device_name=lambda: "cuda",
        get_resource_name=lambda: "GPU",
        get_visible_devices_keyword=lambda: "CUDA_VISIBLE_DEVICES",
        MultiTaskvLLMHttpServer=object,
        MultiTaskCheckpointEngineWorker=object,
        os=__import__("os"),
        subprocess=__import__("subprocess"),
        ray=fake_ray,
    )


def checkpoint_manager_class(**extra_scope):
    class Parent:
        def __init__(self, config=None, actor_wg=None, replicas=None):
            self.config = config
            self.backend = getattr(config, "backend", "nccl")
            self.actor_wg = actor_wg
            self.replicas = list(replicas or [])
            self.build_calls = []

        def add_replicas(self, replicas):
            self.replicas.extend(replicas)

        def remove_replicas(self, replicas):
            replicas = set(replicas)
            self.replicas = [r for r in self.replicas if r not in replicas]

        def build_process_group(self, rollout):
            self.build_calls.append(rollout)

    scope = {
        "ReplicaKey": ReplicaKey,
        "OperationEvidence": OperationEvidence,
        "EvidenceType": EvidenceType,
        "ReplicaKind": ReplicaKind,
        "asyncio": asyncio,
        "hashlib": __import__("hashlib"),
        "json": __import__("json"),
        "os": __import__("os"),
        "get_device_name": lambda: "cuda",
        **extra_scope,
    }
    return isolated(
        "checkpoint/checkpoint_engine_manager.py",
        "MultiTaskCheckpointEngineManager",
        Parent,
        **scope,
    )


def http_server_class(resource_name="GPU"):
    class Parent:
        async def resume_kv_cache(self):
            await self.engine.wake_up(tags=["kv_cache"])
            await self.engine.reset_prefix_cache(reset_connector=True)

    return isolated(
        "rollout/http_server.py",
        "MultiTaskvLLMHttpServer",
        Parent,
        asyncio=asyncio,
        inspect=__import__("inspect"),
        json=__import__("json"),
        ray=FakeRay,
        get_resource_name=lambda: resource_name,
    )


def runtime_replica_class():
    return isolated(
        "rollout/replica.py",
        "MultiTaskvLLMReplica",
        object,
        asyncio=asyncio,
        ReplicaKind=ReplicaKind,
        Lease=Lease,
        FIRST_RELEASE_MAX_COLOCATE_COUNT=FIRST_RELEASE_MAX_COLOCATE_COUNT,
        MultiTaskCheckpointEngineWorker=object,
        get_resource_name=lambda: "GPU",
    )


def native_manager_class():
    return isolated(
        f"{INTEGRATION}/llm_server_manager.py",
        "MultiTaskLLMServerManager",
        object,
        ReplicaKey=ReplicaKey,
        ReplicaKind=ReplicaKind,
        ReplicaState=ReplicaState,
        EvidenceType=EvidenceType,
        OperationEvidence=OperationEvidence,
        native_replica_key=native_replica_key,
        asyncio=asyncio,
    )


def _isolated_group_scheduler_class():
    path = SOURCE / "scheduler/group_scheduler.py"
    tree = ast.parse(path.read_text())
    scheduler = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GroupScheduler"
    )
    scheduler.decorator_list = []
    module = ast.Module(body=[scheduler], type_ignores=[])
    env = {
        "ReplicaKey": ReplicaKey,
        "ReplicaKind": ReplicaKind,
        "ActorHandle": object,
        "Lease": Lease,
        "OperationCommand": OperationCommand,
        "OperationEvidence": OperationEvidence,
        "OperationKind": OperationKind,
        "OperationRecord": OperationRecord,
        "OperationStatus": OperationStatus,
        "RUNTIME_KIND": "test",
        "EvidenceType": EvidenceType,
        "_RELEASE_KINDS": {OperationKind.DONATE, OperationKind.REMOVE},
        "_IDLE_REPORT_MAX_AGE_S": 10.0,
        "CONTROL_RPC_TIMEOUT_S": CONTROL_RPC_TIMEOUT_S,
        "native_replica_key": native_replica_key,
        "time": time,
        "ray": FakeRay,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env["GroupScheduler"]


def _scheduler_test_lease(lease_id="l1", **claim_overrides):
    """Return a new Lease; overrides isolate the fact under test."""
    claim = {
        "claim_id": f"claim-{lease_id}",
        "source_lease_id": f"source-{lease_id}",
        "donor_task_id": "task-a",
        "donor_replica_rank": 0,
        "pg_id": "pg",
        "bundle_index": 0,
        "node_id": "n0",
        "gpu_uuid": "u0",
        "gpu_fraction": FIRST_RELEASE_RAY_GPU_FRACTION,
        "cpu_request": 1.0,
    }
    claim.update(claim_overrides)
    return Lease(lease_id, (claim,))


def _borrowed_test_claim(**changes):
    """Fresh single-rank physical claim for borrowed worker-plan cases."""
    claim = {
        "claim_id": "claim-0",
        "source_lease_id": "source-lease-0",
        "donor_task_id": "donor-task",
        "donor_replica_rank": 0,
        "rank": 0,
        "pg_id": "pg",
        "bundle_index": 0,
        "node_id": "node",
        "gpu_uuid": "GPU-x",
        "node_rank": 0,
        "local_rank": 0,
        "gpu_fraction": FIRST_RELEASE_RAY_GPU_FRACTION,
        "cpu_request": 1.0,
    }
    claim.update(changes)
    return claim



def rollouter_class():
    class Parent:
        def __init__(self, *args, **kwargs):
            self.paused = True
            self.max_concurrent_samples = 8
            self.concurrent_samples_per_replica = 4

    return isolated(
        f"{INTEGRATION}/rollouter.py",
        "MultiTaskFullyAsyncRollouter",
        Parent,
        ReplicaKey=ReplicaKey,
        ReplicaKind=ReplicaKind,
        ReplicaState=ReplicaState,
        AttemptState=AttemptState,
        OperationRecord=OperationRecord,
        OperationEvidence=OperationEvidence,
        EvidenceType=EvidenceType,
        _require_evidence=_test_require_evidence,
        asyncio=asyncio,
        json=__import__("json"),
        ray=FakeRay,
    )

