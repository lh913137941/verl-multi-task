"""CHECKPOINT wiring regression scenarios; shared test fakes live in _wiring_support."""
import ast
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import ast
import asyncio
import pytest
from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    OperationEvidence,
    ReplicaKey,
    ReplicaKind,
)
from _wiring_support import (
    SOURCE,
    isolated,
    FakeRay,
    checkpoint_manager_class,
)


def test_ce_pending_bootstrap_is_not_effective_until_weight_ready():
    ce = checkpoint_manager_class()(replicas=["native"])
    key = ReplicaKey("task-a", "borrowed-0")
    ce.register_pending(key, ["borrowed"], operation_id="op-add")

    assert ce.pending_bootstrap[key] == (("borrowed",), "op-add")
    assert key not in ce.effective_replicas
    assert ce.replicas == ["native"]

    with pytest.raises(ValueError, match="only from WEIGHT_READY"):
        ce.commit_pending(
            key,
            OperationEvidence("op-add", EvidenceType.SERVICE_COMMITTED, 1),
            loaded_version=7,
        )
    assert ce.replicas == ["native"]

    # Model the CE-owned confirmation; real transfer ordering is tested below.
    ce._bootstrap_ready_map[key] = (
        "op-add", 7, OperationEvidence("op-add", EvidenceType.WEIGHT_READY, 2)
    )
    ce.commit_pending(
        key,
        OperationEvidence("op-add", EvidenceType.WEIGHT_READY, 2),
        loaded_version=7,
    )
    assert key not in ce.pending_bootstrap
    assert ce.effective_replicas[key] == (("borrowed",), 7)
    assert ce.replicas == ["native", "borrowed"]

    ce.commit_pending(
        key,
        OperationEvidence("op-add", EvidenceType.WEIGHT_READY, 2),
        loaded_version=7,
    )


def test_ce_pending_bootstrap_rejects_operation_rebind():
    ce = checkpoint_manager_class()(replicas=[])
    key = ReplicaKey("task-a", "borrowed-0")
    ce.register_pending(key, ["borrowed"], operation_id="op-1")

    with pytest.raises(ValueError, match="conflicting pending bootstrap"):
        ce.register_pending(key, ["borrowed"], operation_id="op-2")

    with pytest.raises(ValueError, match="another operation"):
        ce.commit_pending(
            key,
            OperationEvidence("op-2", EvidenceType.WEIGHT_READY, 1),
            loaded_version=3,
        )


def test_ce_target_bootstrap_syncs_only_pending_target_and_is_idempotent():
    calls = []

    class FakeRolloutWG:
        def __init__(self, workers):
            self.workers = workers
            self.world_size = len(workers)

        @classmethod
        def from_detached(cls, *, worker_handles, **kwargs):
            calls.append(("wrap", tuple(worker_handles)))
            return cls(list(worker_handles))

        def update_weights(self, *, global_steps):
            calls.append(("target-update", global_steps))
            return [("target", global_steps)]

        def execute_checkpoint_engine(self, methods):
            calls.append(("target-finalize", tuple(methods)))
            return [("target-finalize", len(methods))]

    class FakeCIA:
        def __init__(self, cls, *args, **kwargs):
            self.cls = cls

    class ActorWG:
        world_size = 1

        def update_weights(self, *, global_steps, mode):
            calls.append(("actor-update", global_steps, mode))
            return [("actor", global_steps)]

        def execute_checkpoint_engine(self, methods):
            calls.append(("actor-finalize", tuple(methods)))
            return [("actor-finalize", len(methods))]

    class FakeRay:
        @staticmethod
        def remote(cls):
            return cls

        @staticmethod
        def get(value):
            calls.append(("ray-get", tuple(value)))
            return list(value)

    class Replica:
        replica_kind = ReplicaKind.BORROWED
        workers = ["worker-0"]

        def get_ray_class_with_init_args(self):
            return object()

        async def release_kv_cache(self):
            calls.append(("release-kv",))

        async def resume_kv_cache(self):
            calls.append(("resume-kv",))

        async def validate_server_runtime(self):
            calls.append(("health",))
            return {"global_steps": 7}

    config = type("Config", (), {"backend": "nccl"})()
    cls = checkpoint_manager_class(
        ray=FakeRay,
        RayClassWithInitArgs=FakeCIA,
        RayWorkerGroup=FakeRolloutWG,
            )
    ce = cls(config=config, actor_wg=ActorWG(), replicas=["native"])
    key = ReplicaKey("task-a", "borrowed-0")
    replica = Replica()
    ce.register_pending(key, [replica], operation_id="op-add")

    first = asyncio.run(
        ce.bootstrap_target(key, operation_id="op-add", loaded_version=7)
    )
    assert first.type is EvidenceType.WEIGHT_READY
    assert key not in ce.effective_replicas
    assert ce.replicas == ["native"]
    assert ce.build_calls and ce.build_calls[0].workers == ["worker-0"]
    assert calls.index(("release-kv",)) < calls.index(("target-update", 7))
    assert calls.index(("target-update", 7)) < calls.index(("resume-kv",))
    assert calls.index(("resume-kv",)) < calls.index(("health",))
    assert ("actor-update", 7, "nccl") in calls
    assert ("target-update", 7) in calls
    assert ("health",) in calls

    before = list(calls)
    second = asyncio.run(
        ce.bootstrap_target(key, operation_id="op-add", loaded_version=7)
    )
    assert second == first
    assert calls == before

    ce.commit_pending(key, first, loaded_version=7)
    assert ce.effective_replicas[key] == ((replica,), 7)
    assert ce.replicas == ["native", replica]

    with pytest.raises(ValueError, match="conflicting target bootstrap replay"):
        asyncio.run(
            ce.bootstrap_target(key, operation_id="op-add", loaded_version=8)
        )


def test_ce_native_restore_wakes_weights_under_bootstrap_and_restores_kv_before_health():
    calls = []

    class FakeRolloutWG:
        world_size = 1

        @classmethod
        def from_detached(cls, *, worker_handles, **kwargs):
            calls.append(("wrap", tuple(worker_handles)))
            return cls()

        def update_weights(self, *, global_steps):
            calls.append(("target-update", global_steps))
            return [None]

        def execute_checkpoint_engine(self, methods):
            calls.append(("target-finalize", tuple(methods)))
            return [None]

    class FakeCIA:
        def __init__(self, cls, *args, **kwargs):
            pass

    class ActorWG:
        world_size = 1

        def update_weights(self, *, global_steps, mode):
            calls.append(("actor-update", global_steps, mode))
            return [None]

        def execute_checkpoint_engine(self, methods):
            calls.append(("actor-finalize", tuple(methods)))
            return [None]

    class FakeRay:
        @staticmethod
        def remote(cls):
            return cls

        @staticmethod
        def get(value):
            return value

    class Replica:
        replica_kind = ReplicaKind.NATIVE
        workers = ["worker-0"]

        def get_ray_class_with_init_args(self):
            return object()

        async def wake_up(self, *, tags=None):
            calls.append(("wake", tuple(tags) if tags is not None else None))
            return ({"sleeping": True, "fully_awake": False},)

        async def release_kv_cache(self):
            calls.append(("release-kv",))

        async def resume_kv_cache(self):
            calls.append(("resume-kv",))

        async def validate_server_runtime(self):
            calls.append(("health",))
            return {"global_steps": 13}

        async def worker_placements(self):
            return ({"node_id": "n0", "gpu_uuid": "u0"},)

        async def sleep(self):
            raise AssertionError("successful RESTORE bootstrap must not rollback")

    config = type("Config", (), {"backend": "nccl"})()
    cls = checkpoint_manager_class(
        ray=FakeRay,
        RayClassWithInitArgs=FakeCIA,
        RayWorkerGroup=FakeRolloutWG,
            )
    ce = cls(config=config, actor_wg=ActorWG(), replicas=[])
    key = ReplicaKey("task-a", "native-0")
    replica = Replica()
    ce.register_pending(key, [replica], operation_id="op-restore")

    evidence = asyncio.run(
        ce.bootstrap_target(key, operation_id="op-restore", loaded_version=13)
    )
    assert evidence.type is EvidenceType.WEIGHT_READY
    assert calls.index(("wake", ("weights",))) < calls.index(("target-update", 13))
    assert calls.index(("target-update", 13)) < calls.index(("resume-kv",))
    assert calls.index(("resume-kv",)) < calls.index(("health",))


def test_ce_native_restore_failure_resleeps_before_escaping():
    calls = []

    class FakeRolloutWG:
        world_size = 1

        @classmethod
        def from_detached(cls, *, worker_handles, **kwargs):
            return cls()

        def update_weights(self, *, global_steps):
            calls.append(("target-update", global_steps))
            raise RuntimeError("target transfer failed")

        def execute_checkpoint_engine(self, methods):
            calls.append(("target-finalize", tuple(methods)))
            return [None]

    class FakeCIA:
        def __init__(self, cls, *args, **kwargs):
            pass

    class ActorWG:
        world_size = 1

        def update_weights(self, *, global_steps, mode):
            calls.append(("actor-update", global_steps, mode))
            return [None]

        def execute_checkpoint_engine(self, methods):
            calls.append(("actor-finalize", tuple(methods)))
            return [None]

    class FakeRay:
        @staticmethod
        def remote(cls):
            return cls

        @staticmethod
        def get(value):
            # Evaluate enough of the fake result list to surface transfer failure.
            return value

    class Replica:
        replica_kind = ReplicaKind.NATIVE
        workers = ["worker-0"]

        def get_ray_class_with_init_args(self):
            return object()

        async def wake_up(self, *, tags=None):
            calls.append(("wake", tuple(tags) if tags is not None else None))
            return ({"sleeping": True, "fully_awake": False},)

        async def release_kv_cache(self):
            calls.append(("release-kv",))

        async def resume_kv_cache(self):
            raise AssertionError("failed transfer must not resume KV")

        async def validate_server_runtime(self):
            raise AssertionError("failed transfer must not reach health")

        async def worker_placements(self):
            return ({"node_id": "n0", "gpu_uuid": "u0"},)

        async def sleep(self):
            calls.append(("sleep",))
            return ({"sleep_level": 2, "sleeping": True},)

    config = type("Config", (), {"backend": "nccl"})()
    cls = checkpoint_manager_class(
        ray=FakeRay,
        RayClassWithInitArgs=FakeCIA,
        RayWorkerGroup=FakeRolloutWG,
            )
    ce = cls(config=config, actor_wg=ActorWG(), replicas=[])
    key = ReplicaKey("task-a", "native-0")
    ce.register_pending(key, [Replica()], operation_id="op-restore")

    evidence = asyncio.run(
        ce.bootstrap_target(key, operation_id="op-restore", loaded_version=13)
    )
    assert evidence.type is EvidenceType.RELEASED
    assert evidence.released_gpu_uuids == ("u0",)
    assert ("wake", ("weights",)) in calls
    assert ("sleep",) in calls


def test_ce_target_bootstrap_requires_server_version_confirmation():
    class FakeRolloutWG:
        world_size = 1

        @classmethod
        def from_detached(cls, *, worker_handles, **kwargs):
            return cls()

        def update_weights(self, *, global_steps):
            return [None]

        def execute_checkpoint_engine(self, methods):
            return [None]

    class FakeCIA:
        def __init__(self, cls, *args, **kwargs):
            pass

    class ActorWG:
        world_size = 1

        def update_weights(self, *, global_steps, mode):
            return [None]

        def execute_checkpoint_engine(self, methods):
            return [None]

    class FakeRay:
        @staticmethod
        def remote(cls):
            return cls

        @staticmethod
        def get(value):
            return value

    class Replica:
        replica_kind = ReplicaKind.BORROWED
        workers = ["worker-0"]

        def get_ray_class_with_init_args(self):
            return object()

        async def release_kv_cache(self):
            return None

        async def resume_kv_cache(self):
            return None

        async def validate_server_runtime(self):
            return {"global_steps": 6}

    config = type("Config", (), {"backend": "nccl"})()
    cls = checkpoint_manager_class(
        ray=FakeRay,
        RayClassWithInitArgs=FakeCIA,
        RayWorkerGroup=FakeRolloutWG,
            )
    ce = cls(config=config, actor_wg=ActorWG(), replicas=[])
    key = ReplicaKey("task-a", "borrowed-0")
    ce.register_pending(key, [Replica()], operation_id="op-add")

    with pytest.raises(RuntimeError, match="published parameter version"):
        asyncio.run(
            ce.bootstrap_target(key, operation_id="op-add", loaded_version=7)
        )
    assert key not in ce._bootstrap_ready_map


def test_ce_rejects_unverified_weight_ready_without_changing_membership():
    ce = checkpoint_manager_class()(replicas=[])
    key = ReplicaKey("task-a", "borrowed-0")
    ce.register_pending(key, ["borrowed"], operation_id="op-add")
    with pytest.raises(ValueError, match="confirmed bootstrap"):
        ce.commit_pending(
            key, OperationEvidence("op-add", EvidenceType.WEIGHT_READY, 2),
            loaded_version=7,
        )
    assert key in ce.pending_bootstrap
    assert ce.effective_replicas == {}
    assert ce.replicas == []


@pytest.mark.parametrize("version", [6, 8, True])
def test_ce_commit_requires_confirmed_bootstrap_version(version):
    ce = checkpoint_manager_class()(replicas=[])
    key = ReplicaKey("task-a", "borrowed-0")
    evidence = OperationEvidence("op-add", EvidenceType.WEIGHT_READY, 2)
    ce.register_pending(key, ["borrowed"], operation_id="op-add")
    ce._bootstrap_ready_map[key] = ("op-add", 7, evidence)
    with pytest.raises(ValueError):
        ce.commit_pending(key, evidence, loaded_version=version)
    assert ce.effective_replicas == {}
    assert ce.replicas == []


def test_ce_remove_invalidates_bootstrap_result_for_reused_key():
    ce = checkpoint_manager_class()(replicas=[])
    key = ReplicaKey("task-a", "borrowed-0")
    evidence = OperationEvidence("op-add", EvidenceType.WEIGHT_READY, 2)
    ce.register_pending(key, ["old-runtime"], operation_id="op-add")
    ce._bootstrap_ready_map[key] = ("op-add", 7, evidence)
    ce.commit_pending(key, evidence, loaded_version=7)
    ce.remove_effective(key)
    ce.register_pending(key, ["new-runtime"], operation_id="op-next")
    assert key not in ce._bootstrap_ready_map
    assert ce.replicas == []


def test_ce_committed_replay_rejects_replaced_evidence():
    ce = checkpoint_manager_class()(replicas=[])
    key = ReplicaKey("task-a", "borrowed-0")
    evidence = OperationEvidence("op-add", EvidenceType.WEIGHT_READY, 2)
    ce.register_pending(key, ["runtime"], operation_id="op-add")
    ce._bootstrap_ready_map[key] = ("op-add", 7, evidence)
    ce.commit_pending(key, evidence, loaded_version=7)
    with pytest.raises(ValueError, match="confirmed bootstrap"):
        ce.commit_pending(
            key, OperationEvidence("op-add", EvidenceType.WEIGHT_READY, 3),
            loaded_version=7,
        )
    assert ce.effective_replicas[key] == (("runtime",), 7)


@pytest.mark.parametrize("stage", ["pending", "effective"])
def test_ce_runtime_cannot_belong_to_two_replica_keys(stage):
    ce = checkpoint_manager_class()(replicas=[])
    first = ReplicaKey("task-a", "first")
    second = ReplicaKey("task-a", "second")
    if stage == "pending":
        ce.register_pending(first, ["runtime"], operation_id="op-first")
    else:
        ce.add_effective(first, ["runtime"], loaded_version=1)
    with pytest.raises(ValueError):
        ce.register_pending(second, ["runtime"], operation_id="op-second")
    with pytest.raises(ValueError, match="another ReplicaKey"):
        ce.add_effective(second, ["runtime"], loaded_version=1)


def test_ce_rejects_duplicate_target_runtimes():
    ce = checkpoint_manager_class()(replicas=[])
    key = ReplicaKey("task-a", "borrowed-0")
    with pytest.raises(ValueError, match="duplicate"):
        ce.register_pending(key, ["runtime", "runtime"], operation_id="op-add")
    with pytest.raises(ValueError, match="duplicate"):
        ce.add_effective(key, ["runtime", "runtime"], loaded_version=1)


def test_ce_failed_native_removal_preserves_membership_for_reconciliation(monkeypatch):
    cls = checkpoint_manager_class()
    ce = cls(replicas=[])
    key = ReplicaKey("task-a", "native-0")
    ce.add_effective(key, ["runtime"], loaded_version=7)
    ready = ("op-original", 7, OperationEvidence(
        "op-original", EvidenceType.WEIGHT_READY, 1
    ))
    ce._bootstrap_ready_map[key] = ready

    native_parent = cls.__mro__[1]
    original_remove = native_parent.remove_replicas

    def failed_remove(_self, _replicas):
        raise RuntimeError("native membership removal failed")

    monkeypatch.setattr(native_parent, "remove_replicas", failed_remove)
    with pytest.raises(RuntimeError, match="native membership removal failed"):
        ce.remove_effective(key)
    assert ce.effective_replicas[key] == (("runtime",), 7)
    assert ce._bootstrap_ready_map[key] == ready
    assert ce.replicas == ["runtime"]

    monkeypatch.setattr(native_parent, "remove_replicas", original_remove)
    ce.remove_effective(key)
    assert key not in ce.effective_replicas
    assert key not in ce._bootstrap_ready_map
    assert ce.replicas == []


def test_task_scoped_checkpoint_adapter_uses_exact_vllm_actor_name():
    """Two concurrent tasks must never route a weight transfer to each other."""
    lookups = []
    registry = {
        "vllm_server_0_0_mt_taskA": object(),
        "vllm_server_0_0_mt_taskB": object(),
    }

    class NativeServerAdapter:
        def __init__(self, *args, **kwargs):
            self._has_server = True
            self.server_handle = None
            self._pd_role = None
            self.replica_rank = kwargs["replica_rank"]
            self.node_rank = 0

        def _get_server_name_prefix(self):
            return "vllm_"

    class ScopedRay:
        @staticmethod
        def get_actor(name):
            lookups.append(name)
            return registry[name]

    cls = isolated(
        "checkpoint/checkpoint_engine_worker.py",
        "_MultiTaskServerAdapter",
        NativeServerAdapter,
        ray=ScopedRay,
    )
    a = cls(replica_rank=0, server_name_suffix="_mt_taskA")
    b = cls(replica_rank=0, server_name_suffix="_mt_taskB")
    assert a._ensure_server_handle() is True
    assert b._ensure_server_handle() is True
    assert a.server_handle is registry["vllm_server_0_0_mt_taskA"]
    assert b.server_handle is registry["vllm_server_0_0_mt_taskB"]
    assert lookups == ["vllm_server_0_0_mt_taskA", "vllm_server_0_0_mt_taskB"]
    # Weight updates reuse the correct cached server; never fall back to
    # unsuffixed "vllm_server_0_0" if the scoped actor is absent.
    assert a._ensure_server_handle() is True
    assert len(lookups) == 2
    with pytest.raises(KeyError, match="vllm_server_1_0_mt_taskA"):
        cls(replica_rank=1, server_name_suffix="_mt_taskA")._ensure_server_handle()
    assert "vllm_server_0_0" not in lookups


def test_task_scoped_checkpoint_adapter_fails_closed_for_unsupported_routes():
    class NativeServerAdapter:
        def __init__(self, *args, **kwargs):
            self._has_server = kwargs.get("has_server", True)
            self.server_handle = None
            self._pd_role = kwargs.get("pd_role")
            self.replica_rank = 0
            self.node_rank = 0

        def _get_server_name_prefix(self):
            return "vllm_"

    class ScopedRay:
        @staticmethod
        def get_actor(name):
            raise AssertionError(f"not allowed to look up {name}")

    cls = isolated(
        "checkpoint/checkpoint_engine_worker.py",
        "_MultiTaskServerAdapter",
        NativeServerAdapter,
        ray=ScopedRay,
    )
    with pytest.raises(ValueError, match="name_suffix"):
        cls(server_name_suffix="")
    assert cls(server_name_suffix="_mt_A", has_server=False)._ensure_server_handle() is False
    with pytest.raises(NotImplementedError, match="PD routing"):
        cls(server_name_suffix="_mt_A", pd_role="decode")._ensure_server_handle()


def test_checkpoint_worker_receives_replica_server_name_suffix():
    """Replica supplies the exact suffix; Worker injects adapter into native CE."""
    from types import SimpleNamespace

    class NativeCheckpointWorker:
        def __init__(self, *args, **kwargs):
            self.received_adapter = kwargs.get("server_adapter")

    class Adapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    cls = isolated(
        "checkpoint/checkpoint_engine_worker.py",
        "MultiTaskCheckpointEngineWorker",
        NativeCheckpointWorker,
        _MultiTaskServerAdapter=Adapter,
        os=__import__("os"),
        register=lambda **kwargs: lambda f: f,
        Dispatch=SimpleNamespace(ONE_TO_ALL=0),
    )
    cfg, model = object(), object()
    worker = cls(
        rollout_config=cfg,
        model_config=model,
        replica_rank=0,
        server_name_suffix="_mt_taskA",
    )
    assert worker.received_adapter.kwargs["server_name_suffix"] == "_mt_taskA"
    assert worker.received_adapter.kwargs["replica_rank"] == 0
    assert worker.received_adapter.kwargs["config"] is cfg
    assert worker.received_adapter.kwargs["model_config"] is model
    with pytest.raises(ValueError, match="explicit task-scoped"):
        cls(
            rollout_config=cfg, model_config=model,
            server_name_suffix="_mt_A", server_adapter=object()
        )
    worker_native = cls(rollout_config=cfg, model_config=model)
    assert worker_native.received_adapter is None

    root = SOURCE / "rollout" / "replica.py"
    tree = ast.parse(root.read_text(encoding="utf-8"))
    replica = next(x for x in tree.body if isinstance(x, ast.ClassDef)
                   and x.name == "MultiTaskvLLMReplica")
    method = next(x for x in replica.body if isinstance(x, ast.FunctionDef)
                  and x.name == "get_ray_class_with_init_args")
    init = next(x for x in ast.walk(method) if isinstance(x, ast.Call)
                and isinstance(x.func, ast.Name) and x.func.id == "RayClassWithInitArgs")
    suffix_arg = next(x.value for x in init.keywords if x.arg == "server_name_suffix")
    assert isinstance(suffix_arg, ast.Attribute)
    assert suffix_arg.attr == "name_suffix"



# --- test_parameter_manifest.py (consolidated boundary scenarios) ---

_MANIFEST_SOURCE = Path(__file__).resolve().parents[2] / "src/multi_task_scheduler/checkpoint/checkpoint_engine_manager.py"


class _Parent:
    def __init__(self, *args, **kwargs):
        self.actor_wg = kwargs.get("actor_wg")
        self.replicas = list(kwargs.get("replicas", []))
        self.backend = getattr(kwargs.get("config"), "backend", "nccl")

    def add_replicas(self, replicas):
        self.replicas.extend(replica for replica in replicas if replica not in self.replicas)

    def remove_replicas(self, replicas):
        self.replicas = [replica for replica in self.replicas if replica not in replicas]


class _Remote:
    def __init__(self, value):
        self.value = value

    def remote(self):
        return self.value


def _manager_class(ray):
    tree = ast.parse(_MANIFEST_SOURCE.read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef))
    node.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    scope = {
        "Parent": _Parent,
        "asyncio": asyncio,
        "hashlib": hashlib,
        "json": json,
        "os": SimpleNamespace(environ={}),
        "ray": ray,
        "ReplicaKey": object,
        "ReplicaKind": object,
        "EvidenceType": object,
        "OperationEvidence": object,
        "RayWorkerGroup": object,
    }
    exec(compile(ast.fix_missing_locations(module), str(_MANIFEST_SOURCE), "exec"), scope)
    return scope["MultiTaskCheckpointEngineManager"]


def _manifest(value="abc", version=3):
    return {
        "complete": True,
        "global_steps": version,
        "parameters": [
            {
                "name": "weight",
                "shape": [2],
                "dtype": "torch.float32",
                "numel": 2,
                "sha256": value,
            }
        ],
        "parameter_count": 1,
        "total_numel": 2,
    }


def test_parameter_manifests_match_across_receivers_and_source():
    source = _manifest()
    workers = [
        SimpleNamespace(get_parameter_manifest=_Remote(_manifest())),
        SimpleNamespace(get_parameter_manifest=_Remote(_manifest())),
    ]
    ray = SimpleNamespace(get=lambda refs: refs)
    manager = _manager_class(ray)(
        config=SimpleNamespace(backend="nccl"),
        actor_wg=SimpleNamespace(),
        replicas=[],
    )
    replicas = [SimpleNamespace(workers=workers)]

    result = asyncio.run(
        manager.validate_parameter_sync(
            replicas,
            expected_version=3,
            source_manifest=source,
        )
    )
    assert result["state"] == "PARAMETERS_VALIDATED"
    assert result["source_state"] == "SOURCE_TO_RECEIVER_VALIDATED"
    assert result["worker_count"] == 2
    assert result["source_manifest_digest"] == result["manifest_digest"]


def test_parameter_source_mismatch_fails_closed():
    source = _manifest()
    worker = SimpleNamespace(
        get_parameter_manifest=_Remote(_manifest(value="different"))
    )
    ray = SimpleNamespace(get=lambda refs: refs)
    manager = _manager_class(ray)(
        config=SimpleNamespace(backend="nccl"),
        actor_wg=SimpleNamespace(),
        replicas=[],
    )

    with pytest.raises(RuntimeError, match="differs from actor source manifest"):
        asyncio.run(
            manager.validate_parameter_sync(
                [SimpleNamespace(workers=[worker])],
                expected_version=3,
                source_manifest=source,
            )
        )


@pytest.mark.parametrize("case", ["empty", "duplicate", "count", "numel", "missing_hash"])
def test_invalid_manifest_cannot_prove_parameter_sync(case):
    manifest = _manifest()
    if case == "empty":
        manifest.update(parameters=[], parameter_count=0, total_numel=0)
    elif case == "duplicate":
        manifest["parameters"] *= 2
        manifest.update(parameter_count=2, total_numel=4)
    elif case == "count":
        manifest["parameter_count"] = 2
    elif case == "numel":
        manifest["total_numel"] = 100
    else:
        del manifest["parameters"][0]["sha256"]
    worker = SimpleNamespace(get_parameter_manifest=_Remote(manifest))
    manager = _manager_class(SimpleNamespace(get=lambda refs: refs))()
    with pytest.raises(RuntimeError, match="manifest"):
        asyncio.run(manager.validate_parameter_sync(
            [SimpleNamespace(workers=[worker])], expected_version=3,
            source_manifest=manifest,
        ))


def test_validation_must_cover_every_requested_replica():
    worker = SimpleNamespace(get_parameter_manifest=_Remote(_manifest()))
    manager = _manager_class(SimpleNamespace(get=lambda refs: refs))()
    with pytest.raises(RuntimeError, match="CE Worker"):
        asyncio.run(manager.validate_parameter_sync(
            [SimpleNamespace(workers=[worker]), SimpleNamespace(workers=[])],
            expected_version=3,
        ))


def test_valid_receivers_do_not_hide_invalid_source_manifest():
    worker = SimpleNamespace(get_parameter_manifest=_Remote(_manifest()))
    manager = _manager_class(SimpleNamespace(get=lambda refs: refs))()
    source = _manifest()
    source["parameter_count"] = 2
    with pytest.raises(RuntimeError, match="manifest counts"):
        asyncio.run(manager.validate_parameter_sync(
            [SimpleNamespace(workers=[worker])], expected_version=3,
            source_manifest=source,
        ))
