"""Opt-in parameter-manifest validation on the simplified CE owner."""

import ast
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "src/multi_task_scheduler/checkpoint/checkpoint_engine_manager.py"


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
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
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
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)
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
