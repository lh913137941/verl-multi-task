"""Regression checks for compatible verl_expansion pieces."""

import ast
from pathlib import Path

import pytest

from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class

ROOT = Path(__file__).resolve().parents[2]


def test_unwrap_native_actor_class_requires_ray_actor_metadata():
    class Native:
        pass

    class ActorClass:
        __ray_actor_class__ = Native

    assert unwrap_native_actor_class(ActorClass) is Native
    with pytest.raises(TypeError, match="Ray ActorClass"):
        unwrap_native_actor_class(object())


def test_replica_selects_multitask_checkpoint_worker_without_importing_verl():
    worker_path = ROOT / "src/multi_task_scheduler/checkpoint/checkpoint_engine_worker.py"
    worker_tree = ast.parse(worker_path.read_text(encoding="utf-8"))
    worker = next(
        node for node in worker_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MultiTaskCheckpointEngineWorker"
    )
    assert any(isinstance(base, ast.Name) and base.id == "CheckpointEngineWorker" for base in worker.bases)
    worker_methods = {
        node.name for node in worker.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"update_weights", "get_parameter_manifest"} <= worker_methods

    replica_path = ROOT / "src/multi_task_scheduler/rollout/replica.py"
    replica_tree = ast.parse(replica_path.read_text(encoding="utf-8"))
    replica_cls = next(
        node for node in replica_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MultiTaskvLLMReplica"
    )
    method = next(
        node for node in replica_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_ray_class_with_init_args"
    )
    names = {node.id for node in ast.walk(method) if isinstance(node, ast.Name)}
    assert {"MultiTaskCheckpointEngineWorker", "RayClassWithInitArgs"} <= names
