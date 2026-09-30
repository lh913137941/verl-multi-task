"""Audit lifetime checks with CPU substitutes; no Ray/tensor transport validation."""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def worker(monkeypatch):
    async def ensure_async_iterator(values):
        async for value in values:
            yield value

    modules = {
        "torch": {"Tensor": object},
        "verl.checkpoint_engine.base": {"CheckpointEngineWorker": object},
        "verl.single_controller.base.decorator": {
            "Dispatch": SimpleNamespace(ONE_TO_ALL=object()),
            "register": lambda **kw: lambda fn: fn,
        },
        "verl.workers.rollout.utils": {"ensure_async_iterator": ensure_async_iterator},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    path = ROOT / "src/multi_task_scheduler/checkpoint/checkpoint_engine_worker.py"
    spec = importlib.util.spec_from_file_location("audit_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.MultiTaskCheckpointEngineWorker()
    result.parameter_validation_enabled = True
    result._tensor_sha256 = lambda tensor: "abc"
    result._last_parameter_manifest = {"complete": True, "global_steps": 3}
    return result


@pytest.mark.parametrize("outcome", ["cancelled", "partial", "empty", "load_failed", "complete"])
def test_receiver_manifest_describes_current_fully_consumed_transfer(worker, outcome):
    async def receive_weights(**kwargs):
        if outcome != "empty":
            for index in range(2):
                yield f"weight.{index}", SimpleNamespace(shape=(2,), dtype="float32", numel=lambda: 2)

    async def update_weights(weights, **kwargs):
        if outcome == "cancelled":
            raise asyncio.CancelledError()
        if outcome == "partial":
            await anext(weights)
            return
        async for _ in weights:
            pass
        if outcome == "load_failed":
            raise RuntimeError("adapter load failed after consuming weights")

    worker.checkpoint_engine = SimpleNamespace(receive_weights=receive_weights)
    worker.server_adapter = SimpleNamespace(update_weights=update_weights)
    if outcome == "complete":
        asyncio.run(worker.update_weights(global_steps=4))
    else:
        error = asyncio.CancelledError if outcome == "cancelled" else RuntimeError
        with pytest.raises(error):
            asyncio.run(worker.update_weights(global_steps=4))
    manifest = worker.get_parameter_manifest()
    assert manifest["global_steps"] == 4
    assert manifest["complete"] is (outcome == "complete")
