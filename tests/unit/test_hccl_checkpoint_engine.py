"""HCCL finalize regression tests with explicit CPU-only dependency substitutes.

These exercise the actual plugin module but do not load Ray, torch_npu or CANN.
Real communicator destruction/recreation must be validated by D3_test.sh.
"""

import importlib.util
import asyncio
import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "src/multi_task_scheduler/checkpoint/hccl_checkpoint_engine.py"


@pytest.fixture
def backend(monkeypatch):
    registry = {"nccl": object()}

    def register(name):
        def decorator(cls):
            registry[name] = cls
            return cls
        return decorator

    class NativeHCCL:
        def prepare(self):
            pass

        def init_process_group(self):
            pass

        async def send_weights(self, *args, **kwargs):
            pass

        def receive_weights(self):
            pass

    npu = SimpleNamespace(
        device=Mock(side_effect=lambda device: nullcontext()),
        synchronize=Mock(),
        empty_cache=Mock(),
    )
    module_attributes = {
        "torch": {"npu": npu, "Tensor": object, "uint8": object},
        "verl": {"__path__": []},
        "verl.checkpoint_engine": {"__path__": []},
        "verl.checkpoint_engine.base": {"CheckpointEngineRegistry": SimpleNamespace(register=register)},
        "verl.checkpoint_engine.hccl_checkpoint_engine": {"HCCLCheckpointEngine": NativeHCCL},
        "verl.workers": {"__path__": []},
        "verl.workers.rollout": {"__path__": []},
        "verl.workers.rollout.utils": {
            "ensure_async_iterator": None,
        },
    }
    async def ensure_async_iterator(iterable):
        if hasattr(iterable, "__aiter__"):
            async for item in iterable:
                yield item
        else:
            for item in iterable:
                yield item

    module_attributes["verl.workers.rollout.utils"]["ensure_async_iterator"] = ensure_async_iterator
    for name, attrs in module_attributes.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("test_hccl_plugin", SOURCE)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    engine = plugin.MultiTaskHCCLCheckpointEngine()
    engine.rebuild_group = True
    engine.rank = 1
    engine.world_size = 3
    engine.device = 2
    engine.send_buf, engine.recv_buf = object(), object()
    communicator = SimpleNamespace(comm=object(), hccl=SimpleNamespace(hcclCommDestroy=Mock()))
    engine.pyhccl = communicator
    return engine, communicator, npu, registry, NativeHCCL


def test_current_ascend_api_destroys_once_and_releases_buffers(backend):
    engine, communicator, npu, _, _ = backend
    events = Mock()
    events.attach_mock(npu.synchronize, "synchronize")
    events.attach_mock(communicator.hccl.hcclCommDestroy, "destroy")
    events.attach_mock(npu.empty_cache, "empty_cache")

    # Reproduces the server API: communicator has no destroyComm method.
    engine.finalize()
    engine.finalize()

    communicator.hccl.hcclCommDestroy.assert_called_once_with(communicator.comm)
    npu.device.assert_called_once_with(2)
    assert [call[0] for call in events.mock_calls] == ["synchronize", "destroy", "empty_cache", "empty_cache"]
    assert engine.pyhccl is None
    assert engine.rank is None and engine.world_size is None
    assert engine.send_buf is None and engine.recv_buf is None


def test_current_vllm_ascend_close_is_preferred_over_raw_library_destroy(backend):
    engine, communicator, npu, _, _ = backend
    communicator.close = Mock()
    engine.finalize()
    engine.finalize()
    communicator.close.assert_called_once_with()
    communicator.hccl.hcclCommDestroy.assert_not_called()
    npu.synchronize.assert_called_once()
    assert engine.pyhccl is None


def test_missing_destroy_method_fails_closed_without_releasing_buffers(backend):
    engine, communicator, npu, _, _ = backend
    communicator.hccl = object()  # no close/destroyComm or hcclCommDestroy
    original_buffers = (engine.send_buf, engine.recv_buf)
    with pytest.raises(RuntimeError, match="no supported HCCL destruction API"):
        engine.finalize()
    assert engine.pyhccl is communicator
    assert (engine.send_buf, engine.recv_buf) == original_buffers
    npu.empty_cache.assert_not_called()


def test_legacy_communicator_api_remains_supported(backend):
    engine, communicator, _, _, _ = backend
    communicator.destroyComm = Mock()
    engine.finalize()
    communicator.destroyComm.assert_called_once_with(communicator.comm)
    communicator.hccl.hcclCommDestroy.assert_not_called()


def test_nonparticipating_training_rank_needs_no_destroy(backend):
    engine, _, npu, _, _ = backend
    engine.rank = -1
    engine.pyhccl = None
    engine.finalize()
    npu.synchronize.assert_not_called()
    assert engine.rank is None and engine.world_size is None
    assert engine.send_buf is None and engine.recv_buf is None


def test_rebuild_disabled_preserves_communicator(backend):
    engine, communicator, npu, _, _ = backend
    engine.rebuild_group = False
    engine.finalize()
    assert engine.pyhccl is communicator
    assert engine.rank == 1 and engine.world_size == 3
    communicator.hccl.hcclCommDestroy.assert_not_called()
    npu.synchronize.assert_not_called()
    assert engine.send_buf is None and engine.recv_buf is None


def test_destroy_failure_is_propagated_without_losing_handle(backend):
    engine, communicator, npu, _, _ = backend
    communicator.hccl.hcclCommDestroy.side_effect = RuntimeError("HCCL destroy failed")
    buffers = engine.send_buf, engine.recv_buf
    with pytest.raises(RuntimeError, match="HCCL destroy failed"):
        engine.finalize()
    assert engine.pyhccl is communicator
    assert engine.rank == 1 and engine.world_size == 3
    assert (engine.send_buf, engine.recv_buf) == buffers
    npu.empty_cache.assert_not_called()


def test_plugin_registers_separate_backend_and_inherits_transfer(backend):
    engine, _, _, registry, native = backend
    assert registry["multitask_hccl"] is type(engine)
    assert registry["nccl"] is not type(engine)
    for method in ("prepare", "init_process_group", "receive_weights"):
        assert getattr(type(engine), method) is getattr(native, method)
    assert getattr(type(engine), "send_weights") is not getattr(native, "send_weights")


@pytest.mark.parametrize("outcome", ["cancelled", "partial", "empty", "send_failed", "complete"])
def test_source_manifest_cannot_reuse_previous_transfer(backend, monkeypatch, outcome):
    engine, _, _, _, native = backend
    engine.rank = 0
    engine.source_validation_enabled = True
    engine._source_manifest = {"complete": True, "global_steps": 3}
    engine._tensor_sha256 = lambda tensor: "abc"
    weights = [] if outcome == "empty" else [
        ("weight", SimpleNamespace(shape=(2,), dtype="float32", numel=lambda: 2))
    ]

    async def send_weights(self, stream, **kwargs):
        if outcome == "cancelled":
            raise asyncio.CancelledError()
        if outcome == "partial":
            await anext(stream)
            return
        async for _ in stream:
            pass
        if outcome == "send_failed":
            raise RuntimeError("send failed after consuming weights")

    monkeypatch.setattr(native, "send_weights", send_weights)
    if outcome == "complete":
        asyncio.run(engine.send_weights(weights, global_steps=4))
    else:
        error = asyncio.CancelledError if outcome == "cancelled" else RuntimeError
        with pytest.raises(error):
            asyncio.run(engine.send_weights(weights, global_steps=4))
    manifest = engine.get_source_manifest()
    assert manifest["global_steps"] == 4
    assert manifest["complete"] is (outcome == "complete")


def test_non_source_rank_clears_previous_canonical_manifest(backend, monkeypatch):
    engine, _, _, _, native = backend
    engine.rank = -1
    engine.source_validation_enabled = True
    engine._source_manifest = {"complete": True, "global_steps": 3}
    weights = [("weight", object())]
    consumed = []

    async def send_weights(self, stream, **kwargs):
        # The native non-source path consumes a synchronous iterator.
        consumed.extend(stream)

    monkeypatch.setattr(native, "send_weights", send_weights)
    asyncio.run(engine.send_weights(weights, global_steps=4))
    assert consumed == weights
    assert engine.get_source_manifest()["global_steps"] == 4
    assert engine.get_source_manifest()["complete"] is False



def test_real_restore_acceptance_loads_opt_in_backend_on_both_sides():
    """Guard against the integration test silently selecting native nccl."""
    source = (
        Path(__file__).resolve().parents[2]
        / "tests/integration/npu/test_native_sleep_npu.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    restore = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_real_npu_restore_reinstalls_current_vpub_and_generates_again"
    )
    config_call = next(
        node for node in ast.walk(restore)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "CheckpointEngineConfig"
    )
    fields = {key.arg: ast.literal_eval(key.value) for key in config_call.keywords}
    assert fields["backend"] == "multitask_hccl"
    assert fields["custom_backend_module"] == (
        "multi_task_scheduler.checkpoint.hccl_checkpoint_engine"
    )
    assert fields["engine_kwargs"]["multitask_hccl"]["rebuild_group"] is True

    # The custom Ray actor must be an importable class in the installed
    # package, not an in-test nested class that cloudpickle might fail to
    # deserialize (producing an unrelated TemporaryActor async-flag error).
    sender_source = (
        Path(__file__).resolve().parents[2]
        / "src/multi_task_scheduler/testing/npu_restore_sender.py"
    )
    sender_tree = ast.parse(sender_source.read_text(encoding="utf-8"))
    sender = next(
        node for node in sender_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "RestoreTrainingWorker"
    )
    sender_loader = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_training_sender_class"
    )
    loader_imports = [
        node for node in ast.walk(sender_loader)
        if isinstance(node, ast.ImportFrom)
    ]
    assert any(
        node.module == "multi_task_scheduler.testing.npu_restore_sender"
        and any(alias.name == "RestoreTrainingWorker" for alias in node.names)
        for node in loader_imports
    )
    assert not any(isinstance(node, ast.ClassDef) for node in ast.walk(sender_loader))
    setup_calls = [node for node in ast.walk(sender) if isinstance(node, ast.Call)]
    import_calls = [
        node for node in setup_calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "import_external_libs"
    ]
    registry_calls = [
        node for node in setup_calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "CheckpointEngineRegistry"
        and node.func.attr == "new"
    ]
    assert len(import_calls) == len(registry_calls) == 1
    assert import_calls[0].lineno < registry_calls[0].lineno
