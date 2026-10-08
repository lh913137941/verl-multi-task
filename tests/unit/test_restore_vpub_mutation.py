"""CPU-only guard for RESTORE's live-FSDP1 Vpub mutation preflight.

The fake FSDP presents temporary unflattened Parameter views while the
summon context is active. They commit to actual flat storage only when
writeback=True; a stale HF embedding accessor must not be mistaken for Vpub.
"""

import ast
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/multi_task_scheduler/testing/npu_restore_sender.py"
)


class Weight:
    def __init__(self, nonzero=1):
        self.nonzero = nonzero

    def zero_(self):
        self.nonzero = 0


class FakeFSDP:
    def __init__(self, live):
        self.live = live
        self.unflattened = None
        self.summon_calls = []

    @staticmethod
    @contextmanager
    def summon_full_params(module, *, recurse, writeback):
        module.summon_calls.append((recurse, writeback))
        module.unflattened = {
            name: Weight(weight.nonzero) for name, weight in module.live.items()
        }
        try:
            yield
        finally:
            if writeback:
                for name, weight in module.unflattened.items():
                    module.live[name].nonzero = weight.nonzero
            module.unflattened = None

    def named_parameters(self):
        if self.unflattened is None:
            raise AssertionError("live params must be accessed under FSDP summon")
        return list(self.unflattened.items())


def _acceptance_helper(monkeypatch):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "zero_and_verify_restore_output_weights"
    )
    torch = SimpleNamespace(
        no_grad=lambda: nullcontext(),
        count_nonzero=lambda tensor: SimpleNamespace(item=lambda: tensor.nonzero),
    )
    fsdp_module = ModuleType("torch.distributed.fsdp")
    fsdp_module.FullyShardedDataParallel = FakeFSDP
    dist_module = ModuleType("torch.distributed")
    dist_module.__path__ = []
    dist_module.fsdp = fsdp_module
    monkeypatch.setitem(sys.modules, "torch.distributed", dist_module)
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", fsdp_module)
    scope = {"torch": torch}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])),
            str(SOURCE),
            "exec",
        ),
        scope,
    )
    return scope[helper.name]


def _engine(*, tied, head, embeddings, exported_input=None, include_input=True, live_input=True):
    live = {"lm_head.weight": head}
    if live_input:
        live["model.embed_tokens.weight"] = embeddings
    module = FakeFSDP(live)
    # This stale accessor is intentionally NOT the authoritative FSDP storage.
    module._fsdp_wrapped_module = SimpleNamespace(
        get_output_embeddings=lambda: SimpleNamespace(weight=Weight(99)),
        get_input_embeddings=lambda: SimpleNamespace(weight=Weight(98)),
    )

    def export():
        params = [("lm_head.weight", head)]
        if include_input:
            params.insert(
                0,
                (
                    "model.embed_tokens.weight",
                    embeddings if exported_input is None else exported_input,
                ),
            )
        return iter(params), None

    return SimpleNamespace(
        module=module,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(tie_word_embeddings=tied)),
        get_per_tensor_param=export,
    )


def test_tied_qwen_mutation_writes_back_from_unflattened_parameters(monkeypatch):
    helper = _acceptance_helper(monkeypatch)
    head, input_weight = Weight(7), Weight(9)
    engine = _engine(tied=True, head=head, embeddings=input_weight)
    stale = engine.module._fsdp_wrapped_module
    result = helper(engine)
    assert result["tied"] is True
    assert result["modified_parameters"] == [
        "lm_head.weight",
        "model.embed_tokens.weight",
    ]
    assert all(value == 0 for value in result["export_nonzero"].values())
    assert head.nonzero == input_weight.nonzero == 0
    assert engine.module.summon_calls == [(True, True)]
    # Previous implementation mutated these obsolete accessor objects.
    assert stale.get_output_embeddings().weight.nonzero == 99
    assert stale.get_input_embeddings().weight.nonzero == 98


def test_untied_model_does_not_change_input_embeddings(monkeypatch):
    helper = _acceptance_helper(monkeypatch)
    head, input_weight = Weight(7), Weight(9)
    result = helper(_engine(tied=False, head=head, embeddings=input_weight))
    assert result["tied"] is False
    assert result["export_nonzero"] == {"lm_head.weight": 0}
    assert head.nonzero == 0
    assert input_weight.nonzero == 9


def test_stale_fsdp_export_is_rejected_before_collective(monkeypatch):
    helper = _acceptance_helper(monkeypatch)
    head, input_weight = Weight(7), Weight(9)
    stale_export = Weight(4)
    with pytest.raises(RuntimeError, match="did not reach the actual Vpub export"):
        helper(_engine(tied=True, head=head, embeddings=input_weight, exported_input=stale_export))


def test_tied_input_missing_from_export_is_rejected(monkeypatch):
    helper = _acceptance_helper(monkeypatch)
    with pytest.raises(RuntimeError, match="omitted canonical input embeddings"):
        helper(_engine(tied=True, head=Weight(4), embeddings=Weight(4), include_input=False))


def test_tied_input_missing_from_live_fsdp_is_rejected(monkeypatch):
    helper = _acceptance_helper(monkeypatch)
    with pytest.raises(RuntimeError, match="lacks live input embedding"):
        helper(_engine(tied=True, head=Weight(4), embeddings=Weight(4), live_input=False))


def test_non_fsdp_source_is_not_mutated_by_accessors(monkeypatch):
    helper = _acceptance_helper(monkeypatch)
    head = Weight(7)
    engine = SimpleNamespace(
        module=SimpleNamespace(
            get_output_embeddings=lambda: SimpleNamespace(weight=head)
        )
    )
    with pytest.raises(NotImplementedError, match="requires the FSDP1"):
        helper(engine)
    assert head.nonzero == 7
