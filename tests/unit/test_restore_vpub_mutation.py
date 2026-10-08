"""CPU-only guard for RESTORE's real FSDP Vpub mutation preflight."""

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

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
    scope = {"torch": torch}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])), str(SOURCE), "exec"),
        scope,
    )
    return scope[helper.name]


def _engine(*, tied, head, embeddings, exported_input=None, include_input=True):
    module = SimpleNamespace(
        get_output_embeddings=lambda: SimpleNamespace(weight=head),
        get_input_embeddings=lambda: SimpleNamespace(weight=embeddings),
    )
    params = [("lm_head.weight", head)]
    if include_input:
        params.insert(
            0,
            ("model.embed_tokens.weight", embeddings if exported_input is None else exported_input),
        )
    return SimpleNamespace(
        module=SimpleNamespace(_fsdp_wrapped_module=module),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(tie_word_embeddings=tied)),
        get_per_tensor_param=lambda: (iter(params), None),
    )


def test_tied_qwen_output_and_canonical_input_are_both_mutated(monkeypatch):
    helper = _acceptance_helper(monkeypatch)
    head, input_weight = Weight(7), Weight(9)
    result = helper(_engine(tied=True, head=head, embeddings=input_weight))
    assert result["tied"] is True
    assert all(value == 0 for value in result["export_nonzero"].values())
    assert head.nonzero == input_weight.nonzero == 0


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
