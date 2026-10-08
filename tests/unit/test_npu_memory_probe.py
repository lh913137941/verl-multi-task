"""CPU-only regression tests for direct NPU smoke preflight.

The fast npu-smi path must avoid importing torch_npu or creating any device
tensor; minimal images fall back to a bounded device-query-only subprocess.
"""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/integration/test_native_sleep_npu.py"

NPU_SMI_SAMPLE = """
+------------------------------------+
| NPU Name           | Health | Power(W) HBM-Usage(MB) |
| Chip Bus-Id        | AIcore | Memory-Usage(MB) HBM-Usage(MB) |
+====================================+
| 0 910B3            | OK     | 102.0  40    0 / 0 |
| 0                   | 0000:C1:00.0 | 0  0 / 0  45000 / 65536 |
+====================================+
| 1 910B3            | OK     | 102.0  40    0 / 0 |
| 0                   | 0000:C2:00.0 | 0  0 / 0  3318 / 65536 |
+====================================+
| 2 910B3            | OK     | 100.0  40    0 / 0 |
| 0                   | 0000:C3:00.0 | 0  0 / 0  3100 / 65536 |
+====================================+
"""


def _functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {
        "_parse_npu_smi_memory",
        "_probe_visible_npu_memory",
        "_select_direct_smoke_npu",
    }
    body = [
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name in names
    ]
    assert len(body) == 3
    namespace = {
        "json": json,
        "os": os,
        "re": re,
        "subprocess": subprocess,
        "sys": sys,
        "pytest": pytest,
        "_visible_npu_devices": lambda: "0,1",
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    return namespace


def test_npu_smi_parses_visible_hbm_only():
    api = _functions()
    records = api["_parse_npu_smi_memory"](
        NPU_SMI_SAMPLE,
        ("0", "1"),
    )
    assert len(records) == 2
    assert records[0] == {
        "device": "0",
        "free_bytes": (65536 - 45000) * 1024 ** 2,
        "total_bytes": 65536 * 1024 ** 2,
    }
    assert records[1]["device"] == "1"


def test_probe_selects_best_visible_npu_without_device_initialization(capsys):
    api = _functions()
    calls = []

    def run(args, **kwargs):
        calls.append(list(args))
        assert args == ["npu-smi", "info"]
        return SimpleNamespace(returncode=0, stdout=NPU_SMI_SAMPLE, stderr="")

    api["subprocess"] = SimpleNamespace(
        run=run,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    selected, utilization = api["_select_direct_smoke_npu"](min_free_gib=6)
    assert selected == "1"
    assert 0.03 <= utilization <= 0.40
    assert calls == [["npu-smi", "info"]]
    assert '"source": "npu-smi"' in capsys.readouterr().out


def test_npu_smi_unrecognized_table_uses_torch_fallback_without_tensor():
    api = _functions()
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[0] == "npu-smi":
            return SimpleNamespace(returncode=0, stdout="unrecognized", stderr="")
        assert args == [sys.executable, "-c", args[2]]
        assert "torch.npu.mem_get_info(index)" in args[2]
        assert "torch.tensor(" not in args[2]
        assert kwargs["timeout"] == 40
        return SimpleNamespace(
            returncode=0,
            stdout='NPU_MEMORY_PROBE [{"index": 0, "free_bytes": 20000000000, "total_bytes": 60000000000}]',
            stderr="",
        )

    api["subprocess"] = SimpleNamespace(
        run=run,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    records = api["_probe_visible_npu_memory"]()
    assert len(calls) == 2
    assert records[0]["device"] == "0"


def test_npu_probe_timeout_reports_source_and_worker_stderr():
    api = _functions()

    def run(args, **kwargs):
        if args[0] == "npu-smi":
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        raise subprocess.TimeoutExpired(
            args,
            kwargs["timeout"],
            stderr=b"ACL launch initialization stalled",
        )

    api["subprocess"] = SimpleNamespace(
        run=run,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    with pytest.raises(RuntimeError, match="ACL launch initialization stalled"):
        api["_probe_visible_npu_memory"]()


def test_npu_probe_invalid_memory_never_uses_fake_available_gpu():
    api = _functions()
    api["_probe_visible_npu_memory"] = lambda: (
        {"device": "0", "free_bytes": None, "total_bytes": 0},
    )
    with pytest.raises(RuntimeError, match="did not obtain valid free-memory"):
        api["_select_direct_smoke_npu"](min_free_gib=1.5)


def test_npu_probe_insufficient_free_memory_does_not_start_model():
    api = _functions()
    api["_probe_visible_npu_memory"] = lambda: (
        {"device": "0", "free_bytes": 256 * 1024**2, "total_bytes": 65536 * 1024**2},
    )
    with pytest.raises(pytest.skip.Exception, match="at least 1.5 GiB"):
        api["_select_direct_smoke_npu"](min_free_gib=1.5)
