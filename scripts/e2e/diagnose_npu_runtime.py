#!/usr/bin/env python3
"""Diagnose the real Ascend runtime before MultiTask NPU acceptance."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


MODEL_ENV = "VERL_MULTITASK_NPU_MODEL_PATH"
RAY_TMPDIR_ENV = "VERL_MULTITASK_RAY_TMPDIR"


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def git_info(module_file: str | None) -> tuple[str | None, str | None, str | None]:
    if not module_file:
        return None, None, None
    path = Path(module_file).resolve()
    for parent in (path.parent, *path.parents):
        if (parent / ".git").exists():
            try:
                head = subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    text=True,
                    timeout=5,
                ).strip()
                branch = subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "--abbrev-ref", "HEAD"],
                    text=True,
                    timeout=5,
                ).strip()
                status = subprocess.check_output(
                    ["git", "-C", str(parent), "status", "--short"],
                    text=True,
                    timeout=5,
                ).strip()
                return head, branch, status or None
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                return None, None, None
    return None, None, None


def verl_expected_pair() -> tuple[str, str] | None:
    try:
        import verl
    except Exception:
        return None
    repo_root = Path(verl.__file__).resolve().parent.parent
    script = repo_root / "scripts" / "install_vllm_mcore_npu.sh"
    if not script.exists():
        return None
    try:
        text = script.read_text(encoding="utf-8")
    except OSError:
        return None
    vllm_version = None
    ascend_version = None
    for line in text.splitlines():
        if "git clone" not in line:
            continue
        if "vllm-ascend.git" in line:
            match = re.search(
                r"(?:-b|--branch)\s+releases/v(?P<version>\d+\.\d+\.\d+)",
                line,
            )
            if match:
                ascend_version = match.group("version")
        elif "vllm.git" in line:
            match = re.search(
                r"(?:-b|--branch)\s+v(?P<version>\d+\.\d+\.\d+)",
                line,
            )
            if match:
                vllm_version = match.group("version")
    if not vllm_version or not ascend_version:
        return None
    return vllm_version, ascend_version


def verl_expected_transformers() -> str | None:
    """Read the final transformers pin from the installed VERL NPU installer."""
    try:
        import verl
    except Exception:
        return None
    repo_root = Path(verl.__file__).resolve().parent.parent
    script = repo_root / "scripts" / "install_vllm_mcore_npu.sh"
    if not script.exists():
        return None
    try:
        text = script.read_text(encoding="utf-8")
    except OSError:
        return None
    matches = re.findall(r"transformers==(?P<version>\d+\.\d+\.\d+)", text)
    return matches[-1] if matches else None


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"__parse_error__": repr(exc)}
    return value if isinstance(value, dict) else {"__value__": value}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get(MODEL_ENV),
        help="local model path; defaults to $" + MODEL_ENV,
    )
    parser.add_argument(
        "--ray-tmpdir",
        default=os.environ.get(RAY_TMPDIR_ENV, "/tmp"),
        help="Ray temp filesystem to inspect; defaults to $" + RAY_TMPDIR_ENV + " or /tmp",
    )
    args = parser.parse_args()

    failures: list[str] = []

    print("=== Python / package versions ===")
    print("python:", sys.version.replace("\n", " "))
    for name in (
        "torch",
        "torch-npu",
        "vllm",
        "vllm-ascend",
        "triton-ascend",
        "transformers",
        "ray",
    ):
        print(f"{name}:", package_version(name))

    print("\n=== VERL runtime contract ===")
    expected = verl_expected_pair()
    print("expected vllm/vllm-ascend:", expected)
    expected_transformers = verl_expected_transformers()
    print("expected transformers:", expected_transformers)

    try:
        import vllm

        vllm_path = str(Path(vllm.__file__).resolve())
        vllm_head, vllm_branch, vllm_status = git_info(vllm.__file__)
    except Exception as exc:
        vllm_path = f"import-error: {exc!r}"
        vllm_head = vllm_branch = vllm_status = None
        failures.append("vllm import failed")
    print("vllm path:", vllm_path)
    print("vllm git head:", vllm_head)
    print("vllm git branch:", vllm_branch)
    print("vllm git dirty:", vllm_status or False)

    try:
        import vllm_ascend

        ascend_path = str(Path(vllm_ascend.__file__).resolve())
        ascend_head, ascend_branch, ascend_status = git_info(vllm_ascend.__file__)
    except Exception as exc:
        ascend_path = f"import-error: {exc!r}"
        ascend_head = ascend_branch = ascend_status = None
        failures.append("vllm_ascend import failed")
    print("vllm-ascend path:", ascend_path)
    print("vllm-ascend git head:", ascend_head)
    print("vllm-ascend git branch:", ascend_branch)
    print("vllm-ascend git dirty:", ascend_status or False)

    if expected is not None:
        from packaging.version import InvalidVersion, Version

        expected_vllm, expected_ascend = expected
        actual_vllm = package_version("vllm")
        actual_ascend = package_version("vllm-ascend")
        try:
            actual_vllm_version = Version(actual_vllm)
            actual_ascend_version = Version(actual_ascend)
            expected_vllm_version = Version(expected_vllm)
            expected_ascend_version = Version(expected_ascend)
        except InvalidVersion:
            actual_vllm_version = actual_ascend_version = None
            expected_vllm_version = expected_ascend_version = None

        if (
            actual_vllm_version is not None
            and actual_vllm_version.release[:3] != expected_vllm_version.release[:3]
        ):
            failures.append(
                f"VERL expects vllm {expected_vllm}, metadata reports {actual_vllm}"
            )
        if (
            actual_ascend_version is not None
            and actual_ascend_version.release[:2] != expected_ascend_version.release[:2]
        ):
            failures.append(
                f"VERL expects vllm-ascend {expected_ascend} lane, "
                f"metadata reports {actual_ascend}"
            )

    if expected_transformers is not None:
        actual_transformers = package_version("transformers")
        if actual_transformers != expected_transformers:
            failures.append(
                f"VERL NPU installer expects transformers {expected_transformers}, "
                f"metadata reports {actual_transformers}"
            )

    print("\n=== NPU platform ===")
    transfer_before = "torch_npu.contrib.transfer_to_npu" in sys.modules
    print("transfer_to_npu loaded before torch_npu import:", transfer_before)
    try:
        import torch
        import torch_npu  # noqa: F401

        print("torch.npu available:", bool(torch.npu.is_available()))
        print("torch.npu count:", int(torch.npu.device_count()))
        print(
            "transfer_to_npu loaded after torch_npu import:",
            "torch_npu.contrib.transfer_to_npu" in sys.modules,
        )
        if not torch.npu.is_available():
            failures.append("torch.npu is unavailable")
    except Exception as exc:
        print("NPU import/probe error:", repr(exc))
        failures.append("torch_npu probe failed")

    print("VERL_PLATFORM:", os.environ.get("VERL_PLATFORM"))
    print("ASCEND_RT_VISIBLE_DEVICES:", os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
    try:
        from verl.utils.device import get_device_name, get_resource_name

        print("VERL device:", get_device_name())
        print("Ray resource:", get_resource_name())
    except Exception as exc:
        print("VERL device probe error:", repr(exc))

    print("\n=== Ray temp filesystem ===")
    ray_tmp = Path(args.ray_tmpdir).expanduser().resolve()
    try:
        ray_tmp.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(ray_tmp)
        free_gib = usage.free / (1024**3)
        free_pct = (usage.free / usage.total * 100) if usage.total else 0.0
        print("path:", ray_tmp)
        print(f"free: {free_gib:.2f} GiB ({free_pct:.2f}%)")
        if free_gib < 4 or free_pct < 5:
            failures.append(
                f"Ray temp filesystem needs >=4 GiB and >=5% free; "
                f"got {free_gib:.2f} GiB / {free_pct:.2f}%"
            )
    except OSError as exc:
        print("disk probe error:", repr(exc))
        failures.append("Ray temp filesystem probe failed")

    print("\n=== Model preflight ===")
    if not args.model:
        print("model: not set (use --model or $" + MODEL_ENV + ")")
        failures.append("model path is not configured")
    else:
        model = Path(args.model).expanduser().resolve()
        print("model:", model)
        if not model.exists():
            failures.append(f"model path does not exist: {model}")
        else:
            config = read_json(model / "config.json") or {}
            quant_desc = read_json(model / "quant_model_description.json")
            print("model_type:", config.get("model_type"))
            print("architectures:", config.get("architectures"))
            print("torch_dtype:", config.get("torch_dtype"))
            print("quantization_config:", config.get("quantization_config"))
            print(
                "quant_model_description:",
                "present" if quant_desc is not None else "absent",
            )
            if quant_desc is not None:
                print("kv_cache_type:", quant_desc.get("kv_cache_type"))
            try:
                from vllm_ascend.quantization.utils import detect_quantization_method

                detected = detect_quantization_method(str(model))
                print("vllm-ascend detected quantization:", detected)
                if detected is not None:
                    failures.append(
                        "lifecycle acceptance requires a plain BF16/FP16 model; "
                        f"backend detected quantization={detected!r}"
                    )
            except Exception as exc:
                print("quantization detector error:", repr(exc))

    print("\n=== Result ===")
    if failures:
        print("BLOCKED")
        for failure in failures:
            print("-", failure)
        return 2

    print("READY")
    print(
        "Run the vanilla backend smoke first: "
        "python -m pytest -q -s -m npu_backend_smoke "
        "tests/integration/test_native_sleep_npu.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
