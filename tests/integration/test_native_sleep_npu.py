"""Real Ascend NPU acceptance for MultiTask lifecycle primitives.

This suite is deliberately opt-in.  It uses vLLM-Ascend/torch_npu and real
devices; ordinary unit/native/ray integration suites must not depend on it.
"""

from __future__ import annotations

import asyncio
import json
from importlib import metadata
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from omegaconf import open_dict
from packaging.version import InvalidVersion, Version

pytestmark = [pytest.mark.native, pytest.mark.npu_integration]


@pytest.fixture(autouse=True)
def _preflight_npu_acceptance(monkeypatch):
    """Pin the platform and reject undersized Ray temp filesystems up front."""
    monkeypatch.setenv("VERL_PLATFORM", "huawei")
    _ray_temp_dir()

MODEL_ENV = "VERL_MULTITASK_NPU_MODEL_PATH"


RAY_TMPDIR_ENV = "VERL_MULTITASK_RAY_TMPDIR"
MIN_RAY_TMP_FREE_GIB = 4
MIN_RAY_TMP_FREE_RATIO = 0.05


def _ray_temp_dir() -> Path:
    """Return one preflighted Ray temp directory shared by all NPU tests."""
    raw = os.environ.get(RAY_TMPDIR_ENV)
    temp_dir = Path(raw).expanduser().resolve() if raw else Path("/tmp").resolve()
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True, exist_ok=True)

    usage = shutil.disk_usage(temp_dir)
    free_gib = usage.free / (1024 ** 3)
    free_ratio = usage.free / usage.total if usage.total else 0.0
    if (
        free_gib < MIN_RAY_TMP_FREE_GIB
        or free_ratio < MIN_RAY_TMP_FREE_RATIO
    ):
        pytest.skip(
            f"Ray temp filesystem {temp_dir} has {free_gib:.2f} GiB free "
            f"({free_ratio * 100:.2f}%); Ray may fail object creation/spilling "
            f"when filesystem usage exceeds 95%. Set {RAY_TMPDIR_ENV} to a "
            f"filesystem with at least {MIN_RAY_TMP_FREE_GIB} GiB and "
            f"{MIN_RAY_TMP_FREE_RATIO * 100:.0f}% free space"
        )
    return temp_dir


def _ray_init_kwargs(num_cpus: int) -> dict:
    """Build Ray init args after the common NPU acceptance preflight."""
    return {
        "num_cpus": num_cpus,
        "_temp_dir": str(_ray_temp_dir()),
        "runtime_env": _runtime_env(),
        "ignore_reinit_error": True,
    }


def _ascend_supports_rl_config() -> bool:
    try:
        from vllm_ascend import ascend_config

        ascend_cls = getattr(ascend_config, "AscendConfig", None)
        dataclass_fields = getattr(ascend_cls, "__dataclass_fields__", {}) or {}
        model_fields = getattr(ascend_cls, "model_fields", {}) or {}
        return bool(
            hasattr(ascend_config, "RlConfig")
            or "rl_config" in dataclass_fields
            or "rl_config" in model_fields
        )
    except Exception:
        return False


def _ascend_rl_engine_kwargs() -> dict:
    """Select exactly one Ascend RL configuration contract."""
    if _ascend_supports_rl_config():
        additional_config = {
            "rl_config": {
                "enabled": True,
            }
        }
    else:
        # Pre-rl_config releases used the top-level ND/NZ switch.
        additional_config = {"weight_nz_mode": 0}

    return {
        "vllm": {
            "additional_config": additional_config,
        }
    }


def _source_git_head(module_file: str) -> str | None:
    path = Path(module_file).resolve()
    for parent in (path.parent, *path.parents):
        if (parent / ".git").exists():
            try:
                return subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    text=True,
                    timeout=5,
                ).strip()
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                return None
    return None


def _verl_expected_vllm_pair() -> tuple[str, str] | None:
    """Read the NPU runtime pair from the installed VERL source checkout."""
    try:
        import verl
    except Exception:
        return None

    repo_root = Path(verl.__file__).resolve().parent.parent
    install_script = repo_root / "scripts" / "install_vllm_mcore_npu.sh"
    if not install_script.exists():
        return None
    try:
        text = install_script.read_text(encoding="utf-8")
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


def _verl_expected_transformers() -> str | None:
    """Read the final transformers pin from the installed VERL NPU installer."""
    try:
        import verl
    except Exception:
        return None
    repo_root = Path(verl.__file__).resolve().parent.parent
    install_script = repo_root / "scripts" / "install_vllm_mcore_npu.sh"
    if not install_script.exists():
        return None
    try:
        text = install_script.read_text(encoding="utf-8")
    except OSError:
        return None
    matches = re.findall(
        r"transformers==(?P<version>\d+\.\d+\.\d+)",
        text,
    )
    return matches[-1] if matches else None


def _validate_vllm_ascend_version_pair() -> None:
    """Fail fast before model load when the NPU runtime sources are unpaired."""
    vllm_raw = metadata.version("vllm")
    ascend_raw = metadata.version("vllm-ascend")
    try:
        vllm_version = Version(vllm_raw)
        ascend_version = Version(ascend_raw)
    except InvalidVersion:
        return

    # vLLM-Ascend release numbering follows the paired vLLM lane. rc/post
    # suffixes may differ, but major/minor must match.
    if (
        vllm_version.release[:2] != (0, 0)
        and ascend_version.release[:2] != (0, 0)
        and vllm_version.release[:2] != ascend_version.release[:2]
    ):
        pytest.fail(
            "unsupported vLLM/vLLM-Ascend release pair: "
            f"vllm={vllm_raw}, vllm-ascend={ascend_raw}. Install matching "
            "release lanes before MultiTask NPU acceptance.",
            pytrace=False,
        )

    expected = _verl_expected_vllm_pair()
    if expected is None:
        return
    expected_vllm, expected_ascend = expected
    expected_vllm_version = Version(expected_vllm)
    expected_ascend_version = Version(expected_ascend)
    if (
        vllm_version.release[:3] not in {(0, 0, 0), expected_vllm_version.release[:3]}
        or (
            ascend_version.release[:2] != (0, 0)
            and ascend_version.release[:2] != expected_ascend_version.release[:2]
        )
    ):
        pytest.fail(
            "installed NPU runtime does not match this VERL checkout: "
            f"VERL scripts expect vllm={expected_vllm} and "
            f"vllm-ascend={expected_ascend} release lane, but metadata reports "
            f"vllm={vllm_raw}, vllm-ascend={ascend_raw}. Align the source "
            "checkouts with scripts/install_vllm_mcore_npu.sh before running "
            "MultiTask lifecycle acceptance.",
            pytrace=False,
        )


def _print_npu_runtime_diagnostics() -> None:
    _validate_vllm_ascend_version_pair()

    expected_transformers = _verl_expected_transformers()
    actual_transformers = metadata.version("transformers")
    if (
        expected_transformers is not None
        and actual_transformers != expected_transformers
    ):
        pytest.fail(
            "installed transformers does not match this VERL NPU checkout: "
            f"expected {expected_transformers}, got {actual_transformers}. "
            "Align the environment with scripts/install_vllm_mcore_npu.sh "
            "before running NPU backend/lifecycle acceptance.",
            pytrace=False,
        )

    def version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "not-installed"

    import torch
    import vllm
    import vllm_ascend

    from verl.utils.device import get_device_name, get_resource_name

    print(
        "NPU_ACCEPTANCE_ENV",
        {
            "torch": getattr(torch, "__version__", "unknown"),
            "torch_npu": version("torch-npu"),
            "vllm": version("vllm"),
            "vllm_ascend": version("vllm-ascend"),
            "vllm_path": str(Path(vllm.__file__).resolve()),
            "vllm_git_head": _source_git_head(vllm.__file__),
            "vllm_ascend_path": str(Path(vllm_ascend.__file__).resolve()),
            "vllm_ascend_git_head": _source_git_head(vllm_ascend.__file__),
            "ascend_rl_config": _ascend_supports_rl_config(),
            "verl_expected_vllm_pair": _verl_expected_vllm_pair(),
            "verl_expected_transformers": expected_transformers,
            "transformers": actual_transformers,
            "verl_device": get_device_name(),
            "ray_resource": get_resource_name(),
            "VERL_PLATFORM": os.environ.get("VERL_PLATFORM"),
            "ASCEND_RT_VISIBLE_DEVICES": os.environ.get(
                "ASCEND_RT_VISIBLE_DEVICES"
            ),
            "VLLM_WORKER_MULTIPROC_METHOD": os.environ.get(
                "VLLM_WORKER_MULTIPROC_METHOD"
            ),
        },
        flush=True,
    )


def _checkpoint_weight_keys(path: Path) -> list[str]:
    index = _load_json_if_present(path / "model.safetensors.index.json")
    if isinstance(index, dict):
        weight_map = index.get("weight_map")
        if isinstance(weight_map, dict):
            return sorted(str(key) for key in weight_map.keys())

    files = sorted(path.glob("*.safetensors"))
    if not files:
        return []
    try:
        from safetensors import safe_open
    except ImportError:
        return []

    keys: list[str] = []
    for file_path in files[:2]:
        try:
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                keys.extend(handle.keys())
        except Exception:
            continue
    return sorted(set(keys))


def _load_json_if_present(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        pytest.fail(f"cannot parse model metadata {path}: {exc}")
    if not isinstance(value, dict):
        pytest.fail(f"model metadata {path} must contain a JSON object")
    return value


def _require_plain_acceptance_model(path: Path) -> None:
    """Keep lifecycle acceptance independent from backend/model-layout drift."""
    config = _load_json_if_present(path / "config.json") or {}
    quant_description = _load_json_if_present(
        path / "quant_model_description.json"
    )

    checkpoint_keys = _checkpoint_weight_keys(path)
    if checkpoint_keys:
        has_model_prefix = any(
            key.startswith("model.layers.") for key in checkpoint_keys
        )
        has_bare_layers = any(key.startswith("layers.") for key in checkpoint_keys)
        if has_bare_layers and not has_model_prefix:
            pytest.fail(
                "Qwen3 checkpoint uses bare 'layers.*' weight keys instead of "
                "the upstream Hugging Face/vLLM 'model.layers.*' layout. Use "
                "an original plain Qwen3 BF16/FP16 checkpoint for NPU backend "
                "and lifecycle acceptance.",
                pytrace=False,
            )

    quantization_config = config.get("quantization_config")
    if quantization_config:
        pytest.skip(
            "NPU lifecycle acceptance requires a plain BF16/FP16 model; "
            "config.json contains quantization_config. Quantized-model loader "
            "compatibility must be validated separately."
        )

    if quant_description:
        kv_cache_type = quant_description.get("kv_cache_type")
        pytest.skip(
            "NPU lifecycle acceptance requires a plain BF16/FP16 model; "
            "quant_model_description.json is present"
            + (
                f" with kv_cache_type={kv_cache_type!r}"
                if kv_cache_type is not None
                else ""
            )
            + ". vLLM-Ascend C8/ModelSlim weight-loader compatibility is "
            "outside this lifecycle acceptance."
        )

    # Ask the installed backend itself. vLLM-Ascend 0.23+ auto-detects
    # quantization during platform config; mirroring that exact decision here
    # prevents lifecycle tests from silently entering a quantized loader path.
    try:
        from vllm_ascend.quantization.utils import detect_quantization_method
    except (ImportError, AttributeError):
        return
    detected = detect_quantization_method(str(path))
    if detected is not None:
        pytest.skip(
            "NPU lifecycle acceptance requires a plain BF16/FP16 model; "
            f"the installed vLLM-Ascend backend auto-detected quantization "
            f"method {detected!r}. Validate that loader separately first."
        )


def _require_model_path() -> str:
    value = os.environ.get(MODEL_ENV)
    if not value:
        pytest.skip(f"set {MODEL_ENV} to a local model path for real NPU acceptance")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        pytest.skip(f"{MODEL_ENV} does not exist: {path}")
    _require_plain_acceptance_model(path)
    return str(path)


def _require_npu(min_devices: int) -> None:
    pytest.importorskip("torch_npu")
    pytest.importorskip("vllm")
    pytest.importorskip("vllm_ascend")

    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    if torch.npu.device_count() < min_devices:
        pytest.skip(
            f"NPU acceptance requires at least {min_devices} NPU device(s)"
        )

    from verl.utils.device import get_device_name, get_resource_name

    if get_device_name() != "npu" or get_resource_name() != "NPU":
        pytest.skip("VERL platform did not resolve to the Ascend NPU backend")


def _visible_npu_devices() -> str:
    configured = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if configured:
        return configured

    import torch

    count = torch.npu.device_count()
    if count <= 0:
        raise RuntimeError("torch.npu reports no visible Ascend devices")
    return ",".join(str(index) for index in range(count))


def _runtime_env() -> dict:
    env_vars = {
        "VERL_PLATFORM": "huawei",
        "ASCEND_RT_VISIBLE_DEVICES": _visible_npu_devices(),
        "TOKENIZERS_PARALLELISM": "true",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_SERVER_DEV_MODE": "1",
        "HCCL_CONNECT_TIMEOUT": "1500",
        "HCCL_HOST_SOCKET_PORT_RANGE": "60000-60050",
        "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61050",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
        "VLLM_LOGGING_LEVEL": "INFO",
        "VLLM_USE_V1": "1",
    }
    if not _ascend_supports_rl_config():
        env_vars["VLLM_ASCEND_ENABLE_NZ"] = "0"
    return {"env_vars": env_vars}


def _dump_recent_vllm_worker_logs() -> str:
    """Print recent Ray log tails and return them for failure classification."""
    temp_dir = _ray_temp_dir()
    logs_dir = temp_dir / "session_latest" / "logs"
    if not logs_dir.exists():
        return ""

    candidates = []
    for pattern in ("*.out", "*.err"):
        candidates.extend(logs_dir.glob(pattern))
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    emitted = 0
    captured = []
    for log_path in candidates:
        if emitted >= 6:
            break
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not any(
            marker in text
            for marker in (
                "VllmWorker",
                "WorkerProc failed",
                "WorkerProc initialization failed",
                "EngineCore failed",
                "Traceback (most recent call last)",
            )
        ):
            continue
        tail = "\n".join(text.splitlines()[-160:])
        captured.append(tail)
        print(
            f"\nNPU_VLLM_WORKER_LOG {log_path}\n{tail}\n",
            flush=True,
        )
        emitted += 1
    return "\n".join(captured)


async def _init_standalone_with_diagnostics(replica) -> None:
    try:
        await replica.init_standalone()
    except BaseException as exc:
        logs = _dump_recent_vllm_worker_logs()
        if (
            "patch_gqa_c8" in logs
            and "gate_up_proj.weight" in logs
            and "KeyError" in logs
        ):
            pytest.fail(
                "Qwen3 fused gate_up_proj is absent during the VERL/vLLM "
                "bootstrap. This occurs before MultiTask sleep/FORCE/RESTORE. "
                "Run the npu_backend_smoke stage and compare its direct "
                "vLLM-Ascend result with the VERL vLLMReplica stage: if direct "
                "vLLM passes but VERL fails, inspect VERL server args / "
                "worker_extension_cls rather than MultiTask lifecycle logic.",
                pytrace=False,
            )
        raise exc


def _run_direct_qwen3_layout_probe(model_path: str) -> None:
    """Inspect the constructed Qwen3 parameter layout before loading weights."""
    env = os.environ.copy()
    visible = env.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible:
        env["ASCEND_RT_VISIBLE_DEVICES"] = visible.split(",", 1)[0].strip()
    else:
        env["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_V1"] = "1"
    env["VERL_MULTITASK_DIRECT_SMOKE_MODEL"] = model_path

    script = r"""
import os

from vllm import LLM
import vllm_ascend  # noqa: F401
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

class _LayoutProbeComplete(RuntimeError):
    pass

_original_load_weights = DefaultModelLoader.load_weights

def _probe(self, model, model_config):
    params = dict(model.named_parameters(remove_duplicate=False))
    layer10 = sorted(
        name for name in params
        if "layers.10.mlp." in name
    )
    modules = dict(model.named_modules())
    gate_name = "model.layers.10.mlp.gate_up_proj"
    gate = modules.get(gate_name)
    print(
        "DIRECT_QWEN3_PRELOAD_LAYOUT",
        {
            "model_type": type(model).__module__ + "." + type(model).__qualname__,
            "parameter_count": len(params),
            "layer10_mlp_params": layer10,
            "gate_up_module_found": gate is not None,
            "gate_up_proj_type": (
                None
                if gate is None
                else type(gate).__module__ + "." + type(gate).__qualname__
            ),
            "gate_up_named_parameters": (
                []
                if gate is None
                else [
                    name
                    for name, _ in gate.named_parameters(
                        recurse=True,
                        remove_duplicate=False,
                    )
                ]
            ),
            "quant_config_type": (
                None
                if getattr(model, "quant_config", None) is None
                else (
                    type(model.quant_config).__module__
                    + "."
                    + type(model.quant_config).__qualname__
                )
            ),
        },
        flush=True,
    )
    raise _LayoutProbeComplete("layout probe complete")

DefaultModelLoader.load_weights = _probe

model = os.environ["VERL_MULTITASK_DIRECT_SMOKE_MODEL"]
try:
    LLM(
        model=model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.4,
        max_model_len=512,
        max_num_seqs=1,
        enforce_eager=True,
        trust_remote_code=True,
    )
except _LayoutProbeComplete:
    print("DIRECT_QWEN3_PRELOAD_LAYOUT_PASS", flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    if result.stdout:
        print("\nDIRECT_QWEN3_LAYOUT_STDOUT\n" + result.stdout, flush=True)
    if result.stderr:
        print("\nDIRECT_QWEN3_LAYOUT_STDERR\n" + result.stderr, flush=True)
    if "DIRECT_QWEN3_PRELOAD_LAYOUT" not in result.stdout:
        pytest.fail(
            "Qwen3 preload layout probe did not reach model construction; "
            "inspect DIRECT_QWEN3_LAYOUT_STDOUT/STDERR above.",
            pytrace=False,
        )


def _run_direct_vllm_ascend_smoke(
    model_path: str,
    *,
    distributed_executor_backend: str | None,
) -> None:
    """Validate vLLM-Ascend directly, without VERL, Ray, or worker extensions."""
    env = os.environ.copy()
    visible = env.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible:
        env["ASCEND_RT_VISIBLE_DEVICES"] = visible.split(",", 1)[0].strip()
    else:
        env["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_USE_V1"] = "1"
    env["VERL_MULTITASK_DIRECT_SMOKE_MODEL"] = model_path
    env["VERL_MULTITASK_DIRECT_SMOKE_BACKEND"] = (
        distributed_executor_backend or ""
    )

    script = r"""
import os

from vllm import LLM, SamplingParams
import vllm_ascend  # noqa: F401
from vllm.model_executor.models.qwen2 import Qwen2Model

_original_qwen2_load_weights = Qwen2Model.load_weights

def _diagnostic_qwen2_load_weights(self, weights):
    params = dict(self.named_parameters(remove_duplicate=False))
    layer10 = sorted(
        name for name in params
        if name.startswith("layers.10.mlp.")
    )
    module = self.layers[10].mlp.gate_up_proj
    print(
        "DIRECT_QWEN3_LAYER10_LAYOUT",
        {
            "gate_up_proj_type": (
                type(module).__module__ + "." + type(module).__qualname__
            ),
            "layer10_mlp_params": layer10,
            "gate_up_named_parameters": [
                name for name, _ in module.named_parameters(
                    recurse=True,
                    remove_duplicate=False,
                )
            ],
            "gate_up_named_buffers": [
                name for name, _ in module.named_buffers(recurse=True)
            ],
            "quant_config_type": (
                None
                if getattr(self, "quant_config", None) is None
                else (
                    type(self.quant_config).__module__
                    + "."
                    + type(self.quant_config).__qualname__
                )
            ),
        },
        flush=True,
    )
    return _original_qwen2_load_weights(self, weights)

Qwen2Model.load_weights = _diagnostic_qwen2_load_weights

model = os.environ["VERL_MULTITASK_DIRECT_SMOKE_MODEL"]
backend = os.environ["VERL_MULTITASK_DIRECT_SMOKE_BACKEND"]
kwargs = {}
if backend:
    kwargs["distributed_executor_backend"] = backend

print(
    "DIRECT_VLLM_ASCEND_CONFIG",
    {"backend": backend or "default", "model": model},
    flush=True,
)
llm = LLM(
    model=model,
    dtype="bfloat16",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.4,
    max_model_len=512,
    max_num_seqs=1,
    enforce_eager=True,
    trust_remote_code=True,
    **kwargs,
)
outputs = llm.generate(
    ["Hello from direct vLLM-Ascend smoke."],
    SamplingParams(temperature=0.0, max_tokens=4),
)
assert outputs and outputs[0].outputs and outputs[0].outputs[0].token_ids
print(
    "DIRECT_VLLM_ASCEND_SMOKE_PASS",
    {"backend": backend or "default"},
    flush=True,
)
"""
    label = distributed_executor_backend or "default"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        text=True,
        capture_output=True,
        timeout=240,
    )
    if result.stdout:
        print(
            f"\nDIRECT_VLLM_ASCEND_STDOUT[{label}]\n" + result.stdout,
            flush=True,
        )
    if result.stderr:
        print(
            f"\nDIRECT_VLLM_ASCEND_STDERR[{label}]\n" + result.stderr,
            flush=True,
        )
    if result.returncode != 0:
        pytest.fail(
            "direct vLLM-Ascend Qwen3 load+generate failed before VERL/Ray "
            f"integration for backend={label!r} (exit={result.returncode}); "
            f"inspect DIRECT_VLLM_ASCEND_STDOUT[{label}] / STDERR above.",
            pytrace=False,
        )


def _require_ray_npus(ray, min_devices: int) -> None:
    resources = ray.cluster_resources()
    count = float(resources.get("NPU", 0.0))
    if count < min_devices:
        pytest.skip(
            f"Ray exposes only {count:g} NPU resource(s); "
            f"{min_devices} required"
        )


def _config(
    model_path: str,
    *,
    enable_sleep_mode: bool = True,
    enforce_eager: bool = False,
):
    from hydra import compose, initialize_config_dir

    import verl

    config_dir = Path(verl.__file__).resolve().parent / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer")

    # The Hydra rollout node is struct-locked and may lag newer RolloutConfig
    # dataclass fields (notably enable_sleep_mode). Acceptance tests need to
    # exercise the real runtime capability, so extend only this test config
    # explicitly instead of mutating VERL's production schema.
    with open_dict(config):
        config.trainer.nnodes = 1
        config.trainer.n_gpus_per_node = 1
        config.actor_rollout_ref.model.path = model_path
        config.actor_rollout_ref.rollout.name = "vllm"
        config.actor_rollout_ref.rollout.mode = "async"
        config.actor_rollout_ref.rollout.nnodes = 1
        config.actor_rollout_ref.rollout.n_gpus_per_node = 1
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
        config.actor_rollout_ref.rollout.data_parallel_size = 1
        config.actor_rollout_ref.rollout.pipeline_model_parallel_size = 1
        config.actor_rollout_ref.rollout.enable_sleep_mode = enable_sleep_mode
        config.actor_rollout_ref.rollout.free_cache_engine = enable_sleep_mode
        config.actor_rollout_ref.rollout.enforce_eager = enforce_eager
        config.actor_rollout_ref.rollout.load_format = "auto"
        config.actor_rollout_ref.rollout.skip_tokenizer_init = False
        config.actor_rollout_ref.rollout.engine_kwargs = (
            _ascend_rl_engine_kwargs() if enable_sleep_mode else {}
        )
    return config


def _exercise_same_npu_borrower(
    *,
    donor_replica,
    donor_key,
    placement: dict,
    rollout_config,
    model_config,
    prompt_ids,
    token: str,
):
    import ray

    from multi_task_scheduler.integration.verl.experimental_fully_async.llm_server_manager import (
        MultiTaskLLMServerManager,
    )
    from multi_task_scheduler.orchestration.contracts import (
        EvidenceType,
        FIRST_RELEASE_MAX_COLOCATE_COUNT,
        FIRST_RELEASE_RAY_GPU_FRACTION,
        ReplicaKey,
        ReplicaState,
    )
    from multi_task_scheduler.rollout.replica import MultiTaskvLLMReplica

    physical_id = placement["gpu_uuid"]
    pg = donor_replica.resource_pool.get_placement_groups()[0]

    borrower_manager = MultiTaskLLMServerManager.__new__(
        MultiTaskLLMServerManager
    )
    borrower_task = f"{token}-borrower"
    borrower_manager.task_session = borrower_task
    borrower_manager.rollout_replica_class = MultiTaskvLLMReplica
    borrower_manager.rollout_config = rollout_config
    borrower_manager.model_config = model_config
    borrower_manager.replica_state = {}
    borrower_manager.replica_kind = {}
    borrower_manager._runtime_inventory = {}
    borrower_manager.next_replica_rank = 0
    borrower_manager._allocated_replica_ranks = set()
    borrower_manager.borrowed_operations = {}
    borrower_manager._native_release_evidence = {}
    borrower_manager.replica_operation_lock = asyncio.Lock()

    borrowed_key = ReplicaKey(borrower_task, "borrowed-0")
    borrowed_spec = {
        "operation_id": f"{token}-add",
        "lease_id": f"{token}-lease",
        "borrower_task_id": borrowed_key.task_session,
        "borrower_replica_id": borrowed_key.replica_id,
        "claims": [
            {
                "claim_id": f"{token}-claim",
                "source_lease_id": f"{token}-source-lease",
                "donor_task_id": donor_key.task_session,
                "donor_replica_rank": 0,
                "pg_id": pg.id.hex(),
                "bundle_index": 0,
                "node_id": placement["node_id"],
                "gpu_uuid": physical_id,
                "rank": 0,
                "node_rank": 0,
                "local_rank": 0,
                "gpu_fraction": FIRST_RELEASE_RAY_GPU_FRACTION,
                "cpu_request": 1.0,
            }
        ],
        "expires_at": 0,
        "placement_epoch": 0,
    }

    create_receipt = asyncio.run(
        borrower_manager.create_borrowed_replica(borrowed_spec)
    )
    assert create_receipt["state"] == "RUNTIME_READY"

    borrowed_runtime = borrower_manager.inspect_runtime(borrowed_key)
    assert borrowed_runtime is not None
    borrowed_placement = asyncio.run(
        borrowed_runtime.validate_worker_placement()
    )
    assert borrowed_placement[0]["gpu_uuid"] == physical_id
    assert borrowed_placement[0]["node_id"] == placement["node_id"]
    assert borrowed_placement[0]["resource_name"] == "NPU"

    output = ray.get(
        borrowed_runtime._server_handle.generate.remote(
            request_id=f"{token}-generate-{uuid4().hex}",
            prompt_ids=prompt_ids,
            sampling_params={"temperature": 0.0, "max_tokens": 16},
            image_data=None,
        ),
        timeout=120,
    )
    assert getattr(output, "token_ids", None)

    destroy_evidence = asyncio.run(
        borrower_manager.destroy(
            borrowed_key,
            operation_id=f"{token}-remove",
        )
    )
    assert destroy_evidence.type is EvidenceType.RELEASED
    assert destroy_evidence.released_gpu_uuids == (physical_id,)
    borrower_manager.transition_replica(
        borrowed_key,
        ReplicaState.RELEASED,
    )
    return destroy_evidence


@pytest.mark.npu_backend_smoke
def test_real_npu_backend_smoke_loads_and_generates_plain_model():
    """Separate direct vLLM-Ascend from the vanilla VERL rollout bootstrap."""

    model_path = _require_model_path()
    _require_npu(1)
    _print_npu_runtime_diagnostics()

    # Stage -1: inspect the actual instantiated parameter layout before
    # checkpoint loading. This distinguishes model construction from loader bugs.
    _run_direct_qwen3_layout_probe(model_path)

    # Stage 0: no VERL vLLMReplica, no Ray actor, no worker_extension_cls,
    # and eager mode to remove graph compilation from the equation.
    _run_direct_vllm_ascend_smoke(
        model_path,
        distributed_executor_backend=None,
    )
    _run_direct_vllm_ascend_smoke(
        model_path,
        distributed_executor_backend="mp",
    )

    import ray
    from transformers import AutoTokenizer

    from verl.utils.tokenizer import normalize_token_ids
    from verl.workers.rollout.vllm_rollout.vllm_async_server import (
        vLLMHttpServer,
        vLLMReplica,
    )

    config = _config(
        model_path,
        enable_sleep_mode=False,
        enforce_eager=True,
    )
    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    prompt_ids = normalize_token_ids(
        tokenizer.encode(
            "Hello from the vanilla VERL NPU backend smoke test.",
            add_special_tokens=True,
        )
    )

    class NoWorkerExtensionServer(vLLMHttpServer):
        def _get_worker_extension_cls(self):
            return ""

    class NoWorkerExtensionReplica(vLLMReplica):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.server_class = ray.remote(NoWorkerExtensionServer)

    def run_verl_stage(replica_cls, label: str):
        ray.shutdown()
        ray.init(**_ray_init_kwargs(4))
        try:
            _require_ray_npus(ray, 1)
            replica = replica_cls(
                replica_rank=0,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=1,
            )
            asyncio.run(_init_standalone_with_diagnostics(replica))
            output = ray.get(
                replica._server_handle.generate.remote(
                    request_id=f"npu-backend-smoke-{label}-{uuid4().hex}",
                    prompt_ids=prompt_ids,
                    sampling_params={"temperature": 0.0, "max_tokens": 8},
                    image_data=None,
                ),
                timeout=120,
            )
            assert getattr(output, "token_ids", None)
            print(
                "VERL_NPU_BACKEND_SMOKE_PASS",
                {"stage": label},
                flush=True,
            )
        finally:
            ray.shutdown()

    # Stage 1: keep VERL server/replica orchestration but remove the colocate
    # worker extension. This isolates VERL CLI/server args from worker patches.
    run_verl_stage(NoWorkerExtensionReplica, "no-worker-extension")

    # Stage 2: exact upstream VERL vLLMReplica baseline.
    run_verl_stage(vLLMReplica, "default-worker-extension")


def test_real_npu_sleep_releases_same_slot_to_borrower():
    """One-NPU DONATE acceptance using vLLM-Ascend level-1 sleep.

    Acceptance is intentionally stronger than a memory counter: after the
    donor sleeps, a borrowed vLLM replica must start on the exact same Ray NPU
    placement and complete generation.
    """

    model_path = _require_model_path()
    _require_npu(1)
    _print_npu_runtime_diagnostics()

    import ray
    from transformers import AutoTokenizer

    from multi_task_scheduler.integration.verl.experimental_fully_async.llm_server_manager import (
        MultiTaskLLMServerManager,
    )
    from multi_task_scheduler.orchestration.contracts import (
        EvidenceType,
        ReplicaKey,
        ReplicaKind,
        ReplicaState,
    )
    from multi_task_scheduler.rollout.replica import MultiTaskvLLMReplica
    from verl.utils.tokenizer import normalize_token_ids

    config = _config(model_path)
    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model

    ray.shutdown()
    ray.init(**_ray_init_kwargs(4))

    try:
        _require_ray_npus(ray, 1)

        replica = MultiTaskvLLMReplica(
            replica_rank=0,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=1,
            replica_kind=ReplicaKind.NATIVE,
        )
        asyncio.run(_init_standalone_with_diagnostics(replica))
        placement = asyncio.run(replica.worker_placements())[0]
        assert placement["resource_name"] == "NPU"
        assert placement["gpu_uuid"].startswith("NPU:")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        prompt_ids = normalize_token_ids(
            tokenizer.encode(
                "Hello from the real NPU sleep acceptance test.",
                add_special_tokens=True,
            )
        )
        baseline = ray.get(
            replica._server_handle.generate.remote(
                request_id=f"npu-sleep-{uuid4().hex}",
                prompt_ids=prompt_ids,
                sampling_params={"temperature": 0.0, "max_tokens": 16},
                image_data=None,
            ),
            timeout=120,
        )
        assert getattr(baseline, "token_ids", None)

        key = ReplicaKey("npu-donor", "native-0")
        manager = MultiTaskLLMServerManager.__new__(MultiTaskLLMServerManager)
        manager.replica_kind = {key: ReplicaKind.NATIVE}
        manager.replica_state = {key: ReplicaState.DRAINING}
        manager._runtime_inventory = {key: replica}
        manager._native_release_evidence = {}

        release = asyncio.run(
            manager.sleep(key, operation_id="npu-donate")
        )
        assert release.type is EvidenceType.RELEASED
        assert release.released_gpu_uuids == (placement["gpu_uuid"],)

        # Exact replay must not execute a second runtime sleep.
        manager.replica_state[key] = ReplicaState.DORMANT
        assert asyncio.run(
            manager.sleep(key, operation_id="npu-donate")
        ) == release

        # The server receipt must prove the platform-selected NPU sleep level.
        sleep_receipt = ray.get(replica.servers[0].sleep.remote())
        assert sleep_receipt["sleeping"] is True
        assert sleep_receipt["sleep_level"] == 1

        destroy_evidence = _exercise_same_npu_borrower(
            donor_replica=replica,
            donor_key=key,
            placement=placement,
            rollout_config=rollout_config,
            model_config=model_config,
            prompt_ids=prompt_ids,
            token="npu-cycle",
        )
        assert destroy_evidence.released_gpu_uuids == release.released_gpu_uuids
    finally:
        ray.shutdown()


def test_real_npu_force_remove_continues_on_another_replica():
    """Two-NPU FORCE acceptance for targeted abort + logical continuation."""

    model_path = _require_model_path()
    _require_npu(2)
    _print_npu_runtime_diagnostics()

    import ray
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    from multi_task_scheduler.integration.verl.experimental_fully_async.rollouter import (
        MultiTaskFullyAsyncRollouter,
        _MultiTaskFullyAsyncLLMServerClient,
    )
    from multi_task_scheduler.orchestration.contracts import (
        AttemptState,
        EvidenceType,
        ReplicaKey,
        ReplicaKind,
        ReplicaState,
    )
    from multi_task_scheduler.rollout.load_balancer import (
        MultiTaskGlobalRequestLoadBalancer,
    )
    from multi_task_scheduler.rollout.replica import MultiTaskvLLMReplica
    from verl.single_controller.ray.base import _unwrap_ray_remote
    from verl.utils.tokenizer import normalize_token_ids

    config = _config(model_path)
    with open_dict(config):
        config.async_training = OmegaConf.create({"partial_rollout": True})

    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model

    ray.shutdown()
    ray.init(**_ray_init_kwargs(8))

    try:
        _require_ray_npus(ray, 2)

        target = MultiTaskvLLMReplica(
            replica_rank=0,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=1,
            replica_kind=ReplicaKind.BORROWED,
        )
        alternate = MultiTaskvLLMReplica(
            replica_rank=1,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=1,
            replica_kind=ReplicaKind.NATIVE,
        )

        async def init_replicas():
            await asyncio.gather(
                _init_standalone_with_diagnostics(target),
                _init_standalone_with_diagnostics(alternate),
            )

        asyncio.run(init_replicas())

        target_key = ReplicaKey("npu-force", "borrowed-0")
        lb = ray.remote(MultiTaskGlobalRequestLoadBalancer).remote(
            {target._server_address: target._server_handle},
            full_determinism=False,
            initial_routes={target_key: target._server_address},
        )
        client = _MultiTaskFullyAsyncLLMServerClient(
            config=config,
            load_balancer_handle=lb,
            client_id="npu-force-client",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        prompt_ids = normalize_token_ids(
            tokenizer.encode(
                "Continue this response long enough to validate NPU FORCE handoff.",
                add_special_tokens=True,
            )
        )
        logical_request_id = f"npu-force-{uuid4().hex}"

        class Manager:
            global_load_balancer = lb

            def __init__(self):
                self.replica_state = {target_key: ReplicaState.ACTIVE}
                self.replica_kind = {target_key: ReplicaKind.BORROWED}

            def replica_meta(self, key):
                return self.replica_kind[key], self.replica_state[key]

            def inspect_runtime(self, key):
                assert key == target_key
                return target

            def transition_replica(self, key, state):
                self.replica_state[key] = state

        async def scenario():
            generation = asyncio.create_task(
                client.generate(
                    request_id=logical_request_id,
                    prompt_ids=prompt_ids,
                    sampling_params={
                        "temperature": 0.0,
                        "max_tokens": 128,
                        "min_tokens": 128,
                    },
                )
            )

            deadline = asyncio.get_running_loop().time() + 30.0
            while True:
                owned = await lb.requests_for_server.remote(
                    target._server_address
                )
                state = await lb.query_attempt.remote(logical_request_id)
                if logical_request_id in owned and state is AttemptState.ADMITTED:
                    break
                if generation.done():
                    raise AssertionError(
                        "generation completed before FORCE observed admission"
                    )
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError(
                        "request was not admitted to NPU FORCE target in time"
                    )
                await asyncio.sleep(0.01)

            await lb.add_servers.remote(
                {alternate._server_address: alternate._server_handle}
            )

            rollouter_cls = _unwrap_ray_remote(MultiTaskFullyAsyncRollouter)
            rollouter = rollouter_cls.__new__(rollouter_cls)
            rollouter.config = config
            rollouter.llm_server_manager = Manager()
            rollouter._pending_operation_targets = {}
            rollouter._force_exit_recovery = {}
            rollouter._force_handoff_timeout_s = 30.0

            exit_ready = await rollouter.prepare_exit(
                target_key,
                operation_id="npu-force-remove",
                force=True,
            )
            assert exit_ready.type is EvidenceType.EXIT_READY
            assert (
                await lb.continuation_handoff_requests.remote(
                    "npu-force-remove"
                )
            ) == (logical_request_id,)
            assert not await lb.has_unsettled_requests.remote(
                target._server_address
            )

            output = await asyncio.wait_for(generation, timeout=120.0)
            assert output.stop_reason not in {"abort", "aborted"}
            assert len(output.token_ids) == 128

            await lb.finish_remove.remote(target_key)
            assert (
                await lb.continuation_handoff_requests.remote(
                    "npu-force-remove"
                )
            ) == ()
            return output

        output = asyncio.run(scenario())
        assert getattr(output, "token_ids", None)
    finally:
        ray.shutdown()


def _training_sender_class():
    import torch

    from verl.checkpoint_engine import CheckpointEngineRegistry
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.engine_workers import TrainingWorker

    class RestoreTrainingWorker(TrainingWorker):
        def __init__(self, config, checkpoint_engine_config):
            super().__init__(config)
            backend = checkpoint_engine_config.backend
            bucket_size = (
                checkpoint_engine_config.update_weights_bucket_megabytes << 20
            )
            engine_kwargs = dict(
                checkpoint_engine_config.engine_kwargs.get(backend, {})
            )
            if torch.distributed.get_rank() == 0:
                engine_kwargs["is_master"] = True
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                backend,
                bucket_size=bucket_size,
                **engine_kwargs,
            )

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def zero_output_weights_for_restore_acceptance(self):
            module = getattr(
                self.engine.module,
                "_fsdp_wrapped_module",
                self.engine.module,
            )
            get_output_embeddings = getattr(
                module,
                "get_output_embeddings",
                None,
            )
            if get_output_embeddings is None:
                raise RuntimeError(
                    "acceptance model does not expose output embeddings"
                )
            output = get_output_embeddings()
            weight = getattr(output, "weight", None)
            if weight is None:
                raise RuntimeError(
                    "acceptance model output embedding has no weight"
                )
            with torch.no_grad():
                weight.zero_()
            local_weight = (
                weight.to_local() if hasattr(weight, "to_local") else weight
            )
            return {
                "shape": tuple(weight.shape),
                "abs_sum": float(
                    local_weight.detach().float().abs().sum().item()
                ),
            }

        @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
        async def update_weights(
            self,
            global_steps: int = None,
            mode: str = "auto",
        ):
            weights, _ = self.engine.get_per_tensor_param()
            await self.checkpoint_engine.send_weights(
                weights,
                global_steps=global_steps,
            )

        @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
        def execute_checkpoint_engine(self, method: str, *args, **kwargs):
            return getattr(self.checkpoint_engine, method)(*args, **kwargs)

    return RestoreTrainingWorker


def test_real_npu_restore_reinstalls_current_vpub_and_generates_again():
    """Two-NPU RESTORE acceptance using the HCCL implementation registered as nccl."""

    model_path = _require_model_path()
    _require_npu(2)
    _print_npu_runtime_diagnostics()

    import ray

    from multi_task_scheduler.checkpoint.checkpoint_engine_manager import (
        MultiTaskCheckpointEngineManager,
    )
    from multi_task_scheduler.integration.verl.experimental_fully_async.llm_server_manager import (
        MultiTaskLLMServerManager,
    )
    from multi_task_scheduler.orchestration.contracts import (
        EvidenceType,
        ReplicaKey,
        ReplicaKind,
        ReplicaState,
    )
    from multi_task_scheduler.rollout.replica import MultiTaskvLLMReplica
    from verl.single_controller.ray import (
        RayClassWithInitArgs,
        RayResourcePool,
        RayWorkerGroup,
    )
    from verl.utils.device import get_device_name
    from verl.utils.tokenizer import normalize_token_ids
    from verl.workers.config import (
        CheckpointEngineConfig,
        FSDPEngineConfig,
        HFModelConfig,
        RolloutConfig,
        TrainingWorkerConfig,
    )

    checkpoint_config = CheckpointEngineConfig(
        backend="nccl",
        update_weights_bucket_megabytes=64,
        engine_kwargs={"nccl": {"rebuild_group": True}},
    )
    model_config = HFModelConfig(
        path=model_path,
        trust_remote_code=True,
        use_remove_padding=True,
    )
    rollout_config = RolloutConfig(
        name="vllm",
        mode="async",
        nnodes=1,
        n_gpus_per_node=1,
        tensor_model_parallel_size=1,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        gpu_memory_utilization=0.5,
        max_num_seqs=16,
        response_length=32,
        load_format="auto",
        skip_tokenizer_init=False,
        enable_sleep_mode=True,
        free_cache_engine=True,
        engine_kwargs=_ascend_rl_engine_kwargs(),
        checkpoint_engine=checkpoint_config,
    )

    ray.shutdown()
    ray.init(**_ray_init_kwargs(8))

    try:
        _require_ray_npus(ray, 2)

        trainer_pool = RayResourcePool(
            process_on_nodes=[1],
            use_gpu=True,
            name_prefix="multitask_npu_restore_sender_",
            max_colocate_count=1,
        )
        engine_config = FSDPEngineConfig(
            forward_only=True,
            fsdp_size=1,
            strategy="fsdp",
            use_torch_compile=False,
        )
        trainer_config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=model_config,
            engine_config=engine_config,
        )
        sender_cls = _training_sender_class()
        sender_init = RayClassWithInitArgs(
            cls=ray.remote(sender_cls),
            config=trainer_config,
            checkpoint_engine_config=checkpoint_config,
        )
        actor_wg = RayWorkerGroup(
            resource_pool=trainer_pool,
            ray_cls_with_init=sender_init,
            device_name=get_device_name(),
        )
        actor_wg.reset()

        replica = MultiTaskvLLMReplica(
            replica_rank=0,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=1,
            replica_kind=ReplicaKind.NATIVE,
        )
        asyncio.run(replica.init_standalone())
        placement = asyncio.run(replica.worker_placements())[0]
        assert placement["resource_name"] == "NPU"

        prompt_ids = normalize_token_ids(
            model_config.tokenizer.encode(
                "Hello from the current-Vpub NPU RESTORE acceptance test.",
                add_special_tokens=True,
            )
        )

        def generate_once(label: str):
            return ray.get(
                replica._server_handle.generate.remote(
                    request_id=f"{label}-{uuid4().hex}",
                    prompt_ids=prompt_ids,
                    sampling_params={"temperature": 0.0, "max_tokens": 16},
                    image_data=None,
                ),
                timeout=120,
            )

        baseline = generate_once("npu-before-restore")
        assert getattr(baseline, "token_ids", None)

        key = ReplicaKey("npu-acceptance", "native-0")
        manager = MultiTaskLLMServerManager.__new__(MultiTaskLLMServerManager)
        manager.replica_kind = {key: ReplicaKind.NATIVE}
        manager.replica_state = {key: ReplicaState.DRAINING}
        manager._runtime_inventory = {key: replica}
        manager._native_release_evidence = {}

        release = asyncio.run(
            manager.sleep(key, operation_id="npu-restore-donate")
        )
        assert release.type is EvidenceType.RELEASED
        assert release.released_gpu_uuids == (placement["gpu_uuid"],)
        manager.replica_state[key] = ReplicaState.DORMANT

        destroy_evidence = _exercise_same_npu_borrower(
            donor_replica=replica,
            donor_key=key,
            placement=placement,
            rollout_config=rollout_config,
            model_config=model_config,
            prompt_ids=prompt_ids,
            token="npu-restore-cycle",
        )
        assert destroy_evidence.released_gpu_uuids == release.released_gpu_uuids

        mutation_receipts = ray.get(
            actor_wg.zero_output_weights_for_restore_acceptance()
        )
        assert all(
            receipt["abs_sum"] == 0.0 for receipt in mutation_receipts
        )

        checkpoint_manager = MultiTaskCheckpointEngineManager(
            config=checkpoint_config,
            actor_wg=actor_wg,
            replicas=[],
        )
        checkpoint_manager.register_pending(
            key,
            [replica],
            operation_id="npu-restore",
        )
        weight_ready = asyncio.run(
            checkpoint_manager.bootstrap_target(
                key,
                operation_id="npu-restore",
                loaded_version=17,
            )
        )
        assert weight_ready.type is EvidenceType.WEIGHT_READY
        assert key not in checkpoint_manager.effective_replicas

        parked_ref = replica._server_handle.generate.remote(
            request_id=f"npu-parked-before-final-wake-{uuid4().hex}",
            prompt_ids=prompt_ids,
            sampling_params={"temperature": 0.0, "max_tokens": 16},
            image_data=None,
        )
        ready, _pending = ray.wait([parked_ref], timeout=1.0)
        assert ready == []

        wake_receipts = asyncio.run(replica.wake_up())
        assert all(
            receipt["fully_awake"] is True
            and receipt["sleeping"] is False
            for receipt in wake_receipts
        )
        health = asyncio.run(replica.validate_server_runtime())
        assert health["global_steps"] == 17

        checkpoint_manager.commit_pending(
            key,
            weight_ready,
            loaded_version=17,
        )
        assert checkpoint_manager.effective_replicas[key][1] == 17

        restored = ray.get(parked_ref, timeout=120)
        assert getattr(restored, "token_ids", None)
        assert restored.token_ids != baseline.token_ids, (
            "NPU RESTORE output did not change after sender Vpub mutation; "
            "global_steps alone is not sufficient acceptance evidence"
        )
    finally:
        ray.shutdown()
