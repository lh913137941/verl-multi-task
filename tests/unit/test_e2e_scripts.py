"""Static acceptance-runner validation for CI without Ray/VERL."""

import ast
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
E2E = ROOT / "scripts" / "e2e"


def test_e2e_shell_scripts_parse_with_bash():
    scripts = sorted(E2E.glob("*.sh"))
    assert scripts
    for path in scripts:
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{path}: {result.stderr}"


def test_e2e_python_drivers_parse_without_importing_runtime_dependencies():
    drivers = sorted(E2E.glob("*.py"))
    assert drivers
    for path in drivers:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))



def test_e2e_common_exposes_checkout_package_to_standalone_drivers(tmp_path):
    # The control-plane and exactly-once drivers run as plain Python scripts,
    # unlike recovery tests which inherit pytest's pythonpath=src setting.
    for inherited in ("", "/existing/project/site-packages"):
        env = dict(os.environ)
        env["PYTHON_BIN"] = sys.executable
        if inherited:
            env["PYTHONPATH"] = inherited
        else:
            env.pop("PYTHONPATH", None)
        script = (
            '. "$1"; '
            '"$PYTHON_BIN" -c '
            "'import importlib.util; "
            "print(importlib.util.find_spec(\"multi_task_scheduler\").submodule_search_locations[0])' "
            '; printf "PYTHONPATH=%s\\n" "$PYTHONPATH"'
        )
        result = subprocess.run(
            ["bash", "-c", script, "bash", str(E2E / "common.sh")],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        module_path, pythonpath_line = result.stdout.strip().splitlines()
        assert Path(module_path).resolve() == ROOT / "src" / "multi_task_scheduler"
        expected_path = str(ROOT / "src") + (":" + inherited if inherited else "")
        assert pythonpath_line == "PYTHONPATH=" + expected_path


def _load_live_fixture_under_test():
    """Isolate the real driver parser; do not import its Ray CLI entrypoint."""
    import json
    import time

    path = E2E / "lifecycle_driver.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    func = next(
        node for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_fixture"
    )
    scope = {"json": json, "time": time, "Path": Path}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        func,
                    ],
                    type_ignores=[],
                )
            ),
            str(path),
            "exec",
        ),
        scope,
    )
    return scope["_load_fixture"]


def test_live_e2e_claim_ids_are_unique_for_consecutive_lease_cycles(tmp_path):
    import json

    sample = {
        "lease_id": "e2e-lifecycle",
        "claims": [
            {
                "claim_id": "fixed-claim-from-example",
                "source_lease_id": "source-0",
                "donor_task_id": "__TASK_SESSION__",
                "donor_replica_rank": 0,
                "pg_id": "same-physical-pg",
                "bundle_index": 0,
                "node_id": "node-0",
                "gpu_uuid": "NPU:node-0:0",
                "gpu_fraction": 0.5,
                "cpu_request": 1.0,
            }
        ],
    }
    path = tmp_path / "lease.json"
    path.write_text(json.dumps(sample), encoding="utf-8")
    load = _load_live_fixture_under_test()
    first, _, _ = load(path, "donor", "borrower", "run-1")
    second, _, _ = load(path, "donor", "borrower", "run-2")

    assert first.lease_id != second.lease_id
    assert first.claim_ids != second.claim_ids
    assert first.claim_ids == ("fixed-claim-from-example-run-1",)
    assert second.claim_ids == ("fixed-claim-from-example-run-2",)
    # Only the logical claim IDs change; do not fabricate physical placement.
    assert first.gpu_uuids == second.gpu_uuids
    assert first.bundle_keys == second.bundle_keys
    assert first.claims[0]["donor_task_id"] == "donor"


def test_live_e2e_explicit_lease_replay_preserves_original_claim_id(tmp_path):
    import json

    sample = {
        "lease_id": "replay-lease",
        "reuse_lease_id": True,
        "claims": [
            {
                "claim_id": "replay-claim",
                "source_lease_id": "source-0",
                "donor_task_id": "donor",
                "donor_replica_rank": 0,
                "pg_id": "pg-0",
                "bundle_index": 0,
                "node_id": "node-0",
                "gpu_uuid": "NPU:node-0:0",
                "gpu_fraction": 0.5,
                "cpu_request": 1.0,
            }
        ],
    }
    path = tmp_path / "lease-replay.json"
    path.write_text(json.dumps(sample), encoding="utf-8")
    load = _load_live_fixture_under_test()
    first, _, _ = load(path, "donor", "borrower", "run-1")
    second, _, _ = load(path, "donor", "borrower", "run-2")

    assert first.lease_id == second.lease_id == "replay-lease"
    assert first.claim_ids == second.claim_ids == ("replay-claim",)


def test_exactly_once_timeout_records_failing_ray_stage(tmp_path, monkeypatch):
    """No Ray/VERL required: exercise the actual driver entrypoint with fake RPCs."""
    import importlib.util
    import json
    import pickle
    from types import ModuleType, SimpleNamespace

    driver_path = E2E / "exactly_once_driver.py"
    module_name = "_e2e_exactly_once_driver_under_test"
    spec = importlib.util.spec_from_file_location(module_name, driver_path)
    driver = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, driver)
    spec.loader.exec_module(driver)

    class RayGetTimeout(Exception):
        pass

    class RemoteMethod:
        def __init__(self, name):
            self.name = name

        def remote(self, *args, **kwargs):
            return self.name

    class QueueHandle:
        get_queue_size = RemoteMethod("ready")
        put_sample_once = RemoteMethod("enqueue")

    class QueueClass:
        @staticmethod
        def remote(*args, **kwargs):
            return QueueHandle()

    queue_module = ModuleType(
        "multi_task_scheduler.integration.verl.experimental_fully_async.message_queue"
    )
    queue_module.MultiTaskMessageQueue = QueueClass
    monkeypatch.setitem(sys.modules, queue_module.__name__, queue_module)

    for timeout_stage, expected_budget in (("ready", 47.0), ("enqueue", 7.0)):
        observed = []
        fake_ray = ModuleType("ray")
        fake_ray.exceptions = SimpleNamespace(GetTimeoutError=RayGetTimeout)
        fake_ray.cloudpickle = SimpleNamespace(dumps=pickle.dumps)
        fake_ray.is_initialized = lambda: False
        fake_ray.init = lambda **kwargs: None
        fake_ray.kill = lambda actor: None
        fake_ray.shutdown = lambda: None

        def get(ref, timeout):
            observed.append((ref, timeout))
            if ref == timeout_stage:
                raise RayGetTimeout("simulated Ray startup or RPC stall")
            assert ref == "ready"
            return 0

        fake_ray.get = get
        monkeypatch.setitem(sys.modules, "ray", fake_ray)
        monkeypatch.setenv("MT_E2E_ACTOR_STARTUP_TIMEOUT_S", "47")
        monkeypatch.setenv("MT_E2E_QUEUE_RPC_TIMEOUT_S", "7")
        result_file = tmp_path / f"{timeout_stage}.json"
        monkeypatch.setattr(
            sys, "argv", ["exactly_once_driver.py", "--result-file", str(result_file)]
        )

        assert driver.main() == 1
        result = json.loads(result_file.read_text(encoding="utf-8"))
        stage = "queue_actor_ready" if timeout_stage == "ready" else "first_enqueue"
        assert result["state"] == "FAILED"
        assert result["last_stage"] == stage
        assert stage in result["detail"]
        assert "timed out" in result["detail"]
        assert observed[-1] == (timeout_stage, expected_budget)


def _load_auto_lease_builder():
    """Execute the pure fixture builder without importing Ray or VERL."""
    path = E2E / "verify_two_verl_jobs.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    target = next(
        node for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_auto_lease"
    )
    scope = {}
    exec(compile(ast.Module(body=[target], type_ignores=[]), str(path), "exec"), scope)
    return scope["build_auto_lease"]


def test_two_real_jobs_can_generate_lease_from_verified_donor_candidate():
    builder = _load_auto_lease_builder()
    candidate = {
        "donor_replica_rank": 0,
        "pg_id": "real-pg-id",
        "node_id": "real-node-id",
        "gpu_uuid": "NPU:real-node-id:3",
        "bundle_index": 0,
        "resource_name": "NPU",
    }
    result = builder("session-donor", [candidate], 0, "unique123", 7200, "multitask-jobs")
    assert result["expires_in_s"] == 7200
    assert result["lease_id"] == "auto-real-e2e-unique123"
    assert len(result["claims"]) == 1
    claim = result["claims"][0]
    assert claim["donor_task_id"] == "session-donor"
    assert claim["donor_replica_rank"] == 0
    assert claim["pg_id"] == "real-pg-id"
    assert claim["gpu_uuid"] == "NPU:real-node-id:3"
    assert claim["pg_namespace"] == "multitask-jobs"
    assert claim["gpu_fraction"] == 0.5
    assert claim["cpu_request"] == 1.0


def test_two_real_jobs_auto_lease_rejects_missing_or_ambiguous_placements():
    builder = _load_auto_lease_builder()
    candidate = {
        "donor_replica_rank": 0,
        "pg_id": "real-pg-id",
        "node_id": "real-node-id",
        "gpu_uuid": "GPU-uuid",
        "bundle_index": 0,
        "resource_name": "GPU",
    }
    for invalid_candidates, rank in (
        ([], 0),
        ([candidate], 1),
        ([candidate, candidate], 0),
        ([dict(candidate, gpu_uuid="")], 0),
        ([dict(candidate, resource_name="CPU")], 0),
        ([dict(candidate, bundle_index=1)], 0),
    ):
        try:
            builder("session-donor", invalid_candidates, rank, "unique123", 7200, "multitask-jobs")
        except RuntimeError:
            pass
        else:
            raise AssertionError("auto Lease accepted an unverified or ambiguous physical placement")

def _load_bridge_installer():
    import importlib.util

    path = E2E / "ensure_verl_multitask_bridge.py"
    spec = importlib.util.spec_from_file_location("_mt_e2e_bridge_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_verl_entry(tmp_path, *, custom_runner=False, with_bridge=False):
    entry = tmp_path / "verl" / "experimental" / "fully_async_policy" / "fully_async_main.py"
    config = entry.parent / "config" / "fully_async_ppo_trainer.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "defaults:\n  - ppo_trainer\n  - _self_\n\nasync_training:\n"
        .replace("\\\n", "\n"), encoding="utf-8"
    )
    runner = "OtherTaskRunner" if custom_runner else "FullyAsyncTaskRunner"
    source = (
        "class FullyAsyncTaskRunner: pass\n"
        "@hydra.main(config_path='config', config_name='fully_async_ppo_trainer')\n"
        "def main(config):\n"
        "    config = migrate_legacy_reward_impl(config)\n"
        "    run_ppo(config, task_runner_class=" + runner + ")\n"
    ).replace("\\\n", "\n")
    if with_bridge:
        installer = _load_bridge_installer()
        patch = (ROOT / "patches" / "verl-v0.10-fully-async-multitask-entry.patch").read_text()
        helper = installer._added_block(patch, "def _resolve_task_runner_class(config):")
        source = source.replace("@hydra.main(", helper + "\n@hydra.main(")
        source = source.replace(
            "task_runner_class=FullyAsyncTaskRunner",
            "task_runner_class=_resolve_task_runner_class(config)",
        )
    entry.write_text(source, encoding="utf-8")
    return entry, config


def test_verl_bridge_automatic_fallback_and_idempotence(tmp_path):
    tool = _load_bridge_installer()
    entry, config = _fake_verl_entry(tmp_path)
    # No .git checkout: force surgical fallback while reusing the .patch contract.
    changed = tool.ensure_current_verl_bridge(entry_path=entry)
    assert changed is True
    py = entry.read_text()
    yaml = config.read_text()
    assert py.count("def _resolve_task_runner_class(config):") == 1
    assert "task_runner_class=_resolve_task_runner_class(config)" in py
    assert "multitask:\n" in yaml.replace("\\\n", "\n")
    assert "  enabled: false" in yaml
    assert list(entry.parent.glob("*.mtbridge-*.bak"))
    assert tool.ensure_current_verl_bridge(entry_path=entry) is False
    assert entry.read_text() == py
    assert config.read_text() == yaml


def test_verl_bridge_partial_install_only_adds_missing_config(tmp_path):
    tool = _load_bridge_installer()
    entry, config = _fake_verl_entry(tmp_path, with_bridge=True)
    before = entry.read_text()
    assert tool.ensure_current_verl_bridge(entry_path=entry) is True
    assert entry.read_text() == before
    assert "multitask:" in config.read_text()


def test_verl_bridge_preserves_custom_runner_and_check_only(tmp_path):
    tool = _load_bridge_installer()
    entry, config = _fake_verl_entry(tmp_path, custom_runner=True)
    before = (entry.read_bytes(), config.read_bytes())
    import pytest

    with pytest.raises(tool.BridgeSetupError, match="custom run_ppo"):
        tool.ensure_current_verl_bridge(entry_path=entry)
    assert (entry.read_bytes(), config.read_bytes()) == before

    entry, config = _fake_verl_entry(tmp_path)
    before = (entry.read_bytes(), config.read_bytes())
    with pytest.raises(tool.BridgeSetupError, match="rerun without --check-only"):
        tool.ensure_current_verl_bridge(entry_path=entry, check_only=True)
    assert (entry.read_bytes(), config.read_bytes()) == before


def test_two_real_jobs_launches_bridge_setup_before_importing_verl():
    path = E2E / "verify_two_verl_jobs.py"
    source = path.read_text()
    setup = source.index("ensure_current_verl_bridge(check_only=a.no_auto_bridge)")
    imported = source.index("import verl.experimental.fully_async_policy.fully_async_main as entry")
    assert setup < imported
    assert "--no-auto-bridge" in source
