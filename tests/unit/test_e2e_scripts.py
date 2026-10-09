"""Static acceptance-runner validation for CI without Ray/VERL."""
import ast
import os
import re
import subprocess
from pathlib import Path
import pytest
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

def _load_ray_connector():
    """Test Ray connection logic without requiring Ray on GitHub Actions."""
    source = E2E.joinpath("verify_two_verl_jobs.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef)
        and n.name == "connect_ray"
    )
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "ray_connector", "exec"), namespace)
    return namespace["connect_ray"]


def test_two_real_jobs_missing_ray_is_blocked_with_one_command_hint():
    import pytest
    connect = _load_ray_connector()

    class MissingRay:
        def init(self, **kwargs):
            raise ConnectionError("no running Ray instance")

    with pytest.raises(RuntimeError, match="--start-local-ray"):
        connect(MissingRay(), address="auto", namespace="multitask-jobs")
    with pytest.raises(ValueError, match="requires --ray-address auto"):
        connect(MissingRay(), address="ray-head:6379", namespace="multitask-jobs", start_local=True)


def test_two_real_jobs_optional_local_ray_passes_real_address_to_children():
    connect = _load_ray_connector()

    class LocalRay:
        def __init__(self, resources):
            self.calls = []
            self.resources = resources

        def init(self, **kwargs):
            self.calls.append(kwargs["address"])
            if kwargs["address"] == "auto":
                raise ConnectionError("no running Ray instance")
            return type("LocalContext", (), {"address_info": {"address": "127.0.0.1:6379"}})()

        def get_runtime_context(self):
            return type("Context", (), {"gcs_address": "127.0.0.1:6379"})()

        def cluster_resources(self):
            return self.resources

    ray = LocalRay({"NPU": 8, "CPU": 32})
    address, started = connect(ray, address="auto", namespace="multitask-jobs", start_local=True)
    assert (address, started) == ("127.0.0.1:6379", True)
    assert ray.calls == ["auto", "local"]
    import pytest

    with pytest.raises(RuntimeError, match="no GPU/NPU resources"):
        connect(LocalRay({"CPU": 32}), address="auto", namespace="multitask-jobs", start_local=True)


def test_two_real_jobs_existing_ray_not_restarted_even_if_local_opt_in():
    connect = _load_ray_connector()

    class ExistingRay:
        def __init__(self):
            self.calls = []

        def init(self, **kwargs):
            self.calls.append(kwargs["address"])
            return object()

    ray = ExistingRay()
    assert connect(ray, address="auto", namespace="multitask-jobs", start_local=True) == ("auto", False)
    assert ray.calls == ["auto"]


def _load_training_inputs_validator():
    """Extract pure preflight functions without importing VERL, Ray or NPU."""
    path = E2E / "verify_two_verl_jobs.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    target = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_effective_override", "validate_training_inputs"}
    ]
    assert len(target) == 2
    ns = {"Path": Path}
    exec(compile(ast.Module(body=target, type_ignores=[]), str(path), "exec"), ns)
    return ns["validate_training_inputs"]


def test_two_real_jobs_inputs_catch_placeholder_before_ray(tmp_path):
    import pytest

    validate = _load_training_inputs_validator()
    model = tmp_path / "real-model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    train = tmp_path / "train.parquet"
    validation = tmp_path / "test.parquet"
    train.touch()
    validation.touch()
    args = [
        "actor_rollout_ref.model.path=/REPLACE_WITH_VERL_REPO_DIR/Qwen3-0.6B",
        f"data.train_files={train}",
        f"data.val_files={validation}",
    ]
    with pytest.raises(ValueError, match="placeholder model"):
        validate(args, role="donor")
    with pytest.raises(ValueError, match="placeholder model"):
        validate(args, role="borrower")

    # An explicit real model override takes precedence without editing fixture.
    validate(args + [f"actor_rollout_ref.model.path={model}"], role="donor")
    with pytest.raises(ValueError, match="lacks config.json"):
        validate(args + [f"actor_rollout_ref.model.path={tmp_path}"], role="donor")
    with pytest.raises(ValueError, match="nonexistent local file"):
        validate(args + [
            f"actor_rollout_ref.model.path={model}",
            f"data.train_files={tmp_path / 'missing.parquet'}",
        ], role="donor")


def test_two_real_jobs_inputs_support_huggingface_id_without_forcing_local_download(tmp_path):
    validate = _load_training_inputs_validator()
    train = tmp_path / "train.parquet"
    validation = tmp_path / "val.parquet"
    train.touch()
    validation.touch()
    validate([
        "actor_rollout_ref.model.path=Qwen/Qwen3-0.6B",
        f"data.train_files={train}",
        f"data.val_files={validation}",
    ], role="borrower")



def _load_e2e_src_bootstrap():
    """Test parent-process import bootstrap without importing heavy VERL/Ray."""
    import importlib

    source = (E2E / "verify_two_verl_jobs.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    func = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "configure_multitask_import")
    scope = {"os": os, "sys": sys, "Path": Path, "importlib": importlib}
    exec(compile(ast.Module(body=[func], type_ignores=[]),
                 "e2e_src_bootstrap", "exec"), scope)
    return scope["configure_multitask_import"]


def test_two_real_jobs_bootstraps_multitask_source_before_imports(monkeypatch):
    import importlib.util

    setup = _load_e2e_src_bootstrap()
    expected = str((ROOT / "src").resolve())
    previous = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", "/other/python/files")
    monkeypatch.setattr(sys, "path", [x for x in sys.path if x != expected])
    pythonpath = setup(ROOT)
    assert pythonpath == expected + os.pathsep + "/other/python/files"
    assert os.environ["PYTHONPATH"] == pythonpath
    assert sys.path[0] == expected
    spec = importlib.util.find_spec("multi_task_scheduler")
    assert Path(spec.origin).resolve().is_relative_to(ROOT / "src")

    # Running twice must not duplicate the checkout on either path.
    setup(ROOT)
    assert sys.path.count(expected) == 1
    assert os.environ["PYTHONPATH"].split(os.pathsep).count(expected) == 1
    if previous is None:
        monkeypatch.delenv("PYTHONPATH")
    else:
        monkeypatch.setenv("PYTHONPATH", previous)


def test_two_real_jobs_bootstrap_rejects_wrong_checkout(tmp_path):
    import pytest

    setup = _load_e2e_src_bootstrap()
    with pytest.raises(ValueError, match="MultiTask source checkout is missing"):
        setup(tmp_path)


def test_two_real_jobs_propagates_multitask_source_to_ray_runtime_env():
    connect = _load_ray_connector()

    class ExistingRay:
        def __init__(self):
            self.calls = []

        def init(self, **kwargs):
            self.calls.append(kwargs)
            return object()

    ray = ExistingRay()
    import_path = "/mounted/checkout/src:/global/site-packages"
    assert connect(ray, address="auto", namespace="multitask-jobs",
                   pythonpath=import_path) == ("auto", False)
    assert ray.calls[0]["runtime_env"] == {
        "env_vars": {"PYTHONPATH": import_path}
    }


def test_two_real_jobs_runs_source_bootstrap_before_verl_or_ray_imports():
    code = (E2E / "verify_two_verl_jobs.py").read_text(encoding="utf-8")
    assert code.index("pythonpath = configure_multitask_import(repo)") < code.index(
        "import ray"
    )
    assert code.index("pythonpath = configure_multitask_import(repo)") < code.index(
        "from multi_task_scheduler.scheduler.discovery import get_or_create_group_scheduler"
    )


def test_two_real_jobs_backend_overrides_keep_npu_hccl_and_cuda_nccl_separate():
    """Regression: the NPU path must not invoke native HCCL.destroyComm."""
    import pytest

    source = (E2E / "verify_two_verl_jobs.py").read_text(encoding="utf-8")
    parsed = ast.parse(source)
    fn = next(node for node in parsed.body if isinstance(node, ast.FunctionDef)
              and node.name == "checkpoint_backend_overrides")
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "checkpoint_overrides", "exec"), namespace)
    overrides = namespace["checkpoint_backend_overrides"]
    npu = overrides("npu")
    cuda = overrides("cuda")
    assert "actor_rollout_ref.rollout.checkpoint_engine.backend=multitask_hccl" in npu
    assert "++actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.multitask_hccl.rebuild_group=true" in npu
    assert "++actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=multi_task_scheduler.checkpoint.hccl_checkpoint_engine" in npu
    assert "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl" in cuda
    assert "++actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group=true" in cuda
    assert not any("custom_backend_module" in option for option in cuda)
    with pytest.raises(ValueError, match="Unsupported checkpoint"):
        overrides("cpu")

    # The actual E2E fixed override list must use the tested helper.
    assert '*checkpoint_backend_overrides(devices["donor"])' in source



def test_control_plane_and_exactly_once_attach_existing_ray_without_cpu_override(
    tmp_path, monkeypatch
):
    """Both independent validation drivers must reuse the parent's Ray GCS.

    A real Ray driver rejects num_cpus when RAY_ADDRESS already points to a
    running cluster. Assert argument shape with a fake init that stops startup.
    """
    import importlib.util
    import json
    from types import ModuleType, SimpleNamespace

    for script_name, expected_exit in (
        ("control_plane_recovery.py", 2),
        ("exactly_once_driver.py", 1),
    ):
        for address in (None, "10.170.27.158:45860"):
            module_name = f"_test_ray_attach_{script_name.replace('.', '_')}"
            spec = importlib.util.spec_from_file_location(module_name, E2E / script_name)
            driver = importlib.util.module_from_spec(spec)
            monkeypatch.setitem(sys.modules, module_name, driver)
            spec.loader.exec_module(driver)

            observed = []
            fake_ray = ModuleType("ray")
            fake_ray.is_initialized = lambda: False

            def fake_init(**kwargs):
                observed.append(kwargs)
                raise RuntimeError("simulated Ray connect stopped before actors")

            fake_ray.init = fake_init
            monkeypatch.setitem(sys.modules, "ray", fake_ray)

            if script_name == "control_plane_recovery.py":
                # The test stops in ray.init: no real Ray or scheduler is needed.
                scheduler_module = ModuleType(
                    "multi_task_scheduler.scheduler.group_scheduler"
                )
                scheduler_module.GroupScheduler = object
                monkeypatch.setitem(sys.modules, scheduler_module.__name__, scheduler_module)
                # Contracts may have been imported by other tests, or may be a
                # lightweight stub if the package is not installed.
                contracts_name = "multi_task_scheduler.orchestration.contracts"
                if contracts_name not in sys.modules:
                    contracts = ModuleType(contracts_name)
                    for symbol in (
                        "EvidenceType", "Lease", "OperationCommand",
                        "OperationEvidence", "OperationKind", "OperationRecord",
                        "OperationStatus", "ReplicaKey", "ReplicaKind",
                    ):
                        setattr(contracts, symbol, object)
                    monkeypatch.setitem(sys.modules, contracts_name, contracts)
            else:
                queue_module = ModuleType(
                    "multi_task_scheduler.integration.verl.experimental_fully_async.message_queue"
                )
                queue_module.MultiTaskMessageQueue = object
                monkeypatch.setitem(sys.modules, queue_module.__name__, queue_module)

            if address:
                monkeypatch.setenv("RAY_ADDRESS", address)
                monkeypatch.setenv("PYTHONPATH", "/mounted/checkout/src")
            else:
                monkeypatch.delenv("RAY_ADDRESS", raising=False)
                monkeypatch.delenv("PYTHONPATH", raising=False)

            output = tmp_path / f"{script_name}-{address or 'local'}.json"
            monkeypatch.setattr(sys, "argv", [
                script_name, "--result-file", str(output)
            ])
            if script_name == "control_plane_recovery.py":
                # Avoid real stale timeout; Ray initialization fails earlier.
                sys.argv.extend(["--stale-wait-s", "0.1"])

            assert driver.main() == expected_exit
            assert len(observed) == 1
            kwargs = observed[0]
            if address:
                assert kwargs["address"] == address
                assert "num_cpus" not in kwargs
                assert kwargs["runtime_env"] == {
                    "env_vars": {"PYTHONPATH": "/mounted/checkout/src"}
                }
            else:
                assert kwargs["address"] == "local"
                assert kwargs["num_cpus"] == (
                    2 if script_name == "control_plane_recovery.py" else 3
                )
                assert "runtime_env" not in kwargs
            record = json.loads(output.read_text(encoding="utf-8"))
            assert record["last_stage"] == "ray_init" if script_name == "exactly_once_driver.py" else record["state"] == "BLOCKED"
            assert "simulated Ray connect" in record["detail"]



# --- test_e2e_cleanup.py (consolidated boundary scenarios) ---

_CLEANUP_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = _CLEANUP_ROOT / "scripts" / "e2e"


def test_only_one_supported_two_job_entrypoint():
    for obsolete in ("verify_cluster.sh", "verify_quick.sh", "list_tasks.py"):
        assert not (SCRIPTS / obsolete).exists()

    runner = (SCRIPTS / "verify_two_verl_jobs.py").read_text(encoding="utf-8")
    assert 'scripts/e2e/run_all.sh' in runner
    assert '--scenarios' in runner
    assert '--require-inflight-force' in runner
    # Keep manual proven fixture support, but no interactive Ray-PG editing.
    assert '--lease' in runner
    assert '--auto-lease' not in runner
    assert '--interactive-lease' not in runner
    assert 'input("Press Enter' not in runner


def test_all_five_validate_scripts_still_reachable_from_real_two_job_runner():
    dispatch = (SCRIPTS / "run_all.sh").read_text(encoding="utf-8")
    expected = {
        "control_plane": ("validate_control_plane.sh", "control_plane_recovery.py"),
        "exactly_once": ("validate_exactly_once.sh", "exactly_once_driver.py"),
        "recovery": ("validate_recovery_faults.sh", "test_taskrunner_wiring.py"),
        "lifecycle": ("validate_lifecycle_cycle.sh", "lifecycle_driver.py"),
        "force": ("validate_force_remove.sh", "lifecycle_driver.py"),
    }
    common = (SCRIPTS / "common.sh").read_text(encoding="utf-8")
    for name, (wrapper, target) in expected.items():
        assert re.search(
            rf"\b{re.escape(name)}\) script=.*{re.escape(wrapper)}",
            dispatch,
        ), (name, wrapper)
        assert (SCRIPTS / wrapper).is_file()
        assert target in (
            (SCRIPTS / wrapper).read_text(encoding="utf-8") + common
        )
    assert 'MT_E2E_SCENARIOS' in dispatch
    assert 'MT_E2E_REQUIRE_COMPLETE' in dispatch

    # This scenario invokes named unit tests rather than the entire suite:
    # moving a test must update its selector, not silently lose recovery proof.
    recovery = (SCRIPTS / "validate_recovery_faults.sh").read_text(encoding="utf-8")
    selectors = re.findall(
        r'"(tests/unit/test_[^"]+\.py)::(test_\w+)"', recovery
    )
    assert len(selectors) == 9
    for relative_path, test_name in selectors:
        test_file = _CLEANUP_ROOT / relative_path
        assert test_file.is_file(), relative_path
        module = ast.parse(test_file.read_text(encoding="utf-8"))
        assert any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == test_name
            for node in module.body
        ), f"stale recovery selector: {relative_path}::{test_name}"


@pytest.mark.parametrize(
    "script", ["validate_lifecycle_cycle.sh", "validate_force_remove.sh"]
)
def test_real_lifecycle_rejects_standalone_without_running_donor_and_borrower(script):
    env = dict(os.environ)
    env.pop("MT_E2E_ATTACH_ONLY", None)
    env.pop("MT_E2E_LEASE_FILE", None)
    env.pop("MULTITASK_LAUNCH_SCRIPT", None)
    result = subprocess.run(
        ["bash", str(SCRIPTS / script)],
        cwd=_CLEANUP_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert "BLOCKED" in result.stderr
    assert "verify_two_verl_jobs.py" in result.stderr


def test_strict_force_receipt_validation_has_one_owner():
    wrapper = (SCRIPTS / "validate_force_remove.sh").read_text(encoding="utf-8")
    runner = (SCRIPTS / "verify_two_verl_jobs.py").read_text(encoding="utf-8")
    assert "MULTITASK_FORCE_HANDOFF" not in wrapper
    assert "MT_E2E_REQUIRE_INFLIGHT_FORCE" not in wrapper
    assert "MULTITASK_FORCE_HANDOFF" in runner
    assert "force_cycle" in runner
    assert "aborted_count" in runner
    assert "confirmed_count" in runner
    assert "MULTITASK_LAUNCH_SCRIPT" not in (
        (SCRIPTS / "common.sh").read_text(encoding="utf-8")
    )
