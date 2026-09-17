"""Entry wiring tests, without importing verl's heavyweight module tree.

The tests execute only the native ``main`` AST with scoped import substitutes.
They prove argument/order forwarding, not real Hydra launch or Ray training.
"""

import ast
import builtins
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import yaml


MAIN = "verl/experimental/fully_async_policy/fully_async_main.py"
CONFIG = "verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml"
# Keep the unmodified upstream baseline stable when the wiring patch is committed.
UPSTREAM_BASELINE = "f92febf50fe3db102273eaf59b1854f392ae761d"
_MISSING = object()


@pytest.fixture
def verl_root():
    selected = os.environ.get("MT_VERL_SOURCE_ROOT")
    assert selected, "Set MT_VERL_SOURCE_ROOT to the explicit native source checkout"
    root = Path(selected)
    assert root.is_absolute() and (root / MAIN).is_file()
    return root


def _main_with_scoped_imports(verl_root, events, resolve, *, import_error=None):
    tree = ast.parse((verl_root / MAIN).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    main.decorator_list = []
    native_actor = object()

    def run_ppo(config, *, task_runner_class):
        events.append(("run_ppo", task_runner_class, config))

    def import_only_entry_dependencies(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "verl.trainer.main_ppo":
            return SimpleNamespace(run_ppo=run_ppo)
        if name == "multi_task_scheduler.integration.verl.runtime_profile":
            events.append(("companion_import",))
            if import_error is not None:
                raise import_error
            return SimpleNamespace(resolve_runtime_profile=resolve)
        if name == "time":
            return builtins.__import__(name, globals, locals, fromlist, level)
        raise AssertionError(f"Unexpected entry dependency: {name}")

    def migrate(config):
        assert config.actor_rollout_ref.rollout.nnodes == config.rollout.nnodes
        assert config.actor_rollout_ref.rollout.n_gpus_per_node == config.rollout.n_gpus_per_node
        events.append(("reward_migration",))
        return config

    scope = {
        "__builtins__": dict(vars(builtins), __import__=import_only_entry_dependencies),
        "OmegaConf": OmegaConf,
        "FullyAsyncTaskRunner": native_actor,
        "auto_set_device": lambda config: events.append(("auto_set_device",)),
        "migrate_legacy_reward_impl": migrate,
    }
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(verl_root / MAIN), "exec"), scope)
    return scope["main"], native_actor


def _config(multitask=_MISSING):
    values = {
        "async_training": {},
        "rollout": {"nnodes": 2, "n_gpus_per_node": 8},
        "actor_rollout_ref": {"rollout": {"nnodes": 0, "n_gpus_per_node": 0}},
    }
    if multitask is not _MISSING:
        values["multitask"] = multitask
    return OmegaConf.create(values)


@pytest.mark.parametrize("multitask", [
    _MISSING, None, {"runtime": None}, {"runtime": {"profile": None}},
    {"enabled": False},
    {"enabled": False, "runtime": {"profile": "experimental_fully_async_standalone"}},
    {"enabled": False, "runtime": {"profile": "unknown"}},
])
def test_disabled_entry_keeps_native_actor_and_never_imports_companion(verl_root, multitask):
    events = []

    def forbidden_resolver(config):
        pytest.fail("Disabled profile must not call the companion resolver")

    main, native_actor = _main_with_scoped_imports(verl_root, events, forbidden_resolver)
    config = _config(multitask)
    main(config)
    assert [event[0] for event in events] == ["auto_set_device", "reward_migration", "run_ppo"]
    assert events[-1][1] is native_actor
    assert events[-1][2] is config


@pytest.mark.parametrize("multitask", ["not-a-mapping", {"runtime": "not-a-mapping"}])
def test_entry_rejects_nonmapping_profile_parents_without_import_or_native_fallback(verl_root, multitask):
    events = []

    def forbidden_resolver(config):
        pytest.fail("Malformed parent configuration must fail before importing the companion")

    main, _ = _main_with_scoped_imports(verl_root, events, forbidden_resolver)
    with pytest.raises(ValueError):
        main(_config(multitask))
    assert not any(event[0] in {"run_ppo", "companion_import"} for event in events)


@pytest.mark.parametrize("multitask", [
    {"enabled": True},
    {"enabled": True, "runtime": {"profile": None}},
    {"runtime": {"profile": "experimental_fully_async_standalone"}},
])
def test_enabled_entry_selects_root_only_after_native_config_mapping(verl_root, multitask):
    events = []
    selected_actor = object()  # Explicit entry-boundary substitute, not an ActorClass.

    def resolve(config):
        assert config.actor_rollout_ref.rollout.nnodes == 2
        assert config.actor_rollout_ref.rollout.n_gpus_per_node == 8
        events.append(("resolve",))
        return selected_actor

    main, _ = _main_with_scoped_imports(verl_root, events, resolve)
    main(_config(multitask))
    assert [event[0] for event in events] == [
        "auto_set_device", "reward_migration", "companion_import", "resolve", "run_ppo"
    ]
    assert events[-1][1] is selected_actor


def test_enabled_entry_propagates_resolution_failure_without_native_fallback(verl_root):
    events = []

    def resolve(config):
        raise RuntimeError("explicit integration failure")

    main, _ = _main_with_scoped_imports(verl_root, events, resolve)
    with pytest.raises(RuntimeError, match="explicit integration failure"):
        main(_config({"runtime": {"profile": "experimental_fully_async_standalone"}}))
    assert not any(event[0] == "run_ppo" for event in events)


@pytest.mark.parametrize("value", [None, "true", "false", 0, 1, [], {}])
def test_invalid_switch_fails_before_import_or_native_launch(verl_root, value):
    events = []
    main, _ = _main_with_scoped_imports(verl_root, events, lambda config: pytest.fail("must not resolve"))
    with pytest.raises(ValueError, match="multitask.enabled"):
        main(_config({"enabled": value}))
    assert not any(event[0] in {"companion_import", "run_ppo"} for event in events)


def test_enabled_entry_does_not_accept_disabled_resolution(verl_root):
    events = []
    main, _ = _main_with_scoped_imports(verl_root, events, lambda config: None)
    with pytest.raises(RuntimeError, match="TaskRunner"):
        main(_config({"enabled": True}))
    assert not any(event[0] == "run_ppo" for event in events)


@pytest.mark.parametrize("missing", ["multi_task_scheduler", "ray", "vllm"])
def test_enabled_entry_reports_missing_package_without_hiding_backend_import_failures(verl_root, missing):
    events = []
    error = ModuleNotFoundError(f"No module named {missing!r}", name=missing)
    main, _ = _main_with_scoped_imports(verl_root, events, None, import_error=error)
    with pytest.raises(ImportError) as raised:
        main(_config({"enabled": True}))
    if missing == "multi_task_scheduler":
        assert "install" in str(raised.value).lower()
        assert raised.value.__cause__ is error
    else:
        assert raised.value is error
    assert not any(event[0] == "run_ppo" for event in events)


def test_native_hydra_config_and_example_reach_the_real_plugin_resolver(verl_root, monkeypatch):
    from multi_task_scheduler.integration.verl.runtime_profile import resolve_runtime_profile

    # Use the actual primary and defaults tree. Only map Hydra's config search to
    # files, avoiding import of GPU-dependent verl.__init__ in this CPU test.
    searchpath = "file://" + (verl_root / "verl/trainer/config").as_posix()
    overrides = [f"hydra.searchpath=['{searchpath}']"]
    with initialize_config_dir(config_dir=str((verl_root / CONFIG).parent), version_base=None):
        disabled = compose(config_name="fully_async_ppo_trainer", overrides=overrides)
        example = Path(__file__).resolve().parents[2] / "examples/experimental_fully_async/native_entry_overrides.txt"
        selected = compose(config_name="fully_async_ppo_trainer", overrides=overrides + [
            line.strip() for line in example.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ])
    assert disabled.multitask.enabled is False
    assert selected.multitask.enabled is True
    events = []
    main, native_actor = _main_with_scoped_imports(verl_root, events, resolve_runtime_profile)
    main(disabled)
    assert events[-1][1] is native_actor
    assert not any(event[0] == "companion_import" for event in events)
    # Replace only the GPU ActorClass import; execute the actual selection and
    # precondition validation on the composed native config.
    selected_actor = object()
    monkeypatch.setitem(sys.modules,
                        "multi_task_scheduler.integration.verl.experimental_fully_async.task_runner",
                        SimpleNamespace(MultiTaskFullyAsyncTaskRunner=selected_actor))
    main(selected)
    assert events[-1][1] is selected_actor


@pytest.mark.parametrize("relative", [
    MAIN,
    "verl/experimental/fully_async_policy/fully_async_trainer.py",
    "verl/experimental/fully_async_policy/fully_async_rollouter.py",
    "verl/workers/rollout/llm_server.py",
    "verl/workers/rollout/replica.py",
    "verl/workers/rollout/vllm_rollout/vllm_async_server.py",
    "verl/checkpoint_engine/base.py",
])
def test_native_class_implementations_are_unchanged_from_upstream_baseline(verl_root, relative):
    committed = subprocess.run(
        ["git", "-C", str(verl_root), "show", f"{UPSTREAM_BASELINE}:{relative}"],
        check=True, capture_output=True, text=True,
    ).stdout

    def classes(source):
        return [ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                if isinstance(node, ast.ClassDef)]

    assert classes((verl_root / relative).read_text()) == classes(committed)


def test_native_primary_adds_only_disabled_plugin_defaults(verl_root):
    committed = subprocess.run(
        ["git", "-C", str(verl_root), "show", f"{UPSTREAM_BASELINE}:{CONFIG}"],
        check=True, capture_output=True, text=True,
    ).stdout
    original = yaml.safe_load(committed)
    current = yaml.safe_load((verl_root / CONFIG).read_text())
    assert current.pop("multitask") == {"enabled": False, "runtime": {"profile": None}}
    assert current == original
