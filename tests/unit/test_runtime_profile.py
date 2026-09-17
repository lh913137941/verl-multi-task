"""Validate configuration without importing native GPU/backend modules."""

import copy
import sys

import pytest
from omegaconf import OmegaConf

from multi_task_scheduler.integration.verl.runtime_profile import (
    PROFILE_ID, ProfileConfigurationError, resolve_runtime_profile, validate_runtime_profile,
)


@pytest.fixture
def config():
    return {
        "multitask": {"runtime": {"profile": PROFILE_ID}},
        "actor_rollout_ref": {
            "hybrid_engine": False,
            "rollout": {
                "mode": "async", "name": "vllm", "calculate_log_probs": True,
                "nnodes": 2, "n_gpus_per_node": 8,
                "tensor_model_parallel_size": 4, "data_parallel_size": 1,
                "pipeline_model_parallel_size": 1, "disaggregation": {"enabled": False},
                "checkpoint_engine": {"backend": "nccl"},
            },
        },
        "rollout": {"nnodes": 2, "n_gpus_per_node": 8},
        "async_training": {"use_trainer_do_validate": False, "use_dynamic_resource_scheduling": False},
        "data": {"train_batch_size": 0, "gen_batch_size": 1},
    }


@pytest.mark.parametrize("disabled", [
    {}, {"multitask": None}, {"multitask": {"runtime": None}},
    {"multitask": {"runtime": {"profile": None}}},
    {"multitask": {"enabled": False}},
    {"multitask": {"enabled": False, "runtime": {"profile": PROFILE_ID}}},
    {"multitask": {"enabled": False, "runtime": {"profile": "unknown"}}},
])
def test_disabled_resolution_has_no_runtime_import(disabled):
    before = set(sys.modules)
    assert resolve_runtime_profile(disabled) is None
    assert not any(name == "ray" or name.startswith("verl.") for name in set(sys.modules) - before)


@pytest.mark.parametrize("value", ["", "unknown", False, 123, [], {}])
def test_invalid_selection_fails_before_runtime_import(config, value):
    config["multitask"]["runtime"]["profile"] = value
    with pytest.raises(ProfileConfigurationError):
        resolve_runtime_profile(config)


@pytest.mark.parametrize("use_omegaconf", [False, True])
def test_valid_configuration_is_not_mutated(config, use_omegaconf):
    before = copy.deepcopy(config)
    selected = OmegaConf.create(config) if use_omegaconf else config
    assert validate_runtime_profile(selected)
    assert (OmegaConf.to_container(selected) if use_omegaconf else selected) == before


@pytest.mark.parametrize("runtime", [None, {}, {"profile": None}, {"profile": PROFILE_ID}])
def test_enabled_switch_selects_supported_profile_without_mutating_config(config, runtime):
    config["multitask"] = {"enabled": True, "runtime": runtime}
    selected = OmegaConf.create(config)
    before = OmegaConf.to_container(selected)
    assert validate_runtime_profile(selected) is True
    assert OmegaConf.to_container(selected) == before


@pytest.mark.parametrize("value", [None, "true", "false", 0, 1, [], {}])
def test_switch_requires_a_boolean_even_with_a_valid_profile(config, value):
    config["multitask"]["enabled"] = value
    with pytest.raises(ProfileConfigurationError, match="multitask.enabled"):
        resolve_runtime_profile(config)


@pytest.mark.parametrize("path,value", [
    ("actor_rollout_ref.hybrid_engine", True),
    ("async_training.use_trainer_do_validate", True),
    ("async_training.use_dynamic_resource_scheduling", True),
    ("actor_rollout_ref.rollout.name", "sglang"),
    ("actor_rollout_ref.rollout.disaggregation.enabled", True),
    ("actor_rollout_ref.rollout.mode", "sync"),
    ("actor_rollout_ref.rollout.checkpoint_engine.backend", "naive"),
    ("actor_rollout_ref.rollout.calculate_log_probs", False),
    ("data.train_batch_size", 1),
    ("data.gen_batch_size", True),
    ("actor_rollout_ref.rollout.tensor_model_parallel_size", 17),
    ("actor_rollout_ref.rollout.data_parallel_size", 0),
    ("actor_rollout_ref.rollout.nnodes", 3),
    ("rollout.n_gpus_per_node", True),
    ("multitask.runtime.task_runner_class", "custom.Class"),
    ("multitask.rollout_deployment", "standalone"),
])
def test_conflicting_configuration_is_rejected(config, path, value):
    target = config
    keys = path.split(".")
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(ProfileConfigurationError):
        validate_runtime_profile(config)


def test_resource_remainder_does_not_invent_native_divisibility_rules(config):
    config["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] = 6
    assert validate_runtime_profile(config)


def test_partial_flag_is_native_behavior_not_another_profile(config):
    for partial in (False, True):
        config["async_training"]["partial_rollout"] = partial
        assert validate_runtime_profile(config)


def test_missing_required_field_is_an_explicit_error(config):
    del config["actor_rollout_ref"]["rollout"]["checkpoint_engine"]
    with pytest.raises(ProfileConfigurationError, match="Missing configuration"):
        validate_runtime_profile(config)


@pytest.mark.parametrize("malformed", [{"multitask": "x"}, {"multitask": {"runtime": "x"}}, None])
def test_malformed_parent_sections_are_not_silently_disabled(malformed):
    with pytest.raises(ProfileConfigurationError):
        resolve_runtime_profile(malformed)
