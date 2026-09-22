import copy

import pytest

from multi_task_scheduler.integration.verl.runtime_profile import (
    PROFILE_ID,
    ProfileConfigurationError,
    resolve_runtime_profile,
    validate_runtime_profile,
)


def config():
    return {
        "multitask": {"runtime": {"profile": PROFILE_ID}},
        "actor_rollout_ref": {
            "hybrid_engine": False,
            "rollout": {
                "mode": "async",
                "name": "vllm",
                "calculate_log_probs": True,
                "nnodes": 1,
                "n_gpus_per_node": 8,
                "tensor_model_parallel_size": 4,
                "data_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "disaggregation": {"enabled": False},
                "checkpoint_engine": {"backend": "nccl"},
            },
        },
        "rollout": {"nnodes": 1, "n_gpus_per_node": 8},
        "async_training": {
            "use_trainer_do_validate": False,
            "use_dynamic_resource_scheduling": False,
        },
        "data": {"train_batch_size": 0, "gen_batch_size": 1},
    }


def test_valid_profile_is_not_mutated():
    current = config()
    before = copy.deepcopy(current)
    assert validate_runtime_profile(current)
    assert current == before


@pytest.mark.parametrize(
    "disabled",
    [
        {},
        {"multitask": None},
        {"multitask": {"runtime": None}},
        {"multitask": {"runtime": {"profile": None}}},
        {"multitask": {"enabled": False}},
        {
            "multitask": {
                "enabled": False,
                "runtime": {"profile": PROFILE_ID},
            }
        },
    ],
)
def test_disabled_resolution_does_not_import_runtime(disabled):
    assert resolve_runtime_profile(disabled) is None


@pytest.mark.parametrize("value", [None, "true", "false", 0, 1, [], {}])
def test_enabled_switch_requires_boolean(value):
    current = config()
    current["multitask"]["enabled"] = value
    with pytest.raises(ProfileConfigurationError, match="multitask.enabled"):
        validate_runtime_profile(current)


@pytest.mark.parametrize(
    "path,value",
    [
        ("actor_rollout_ref.hybrid_engine", True),
        ("async_training.use_trainer_do_validate", True),
        ("async_training.use_dynamic_resource_scheduling", True),
        ("actor_rollout_ref.rollout.mode", "sync"),
        ("actor_rollout_ref.rollout.name", "sglang"),
        ("actor_rollout_ref.rollout.calculate_log_probs", False),
        ("data.train_batch_size", 1),
        ("data.gen_batch_size", 2),
    ],
)
def test_conflicting_profile_configuration_is_rejected(path, value):
    current = config()
    target = current
    keys = path.split(".")
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(ProfileConfigurationError):
        validate_runtime_profile(current)


@pytest.mark.parametrize(
    "field,value",
    [
        ("nnodes", 2),
        ("data_parallel_size", 2),
        ("pipeline_model_parallel_size", 2),
    ],
)
def test_first_release_scope_rejects_cross_node_dp_pp(field, value):
    current = config()
    current["actor_rollout_ref"]["rollout"][field] = value
    if field == "nnodes":
        current["rollout"]["nnodes"] = value
    with pytest.raises(ProfileConfigurationError):
        validate_runtime_profile(current)


def test_tp_must_fit_single_node():
    current = config()
    current["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] = 9
    with pytest.raises(ProfileConfigurationError, match="TP replica"):
        validate_runtime_profile(current)


def test_router_plugin_is_rejected_in_first_release():
    current = config()
    current["actor_rollout_ref"]["rollout"]["router_config_path"] = "router.yaml"
    with pytest.raises(ProfileConfigurationError, match="router_config_path"):
        validate_runtime_profile(current)


def test_non_naive_checkpoint_engine_is_required():
    current = config()
    current["actor_rollout_ref"]["rollout"]["checkpoint_engine"]["backend"] = "naive"
    with pytest.raises(ProfileConfigurationError, match="non-naive"):
        validate_runtime_profile(current)


def test_native_main_must_map_rollout_resources_before_selection():
    current = config()
    current["rollout"]["n_gpus_per_node"] = 4
    with pytest.raises(ProfileConfigurationError, match="Native main must map"):
        validate_runtime_profile(current)


def test_unknown_profile_and_extra_runtime_selector_are_rejected():
    current = config()
    current["multitask"]["runtime"]["profile"] = "unknown"
    with pytest.raises(ProfileConfigurationError, match="Unknown"):
        validate_runtime_profile(current)

    current = config()
    current["multitask"]["runtime"]["task_runner_class"] = "custom.Class"
    with pytest.raises(ProfileConfigurationError, match="accepts only profile"):
        validate_runtime_profile(current)


@pytest.mark.parametrize(
    "malformed",
    [
        None,
        {"multitask": "x"},
        {"multitask": {"runtime": "x"}},
    ],
)
def test_malformed_parent_sections_fail_explicitly(malformed):
    with pytest.raises(ProfileConfigurationError):
        resolve_runtime_profile(malformed)
