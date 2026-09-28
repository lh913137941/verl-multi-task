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
        "multitask": {"enabled": True, "runtime": {"profile": PROFILE_ID}},
        "actor_rollout_ref": {
            "hybrid_engine": False,
            "rollout": {
                "mode": "async",
                "name": "vllm",
                "calculate_log_probs": True,
                "enable_sleep_mode": True,
                "free_cache_engine": True,
                "nnodes": 1,
                "n_gpus_per_node": 8,
                "tensor_model_parallel_size": 1,
                "data_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "disaggregation": {"enabled": False},
                "checkpoint_engine": {
                    "backend": "nccl",
                    "engine_kwargs": {"nccl": {"rebuild_group": True}},
                },
            },
        },
        "rollout": {"nnodes": 1, "n_gpus_per_node": 8},
        "trainer": {"device": "cuda"},
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
        {"multitask": {"runtime": {"profile": PROFILE_ID}}},
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
        ("actor_rollout_ref.rollout.enable_sleep_mode", False),
        ("actor_rollout_ref.rollout.free_cache_engine", False),
        ("trainer.device", "npu"),
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


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_unverified_tensor_parallelism_is_rejected(tp):
    current = config()
    current["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] = tp
    with pytest.raises(ProfileConfigurationError, match="tensor_model_parallel_size=1"):
        validate_runtime_profile(current)


def test_router_plugin_is_rejected_in_first_release():
    current = config()
    current["actor_rollout_ref"]["rollout"]["router_config_path"] = "router.yaml"
    with pytest.raises(ProfileConfigurationError, match="router_config_path"):
        validate_runtime_profile(current)


def test_level2_sleep_profile_rejects_mtp_rollout():
    current = config()
    current["actor_rollout_ref"]["rollout"]["mtp"] = {
        "enable": True,
        "enable_rollout": True,
    }
    with pytest.raises(ProfileConfigurationError, match="MTP rollout"):
        validate_runtime_profile(current)


def test_level2_sleep_profile_allows_mtp_loaded_but_not_used_for_rollout():
    current = config()
    current["actor_rollout_ref"]["rollout"]["mtp"] = {
        "enable": True,
        "enable_rollout": False,
    }
    assert validate_runtime_profile(current)


@pytest.mark.parametrize(
    "model",
    [
        {"lora_rank": 8},
        {"lora": {"rank": 8}},
    ],
)
def test_level2_sleep_profile_rejects_lora_rollout(model):
    current = config()
    current["actor_rollout_ref"]["model"] = model
    with pytest.raises(ProfileConfigurationError, match="unmerged LoRA rollout"):
        validate_runtime_profile(current)


@pytest.mark.parametrize(
    "model",
    [
        {"lora_rank": 8, "lora": {"merge": True}},
        {"lora": {"rank": 8, "merge": True}},
    ],
)
def test_level2_sleep_profile_allows_merged_lora(model):
    current = config()
    current["actor_rollout_ref"]["model"] = model
    assert validate_runtime_profile(current)


@pytest.mark.parametrize("backend", ["naive", "hccl", "nixl"])
def test_only_nccl_checkpoint_engine_is_in_first_release(backend):
    current = config()
    current["actor_rollout_ref"]["rollout"]["checkpoint_engine"]["backend"] = backend
    with pytest.raises(ProfileConfigurationError, match="backend='nccl'"):
        validate_runtime_profile(current)


@pytest.mark.parametrize("value", [None, False, "true", 1])
def test_dynamic_collective_membership_requires_explicit_rebuild(value):
    current = config()
    ce = current["actor_rollout_ref"]["rollout"]["checkpoint_engine"]
    ce["engine_kwargs"] = {"nccl": {"rebuild_group": value}}
    with pytest.raises(ProfileConfigurationError, match="rebuild_group"):
        validate_runtime_profile(current)


def test_collective_rebuild_missing_is_rejected():
    current = config()
    ce = current["actor_rollout_ref"]["rollout"]["checkpoint_engine"]
    ce.pop("engine_kwargs", None)
    with pytest.raises(ProfileConfigurationError, match="rebuild_group"):
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
