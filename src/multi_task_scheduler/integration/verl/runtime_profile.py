"""One explicit runtime selection; disabled imports never load Ray or verl."""

from collections.abc import Mapping

PROFILE_ID = "experimental_fully_async_standalone"
_MISSING = object()


class ProfileConfigurationError(ValueError):
    """The requested profile is outside the first-release verified boundary."""


def _select(config, path, default=_MISSING):
    value = config
    for key in path.split("."):
        if value is None and default is not _MISSING:
            return default
        if not isinstance(value, Mapping):
            raise ProfileConfigurationError(
                f"{path}: parent of {key} must be a mapping"
            )
        if key not in value:
            if default is _MISSING:
                raise ProfileConfigurationError(f"Missing configuration: {path}")
            return default
        value = value[key]
    return value


def validate_runtime_profile(config) -> bool:
    if not isinstance(config, Mapping):
        raise ProfileConfigurationError("Configuration must be a mapping")
    multitask = _select(config, "multitask", None)
    if multitask is None:
        return False
    if not isinstance(multitask, Mapping):
        raise ProfileConfigurationError("multitask must be a mapping")
    enabled = multitask.get("enabled", False)
    if "enabled" in multitask:
        if type(enabled) is not bool:
            raise ProfileConfigurationError("multitask.enabled must be a boolean")
        if not enabled:
            return False
    profile = _select(config, "multitask.runtime.profile", None)
    if profile is None:
        if not enabled:
            return False
        profile = PROFILE_ID
    if not isinstance(profile, str) or profile != PROFILE_ID:
        raise ProfileConfigurationError(
            f"Unknown multitask.runtime.profile: {profile!r}"
        )
    runtime = _select(config, "multitask.runtime", None)
    if runtime is not None and set(runtime) - {"profile"}:
        raise ProfileConfigurationError("multitask.runtime accepts only profile")
    if _select(config, "multitask.rollout_deployment", None) is not None:
        raise ProfileConfigurationError(
            "Use only multitask.runtime.profile to select the deployment"
        )

    for path, expected in (
        ("actor_rollout_ref.hybrid_engine", False),
        ("async_training.use_trainer_do_validate", False),
        ("async_training.use_dynamic_resource_scheduling", False),
        ("actor_rollout_ref.rollout.mode", "async"),
        ("actor_rollout_ref.rollout.name", "vllm"),
        ("actor_rollout_ref.rollout.calculate_log_probs", True),
        ("data.train_batch_size", 0),
        ("data.gen_batch_size", 1),
    ):
        value = _select(config, path)
        if type(value) is not type(expected) or value != expected:
            raise ProfileConfigurationError(
                f"{path} must be {expected!r}, got {value!r}"
            )

    prefix = "actor_rollout_ref.rollout"
    disaggregation = _select(config, f"{prefix}.disaggregation", None)
    if disaggregation is not None:
        if (
            not isinstance(disaggregation, Mapping)
            or disaggregation.get("enabled", False) is not False
        ):
            raise ProfileConfigurationError(
                "first release does not support PD/disaggregated rollout"
            )

    router_config_path = _select(config, f"{prefix}.router_config_path", None)
    if router_config_path not in (None, ""):
        raise ProfileConfigurationError(
            "first release owns the request-state LB and does not support router_config_path"
        )

    backend = _select(config, f"{prefix}.checkpoint_engine.backend")
    if not isinstance(backend, str) or not backend.strip() or backend == "naive":
        raise ProfileConfigurationError(
            "pure STANDALONE requires a non-naive checkpoint engine"
        )

    sizes = {}
    for field in (
        "nnodes",
        "n_gpus_per_node",
        "tensor_model_parallel_size",
        "data_parallel_size",
        "pipeline_model_parallel_size",
    ):
        value = _select(config, f"{prefix}.{field}")
        if type(value) is not int or value <= 0:
            raise ProfileConfigurationError(
                f"{prefix}.{field} must be a positive integer"
            )
        sizes[field] = value

    if sizes["nnodes"] != 1:
        raise ProfileConfigurationError(
            "first release supports single-node rollout only"
        )
    if sizes["data_parallel_size"] != 1:
        raise ProfileConfigurationError(
            "first release requires data_parallel_size=1"
        )
    if sizes["pipeline_model_parallel_size"] != 1:
        raise ProfileConfigurationError(
            "first release requires pipeline_model_parallel_size=1"
        )

    for field in ("nnodes", "n_gpus_per_node"):
        value = _select(config, f"rollout.{field}")
        if type(value) is not int or value != sizes[field]:
            raise ProfileConfigurationError(
                f"Native main must map rollout.{field} before selecting MultiTask"
            )

    if sizes["tensor_model_parallel_size"] > sizes["n_gpus_per_node"]:
        raise ProfileConfigurationError(
            "TP replica must fit on the single supported node"
        )
    return True


def resolve_runtime_profile(config):
    if not validate_runtime_profile(config):
        return None
    from .experimental_fully_async.task_runner import MultiTaskFullyAsyncTaskRunner

    return MultiTaskFullyAsyncTaskRunner
