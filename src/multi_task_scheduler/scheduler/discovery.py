"""Discover one shared GroupScheduler after the native driver starts Ray."""

GROUP_SCHEDULER_NAME = "verl-multi-task-group-scheduler"
GROUP_SCHEDULER_NAMESPACE = "verl-multi-task"


def get_or_create_group_scheduler():
    """Return the named detached ActorHandle, including concurrent first callers.

    The fixed namespace lets tasks from different Ray jobs discover the same
    Actor. The Actor survives individual task completion; a cluster operator
    manages its lifetime. Importing this module never starts Ray or the Actor.
    """
    import ray

    from multi_task_scheduler.orchestration.contracts import CONTROL_RPC_TIMEOUT_S

    from .group_scheduler import RUNTIME_KIND, GroupScheduler

    if not ray.is_initialized():
        raise RuntimeError("GroupScheduler discovery requires native Ray initialization first")

    scheduler = GroupScheduler.options(
        name=GROUP_SCHEDULER_NAME,
        namespace=GROUP_SCHEDULER_NAMESPACE,
        lifetime="detached",
        get_if_exists=True,
    ).remote()
    if (
        ray.get(scheduler.runtime_kind.remote(), timeout=CONTROL_RPC_TIMEOUT_S)
        != RUNTIME_KIND
    ):
        raise RuntimeError("The named GroupScheduler belongs to an incompatible runtime")
    return scheduler
