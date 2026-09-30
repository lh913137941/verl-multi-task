#!/usr/bin/env python3
"""List MultiTask TaskRunner sessions attached to the shared GroupScheduler."""

import os

import ray

from multi_task_scheduler.scheduler.discovery import (
    GROUP_SCHEDULER_NAME,
    GROUP_SCHEDULER_NAMESPACE,
)


if not ray.is_initialized():
    ray.init(
        address=os.environ.get("RAY_ADDRESS", "auto"),
        ignore_reinit_error=True,
        log_to_driver=False,
    )

gs = ray.get_actor(
    GROUP_SCHEDULER_NAME,
    namespace=GROUP_SCHEDULER_NAMESPACE,
)
runners = ray.get(gs.get_task_runners.remote(), timeout=10)
for task_session in sorted(runners):
    print(task_session)
