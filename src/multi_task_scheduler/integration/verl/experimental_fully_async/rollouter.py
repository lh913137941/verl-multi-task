"""Experimental Fully Async Rollouter with the current lifecycle surface.

Native generation/queue behavior stays inherited. Device/runtime actions remain
explicit failures until a verified native backend is available. Rollouter owns
ProductionWindow facts; LB owns idle-candidate construction and reporting.
"""

import ray

from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncAgentLoopManager,
    FullyAsyncRollouter,
)
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.contracts import ServiceAction
from multi_task_scheduler.orchestration.production_window import ProductionWindow

from .llm_server_manager import MultiTaskLLMServerManager


@ray.remote(num_cpus=10, max_concurrency=100)
class MultiTaskFullyAsyncRollouter(unwrap_native_actor_class(FullyAsyncRollouter)):
    """Own production/capacity/lifecycle coordination for the task."""

    def __init__(self, config, tokenizer, processor=None, device_name=None, *, group_scheduler=None):
        self.group_scheduler = group_scheduler
        self._production_window = None
        super().__init__(config, tokenizer, processor=processor, device_name=device_name)

    async def _init_async_rollout_manager(self):
        enable_agent_reward_loop = (
            not self.use_rm or self.config.reward.reward_model.enable_resource_pool
        )
        reward_loop_worker_handles = (
            self.reward_loop_manager.reward_loop_workers
            if enable_agent_reward_loop
            else None
        )
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        self.async_rollout_mode = True
        self.llm_server_manager = await MultiTaskLLMServerManager.create(
            config=self.config,
            worker_group=self.get_hybrid_worker_group(),
            group_scheduler=self.group_scheduler,
        )
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(
                client_cls=FullyAsyncLLMServerClient
            ),
            reward_loop_worker_handles=reward_loop_worker_handles,
            teacher_client=(
                self.teacher_model_manager.get_client()
                if self.teacher_model_manager
                else None
            ),
        )

    @property
    def production_window(self) -> ProductionWindow | None:
        """Return the latest task-session production fact, if one is established."""
        return self._production_window

    def set_production_window(self, window: ProductionWindow) -> None:
        if not isinstance(window, ProductionWindow):
            raise TypeError("set_production_window requires ProductionWindow")
        current = self._production_window
        if current is not None:
            if window.task_session != current.task_session:
                raise ValueError("production window task_session cannot change in-place")
            if window.epoch < current.epoch:
                raise ValueError("production window epoch cannot move backwards")
            if window.epoch == current.epoch and window.revision < current.revision:
                raise ValueError("production window revision cannot move backwards")
        self._production_window = window

    def prepare_replica(self, ctx, key, placement):
        """Hidden-create one borrowed runtime without publishing service."""
        raise NotImplementedError("prepare_replica requires verified native backend")

    def prepare_exit(self, command):
        """Unified DONATE/REMOVE drain and verified continuation coordinator."""
        raise NotImplementedError("prepare_exit requires verified native backend")

    def commit_service_change(
        self,
        ctx,
        action: ServiceAction,
        prerequisite,
        ce_commit,
        prepared=None,
    ):
        """Canonical ADD/REMOVE service commit; caller must hold Trainer G.

        REMOVE performs the required owner-side exit revalidation internally;
        it is intentionally not exposed as a second coordination API.
        """
        action = ServiceAction(action)
        raise NotImplementedError(
            f"commit_service_change({action.value}) requires verified owner commit wiring"
        )

    def finalize_release(self, ctx, service):
        """Sleep native or destroy borrowed only after ServiceEvidence(REMOVE)."""
        raise NotImplementedError("finalize_release requires verified runtime release backend")

    def query_phase(self, ctx, phase):
        """Return owner-backed phase facts once owner-side journals are wired."""
        raise NotImplementedError("query_phase requires owner-side journal wiring")
