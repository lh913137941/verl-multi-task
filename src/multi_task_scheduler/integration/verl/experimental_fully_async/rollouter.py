"""Experimental Fully Async Rollouter with production-window sensing.

Native generation/queue behavior stays inherited. This binding exposes the
simplified lifecycle coordination surface while keeping device/runtime actions
as explicit failures until a verified native backend is available.
"""

import ray

from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncAgentLoopManager,
    FullyAsyncRollouter,
)
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.production_window import ProductionWindow, select_idle_candidates

from .llm_server_manager import MultiTaskLLMServerManager


@ray.remote(num_cpus=10, max_concurrency=100)
class MultiTaskFullyAsyncRollouter(unwrap_native_actor_class(FullyAsyncRollouter)):
    """Real Ray Actor; lifecycle facts are overlays, native behavior is inherited."""

    def __init__(self, config, tokenizer, processor=None, device_name=None, *, group_scheduler=None):
        self.group_scheduler = group_scheduler
        super().__init__(config, tokenizer, processor=processor, device_name=device_name)

    def _ensure_production_window(self) -> ProductionWindow:
        if not hasattr(self, "_production_window"):
            self._production_window = ProductionWindow()
        return self._production_window

    async def _init_async_rollout_manager(self):
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        self.async_rollout_mode = True
        self.llm_server_manager = await MultiTaskLLMServerManager.create(
            config=self.config,
            worker_group=self.get_hybrid_worker_group(),
            group_scheduler=self.group_scheduler,
        )
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient),
            reward_loop_worker_handles=reward_loop_worker_handles,
            teacher_client=self.teacher_model_manager.get_client() if self.teacher_model_manager else None,
        )

    @property
    def production_window(self) -> ProductionWindow:
        return self._ensure_production_window()

    def set_production_window(self, window: ProductionWindow) -> None:
        self._production_window = window

    def report_idle_candidates(
        self,
        window: ProductionWindow,
        replicas,
        *,
        observations_fresh: bool,
        min_active_gpus: int,
        current_active_gpus: int,
        routable_count: int,
    ):
        return select_idle_candidates(
            window,
            replicas,
            observations_fresh=observations_fresh,
            min_active_gpus=min_active_gpus,
            current_active_gpus=current_active_gpus,
            routable_count=routable_count,
        )

    # Simplified lifecycle coordination surface. The methods are intentionally
    # present now so callers do not bind to the superseded begin_drain workflow.
    def prepare_replica(self, ctx, key, placement):
        """Hidden-create a borrowed runtime; success must still be non-routable."""
        raise NotImplementedError("hidden prepare requires verified native backend")

    def prepare_exit(self, command):
        """Unified DONATE/REMOVE exit preparation, including natural/force policy."""
        raise NotImplementedError("exit preparation requires verified native backend")

    def revalidate_exit(self, ctx, proof):
        """Revalidate exit evidence while Trainer owns G."""
        raise NotImplementedError("exit evidence validation requires verified native backend")

    def commit_service(self, ctx, prepared, weight, ce_commit):
        """Commit R then C/M after CE ADD evidence has been verified."""
        raise NotImplementedError("service commit requires verified native backend")

    def commit_removal(self, ctx, proof, ce_commit):
        """Commit R=REMOVED then C/M after CE REMOVE evidence."""
        raise NotImplementedError("service removal commit requires verified native backend")

    def finalize_release(self, ctx, service):
        """Sleep native or destroy borrowed only after ServiceEvidence(REMOVE)."""
        raise NotImplementedError("physical release requires verified native backend")

    def query_phase(self, ctx, phase):
        """Owner-side reconciliation hook; absence is never inferred as not-applied."""
        raise NotImplementedError("phase reconciliation requires owner-side journal wiring")

    # Compatibility only. New orchestration must call prepare_exit(command).
    def begin_drain(self, ctx, replica_id: str):
        raise NotImplementedError("begin_drain is superseded by prepare_exit(command)")
