"""Experimental Fully Async Rollouter with the simplified lifecycle surface.

Native generation/queue behavior stays inherited. Device/runtime actions remain
explicit failures until a verified native backend is available. Idle reporting
uses the canonical evidence-bearing ``IdleCandidateReport`` contract.
"""

import ray

from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncAgentLoopManager,
    FullyAsyncRollouter,
)
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.contracts import IdleCandidateReport
from multi_task_scheduler.orchestration.production_window import (
    ProductionWindow,
    select_idle_candidates,
)

from .llm_server_manager import MultiTaskLLMServerManager


@ray.remote(num_cpus=10, max_concurrency=100)
class MultiTaskFullyAsyncRollouter(unwrap_native_actor_class(FullyAsyncRollouter)):
    """Own production/capacity/lifecycle coordination for the task."""

    def __init__(self, config, tokenizer, processor=None, device_name=None, *, group_scheduler=None):
        self.group_scheduler = group_scheduler
        super().__init__(config, tokenizer, processor=processor, device_name=device_name)

    def _ensure_production_window(self) -> ProductionWindow:
        if not hasattr(self, "_production_window"):
            self._production_window = ProductionWindow()
        return self._production_window

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
    def production_window(self) -> ProductionWindow:
        return self._ensure_production_window()

    def set_production_window(self, window: ProductionWindow) -> None:
        if not isinstance(window, ProductionWindow):
            raise TypeError("set_production_window requires ProductionWindow")
        self._production_window = window

    def report_idle_candidates(
        self,
        window: ProductionWindow,
        replicas,
        *,
        task_session: str,
        gs_epoch: str,
        lb_session: str,
        valid_for_ms: int,
        observations_fresh: bool,
        min_active_gpus: int,
        current_active_gpus: int,
        routable_count: int,
    ) -> IdleCandidateReport:
        """Freeze the complete current candidate set for GS replacement semantics."""
        if window is not self.production_window:
            raise ValueError("idle report must use the Rollouter current production window")
        candidates = select_idle_candidates(
            window,
            replicas,
            observations_fresh=observations_fresh,
            min_active_gpus=min_active_gpus,
            current_active_gpus=current_active_gpus,
            routable_count=routable_count,
        )
        return IdleCandidateReport(
            task_session=task_session,
            gs_epoch=gs_epoch,
            lb_session=lb_session,
            source_seq=window.source_seq,
            production_revision=window.production_revision,
            valid_for_ms=valid_for_ms,
            candidates=candidates,
        )

    def prepare_replica(self, ctx, key, placement):
        """Hidden-create one borrowed runtime without publishing service."""
        raise NotImplementedError("prepare_replica requires verified native backend")

    def prepare_exit(self, command):
        """Unified DONATE/REMOVE drain and verified continuation coordinator."""
        raise NotImplementedError("prepare_exit requires verified native backend")

    def revalidate_exit(self, ctx, proof):
        """Revalidate exit evidence while Trainer owns G."""
        raise NotImplementedError("revalidate_exit requires owner-side drain evidence wiring")

    def commit_service(self, ctx, prepared, weight, ce_commit):
        """Commit LB route then C/M after a valid CE ADD receipt."""
        raise NotImplementedError("commit_service requires verified owner commit wiring")

    def commit_removal(self, ctx, proof, ce_commit):
        """Commit LB removal then C/M after a valid CE REMOVE receipt."""
        raise NotImplementedError("commit_removal requires verified owner commit wiring")

    def finalize_release(self, ctx, service):
        """Sleep native or destroy borrowed only after ServiceEvidence(REMOVE)."""
        raise NotImplementedError("finalize_release requires verified runtime release backend")

    def query_phase(self, ctx, phase):
        """Return owner-backed phase facts once owner-side journals are wired."""
        raise NotImplementedError("query_phase requires owner-side journal wiring")
