"""Experimental Fully Async Rollouter with production-window sensing.

The manager/agent-loop initialization below follows verl's Apache-2.0-licensed
FullyAsyncRollouter; only the manager type and its GS handle are added. The
idle-bubble overlay (section 5.1) merges the load balancer's versioned request
facts into a ``ProductionWindow`` and derives the candidate set; the actual
drain/prepare primitives stay explicit failures until a native backend is
verified (``init_workers``/``fit`` remain inherited).
"""

import ray

from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncAgentLoopManager,
    FullyAsyncRollouter,
)
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.production_window import (
    ProductionWindow,
    select_idle_candidates,
)

from .llm_server_manager import MultiTaskLLMServerManager


@ray.remote(num_cpus=10, max_concurrency=100)
class MultiTaskFullyAsyncRollouter(unwrap_native_actor_class(FullyAsyncRollouter)):
    """Real Ray Actor; native generation, queue and training methods stay inherited."""

    def __init__(self, config, tokenizer, processor=None, device_name=None, *, group_scheduler=None):
        self.group_scheduler = group_scheduler
        super().__init__(config, tokenizer, processor=processor, device_name=device_name)

    def _ensure_production_window(self) -> ProductionWindow:
        # Lazy so the inherited native Rollouter constructor stays untouched.
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

    # -- idle-bubble sensing overlay (section 5.1) ------------------------- #

    @property
    def production_window(self) -> ProductionWindow:
        return self._ensure_production_window()

    def set_production_window(self, window: ProductionWindow) -> None:
        """Authoritative merge of the LB's versioned facts (section 5.1)."""
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
        """Derive the idle candidate set from the production window (pure)."""
        return select_idle_candidates(
            window,
            replicas,
            observations_fresh=observations_fresh,
            min_active_gpus=min_active_gpus,
            current_active_gpus=current_active_gpus,
            routable_count=routable_count,
        )

    def prepare_replica(self, ctx, placement):
        """Materialize a hidden runtime for a lease; never published here."""
        raise NotImplementedError(
            "hidden prepare requires verified native backend"
        )

    def begin_drain(self, ctx, replica_id: str):
        """Drop a target from routable and observe it to drained (section 5.2)."""
        raise NotImplementedError(
            "drain observation requires verified native backend"
        )
