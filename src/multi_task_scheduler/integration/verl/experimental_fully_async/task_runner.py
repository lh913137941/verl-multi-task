# Copyright 2025 Meituan Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""TaskRunner binding for the single experimental Fully Async profile."""

import logging
import ray
from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.separation.utils import create_resource_pool_manager
from verl.trainer.ppo.utils import Role
from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.contracts import OperationCommand, OperationRecord, OperationStatus
from multi_task_scheduler.orchestration.operation_journal import OperationJournal
from multi_task_scheduler.scheduler.discovery import get_or_create_group_scheduler
from .rollouter import MultiTaskFullyAsyncRollouter
from .trainer import MultiTaskFullyAsyncTrainer

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class MultiTaskFullyAsyncTaskRunner(unwrap_native_actor_class(FullyAsyncTaskRunner)):
    def __init__(self):
        super().__init__()
        self.group_scheduler = None

    def _ensure_journal(self) -> OperationJournal:
        if not hasattr(self, "_operation_journal"):
            self._operation_journal = OperationJournal()
        return self._operation_journal

    def submit_operation(self, command: OperationCommand) -> OperationRecord:
        if not isinstance(command, OperationCommand):
            raise TypeError("submit_operation requires OperationCommand")
        return self._ensure_journal().begin(command)

    def query_operation(self, operation_id: str) -> OperationRecord:
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id must be a nonempty string")
        record = self._ensure_journal().query(operation_id)
        if record is not None:
            return record
        return OperationRecord(operation_id=operation_id, status=OperationStatus.UNKNOWN, result="operation not found in TaskRunner journal")

    def run(self, config):
        self.group_scheduler = get_or_create_group_scheduler()
        context = ray.get_runtime_context()
        task_id = str(context.get_actor_id())
        try:
            ray.get(self.group_scheduler.attach_task.remote(task_id, context.current_actor), timeout=30)
            return super().run(config)
        finally:
            try:
                ray.get(self.group_scheduler.detach_task.remote(task_id), timeout=30)
            except Exception:
                logger.warning("Could not detach TaskRunner %s from GroupScheduler", task_id, exc_info=True)

    def _create_rollouter(self, config) -> None:
        print("[ASYNC MAIN] Starting create rollouter...")
        rollouter = MultiTaskFullyAsyncRollouter.remote(config=config, tokenizer=self.components["tokenizer"], processor=self.components["processor"], device_name=config.trainer.device, group_scheduler=self.group_scheduler)
        if "hybrid_worker_group" in self.components:
            ray.get(rollouter.set_hybrid_worker_group.remote(self.components["hybrid_worker_group"]))
            print("[ASYNC MAIN] Hybrid worker group injected into rollouter")
        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())
        self.components["rollouter"] = rollouter
        print("[ASYNC MAIN] Rollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        print("[ASYNC MAIN] Starting create trainer...")
        trainer_role_mapping = {role: worker_cls for role, worker_cls in self.components["role_worker_mapping"].items() if role != Role.Rollout}
        trainer = MultiTaskFullyAsyncTrainer.remote(config=config, tokenizer=self.components["tokenizer"], role_worker_mapping=trainer_role_mapping, resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())), ray_worker_group_cls=self.components["ray_worker_group_cls"], device_name=config.trainer.device)
        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ASYNC MAIN] FullyAsyncTrainer created and initialized successfully")
