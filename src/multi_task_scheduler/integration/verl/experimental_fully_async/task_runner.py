# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""TaskRunner binding for the single experimental Fully Async profile."""

import logging

import ray

from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.separation.utils import create_resource_pool_manager
from verl.trainer.ppo.utils import Role

from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.contracts import OperationResult
from multi_task_scheduler.orchestration.operation_journal import OperationJournal
from multi_task_scheduler.scheduler.discovery import get_or_create_group_scheduler

from .rollouter import MultiTaskFullyAsyncRollouter
from .trainer import MultiTaskFullyAsyncTrainer

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class MultiTaskFullyAsyncTaskRunner(unwrap_native_actor_class(FullyAsyncTaskRunner)):
    """Own GS/Trainer/Rollouter handles and the one authoritative operation journal."""

    def __init__(self):
        super().__init__()
        self.group_scheduler = None

    def _ensure_journal(self) -> OperationJournal:
        if not hasattr(self, "_operation_journal"):
            self._operation_journal = OperationJournal()
        return self._operation_journal

    def _record_operation(self, command):
        return self._ensure_journal().begin(
            command.operation_id,
            command.lease_epoch,
            command.replica_id,
            command.kind,
            payload_digest=command.payload_digest,
            command_seq=command.command_seq,
        )

    def submit_operation(self, command):
        """Canonical GS -> TaskRunner control entry returning ACCEPTED progress.

        This pass records and fences the operation idempotently. Real ADD /
        DONATE / REMOVE / RESTORE execution remains unavailable until the
        verified native runtime backend is wired; an ACCEPTED result must never
        be confused with completed lifecycle work.
        """
        record = self._record_operation(command)
        context = getattr(command, "context", None)
        if context is None:
            # Old ad-hoc callers are supported only by begin_operation(); the
            # public submit_operation contract requires OperationCommand.
            raise TypeError("submit_operation requires an OperationCommand with context")
        return OperationResult(
            identity_fields=context,
            phase=record.phase,
            phase_revision=record.phase_revision,
            state=record.status,
            actual_replica_state=None,
        )

    def begin_operation(self, command):
        """Compatibility entry returning the internal journal record."""
        return self._record_operation(command)

    def query_operation(self, task_session: str, operation_id: str | None = None):
        """Read the operation journal without waiting for G or GPU work.

        Canonical form is ``query_operation(task_session, operation_id)``.
        For callers from the previous binding pass, a single positional argument
        is still interpreted as ``operation_id``.
        """
        if operation_id is None:
            operation_id = task_session
        return self._ensure_journal().query(operation_id)

    def probe_task(self, task_session: str | None = None):
        """Return only facts this binding can currently prove."""
        trainer = self.components.get("trainer") if hasattr(self, "components") else None
        rollouter = self.components.get("rollouter") if hasattr(self, "components") else None
        return {
            "task_session": task_session,
            "trainer_attached": trainer is not None,
            "rollouter_attached": rollouter is not None,
            "operation_count": len(self._ensure_journal()._records),
        }

    def run(self, config):
        """Attach this Actor to GS, then execute verl's original run method."""
        self.group_scheduler = get_or_create_group_scheduler()
        context = ray.get_runtime_context()
        task_id = context.get_actor_id()
        try:
            ray.get(self.group_scheduler.attach_task.remote(task_id, context.current_actor), timeout=30)
            return super().run(config)
        finally:
            try:
                ray.get(self.group_scheduler.detach_task.remote(task_id), timeout=30)
            except Exception:
                logger.warning(
                    "Could not detach TaskRunner %s from GroupScheduler", task_id, exc_info=True
                )

    def _create_rollouter(self, config) -> None:
        print("[ASYNC MAIN] Starting create rollouter...")
        rollouter = MultiTaskFullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
            group_scheduler=self.group_scheduler,
        )
        if "hybrid_worker_group" in self.components:
            ray.get(
                rollouter.set_hybrid_worker_group.remote(
                    self.components["hybrid_worker_group"]
                )
            )
            print("[ASYNC MAIN] Hybrid worker group injected into rollouter")
        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())
        self.components["rollouter"] = rollouter
        print("[ASYNC MAIN] Rollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        print("[ASYNC MAIN] Starting create trainer...")
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }
        trainer = MultiTaskFullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(
                config, roles=list(trainer_role_mapping.keys())
            ),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )
        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ASYNC MAIN] FullyAsyncTrainer created and initialized successfully")
