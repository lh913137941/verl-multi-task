# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""TaskRunner binding for the single experimental Fully Async profile."""

import logging

import ray

from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.separation.utils import create_resource_pool_manager
from verl.trainer.ppo.utils import Role

from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.contracts import (
    OperationCommand,
    OperationResult,
    QueryResult,
)
from multi_task_scheduler.orchestration.operation_journal import (
    OperationJournal,
    OperationStatus,
    Outcome,
)
from multi_task_scheduler.orchestration.receipts import ReleaseEvidence, ServiceEvidence
from multi_task_scheduler.scheduler.discovery import get_or_create_group_scheduler

from .rollouter import MultiTaskFullyAsyncRollouter
from .trainer import MultiTaskFullyAsyncTrainer

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class MultiTaskFullyAsyncTaskRunner(unwrap_native_actor_class(FullyAsyncTaskRunner)):
    """Own GS/Trainer/Rollouter handles and the authoritative operation journal."""

    def __init__(self):
        super().__init__()
        self.group_scheduler = None

    def _ensure_journal(self) -> OperationJournal:
        if not hasattr(self, "_operation_journal"):
            self._operation_journal = OperationJournal()
        return self._operation_journal

    @staticmethod
    def _result_from_record(record) -> OperationResult:
        """Project the journal's latest lifecycle proofs into the public result."""
        service = None
        release = None
        for phase_result in record.phase_results.values():
            if isinstance(phase_result, ServiceEvidence):
                if (
                    service is None
                    or phase_result.header.phase_revision
                    >= service.header.phase_revision
                ):
                    service = phase_result
            elif isinstance(phase_result, ReleaseEvidence):
                if (
                    release is None
                    or phase_result.header.phase_revision
                    >= release.header.phase_revision
                ):
                    release = phase_result

        return OperationResult(
            ctx=record.command.ctx,
            target=record.command.target,
            status=record.status,
            phase=record.phase,
            phase_revision=record.phase_revision,
            service=service,
            release=release,
            error=record.error,
        )

    @staticmethod
    def _query_outcome(record) -> Outcome:
        if record.status is OperationStatus.ACCEPTED:
            return Outcome.KNOWN_NOT_APPLIED
        if record.status is OperationStatus.SUCCEEDED:
            return Outcome.KNOWN_APPLIED
        if record.status is OperationStatus.FAILED:
            error_outcome = getattr(record.error, "outcome", None)
            return Outcome.UNKNOWN if error_outcome is None else Outcome(error_outcome)
        return Outcome.UNKNOWN

    def submit_operation(self, command: OperationCommand) -> OperationResult:
        """Validate/idempotently accept one complete lifecycle command.

        OperationJournal is the single authority for replay identity, one-active-
        operation-per-task serialization and command_seq fencing. ACCEPTED proves
        only journal acceptance; runtime side effects remain owned by the native
        lifecycle implementation.
        """
        if not isinstance(command, OperationCommand):
            raise TypeError("submit_operation requires OperationCommand")
        return self._result_from_record(self._ensure_journal().begin(command))

    def query_operation(
        self, task_session: str, operation_id: str
    ) -> QueryResult[OperationResult]:
        """Read the authoritative operation record without waiting for G/GPU work."""
        record = self._ensure_journal().query(operation_id)
        if record is None or record.command.ctx.task_session != task_session:
            return QueryResult(found=False, value=None, outcome=Outcome.UNKNOWN)
        return QueryResult(
            found=True,
            value=self._result_from_record(record),
            outcome=self._query_outcome(record),
        )

    def probe_task(self, task_session: str):
        """The full TaskSnapshot needs real M/E/R/C owner observations."""
        raise NotImplementedError(
            "probe_task requires M/E/R/C + production/sync snapshot wiring"
        )

    def run(self, config):
        """Attach this Actor to GS, then execute verl's original run method."""
        self.group_scheduler = get_or_create_group_scheduler()
        context = ray.get_runtime_context()
        task_id = context.get_actor_id()
        try:
            ray.get(
                self.group_scheduler.attach_task.remote(task_id, context.current_actor),
                timeout=30,
            )
            return super().run(config)
        finally:
            try:
                ray.get(
                    self.group_scheduler.detach_task.remote(task_id),
                    timeout=30,
                )
            except Exception:
                logger.warning(
                    "Could not detach TaskRunner %s from GroupScheduler",
                    task_id,
                    exc_info=True,
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
