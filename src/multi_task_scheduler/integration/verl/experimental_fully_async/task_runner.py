# Copyright 2025 Meituan Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""TaskRunner binding for the single experimental Fully Async profile."""

import logging
import threading

import ray
from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.separation.utils import create_resource_pool_manager
from verl.trainer.ppo.utils import Role

from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    Lease,
    OperationCommand,
    OperationEvidence,
    OperationKind,
    OperationRecord,
    OperationStatus,
)
from multi_task_scheduler.orchestration.operation_journal import OperationJournal
from multi_task_scheduler.scheduler.discovery import get_or_create_group_scheduler

from .message_queue import MultiTaskMessageQueue
from .rollouter import MultiTaskFullyAsyncRollouter
from .trainer import MultiTaskFullyAsyncTrainer

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1, max_concurrency=8)
class MultiTaskFullyAsyncTaskRunner(unwrap_native_actor_class(FullyAsyncTaskRunner)):
    """Native run thread plus a small concurrent control surface for GS commands."""

    def __init__(self):
        super().__init__()
        self.group_scheduler = None
        self.task_session = None
        self._actor_handle = None
        self._control_ready = False
        self._attached_to_gs = False
        self._journal_lock = threading.RLock()
        self._operation_threads: dict[str, threading.Thread] = {}
        self._operation_leases: dict[str, Lease] = {}

    def _ensure_journal(self) -> OperationJournal:
        if not hasattr(self, "_operation_journal"):
            self._operation_journal = OperationJournal()
        return self._operation_journal

    @staticmethod
    def _snapshot_record(record: OperationRecord) -> OperationRecord:
        return OperationRecord(
            operation_id=record.operation_id,
            status=record.status,
            result=record.result,
        )

    @staticmethod
    def _build_borrowed_spec(
        command: OperationCommand,
        lease: Lease,
    ) -> dict:
        """Resolve lease_id into one internal placement spec without changing the command wire."""
        if command.kind is not OperationKind.ADD:
            raise ValueError("borrowed placement spec is valid only for ADD")
        if lease.lease_id != command.lease_id:
            raise ValueError("operation lease snapshot does not match command.lease_id")

        selected_slots = []
        for rank, claim in enumerate(lease.claims):
            slot = dict(claim)
            slot.setdefault("rank", rank)
            slot.setdefault("node_rank", 0)
            slot.setdefault("local_rank", rank)
            selected_slots.append(slot)

        return {
            "operation_id": command.operation_id,
            "lease_id": lease.lease_id,
            "lease_ids": list(lease.source_lease_ids),
            "borrower_task_id": command.target.task_session,
            "borrower_replica_id": command.target.replica_id,
            "replica_rank": None,
            "selected_slots": selected_slots,
            "world_size": len(selected_slots),
            "max_colocate_count": 1,
            "expires_at": lease.expires_at,
            "placement_epoch": command.target.runtime_epoch,
        }

    @staticmethod
    def _require_evidence(
        value,
        *,
        operation_id: str,
        expected: EvidenceType,
    ) -> OperationEvidence:
        if not isinstance(value, OperationEvidence):
            raise TypeError(f"expected OperationEvidence({expected.value})")
        if value.operation_id != operation_id:
            raise ValueError("operation evidence belongs to another operation")
        if value.type is not expected:
            raise ValueError(
                f"expected {expected.value} evidence, got {value.type.value}"
            )
        return value

    @staticmethod
    def _failure_status(exc: BaseException) -> OperationStatus:
        # Once a lifecycle worker starts, an exception does not prove which
        # owner-side effects completed. Keep the task fenced for reconciliation.
        return OperationStatus.UNKNOWN

    def _launch_operation(self, operation_id: str) -> None:
        worker = threading.Thread(
            target=self._execute_operation,
            args=(operation_id,),
            name=f"multitask-operation-{operation_id}",
            daemon=True,
        )
        self._operation_threads[operation_id] = worker
        worker.start()

    def submit_operation(
        self,
        command: OperationCommand,
        *,
        lease: Lease | None = None,
    ) -> OperationRecord:
        if not isinstance(command, OperationCommand):
            raise TypeError("submit_operation requires OperationCommand")
        if lease is not None:
            if not isinstance(lease, Lease):
                raise TypeError("lease snapshot must be Lease")
            if lease.lease_id != command.lease_id:
                raise ValueError("lease snapshot does not match command.lease_id")
        if command.kind is OperationKind.ADD and lease is None:
            raise ValueError("ADD requires the GS-resolved Lease snapshot")

        launch = False
        with self._journal_lock:
            if not self._control_ready or self.task_session is None:
                raise RuntimeError("TaskRunner control plane is not ready")
            if command.target.task_session != self.task_session:
                raise ValueError("operation targets another task session")

            journal = self._ensure_journal()
            existing = journal.query(command.operation_id)
            record = journal.begin(command)

            if lease is not None:
                previous_lease = self._operation_leases.get(command.operation_id)
                if previous_lease is not None and previous_lease != lease:
                    raise ValueError("conflicting lease snapshot replay")
                self._operation_leases.setdefault(command.operation_id, lease)

            snapshot = self._snapshot_record(record)
            launch = existing is None

        if launch:
            try:
                self._launch_operation(command.operation_id)
            except BaseException as exc:
                with self._journal_lock:
                    self._ensure_journal().finish(
                        command.operation_id,
                        OperationStatus.FAILED,
                        f"failed to launch lifecycle worker: {type(exc).__name__}: {exc}",
                    )
                raise
        return snapshot

    def query_operation(self, operation_id: str) -> OperationRecord:
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id must be a nonempty string")
        with self._journal_lock:
            record = self._ensure_journal().query(operation_id)
            if record is not None:
                return self._snapshot_record(record)
        return OperationRecord(
            operation_id=operation_id,
            status=OperationStatus.UNKNOWN,
            result="operation not found in TaskRunner journal",
        )

    def _execute_operation(self, operation_id: str) -> None:
        with self._journal_lock:
            journal = self._ensure_journal()
            command = journal.command(operation_id)
            record = journal.mark_running(operation_id)
            operation = self._snapshot_record(record)

        try:
            trainer = self.components["trainer"]
            rollouter = self.components["rollouter"]

            if command.kind is OperationKind.ADD:
                lease = self._operation_leases.get(operation_id)
                if lease is None:
                    raise RuntimeError("ADD lost its immutable Lease snapshot")
                placement_spec = self._build_borrowed_spec(command, lease)
                ray.get(
                    rollouter.prepare_replica.remote(
                        command.target,
                        operation_id=operation_id,
                        spec=placement_spec,
                    )
                )
                evidence = ray.get(trainer.bootstrap_and_publish.remote(operation))
                final_evidence = self._require_evidence(
                    evidence,
                    operation_id=operation_id,
                    expected=EvidenceType.SERVICE_COMMITTED,
                )

            elif command.kind in {OperationKind.DONATE, OperationKind.REMOVE}:
                exit_evidence = ray.get(
                    rollouter.prepare_exit.remote(
                        command.target,
                        operation_id=operation_id,
                        force=bool(command.force),
                    )
                )
                self._require_evidence(
                    exit_evidence,
                    operation_id=operation_id,
                    expected=EvidenceType.EXIT_READY,
                )

                service_evidence = ray.get(trainer.remove_and_commit.remote(operation))
                self._require_evidence(
                    service_evidence,
                    operation_id=operation_id,
                    expected=EvidenceType.SERVICE_COMMITTED,
                )

                release_evidence = ray.get(rollouter.finalize_release.remote(operation))
                final_evidence = self._require_evidence(
                    release_evidence,
                    operation_id=operation_id,
                    expected=EvidenceType.RELEASED,
                )
                ray.get(
                    self.group_scheduler.advance_lease.remote(
                        command.lease_id,
                        final_evidence,
                    ),
                    timeout=30,
                )

            elif command.kind is OperationKind.RESTORE:
                evidence = ray.get(trainer.restore_and_publish.remote(operation))
                final_evidence = self._require_evidence(
                    evidence,
                    operation_id=operation_id,
                    expected=EvidenceType.SERVICE_COMMITTED,
                )

            else:  # pragma: no cover - OperationKind construction already fences this.
                raise ValueError(f"unsupported operation kind: {command.kind!r}")

            with self._journal_lock:
                self._ensure_journal().finish(
                    operation_id,
                    OperationStatus.SUCCEEDED,
                    final_evidence.type.value,
                )

        except BaseException as exc:
            status = self._failure_status(exc)
            with self._journal_lock:
                current = self._ensure_journal().query(operation_id)
                if current is not None and current.status not in {
                    OperationStatus.SUCCEEDED,
                    OperationStatus.FAILED,
                    OperationStatus.UNKNOWN,
                }:
                    self._ensure_journal().finish(
                        operation_id,
                        status,
                        f"{type(exc).__name__}: {exc}",
                    )
            logger.exception(
                "Lifecycle operation %s failed with %s",
                operation_id,
                status.value,
            )
        finally:
            self._operation_threads.pop(operation_id, None)

    def run(self, config):
        self.group_scheduler = get_or_create_group_scheduler()
        context = ray.get_runtime_context()
        self.task_session = str(context.get_actor_id())
        self._actor_handle = context.current_actor
        try:
            return super().run(config)
        finally:
            self._control_ready = False
            if self._attached_to_gs:
                try:
                    ray.get(
                        self.group_scheduler.detach_task.remote(self.task_session),
                        timeout=30,
                    )
                except Exception:
                    logger.warning(
                        "Could not detach TaskRunner %s from GroupScheduler",
                        self.task_session,
                        exc_info=True,
                    )
                self._attached_to_gs = False

    def _replace_message_queue(self, config) -> None:
        """Swap the native empty startup queue for the queue-owned idempotent variant."""
        old_queue = self.components["message_queue"]
        old_size = ray.get(old_queue.get_queue_size.remote(), timeout=30)
        if old_size != 0:
            raise RuntimeError(
                "cannot replace native MessageQueue after samples have been enqueued"
            )

        max_queue_size = ray.get(
            self.components["rollouter"].get_max_queue_size.remote(),
            timeout=30,
        )
        ray.kill(old_queue, no_restart=True)

        queue = MultiTaskMessageQueue.remote(
            config,
            max_queue_size,
            task_session=self.task_session,
        )
        client = MessageQueueClient(queue)
        self.components["message_queue"] = queue
        self.components["message_queue_client"] = client
        ray.get(
            [
                self.components["rollouter"].set_message_queue_client.remote(client),
                self.components["trainer"].set_message_queue_client.remote(client),
            ]
        )

    def _initialize_components(self, config) -> None:
        super()._initialize_components(config)
        # Parent initialization performs checkpoint restore, initial weight sync and
        # optional validation before training starts. At this point the training
        # sample queue must still be empty, so replacing it cannot lose samples.
        self._replace_message_queue(config)
        ray.get(
            self.group_scheduler.attach_task.remote(
                self.task_session,
                self._actor_handle,
            ),
            timeout=30,
        )
        self._attached_to_gs = True
        self._control_ready = True

    def _create_rollouter(self, config) -> None:
        print("[ASYNC MAIN] Starting create rollouter...")
        rollouter = MultiTaskFullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
            group_scheduler=self.group_scheduler,
            task_session=self.task_session,
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
                config,
                roles=list(trainer_role_mapping.keys()),
            ),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
            task_session=self.task_session,
        )
        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ASYNC MAIN] FullyAsyncTrainer created and initialized successfully")
