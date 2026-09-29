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

from verl.single_controller.ray.base import _unwrap_ray_remote
from multi_task_scheduler.orchestration.contracts import (
    CONTROL_RPC_TIMEOUT_S,
    EvidenceType,
    Lease,
    OperationCommand,
    OperationEvidence,
    OperationKind,
    OperationRecord,
    OperationStatus,
    require_evidence,
)
from multi_task_scheduler.orchestration.operation_journal import OperationJournal
from multi_task_scheduler.scheduler.discovery import get_or_create_group_scheduler

from .message_queue import MultiTaskMessageQueue
from .rollouter import MultiTaskFullyAsyncRollouter
from .trainer import MultiTaskFullyAsyncTrainer

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1, max_concurrency=8)
class MultiTaskFullyAsyncTaskRunner(_unwrap_ray_remote(FullyAsyncTaskRunner)):
    """Native run thread plus a small concurrent control surface for GS commands."""

    def __init__(self):
        super().__init__()
        self.group_scheduler = None
        self.task_session = None
        self._control_ready = False
        self._attached_to_gs = False
        self._journal_lock = threading.RLock()
        self._operation_threads: dict[str, threading.Thread] = {}
        self._operation_leases: dict[str, Lease] = {}
        self._operation_journal = OperationJournal()

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

        # Donor/source rank metadata is not borrower topology.
        claims = [
            dict(claim, rank=rank, node_rank=0, local_rank=rank)
            for rank, claim in enumerate(lease.claims)
        ]

        return {
            "operation_id": command.operation_id,
            "lease_id": lease.lease_id,
            "borrower_task_id": command.target.task_session,
            "borrower_replica_id": command.target.replica_id,
            "claims": claims,
            "expires_at": lease.expires_at,
            "placement_epoch": command.target.runtime_epoch,
        }

    def _close_rolled_back(
        self,
        command: OperationCommand,
        operation_id: str,
        evidence,
        label: str,
        result: str,
    ) -> None:
        """Close an operation whose owner-side commit was verifiably rolled back.

        The owner answered RELEASED instead of SERVICE_COMMITTED, which is the
        documented proof that no hidden/native runtime survived the attempt. GS
        records the same RELEASED receipt as a rollback, so the lease is advanced
        -- not abandoned -- before the journal closes FAILED.
        """
        rollback_evidence = require_evidence(
            evidence,
            operation_id,
            EvidenceType.RELEASED,
            label,
        )
        self._advance_lease(command, rollback_evidence)
        with self._journal_lock:
            self._operation_journal.finish(
                operation_id,
                OperationStatus.FAILED,
                result,
            )

    def _advance_lease(
        self,
        command: OperationCommand,
        evidence: OperationEvidence,
    ) -> None:
        # GS advance_lease is evidence-idempotent. Retry the exact same evidence
        # once so an ACK loss after a committed ledger write does not immediately
        # turn a resolved lifecycle operation into UNKNOWN.
        last_error = None
        for _attempt in range(2):
            try:
                ray.get(
                    self.group_scheduler.advance_lease.remote(
                        command.lease_id,
                        evidence,
                    ),
                    timeout=CONTROL_RPC_TIMEOUT_S,
                )
                return
            except Exception as exc:
                last_error = exc
        raise last_error

    def _launch_operation(self, operation_id: str) -> None:
        worker = threading.Thread(
            target=self._execute_operation,
            args=(operation_id,),
            name=f"multitask-operation-{operation_id}",
            daemon=True,
        )
        with self._journal_lock:
            self._operation_threads[operation_id] = worker
        try:
            worker.start()
        except BaseException:
            with self._journal_lock:
                self._operation_threads.pop(operation_id, None)
            raise

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
        reconciliation_launch = False
        with self._journal_lock:
            if not self._control_ready or self.task_session is None:
                raise RuntimeError("TaskRunner control plane is not ready")
            if command.target.task_session != self.task_session:
                raise ValueError("operation targets another task session")

            journal = self._operation_journal
            existing = journal.query(command.operation_id)
            record = journal.begin(command)

            if command.kind is OperationKind.ADD:
                previous_lease = self._operation_leases.get(command.operation_id)
                if previous_lease is not None and previous_lease != lease:
                    raise ValueError("conflicting lease snapshot replay")
                self._operation_leases.setdefault(command.operation_id, lease)

            snapshot = self._snapshot_record(record)
            reconciliation_launch = (
                existing is not None
                and existing.status is OperationStatus.UNKNOWN
                and command.operation_id not in self._operation_threads
            )
            launch = existing is None or reconciliation_launch

        if launch:
            try:
                self._launch_operation(command.operation_id)
            except BaseException as exc:
                with self._journal_lock:
                    self._operation_threads.pop(command.operation_id, None)
                    if reconciliation_launch:
                        # The prior UNKNOWN owner outcome still exists. A local
                        # thread-launch failure creates no new owner fact and must
                        # not erase preserved reconciliation inputs or rewrite
                        # the old UNKNOWN into a resolved FAILED result.
                        pass
                    else:
                        self._operation_leases.pop(command.operation_id, None)
                        self._operation_journal.finish(
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
            record = self._operation_journal.query(operation_id)
            if record is not None:
                return self._snapshot_record(record)
        return OperationRecord(
            operation_id=operation_id,
            status=OperationStatus.UNKNOWN,
            result=None,
        )

    def _execute_operation(self, operation_id: str) -> None:
        try:
            with self._journal_lock:
                journal = self._operation_journal
                command = journal.command(operation_id)
                current = journal.query(operation_id)
                reconciling = (
                    current is not None
                    and current.status is OperationStatus.UNKNOWN
                )
                if reconciling:
                    record = journal.reopen_unknown(operation_id)
                else:
                    record = journal.mark_running(operation_id)
                operation = self._snapshot_record(record)

            trainer = self.components["trainer"]
            rollouter = self.components["rollouter"]

            if command.kind is OperationKind.ADD:
                lease = self._operation_leases.get(operation_id)
                if lease is None:
                    raise RuntimeError("ADD lost its immutable Lease snapshot")
                try:
                    placement_spec = self._build_borrowed_spec(command, lease)
                except (TypeError, ValueError) as exc:
                    with self._journal_lock:
                        self._operation_journal.finish(
                            operation_id,
                            OperationStatus.FAILED,
                            f"ADD placement validation failed: {type(exc).__name__}: {exc}",
                        )
                    return
                prepare_result = ray.get(
                    rollouter.prepare_replica.remote(
                        command.target,
                        operation_id=operation_id,
                        spec=placement_spec,
                    )
                )
                if (
                    isinstance(prepare_result, OperationEvidence)
                    and prepare_result.type is EvidenceType.RELEASED
                ):
                    self._close_rolled_back(
                        command,
                        operation_id,
                        prepare_result,
                        "prepare_replica",
                        "ADD prepare failed; no borrower runtime remains",
                    )
                    return
                evidence = ray.get(trainer.bootstrap_and_publish.remote(operation))
                if (
                    isinstance(evidence, OperationEvidence)
                    and evidence.type is EvidenceType.RELEASED
                ):
                    self._close_rolled_back(
                        command,
                        operation_id,
                        evidence,
                        "bootstrap_and_publish",
                        "ADD bootstrap failed; hidden runtime was verified RELEASED",
                    )
                    return
                final_evidence = require_evidence(
                    evidence,
                    operation_id,
                    EvidenceType.SERVICE_COMMITTED,
                    "bootstrap_and_publish",
                )

            elif command.kind in {OperationKind.DONATE, OperationKind.REMOVE}:
                service_evidence = None
                if reconciling:
                    # The remote release may have committed before TaskRunner lost
                    # its ACK. Manager's release ledger is authoritative and makes
                    # this replay side-effect free.
                    prior_release = ray.get(
                        rollouter.query_release_operation.remote(
                            command.target,
                            operation_id,
                        )
                    )
                    if prior_release is not None:
                        final_evidence = require_evidence(
                            prior_release,
                            operation_id,
                            EvidenceType.RELEASED,
                            "query_release_operation",
                        )
                        self._advance_lease(command, final_evidence)
                        with self._journal_lock:
                            self._operation_journal.finish(
                                operation_id,
                                OperationStatus.SUCCEEDED,
                                final_evidence.type.value,
                            )
                        return

                    # If E removal happened and G was fenced around an uncertain
                    # R/C commit, reconcile that owner state before repeating
                    # prepare_exit. A healthy gate returns None and we continue
                    # through the ordinary idempotent exit path.
                    service_evidence = ray.get(
                        trainer.reconcile_exit.remote(operation)
                    )

                if service_evidence is None:
                    try:
                        exit_evidence = ray.get(
                            rollouter.prepare_exit.remote(
                                command.target,
                                operation_id=operation_id,
                                force=bool(command.force),
                            )
                        )
                    except BaseException as exc:
                        # prepare_exit binds operation -> target immediately when
                        # it begins owner mutation (before drain/abort). If no
                        # binding exists after rejection, the error is a proven
                        # no-side-effect preflight failure and may close FAILED
                        # without fencing this task.
                        try:
                            ray.get(
                                rollouter.get_pending_target.remote(operation_id)
                            )
                        except BaseException:
                            with self._journal_lock:
                                self._operation_journal.finish(
                                    operation_id,
                                    OperationStatus.FAILED,
                                    "exit preflight rejected before owner mutation: "
                                    f"{type(exc).__name__}: {exc}",
                                )
                            return
                        raise
                    require_evidence(
                        exit_evidence,
                        operation_id,
                        EvidenceType.EXIT_READY,
                        "prepare_exit",
                    )
                    service_evidence = ray.get(
                        trainer.remove_and_commit.remote(operation)
                    )

                require_evidence(
                    service_evidence,
                    operation_id,
                    EvidenceType.SERVICE_COMMITTED,
                    "exit service commit",
                )

                release_evidence = ray.get(rollouter.finalize_release.remote(operation))
                final_evidence = require_evidence(
                    release_evidence,
                    operation_id,
                    EvidenceType.RELEASED,
                    "finalize_release",
                )
                self._advance_lease(command, final_evidence)

            elif command.kind is OperationKind.RESTORE:
                ray.get(
                    rollouter.prepare_replica.remote(
                        command.target,
                        operation_id=operation_id,
                        spec=None,
                    )
                )
                evidence = ray.get(trainer.restore_and_publish.remote(operation))
                if (
                    isinstance(evidence, OperationEvidence)
                    and evidence.type is EvidenceType.RELEASED
                ):
                    self._close_rolled_back(
                        command,
                        operation_id,
                        evidence,
                        "restore_and_publish",
                        "RESTORE bootstrap failed; native runtime was verified re-slept",
                    )
                    return
                final_evidence = require_evidence(
                    evidence,
                    operation_id,
                    EvidenceType.SERVICE_COMMITTED,
                    "restore_and_publish",
                )
                self._advance_lease(command, final_evidence)

            else:  # pragma: no cover - OperationKind construction already fences this.
                raise ValueError(f"unsupported operation kind: {command.kind!r}")

            with self._journal_lock:
                self._operation_journal.finish(
                    operation_id,
                    OperationStatus.SUCCEEDED,
                    final_evidence.type.value,
                )

        except BaseException as exc:
            status = OperationStatus.UNKNOWN
            with self._journal_lock:
                current = self._operation_journal.query(operation_id)
                if current is not None and current.status not in {
                    OperationStatus.SUCCEEDED,
                    OperationStatus.FAILED,
                    OperationStatus.UNKNOWN,
                }:
                    self._operation_journal.finish(
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
            with self._journal_lock:
                self._operation_threads.pop(operation_id, None)
                current = self._operation_journal.query(operation_id)
                if current is None or current.status is not OperationStatus.UNKNOWN:
                    self._operation_leases.pop(operation_id, None)

    def run(self, config):
        self.group_scheduler = get_or_create_group_scheduler()
        context = ray.get_runtime_context()
        self.task_session = str(context.get_actor_id())
        try:
            return super().run(config)
        finally:
            with self._journal_lock:
                self._control_ready = False
            if self._attached_to_gs:
                try:
                    ray.get(
                        self.group_scheduler.detach_task.remote(self.task_session),
                        timeout=CONTROL_RPC_TIMEOUT_S,
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
        old_size = ray.get(
            old_queue.get_queue_size.remote(),
            timeout=CONTROL_RPC_TIMEOUT_S,
        )
        if old_size != 0:
            raise RuntimeError(
                "cannot replace native MessageQueue after samples have been enqueued"
            )

        max_queue_size = ray.get(
            self.components["rollouter"].get_max_queue_size.remote(),
            timeout=CONTROL_RPC_TIMEOUT_S,
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
                ray.get_runtime_context().current_actor,
            ),
            timeout=CONTROL_RPC_TIMEOUT_S,
        )
        self._attached_to_gs = True
        with self._journal_lock:
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
