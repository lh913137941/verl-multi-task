"""Shared GroupScheduler actor for simplified design §3/§8.

The actor owns the GS ledger and lease authorization. It records intent before
issuing work and validates protocol/session fencing at the public operation
entry; it does not infer device release from task health.
"""

import time
import uuid

import ray
from ray.actor import ActorHandle

from multi_task_scheduler.orchestration.contracts import OperationCommand, OperationResult
from multi_task_scheduler.orchestration.operation_journal import OperationStatus
from multi_task_scheduler.scheduler.ledger import (
    IdleReport,
    Ledger,
    LeaseRecord,
    ProtocolInstance,
    ResourceManifest,
)
from multi_task_scheduler.scheduler.lease import LeaseState, LeaseStateMachine

RUNTIME_KIND = "verl-multi-task:experimental_fully_async_standalone:p1"
PROTOCOL_VERSION = 1


@ray.remote(num_cpus=0)
class GroupScheduler:
    """Keep task controllers and authoritative GS ledger/lease state."""

    def __init__(self) -> None:
        self.task_runners: dict[str, ActorHandle] = {}
        self.controllers: dict[tuple[str, str], ActorHandle] = {}
        self.ledger = Ledger(
            ProtocolInstance(
                protocol_version=PROTOCOL_VERSION,
                runtime_kind=RUNTIME_KIND,
                gs_epoch=f"gs-{uuid.uuid4().hex}",
                sharing_namespace="verl-multi-task",
                recovery_state="RECOVERY_ONLY",
            )
        )
        self.lease_sm = LeaseStateMachine()

    # -- discovery / legacy ------------------------------------------------- #

    def runtime_kind(self) -> str:
        return RUNTIME_KIND

    def attach_task(self, task_id: str, task_runner: ActorHandle) -> None:
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be a nonempty TaskRunner Actor ID")
        if not isinstance(task_runner, ActorHandle):
            raise TypeError("GroupScheduler requires a real TaskRunner ActorHandle")
        existing = self.task_runners.get(task_id)
        if existing is not None and existing != task_runner:
            raise ValueError(f"TaskRunner reference already exists for {task_id}")
        self.task_runners[task_id] = task_runner

    def detach_task(self, task_id: str) -> None:
        self.task_runners.pop(task_id, None)

    def get_task_runners(self) -> dict[str, ActorHandle]:
        return dict(self.task_runners)

    def schedule(self) -> list:
        """No automatic cross-task policy is implemented in this pass."""
        return []

    # -- §8.1 GS <-> task control interfaces ------------------------------- #

    def attach_controller(
        self,
        task_id: str,
        task_session: str,
        handle: ActorHandle,
        protocol: dict | None = None,
    ) -> dict:
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be a nonempty string")
        if not isinstance(task_session, str) or not task_session:
            raise ValueError("task_session must be a nonempty string")
        if not isinstance(handle, ActorHandle):
            raise TypeError("attach_controller requires a real TaskRunner ActorHandle")
        self.ledger.register_task(task_id, task_session, task_runner=handle)
        self.controllers[(task_id, task_session)] = handle
        return {
            "status": "INITIALIZING",
            "task_id": task_id,
            "task_session": task_session,
            "protocol_version": self.ledger.protocol.protocol_version,
            "gs_epoch": self.ledger.protocol.gs_epoch,
        }

    def register_resources(self, registration_id: str, manifest: ResourceManifest) -> dict:
        if not isinstance(manifest, ResourceManifest):
            raise TypeError("register_resources requires a ResourceManifest")
        self.ledger.register_resources(manifest)
        return {"registration_id": registration_id, "status": "READY"}

    def probe_task(self, probe_id: str, known_revisions: dict | None = None) -> dict:
        tasks = [
            {
                "task_id": r.task_id,
                "task_session": r.task_session,
                "status": r.status,
            }
            for r in self.ledger.tasks.values()
        ]
        leases = [
            {"lease_id": lease.lease_id, "state": lease.state}
            for lease in self.ledger.leases.values()
        ]
        return {"probe_id": probe_id, "tasks": tasks, "leases": leases}

    def report_idle_candidates(self, report: IdleReport) -> dict:
        if not isinstance(report, IdleReport):
            raise TypeError("report_idle_candidates requires an IdleReport")
        if type(report.valid_for_ms) is not int or report.valid_for_ms <= 0:
            raise ValueError("valid_for_ms must be a positive integer")
        self.ledger.upsert_idle_observation(
            report, valid_until=time.monotonic() + report.valid_for_ms / 1000
        )
        return {"source_seq": report.source_seq, "candidates": list(report.candidate_ids)}

    def _validate_command_fence(self, command: OperationCommand) -> None:
        protocol = self.ledger.protocol
        if command.protocol_version != protocol.protocol_version:
            raise ValueError(
                f"protocol version mismatch: {command.protocol_version} != {protocol.protocol_version}"
            )
        if command.gs_epoch != protocol.gs_epoch:
            raise ValueError("stale gs_epoch")
        if (command.target_task_id, command.target_task_session) not in self.ledger.tasks:
            raise ValueError("unknown target task session")

    def submit_operation(self, command: OperationCommand) -> dict:
        """Validate/fence intent and return quickly without claiming completion."""
        if not isinstance(command, OperationCommand):
            raise TypeError("submit_operation requires an OperationCommand")
        try:
            self._validate_command_fence(command)
            self.ledger.record_operation(command)
        except ValueError as exc:
            return {
                "status": OperationStatus.FAILED.value,
                "state": OperationStatus.FAILED.value,  # compatibility
                "operation_id": command.operation_id,
                "error": str(exc),
            }
        return {
            "status": OperationStatus.ACCEPTED.value,
            "state": OperationStatus.ACCEPTED.value,  # compatibility
            "operation_id": command.operation_id,
        }

    def query_operation(self, operation_id: str) -> dict | None:
        record = self.ledger.operations.get(operation_id)
        if record is None:
            return None
        final = record.final_result
        return {
            "operation_id": operation_id,
            "phase": record.phase.value,
            "status": (
                final.status.value if final is not None else OperationStatus.ACCEPTED.value
            ),
            "final_result": final,
        }

    def get_resource_snapshot(self, expected_session: str) -> dict:
        return {
            "protocol_version": self.ledger.protocol.protocol_version,
            "gs_epoch": self.ledger.protocol.gs_epoch,
            "expected_session": expected_session,
            "tasks": [
                {
                    "task_id": r.task_id,
                    "task_session": r.task_session,
                    "status": r.status,
                    "initial_gpus": r.initial_gpus,
                }
                for r in self.ledger.tasks.values()
            ],
            "gpus": [
                {
                    "node_id": g.node_id,
                    "gpu_uuid": g.gpu_uuid,
                    "state": g.state,
                    "current_user": g.current_user,
                }
                for g in self.ledger.gpus.values()
            ],
        }

    def report_operation_result(self, result: OperationResult) -> dict:
        if not isinstance(result, OperationResult):
            raise TypeError("report_operation_result requires an OperationResult")
        self.ledger.merge_operation_result(result.identity_fields.operation_id, result)
        return {"operation_id": result.identity_fields.operation_id, "merged": True}

    # -- lease authorization ------------------------------------------------ #

    def open_lease(self, lease: LeaseRecord) -> dict:
        if not isinstance(lease, LeaseRecord):
            raise TypeError("open_lease requires a LeaseRecord")
        self.ledger.open_lease(lease)
        self.lease_sm.register(lease)
        return {"lease_id": lease.lease_id, "state": lease.state}

    def advance_lease(
        self,
        lease_id: str,
        new_state: str,
        supporting_result_state: str | None = None,
        operation_id: str | None = None,
    ) -> dict:
        self.lease_sm.advance(
            lease_id,
            LeaseState(new_state),
            supporting_result_state=supporting_result_state,
            operation_id=operation_id,
        )
        return {"lease_id": lease_id, "state": new_state}
