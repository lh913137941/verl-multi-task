"""Shared GroupScheduler actor for simplified design §3/§8.

GS owns global resource authorization and intent. It forwards lifecycle commands
to the attached TaskRunner and never converts ACK/health into fake completion.
"""

import time
import uuid

import ray
from ray.actor import ActorHandle

from multi_task_scheduler.orchestration.contracts import (
    IdleCandidateReport,
    OperationCommand,
    OperationResult,
    QueryResult,
)
from multi_task_scheduler.orchestration.operation_journal import (
    OperationStatus,
    Outcome,
    Phase,
)
from multi_task_scheduler.orchestration.receipts import Ack
from multi_task_scheduler.scheduler.ledger import (
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
    """Keep task controllers plus authoritative GS ledger/lease state."""

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

    def runtime_kind(self) -> str:
        return RUNTIME_KIND

    def attach_task(self, task_id: str, task_runner: ActorHandle) -> None:
        """Runtime discovery hook used before task_session registration exists."""
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
        """Cross-task policy is outside this implementation pass."""
        return []

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
            "task_id": task_id,
            "task_session": task_session,
            "protocol_version": self.ledger.protocol.protocol_version,
            "gs_epoch": self.ledger.protocol.gs_epoch,
        }

    def register_resources(self, registration_id: str, manifest: ResourceManifest) -> dict:
        if not isinstance(manifest, ResourceManifest):
            raise TypeError("register_resources requires ResourceManifest")
        self.ledger.register_resources(manifest)
        for record in self.ledger.tasks.values():
            if record.task_session == manifest.owner_task_session:
                record.status = "READY"
        return {"registration_id": registration_id, "status": "READY"}

    def report_idle_candidates(self, report: IdleCandidateReport) -> Ack:
        if not isinstance(report, IdleCandidateReport):
            raise TypeError("report_idle_candidates requires IdleCandidateReport")
        self.ledger.upsert_idle_report(
            report, valid_until=time.monotonic() + report.valid_for_ms / 1000
        )
        return Ack(accepted=True, revision=report.source_seq)

    def _validate_command_fence(self, command: OperationCommand) -> None:
        protocol = self.ledger.protocol
        if command.ctx.protocol_version != protocol.protocol_version:
            raise ValueError(
                "protocol version mismatch: "
                f"{command.ctx.protocol_version} != {protocol.protocol_version}"
            )
        if command.ctx.gs_epoch != protocol.gs_epoch:
            raise ValueError("stale gs_epoch")
        task_key = (command.ctx.task_id, command.ctx.task_session)
        task = self.ledger.tasks.get(task_key)
        if task is None:
            raise ValueError("unknown target task session")
        if task.status != "READY":
            raise ValueError("target task is not READY")
        if command.target.task_session != command.ctx.task_session:
            raise ValueError("target runtime belongs to a different task_session")
        if task_key not in self.controllers:
            raise ValueError("target TaskRunner controller is not attached")

    def submit_operation(self, command: OperationCommand) -> OperationResult:
        """Record GS intent, forward once to TR, and return TR's ACCEPTED result."""
        if not isinstance(command, OperationCommand):
            raise TypeError("submit_operation requires OperationCommand")
        self._validate_command_fence(command)
        record = self.ledger.record_operation(command)
        if record.final_result is not None:
            return record.final_result
        controller = self.controllers[(command.ctx.task_id, command.ctx.task_session)]
        result = ray.get(controller.submit_operation.remote(command), timeout=30)
        if not isinstance(result, OperationResult):
            raise TypeError("TaskRunner returned a non-OperationResult")
        self.ledger.merge_operation_result(command.ctx.operation_id, result)
        return result

    def query_operation(
        self, task_session: str, operation_id: str
    ) -> QueryResult[OperationResult]:
        record = self.ledger.operations.get(operation_id)
        if record is None or record.command.ctx.task_session != task_session:
            return QueryResult(found=False, value=None, outcome=Outcome.UNKNOWN)
        if record.final_result is not None:
            result = record.final_result
        else:
            result = OperationResult(
                ctx=record.command.ctx,
                target=record.command.target,
                status=OperationStatus.ACCEPTED,
                phase=Phase.VALIDATE,
                phase_revision=0,
            )
        outcome = (
            Outcome.KNOWN_APPLIED
            if result.status in {OperationStatus.SUCCEEDED, OperationStatus.FAILED}
            else Outcome.UNKNOWN
        )
        return QueryResult(found=True, value=result, outcome=outcome)

    def probe_task(self, task_id: str, task_session: str):
        controller = self.controllers.get((task_id, task_session))
        if controller is None:
            raise ValueError("unknown target task session")
        return ray.get(controller.probe_task.remote(task_session), timeout=30)

    def get_resource_snapshot(self, expected_session: str) -> dict:
        """GS physical-ledger inspection; not a ReleaseEvidence substitute."""
        return {
            "protocol_version": self.ledger.protocol.protocol_version,
            "gs_epoch": self.ledger.protocol.gs_epoch,
            "expected_session": expected_session,
            "tasks": [
                {
                    "task_id": record.task_id,
                    "task_session": record.task_session,
                    "status": record.status,
                    "initial_gpus": record.initial_gpus,
                }
                for record in self.ledger.tasks.values()
            ],
            "gpus": [
                {
                    "node_id": gpu.node_id,
                    "gpu_uuid": gpu.gpu_uuid,
                    "state": gpu.state,
                    "current_user": gpu.current_user,
                }
                for gpu in self.ledger.gpus.values()
            ],
        }

    def report_operation_result(self, result: OperationResult) -> Ack:
        if not isinstance(result, OperationResult):
            raise TypeError("report_operation_result requires OperationResult")
        self.ledger.merge_operation_result(result.ctx.operation_id, result)
        return Ack(accepted=True, revision=result.phase_revision)

    def open_lease(self, lease: LeaseRecord) -> dict:
        if not isinstance(lease, LeaseRecord):
            raise TypeError("open_lease requires LeaseRecord")
        self.ledger.open_lease(lease)
        self.lease_sm.register(lease)
        return {"lease_id": lease.lease_id, "state": lease.state}

    def advance_lease(
        self,
        lease_id: str,
        new_state: str,
        supporting_result_state: OperationStatus | None = None,
        operation_id: str | None = None,
    ) -> dict:
        self.lease_sm.advance(
            lease_id,
            LeaseState(new_state),
            supporting_result_state=supporting_result_state,
            operation_id=operation_id,
        )
        return {"lease_id": lease_id, "state": new_state}
