"""Shared GroupScheduler actor for simplified design §3/§8.

GS owns global resource authorization and intent. It forwards lifecycle commands
to the attached TaskRunner and never converts ACK/health/status into physical
release proof.
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
    OperationKind,
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

_EXPECTED_LEASE_STATE = {
    OperationKind.DONATE: LeaseState.DONOR_DRAINING,
    OperationKind.ADD: LeaseState.BORROWER_PREPARING,
    OperationKind.REMOVE: LeaseState.RECALLING,
    OperationKind.RESTORE: LeaseState.DONOR_RESTORING,
}


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

    def _validate_lease_authorization(self, command: OperationCommand) -> None:
        lease = self.ledger.leases.get(command.ctx.lease_id)
        if lease is None:
            raise ValueError("unknown lease")
        authorization = command.authorization
        if lease.lease_epoch != command.ctx.lease_epoch:
            raise ValueError("stale lease_epoch")
        if authorization.donor_session != lease.donor_session:
            raise ValueError("authorization donor_session does not match lease")
        if lease.borrower_session is None:
            raise ValueError("lease has no borrower_session")
        if authorization.borrower_session != lease.borrower_session:
            raise ValueError("authorization borrower_session does not match lease")
        if lease.placement is None:
            raise ValueError("lease has no placement evidence")
        if authorization.placement_digest != lease.placement.placement_digest:
            raise ValueError("authorization placement does not match lease")

        expected_state = _EXPECTED_LEASE_STATE[command.kind]
        if LeaseState(lease.state) is not expected_state:
            raise ValueError(
                f"lease state {lease.state} does not authorize {command.kind.value}"
            )

        expected_session = (
            lease.donor_session
            if command.kind in {OperationKind.DONATE, OperationKind.RESTORE}
            else lease.borrower_session
        )
        if command.ctx.task_session != expected_session:
            raise ValueError("operation targets the wrong lease participant")

        if command.kind in {OperationKind.ADD, OperationKind.RESTORE}:
            if lease.last_release_digest is None:
                raise ValueError("lease has no GS-confirmed prior release evidence")
            if authorization.prior_release_digest != lease.last_release_digest:
                raise ValueError("prior_release_digest does not match GS-confirmed release")

        self.lease_sm.validate_authorization(
            command.ctx.lease_id,
            command.ctx.operation_id,
            authorization.authorization_seq,
        )

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
        self._validate_lease_authorization(command)

    def submit_operation(self, command: OperationCommand) -> OperationResult:
        """Record/forward a new intent or replay an already accepted operation.

        Current lease-state authorization is evaluated only for a new operation.
        Once an operation_id has been accepted, later retries are fenced by its
        immutable recorded command and must keep returning/querying that same
        progress even after the lease moves to a later state.
        """
        if not isinstance(command, OperationCommand):
            raise TypeError("submit_operation requires OperationCommand")

        existing = self.ledger.operations.get(command.ctx.operation_id)
        if existing is None:
            self._validate_command_fence(command)
            record = self.ledger.record_operation(command)
            self.lease_sm.record_authorization(
                command.ctx.lease_id,
                command.ctx.operation_id,
                command.authorization.authorization_seq,
            )
        else:
            record = self.ledger.record_operation(command)

        if record.final_result is not None:
            return record.final_result

        controller = self.controllers.get(
            (record.command.ctx.task_id, record.command.ctx.task_session)
        )
        if controller is None:
            return self._local_query_result(record).value

        result = ray.get(controller.submit_operation.remote(record.command), timeout=30)
        if not isinstance(result, OperationResult):
            raise TypeError("TaskRunner returned a non-OperationResult")
        self.ledger.merge_operation_result(record.operation_id, result)
        return result

    @staticmethod
    def _local_query_result(record) -> QueryResult[OperationResult]:
        if record.final_result is not None:
            result = record.final_result
            if result.status is OperationStatus.SUCCEEDED:
                outcome = Outcome.KNOWN_APPLIED
            elif result.status is OperationStatus.ACCEPTED:
                outcome = Outcome.KNOWN_NOT_APPLIED
            elif result.status is OperationStatus.FAILED:
                error_outcome = getattr(result.error, "outcome", None)
                outcome = (
                    Outcome.UNKNOWN
                    if error_outcome is None
                    else Outcome(error_outcome)
                )
            else:
                outcome = Outcome.UNKNOWN
        else:
            result = OperationResult(
                ctx=record.command.ctx,
                target=record.command.target,
                status=OperationStatus.ACCEPTED,
                phase=Phase.VALIDATE,
                phase_revision=0,
            )
            outcome = Outcome.UNKNOWN
        return QueryResult(found=True, value=result, outcome=outcome)

    def query_operation(
        self, task_session: str, operation_id: str
    ) -> QueryResult[OperationResult]:
        """Prefer the TaskRunner authoritative journal; never infer from health."""
        record = self.ledger.operations.get(operation_id)
        if record is None or record.command.ctx.task_session != task_session:
            return QueryResult(found=False, value=None, outcome=Outcome.UNKNOWN)

        controller = self.controllers.get(
            (record.command.ctx.task_id, record.command.ctx.task_session)
        )
        if controller is None:
            return self._local_query_result(record)
        try:
            queried = ray.get(
                controller.query_operation.remote(task_session, operation_id),
                timeout=30,
            )
        except Exception:
            return self._local_query_result(record)
        if not isinstance(queried, QueryResult):
            raise TypeError("TaskRunner query returned a non-QueryResult")
        if queried.found and queried.value is not None:
            self.ledger.merge_operation_result(operation_id, queried.value)
        return queried

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
        operation_id: str | None = None,
    ) -> dict:
        """Advance authorization using the authoritative operation result when required."""
        supporting_result = None
        if operation_id is not None:
            record = self.ledger.operations.get(operation_id)
            if record is None:
                raise ValueError(f"unknown operation {operation_id!r}")
            queried = self.query_operation(record.command.ctx.task_session, operation_id)
            if queried.found:
                supporting_result = queried.value
        lease = self.lease_sm.advance(
            lease_id,
            LeaseState(new_state),
            supporting_result=supporting_result,
        )
        return {
            "lease_id": lease_id,
            "state": lease.state,
            "last_release_digest": lease.last_release_digest,
        }
