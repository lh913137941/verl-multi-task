"""The shared GroupScheduler Actor: ledger, leases, and control interfaces.

The Actor keeps two concerns separate (section 3.3): *intent* (what GS decided,
stored in the ledger before a command is issued) and *observed* (what tasks
report, merged by receipt). Resource scheduling policy is still unimplemented:
``schedule`` returns no decisions, and ``submit_operation`` records intent and
returns immediately rather than driving a task (section 4.5 — no long RPC
under a ledger mutation).

Each method mutates local state and returns, so a single synchronous Ray Actor
already serializes ledger writes; there is no second lock to hold across a
network call.
"""

import ray
import time
from ray.actor import ActorHandle

from multi_task_scheduler.orchestration.contracts import Command, OperationResult
from multi_task_scheduler.scheduler.ledger import (
    IdleReport,
    Ledger,
    LeaseRecord,
    ProtocolInstance,
    ResourceManifest,
)
from multi_task_scheduler.scheduler.lease import LeaseState, LeaseStateMachine

RUNTIME_KIND = "verl-multi-task:experimental_fully_async_standalone:p1"
PROTOCOL_VERSION = "p1"


@ray.remote(num_cpus=0)
class GroupScheduler:
    """Keep TaskRunner controllers and the global ledger/lease state.

    It is not a heartbeat monitor or a scheduling policy; those stay out of
    scope for this pass.
    """

    def __init__(self) -> None:
        self.task_runners: dict[str, ActorHandle] = {}
        self.controllers: dict[tuple[str, str], ActorHandle] = {}
        self.ledger = Ledger(
            ProtocolInstance(
                protocol_version=PROTOCOL_VERSION,
                runtime_kind=RUNTIME_KIND,
                gs_epoch=1,
                sharing_namespace="verl-multi-task",
                recovery_state="RECOVERY_ONLY",
            )
        )
        self.lease_sm = LeaseStateMachine()

    # -- discovery / legacy ------------------------------------------------- #

    def runtime_kind(self) -> str:
        """Identify this implementation when a job discovers a named Actor."""
        return RUNTIME_KIND

    def attach_task(self, task_id: str, task_runner: ActorHandle) -> None:
        """Legacy single-key reference; kept for the passthrough integration."""
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be a nonempty TaskRunner Actor ID")
        if not isinstance(task_runner, ActorHandle):
            raise TypeError("GroupScheduler requires a real TaskRunner ActorHandle")
        existing = self.task_runners.get(task_id)
        if existing is not None and existing != task_runner:
            raise ValueError(f"TaskRunner reference already exists for {task_id}")
        self.task_runners[task_id] = task_runner

    def detach_task(self, task_id: str) -> None:
        """Release a completed TaskRunner reference without changing resources."""
        self.task_runners.pop(task_id, None)

    def get_task_runners(self) -> dict[str, ActorHandle]:
        """Return the current reference map for initialization verification."""
        return dict(self.task_runners)

    def schedule(self) -> list:
        """Empty extension: no automatic scheduling policy is implemented yet."""
        return []

    # -- 4.4 GS <-> task control interfaces -------------------------------- #

    def attach_controller(
        self,
        task_id: str,
        task_session: str,
        handle: ActorHandle,
        protocol: dict | None = None,
    ) -> dict:
        """Establish a control reference; status INITIALIZING (section 4.4)."""
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be a nonempty string")
        if not isinstance(task_session, str) or not task_session:
            raise ValueError("task_session must be a nonempty string")
        if not isinstance(handle, ActorHandle):
            raise TypeError("attach_controller requires a real TaskRunner ActorHandle")
        self.ledger.register_task(task_id, task_session, task_runner=handle)
        self.controllers[(task_id, task_session)] = handle
        return {"status": "INITIALIZING", "task_id": task_id, "task_session": task_session}

    def register_resources(self, registration_id: str, manifest: ResourceManifest) -> dict:
        """Validate GPU uniqueness and capability match, then READY (section 4.4)."""
        if not isinstance(manifest, ResourceManifest):
            raise TypeError("register_resources requires a ResourceManifest")
        self.ledger.register_resources(manifest)
        return {"registration_id": registration_id, "status": "READY"}

    def probe_task(self, probe_id: str, known_revisions: dict | None = None) -> dict:
        """Return sessions, status, and resource/lease summaries (section 4.4)."""
        tasks = [
            {
                "task_id": r.task_id,
                "task_session": r.task_session,
                "status": r.status,
            }
            for r in self.ledger.tasks.values()
        ]
        leases = [
            {"lease_id": l.lease_id, "state": l.state} for l in self.ledger.leases.values()
        ]
        return {"probe_id": probe_id, "tasks": tasks, "leases": leases}

    def report_idle_candidates(self, report: IdleReport) -> dict:
        """Only refresh IdleObservation; never drain, sleep, or transfer (5.1)."""
        if not isinstance(report, IdleReport):
            raise TypeError("report_idle_candidates requires an IdleReport")
        if type(report.valid_for_ms) is not int or report.valid_for_ms <= 0:
            raise ValueError("valid_for_ms must be a positive integer")
        self.ledger.upsert_idle_observation(
            report, valid_until=time.monotonic() + report.valid_for_ms / 1000
        )
        return {"source_seq": report.source_seq, "candidates": list(report.candidate_ids)}

    def submit_operation(self, command: Command) -> dict:
        """Record intent and return ACCEPTED/REJECTED without executing it."""
        if not isinstance(command, Command):
            raise TypeError("submit_operation requires a Command")
        try:
            self.ledger.record_operation(command)
        except ValueError as exc:
            return {"state": "REJECTED", "operation_id": command.operation_id, "error": str(exc)}
        return {"state": "ACCEPTED", "operation_id": command.operation_id}

    def query_operation(self, operation_id: str) -> dict | None:
        """Return the recorded phase and final result for reconciliation."""
        record = self.ledger.operations.get(operation_id)
        if record is None:
            return None
        return {
            "operation_id": operation_id,
            "phase": record.phase,
            "final_result": record.final_result,
        }

    def get_resource_snapshot(self, expected_session: str) -> dict:
        """Return manager/CE/LB metadata and revisions (section 4.4)."""
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
        """Idempotently merge a task-reported result; older revisions are history."""
        if not isinstance(result, OperationResult):
            raise TypeError("report_operation_result requires an OperationResult")
        self.ledger.merge_operation_result(result.identity_fields.operation_id, result)
        return {"operation_id": result.identity_fields.operation_id, "merged": True}

    # -- lease authorization (receipt-driven) ------------------------------- #

    def open_lease(self, lease: LeaseRecord) -> dict:
        """Write the lease intent before issuing any command (section 5.6)."""
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
        """Advance a lease along the section 5.6 edges."""
        self.lease_sm.advance(
            lease_id,
            LeaseState(new_state),
            supporting_result_state=supporting_result_state,
            operation_id=operation_id,
        )
        return {"lease_id": lease_id, "state": new_state}
