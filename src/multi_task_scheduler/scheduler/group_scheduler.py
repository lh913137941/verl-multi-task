"""Shared GroupScheduler actor for the minimal 092203 lease/command contract."""

from __future__ import annotations

import time

import ray
from ray.actor import ActorHandle

from multi_task_scheduler.orchestration.contracts import (
    EvidenceType,
    Lease,
    OperationCommand,
    OperationEvidence,
    OperationKind,
    OperationRecord,
)

RUNTIME_KIND = "verl-multi-task:experimental_fully_async_standalone:092203"
_RELEASE_KINDS = {OperationKind.DONATE, OperationKind.REMOVE}


@ray.remote(num_cpus=0)
class GroupScheduler:
    def __init__(self) -> None:
        self.task_runners: dict[str, ActorHandle] = {}
        self.leases: dict[str, Lease] = {}
        self.idle_reports: dict[str, object] = {}
        # GS keeps intent so later RELEASED evidence can be bound back to the
        # exact operation and lease instead of accepting a bare GPU list.
        self.operation_commands: dict[str, OperationCommand] = {}
        self.release_evidence: dict[tuple[str, str], OperationEvidence] = {}
        self.release_history: dict[str, list[str]] = {}

    def runtime_kind(self) -> str:
        return RUNTIME_KIND

    def attach_task(self, task_id: str, task_runner: ActorHandle) -> None:
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be a nonempty string")
        if not isinstance(task_runner, ActorHandle):
            raise TypeError("task_runner must be a real Ray ActorHandle")
        existing = self.task_runners.get(task_id)
        if existing is not None and existing != task_runner:
            raise ValueError(f"TaskRunner already attached for {task_id}")
        self.task_runners[task_id] = task_runner

    def detach_task(self, task_id: str) -> None:
        self.task_runners.pop(str(task_id), None)

    def get_task_runners(self) -> dict[str, ActorHandle]:
        return dict(self.task_runners)

    def schedule(self) -> list:
        return []

    def submit_idle_report(self, report):
        if not isinstance(report, dict):
            raise TypeError("submit_idle_report requires a metadata dict")
        task_session = report.get("task_session")
        candidates = report.get("candidates")
        if not isinstance(task_session, str) or not task_session:
            raise ValueError("idle report requires task_session")
        if not isinstance(candidates, (tuple, list)):
            raise ValueError("idle report requires candidates")
        self.idle_reports[task_session] = {
            "observed_at": time.monotonic(),
            "candidates": tuple(candidates),
        }
        return {"accepted": True, "candidate_count": len(candidates)}

    def submit_operation(self, command: OperationCommand) -> OperationRecord:
        if not isinstance(command, OperationCommand):
            raise TypeError("submit_operation requires OperationCommand")
        lease = self.leases.get(command.lease_id)
        if lease is None:
            raise ValueError(f"unknown lease {command.lease_id!r}")
        if lease.expires_at and time.time() >= lease.expires_at:
            raise ValueError(f"lease {command.lease_id!r} has expired")

        previous = self.operation_commands.get(command.operation_id)
        if previous is not None and previous != command:
            raise ValueError("conflicting operation replay at GroupScheduler")
        if previous is None:
            self.operation_commands[command.operation_id] = command

        task_runner = self.task_runners.get(command.target.task_session)
        if task_runner is None:
            raise ValueError(
                "no TaskRunner is attached under target.task_session; "
                "first release uses the attached task id as task_session"
            )
        result = ray.get(task_runner.submit_operation.remote(command), timeout=30)
        if not isinstance(result, OperationRecord):
            raise TypeError("TaskRunner returned a non-OperationRecord")
        return result

    def open_lease(self, lease: Lease) -> Lease:
        """GS-internal ledger action; scheduler policy calls this before command issue."""
        if not isinstance(lease, Lease):
            raise TypeError("open_lease requires Lease")
        existing = self.leases.get(lease.lease_id)
        if existing is not None:
            if existing != lease:
                raise ValueError("conflicting lease replay")
            return existing
        self.leases[lease.lease_id] = lease
        self.release_history.setdefault(lease.lease_id, [])
        return lease

    def advance_lease(self, lease_id: str, evidence: OperationEvidence) -> dict:
        """GS-internal release commit after exact operation/lease/GPU validation."""
        lease = self.leases.get(lease_id)
        if lease is None:
            raise ValueError(f"unknown lease {lease_id!r}")
        if not isinstance(evidence, OperationEvidence):
            raise TypeError("advance_lease requires OperationEvidence")
        if evidence.type is not EvidenceType.RELEASED:
            raise ValueError("lease handoff requires RELEASED evidence")

        command = self.operation_commands.get(evidence.operation_id)
        if command is None:
            raise ValueError("RELEASED evidence references an unknown operation")
        if command.lease_id != lease_id:
            raise ValueError("RELEASED evidence operation belongs to another lease")
        if command.kind not in _RELEASE_KINDS:
            raise ValueError("only DONATE/REMOVE operations may release lease GPUs")
        if set(evidence.released_gpu_uuids) != set(lease.gpu_uuids):
            raise ValueError("RELEASED evidence must exactly cover lease GPU claims")

        evidence_key = (lease_id, evidence.operation_id)
        existing = self.release_evidence.get(evidence_key)
        if existing is not None:
            if existing != evidence:
                raise ValueError("conflicting release evidence replay")
            return {
                "lease_id": lease_id,
                "operation_id": evidence.operation_id,
                "released": True,
            }

        self.release_evidence[evidence_key] = evidence
        self.release_history.setdefault(lease_id, []).append(evidence.operation_id)
        return {
            "lease_id": lease_id,
            "operation_id": evidence.operation_id,
            "released": True,
        }
