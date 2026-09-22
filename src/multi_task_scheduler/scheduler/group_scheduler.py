"""Shared GroupScheduler actor for the minimal 092203 lease/command contract."""

from __future__ import annotations

import time
import ray
from ray.actor import ActorHandle
from multi_task_scheduler.orchestration.contracts import EvidenceType, Lease, OperationCommand, OperationEvidence, OperationRecord

RUNTIME_KIND = "verl-multi-task:experimental_fully_async_standalone:092203"


@ray.remote(num_cpus=0)
class GroupScheduler:
    def __init__(self) -> None:
        self.task_runners: dict[str, ActorHandle] = {}
        self.leases: dict[str, Lease] = {}
        self.idle_reports: dict[str, object] = {}
        self.release_evidence: dict[str, OperationEvidence] = {}

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
        self.idle_reports[task_session] = {"observed_at": time.monotonic(), "candidates": tuple(candidates)}
        return {"accepted": True, "candidate_count": len(candidates)}

    def submit_operation(self, command: OperationCommand) -> OperationRecord:
        if not isinstance(command, OperationCommand):
            raise TypeError("submit_operation requires OperationCommand")
        if command.lease_id not in self.leases:
            raise ValueError(f"unknown lease {command.lease_id!r}")
        task_runner = self.task_runners.get(command.target.task_session)
        if task_runner is None:
            raise ValueError("no TaskRunner is attached under target.task_session; first release uses the attached task id as task_session")
        result = ray.get(task_runner.submit_operation.remote(command), timeout=30)
        if not isinstance(result, OperationRecord):
            raise TypeError("TaskRunner returned a non-OperationRecord")
        return result

    def open_lease(self, lease: Lease) -> Lease:
        if not isinstance(lease, Lease):
            raise TypeError("open_lease requires Lease")
        existing = self.leases.get(lease.lease_id)
        if existing is not None:
            if existing != lease:
                raise ValueError("conflicting lease replay")
            return existing
        self.leases[lease.lease_id] = lease
        return lease

    def advance_lease(self, lease_id: str, evidence: OperationEvidence) -> dict:
        lease = self.leases.get(lease_id)
        if lease is None:
            raise ValueError(f"unknown lease {lease_id!r}")
        if not isinstance(evidence, OperationEvidence):
            raise TypeError("advance_lease requires OperationEvidence")
        if evidence.type is not EvidenceType.RELEASED:
            raise ValueError("lease handoff requires RELEASED evidence")
        if set(evidence.released_gpu_uuids) != set(lease.gpu_uuids):
            raise ValueError("RELEASED evidence must exactly cover lease GPU claims")
        existing = self.release_evidence.get(lease_id)
        if existing is not None and existing != evidence:
            raise ValueError("conflicting release evidence for lease")
        self.release_evidence[lease_id] = evidence
        return {"lease_id": lease_id, "released": True}
