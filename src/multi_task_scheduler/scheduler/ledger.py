"""GS global ledger aligned with simplified design §6 and §8.

The ledger stores GS intent separately from task-observed results. Replays are
accepted only for the same immutable business operation; result merging is
fenced by the exact OperationContext and ReplicaKey before revision ordering.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Tuple

from multi_task_scheduler.orchestration.contracts import (
    IdleCandidate,
    IdleCandidateReport,
    OperationCommand,
    OperationResult,
    PlacementSpec,
    ReplicaKey,
    RuntimeCapabilities,
)
from multi_task_scheduler.orchestration.operation_journal import Phase, command_identity


@dataclass
class TaskRecord:
    task_id: str
    task_session: str
    status: str = "INITIALIZING"
    capabilities: RuntimeCapabilities | None = None
    initial_gpus: int = 0
    task_runner: Any = None


@dataclass
class GpuRecord:
    cluster_epoch: str
    node_id: str
    gpu_uuid: str
    physical_id: int
    native_owner: str
    current_user: str | None = None
    lease_id: str | None = None
    lease_epoch: int = 0
    state: str = "FREE"
    reserved_residual: int = 0

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.cluster_epoch, self.node_id, self.gpu_uuid)


@dataclass
class LeaseRecord:
    lease_id: str
    donor_session: str
    borrower_session: str | None = None
    lease_epoch: int = 0
    state: str = "PLANNED"
    last_release_digest: str | None = None
    placement: PlacementSpec | None = None

    def __post_init__(self) -> None:
        if not self.lease_id or not self.donor_session:
            raise ValueError("lease_id and donor_session must be nonempty")
        if self.lease_epoch < 0:
            raise ValueError("lease_epoch must be nonnegative")
        if self.borrower_session is not None and not self.borrower_session:
            raise ValueError("borrower_session must be nonempty when present")
        if self.last_release_digest is not None and not self.last_release_digest:
            raise ValueError("last_release_digest must be nonempty when present")


@dataclass(frozen=True)
class IdleObservation:
    candidate: IdleCandidate
    source_seq: int
    valid_until: float

    @property
    def key(self) -> ReplicaKey:
        return self.candidate.key


@dataclass
class OperationRecord:
    operation_id: str
    command: OperationCommand
    accepted_at: float = field(default_factory=time.monotonic)
    phase: Phase = Phase.VALIDATE
    final_result: OperationResult | None = None

    @property
    def deadline_at(self) -> float:
        """Deadline fixed by the first accepted command; retries never reset it."""
        return self.accepted_at + self.command.remaining_budget_ms / 1000.0


@dataclass
class ProtocolInstance:
    protocol_version: int
    runtime_kind: str
    gs_epoch: str
    sharing_namespace: str
    recovery_state: str = "RECOVERY_ONLY"

    def __post_init__(self) -> None:
        if type(self.protocol_version) is not int or self.protocol_version < 0:
            raise ValueError("protocol_version must be a nonnegative integer")
        if not self.gs_epoch:
            raise ValueError("gs_epoch must be a nonempty string")


@dataclass(frozen=True)
class ResourceManifest:
    owner_task_session: str
    placement: PlacementSpec
    replica_id: str
    runtime_epoch: int = 0

    def __post_init__(self) -> None:
        if not self.owner_task_session or not self.replica_id:
            raise ValueError("resource manifest identities must be nonempty")
        if self.runtime_epoch < 0:
            raise ValueError("runtime_epoch must be nonnegative")


class Ledger:
    """Intent + observed storage with owner/fencing rules from the simplified design."""

    def __init__(self, protocol: ProtocolInstance) -> None:
        self.protocol = protocol
        self.tasks: dict[Tuple[str, str], TaskRecord] = {}
        self.gpus: dict[Tuple[str, str, str], GpuRecord] = {}
        self.leases: dict[str, LeaseRecord] = {}
        self.idle_observations: dict[ReplicaKey, IdleObservation] = {}
        self.operations: dict[str, OperationRecord] = {}
        self._idle_reports: dict[Tuple[str, str], IdleCandidateReport] = {}

    def register_task(
        self,
        task_id: str,
        task_session: str,
        *,
        capabilities: RuntimeCapabilities | None = None,
        task_runner: Any = None,
    ) -> TaskRecord:
        key = (task_id, task_session)
        record = self.tasks.get(key)
        if record is None:
            record = TaskRecord(task_id=task_id, task_session=task_session)
            self.tasks[key] = record
        if capabilities is not None:
            record.capabilities = capabilities
        if task_runner is not None:
            record.task_runner = task_runner
        return record

    def register_resources(self, manifest: ResourceManifest) -> None:
        """Validate single-node physical identities atomically, then record GS GPU facts."""
        owner = next(
            (
                record
                for record in self.tasks.values()
                if record.task_session == manifest.owner_task_session
            ),
            None,
        )
        if owner is None:
            raise ValueError(f"unknown owner task session {manifest.owner_task_session!r}")

        node = manifest.placement.node
        seen: set[Tuple[str, str, str]] = set()
        for gpu_uuid in node.gpu_uuids:
            gkey = (self.protocol.gs_epoch, node.node_id, gpu_uuid)
            if gkey in self.gpus or gkey in seen:
                raise ValueError(f"GPU already registered: {gkey}")
            seen.add(gkey)

        new_gpus = {
            (self.protocol.gs_epoch, node.node_id, gpu_uuid): GpuRecord(
                cluster_epoch=self.protocol.gs_epoch,
                node_id=node.node_id,
                gpu_uuid=gpu_uuid,
                physical_id=physical_id,
                native_owner=manifest.owner_task_session,
            )
            for gpu_uuid, physical_id in zip(node.gpu_uuids, node.physical_gpu_ids)
        }
        self.gpus.update(new_gpus)
        owner.initial_gpus += len(new_gpus)

    def set_current_user(
        self,
        gpu_key: Tuple[str, str, str],
        task_session: str,
        lease_id: str,
        lease_epoch: int,
    ) -> None:
        gpu = self.gpus[gpu_key]
        if gpu.current_user is not None and gpu.current_user != task_session:
            raise ValueError(f"GPU {gpu_key} is already authorized to {gpu.current_user}")
        gpu.current_user = task_session
        gpu.lease_id = lease_id
        gpu.lease_epoch = lease_epoch
        gpu.state = "LENT"

    def clear_current_user(self, gpu_key: Tuple[str, str, str]) -> None:
        gpu = self.gpus[gpu_key]
        gpu.current_user = None
        gpu.lease_id = None
        gpu.lease_epoch = 0
        gpu.state = "FREE"

    def _lease_gpu_records(
        self, lease: LeaseRecord
    ) -> Tuple[Tuple[Tuple[str, str, str], GpuRecord], ...]:
        """Resolve a lease placement to registered GS GPU records.

        ``current_user is None`` means the registered native owner retains the
        implicit authorization. A borrower authorization is always explicit and
        carries the matching lease identity on every GPU record.
        """
        if lease.placement is None:
            raise ValueError("lease placement is required for GPU authorization")
        node = lease.placement.node
        records = []
        for gpu_uuid in node.gpu_uuids:
            gpu_key = (self.protocol.gs_epoch, node.node_id, gpu_uuid)
            gpu = self.gpus.get(gpu_key)
            if gpu is None:
                raise ValueError(f"lease references unregistered GPU {gpu_key}")
            if gpu.native_owner != lease.donor_session:
                raise ValueError(
                    f"GPU {gpu_key} native owner {gpu.native_owner!r} "
                    f"does not match donor {lease.donor_session!r}"
                )
            records.append((gpu_key, gpu))
        return tuple(records)

    def validate_borrower_gpu_authorization(
        self, lease: LeaseRecord
    ) -> Tuple[Tuple[str, str, str], ...]:
        """Preflight donor->borrower authorization without mutating GS state."""
        if lease.borrower_session is None:
            raise ValueError("borrower_session is required for borrower authorization")
        records = self._lease_gpu_records(lease)
        for gpu_key, gpu in records:
            if gpu.current_user is not None:
                raise ValueError(f"GPU {gpu_key} already has current_user {gpu.current_user!r}")
            if gpu.lease_id is not None or gpu.state != "FREE":
                raise ValueError(f"GPU {gpu_key} is not free for borrower authorization")
        return tuple(gpu_key for gpu_key, _ in records)

    def authorize_borrower_gpus(
        self,
        lease: LeaseRecord,
        gpu_keys: Tuple[Tuple[str, str, str], ...],
    ) -> None:
        """Commit a previously validated donor->borrower authorization."""
        if lease.borrower_session is None:
            raise ValueError("borrower_session is required for borrower authorization")
        for gpu_key in gpu_keys:
            self.set_current_user(
                gpu_key,
                lease.borrower_session,
                lease.lease_id,
                lease.lease_epoch,
            )

    def validate_native_gpu_restoration(
        self, lease: LeaseRecord
    ) -> Tuple[Tuple[str, str, str], ...]:
        """Preflight borrower->native authorization without mutating GS state."""
        if lease.borrower_session is None:
            raise ValueError("borrower_session is required for native restoration")
        records = self._lease_gpu_records(lease)
        for gpu_key, gpu in records:
            if gpu.current_user != lease.borrower_session:
                raise ValueError(
                    f"GPU {gpu_key} is not authorized to borrower {lease.borrower_session!r}"
                )
            if gpu.lease_id != lease.lease_id or gpu.lease_epoch != lease.lease_epoch:
                raise ValueError(f"GPU {gpu_key} borrower authorization belongs to another lease")
            if gpu.state != "LENT":
                raise ValueError(f"GPU {gpu_key} is not in LENT state")
        return tuple(gpu_key for gpu_key, _ in records)

    def restore_native_gpus(
        self,
        lease: LeaseRecord,
        gpu_keys: Tuple[Tuple[str, str, str], ...],
    ) -> None:
        """Commit a previously validated borrower->native authorization."""
        for gpu_key in gpu_keys:
            self.clear_current_user(gpu_key)

    def open_lease(self, lease: LeaseRecord) -> None:
        if lease.lease_id in self.leases:
            raise ValueError(f"lease already exists: {lease.lease_id}")
        self.leases[lease.lease_id] = lease

    def get_lease(self, lease_id: str) -> LeaseRecord:
        return self.leases[lease_id]

    def upsert_idle_report(self, report: IdleCandidateReport, valid_until: float) -> None:
        if report.gs_epoch != self.protocol.gs_epoch:
            raise ValueError("stale idle report gs_epoch")
        if not math.isfinite(valid_until):
            raise ValueError("idle observations require a finite expiry")
        report_key = (report.task_session, report.lb_session)
        previous = self._idle_reports.get(report_key)
        if previous is not None:
            if report.source_seq < previous.source_seq:
                return
            if report.source_seq == previous.source_seq:
                if report != previous:
                    raise ValueError("conflicting idle report replay")
                return
            if report.production_revision < previous.production_revision:
                raise ValueError("stale production revision")

        self.idle_observations = {
            key: value
            for key, value in self.idle_observations.items()
            if key.task_session != report.task_session
        }
        self._idle_reports[report_key] = copy.deepcopy(report)
        for candidate in report.candidates:
            self.idle_observations[candidate.key] = IdleObservation(
                candidate=candidate,
                source_seq=report.source_seq,
                valid_until=valid_until,
            )

    def record_operation(self, command: OperationCommand) -> OperationRecord:
        existing = self.operations.get(command.ctx.operation_id)
        if existing is not None:
            if command_identity(existing.command) != command_identity(command):
                raise ValueError(f"conflicting replay for operation {command.ctx.operation_id}")
            return existing
        record = OperationRecord(
            operation_id=command.ctx.operation_id,
            command=command,
        )
        self.operations[command.ctx.operation_id] = record
        return record

    @staticmethod
    def _result_matches_command(command: OperationCommand, result: OperationResult) -> bool:
        return result.ctx == command.ctx and result.target == command.target

    def merge_operation_result(
        self, operation_id: str, result: OperationResult
    ) -> OperationRecord:
        record = self.operations.get(operation_id)
        if record is None:
            raise ValueError(f"unknown operation {operation_id!r}")
        if result.ctx.operation_id != operation_id:
            raise ValueError("operation result id does not match merge target")
        if not self._result_matches_command(record.command, result):
            raise ValueError(f"stale or mismatched operation result for {operation_id!r}")

        if record.final_result is not None:
            if result.phase_revision < record.final_result.phase_revision:
                return record
            if result.phase_revision == record.final_result.phase_revision:
                if result != record.final_result:
                    raise ValueError(
                        f"conflicting result replay at revision {result.phase_revision}"
                    )
                return record
        record.final_result = result
        record.phase = result.phase
        return record
