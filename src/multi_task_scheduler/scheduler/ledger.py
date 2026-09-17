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
    NodePlacement,
    OperationCommand,
    OperationResult,
    PlacementSpec,
    ReplicaKey,
    RuntimeCapabilities,
)
from multi_task_scheduler.orchestration.operation_journal import Phase


@dataclass
class TaskRecord:
    task_id: str
    task_session: str
    status: str = "INITIALIZING"
    config_summary: str = ""
    model_signature: str = ""
    capabilities: RuntimeCapabilities | None = None
    initial_gpus: int = 0
    min_active_gpus: int = 0
    max_active_gpus: int = 0
    last_probe: float = 0.0
    task_runner: Any = None


@dataclass
class NativeReplicaRecord:
    key: ReplicaKey
    owner_task_session: str
    node: NodePlacement
    gpu_uuids: Tuple[str, ...]
    status: str = "ACTIVE"
    revision: int = 0


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
    gpu_keys: Tuple[Tuple[str, str], ...] = ()
    borrower_session: str | None = None
    lease_epoch: int = 0
    state: str = "PLANNED"
    operation_ids: Tuple[str, ...] = ()
    last_operation_id: str | None = None
    deadline: float | None = None
    placement: PlacementSpec | None = None
    model_constraints: str = ""


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
        self.native_replicas: dict[ReplicaKey, NativeReplicaRecord] = {}
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
        config_summary: str = "",
        model_signature: str = "",
        min_active_gpus: int = 0,
        max_active_gpus: int = 0,
        task_runner: Any = None,
    ) -> TaskRecord:
        key = (task_id, task_session)
        record = self.tasks.get(key)
        if record is None:
            record = TaskRecord(task_id=task_id, task_session=task_session)
            self.tasks[key] = record
        if capabilities is not None:
            record.capabilities = capabilities
        record.config_summary = config_summary or record.config_summary
        record.model_signature = model_signature or record.model_signature
        record.min_active_gpus = min_active_gpus
        record.max_active_gpus = max_active_gpus
        if task_runner is not None:
            record.task_runner = task_runner
        record.last_probe = time.monotonic()
        return record

    def register_resources(self, manifest: ResourceManifest) -> NativeReplicaRecord:
        """Validate single-node physical identities atomically, then record GS facts."""
        if not any(
            record.task_session == manifest.owner_task_session
            for record in self.tasks.values()
        ):
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

        key = ReplicaKey(
            task_session=manifest.owner_task_session,
            replica_id=manifest.replica_id,
            runtime_epoch=manifest.runtime_epoch,
        )
        replica = NativeReplicaRecord(
            key=key,
            owner_task_session=manifest.owner_task_session,
            node=node,
            gpu_uuids=node.gpu_uuids,
        )
        self.native_replicas[key] = replica
        return replica

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

    def stale_idle_observations(self, now: float) -> Tuple[IdleObservation, ...]:
        return tuple(
            observation
            for observation in self.idle_observations.values()
            if observation.valid_until <= now
        )

    @staticmethod
    def _command_identity(command: OperationCommand) -> tuple:
        """Business identity; remaining transport budget may decrease on retry."""
        return (
            command.ctx,
            command.kind,
            command.target,
            command.authorization,
            command.placement,
            command.candidate,
            command.recall_mode,
            command.payload_digest,
        )

    def record_operation(self, command: OperationCommand) -> OperationRecord:
        existing = self.operations.get(command.ctx.operation_id)
        if existing is not None:
            if self._command_identity(existing.command) != self._command_identity(command):
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
