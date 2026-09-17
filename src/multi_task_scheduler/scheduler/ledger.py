"""GS global data model and ledger (section 3.3), pure Python.

The ledger records *intent* (what GS decided) and *observed* (what tasks
reported) separately, and never lets a desired state overwrite actual GPU
occupancy. ``native_owner`` comes from registration and is immutable while a
lease is active; ``current_user`` changes only through GS authorization plus a
verified receipt, never from a task's claim.

No Ray, verl, torch, or vLLM import here: the Actor passes opaque handles in
and reads summaries out.
"""

from __future__ import annotations

import time
import math
import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from multi_task_scheduler.orchestration.contracts import (
    NodeBlock,
    OperationResult,
    PlacementSpec,
    RuntimeCapabilities,
)


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
    owner_task_session: str
    replica_id: str
    node_blocks: Tuple[NodeBlock, ...]
    gpu_uuids: Tuple[str, ...]
    runtime_epoch: int = 0
    status: str = "ACTIVE"
    revision: int = 0


@dataclass
class GpuRecord:
    cluster_epoch: int
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
    def key(self) -> Tuple[int, str, str]:
        return (self.cluster_epoch, self.node_id, self.gpu_uuid)


@dataclass
class LeaseRecord:
    lease_id: str
    donor_session: str
    gpu_keys: Tuple[Tuple[str, str], ...] = ()  # (node_id, gpu_uuid)
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
    source_session: str
    replica_id: str
    source_seq: int
    production_epoch: int
    observed_age: float
    valid_until: float
    reason: str = ""

    @property
    def key(self) -> Tuple[str, str]:
        return (self.source_session, self.replica_id)


@dataclass
class OperationRecord:
    operation_id: str
    command: Any
    payload_digest: str
    command_seq: int
    accepted_at: float = field(default_factory=time.monotonic)
    phase: str = "ACCEPTED"
    final_result: OperationResult | None = None


@dataclass
class ProtocolInstance:
    protocol_version: str
    runtime_kind: str
    gs_epoch: int
    sharing_namespace: str
    recovery_state: str = "RECOVERY_ONLY"


@dataclass(frozen=True)
class ResourceManifest:
    """Registration payload: placement + per-replica native ownership."""

    owner_task_session: str
    placement: PlacementSpec
    replica_id: str
    runtime_epoch: int = 0


@dataclass(frozen=True)
class IdleReport:
    """LB -> GS idle-candidate report (section 4.4)."""

    source_session: str
    source_seq: int
    production_epoch: int
    candidate_ids: Tuple[str, ...] = ()
    candidate_reasons: Mapping[str, str] = field(default_factory=dict)
    valid_for_ms: int = 1000


class Ledger:
    """Intent + observed storage with the section 3.3 update rules."""

    def __init__(self, protocol: ProtocolInstance) -> None:
        self.protocol = protocol
        self.tasks: dict[Tuple[str, str], TaskRecord] = {}
        self.native_replicas: dict[Tuple[str, str], NativeReplicaRecord] = {}
        self.gpus: dict[Tuple[int, str, str], GpuRecord] = {}
        self.leases: dict[str, LeaseRecord] = {}
        self.idle_observations: dict[Tuple[str, str], IdleObservation] = {}
        self.operations: dict[str, OperationRecord] = {}
        self._idle_reports: dict[str, IdleReport] = {}

    # -- tasks ----------------------------------------------------------- #

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

    # -- resources -------------------------------------------------------- #

    def register_resources(self, manifest: ResourceManifest) -> NativeReplicaRecord:
        """Validate GPU uniqueness and capability match, then record natives."""
        if not any(record.task_session == manifest.owner_task_session for record in self.tasks.values()):
            raise ValueError(f"unknown owner task session {manifest.owner_task_session!r}")

        seen: set[Tuple[int, str, str]] = set()
        for block in manifest.placement.node_blocks:
            for gpu in block.gpus:
                gkey = (self.protocol.gs_epoch, block.node_id, gpu.gpu_uuid)
                if gkey in self.gpus or gkey in seen:
                    raise ValueError(f"GPU already registered: {gkey}")
                seen.add(gkey)

        for block in manifest.placement.node_blocks:
            for gpu in block.gpus:
                gkey = (self.protocol.gs_epoch, block.node_id, gpu.gpu_uuid)
                self.gpus[gkey] = GpuRecord(
                    cluster_epoch=self.protocol.gs_epoch,
                    node_id=block.node_id,
                    gpu_uuid=gpu.gpu_uuid,
                    physical_id=gpu.physical_id,
                    native_owner=manifest.owner_task_session,
                )

        replica_key = (manifest.owner_task_session, manifest.replica_id)
        replica = NativeReplicaRecord(
            owner_task_session=manifest.owner_task_session,
            replica_id=manifest.replica_id,
            node_blocks=manifest.placement.node_blocks,
            gpu_uuids=manifest.placement.gpu_uuids,
            runtime_epoch=manifest.runtime_epoch,
        )
        self.native_replicas[replica_key] = replica
        return replica

    # -- GPU authorization (GS-only) ------------------------------------- #

    def set_current_user(
        self,
        gpu_key: Tuple[int, str, str],
        task_session: str,
        lease_id: str,
        lease_epoch: int,
    ) -> None:
        gpu = self.gpus[gpu_key]
        if gpu.current_user is not None and gpu.current_user != task_session:
            raise ValueError(
                f"GPU {gpu_key} is already authorized to {gpu.current_user}"
            )
        gpu.current_user = task_session
        gpu.lease_id = lease_id
        gpu.lease_epoch = lease_epoch
        gpu.state = "LENT"

    def clear_current_user(self, gpu_key: Tuple[int, str, str]) -> None:
        gpu = self.gpus[gpu_key]
        gpu.current_user = None
        gpu.lease_id = None
        gpu.lease_epoch = 0
        gpu.state = "FREE"

    # -- leases ----------------------------------------------------------- #

    def open_lease(self, lease: LeaseRecord) -> None:
        if lease.lease_id in self.leases:
            raise ValueError(f"lease already exists: {lease.lease_id}")
        self.leases[lease.lease_id] = lease

    def get_lease(self, lease_id: str) -> LeaseRecord:
        return self.leases[lease_id]

    # -- idle observations ------------------------------------------------ #

    def upsert_idle_observation(self, report: IdleReport, valid_until: float) -> None:
        if not math.isfinite(valid_until):
            raise ValueError("Idle observations require a finite expiry")
        if type(report.source_seq) is not int or report.source_seq < 0:
            raise ValueError("source_seq must be a nonnegative integer")
        previous = self._idle_reports.get(report.source_session)
        if previous is not None:
            if report.source_seq < previous.source_seq:
                return
            if report.source_seq == previous.source_seq:
                if report != previous:
                    raise ValueError("conflicting idle report replay")
                return  # A repeated observation never renews its TTL.
            if report.production_epoch < previous.production_epoch:
                raise ValueError("stale production epoch")
        # Reports replace the whole set, including an empty withdrawal.
        self.idle_observations = {
            key: value for key, value in self.idle_observations.items()
            if key[0] != report.source_session
        }
        self._idle_reports[report.source_session] = copy.deepcopy(report)
        for replica_id in report.candidate_ids:
            obs = IdleObservation(
                source_session=report.source_session,
                replica_id=replica_id,
                source_seq=report.source_seq,
                production_epoch=report.production_epoch,
                observed_age=0.0,
                valid_until=valid_until,
                reason=report.candidate_reasons.get(replica_id, ""),
            )
            self.idle_observations[obs.key] = obs

    def stale_idle_observations(self, now: float) -> Tuple[IdleObservation, ...]:
        return tuple(o for o in self.idle_observations.values() if o.valid_until <= now)

    # -- operations ------------------------------------------------------- #

    def record_operation(self, command: Any) -> OperationRecord:
        existing = self.operations.get(command.operation_id)
        if existing is not None:
            if existing.payload_digest != command.payload_digest:
                raise ValueError(
                    f"conflicting payload digest for {command.operation_id}"
                )
            return existing
        record = OperationRecord(
            operation_id=command.operation_id,
            command=command,
            payload_digest=command.payload_digest,
            command_seq=command.command_seq,
        )
        self.operations[command.operation_id] = record
        return record

    def merge_operation_result(
        self, operation_id: str, result: OperationResult
    ) -> OperationRecord:
        record = self.operations.get(operation_id)
        if record is None:
            raise ValueError(f"unknown operation {operation_id!r}")
        # Lower phase_revision / older identity fields are history, never overwrite.
        if record.final_result is not None:
            if result.phase_revision < record.final_result.phase_revision:
                return record
            if result.phase_revision == record.final_result.phase_revision:
                return record
        record.final_result = result
        record.phase = result.phase
        return record
