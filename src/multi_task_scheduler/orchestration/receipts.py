"""Canonical lifecycle evidence for the simplified orchestration contract.

Evidence objects represent facts already established by their state owner. There
are no legacy success receipts or boolean shortcuts in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias, Tuple

from .contracts import (
    CleanupInventory,
    OperationContext,
    OperationError,
    PreparedReplica,
    ProcessIdentity,
    RecallMode,
    ReleaseKind,
    ReplicaKey,
    ServiceAction,
)


class CommitOwner(str, Enum):
    CE = "CE"
    LB = "LB"


@dataclass(frozen=True)
class EvidenceHeader:
    ctx: OperationContext
    key: ReplicaKey
    phase_revision: int
    digest: str

    def __post_init__(self) -> None:
        if self.key.task_session != self.ctx.task_session:
            raise ValueError("evidence key task_session must match operation context")
        if self.phase_revision < 0:
            raise ValueError("phase_revision must be nonnegative")
        if not self.digest:
            raise ValueError("evidence digest must be nonempty")


@dataclass(frozen=True)
class DrainTicket:
    header: EvidenceHeader
    drain_id: str
    route_epoch: int
    server_admission_epoch: int

    def __post_init__(self) -> None:
        if not self.drain_id:
            raise ValueError("drain_id must be nonempty")
        if self.route_epoch < 0 or self.server_admission_epoch < 0:
            raise ValueError("route/admission epochs must be nonnegative")


@dataclass(frozen=True)
class WeightEvidence:
    header: EvidenceHeader
    transfer_id: str
    snapshot_id: str
    manifest_digest: str
    version: int
    receiver_versions: dict[str, int]
    device_complete: bool
    temporary_topology_clean: bool

    def __post_init__(self) -> None:
        if not self.transfer_id or not self.snapshot_id or not self.manifest_digest:
            raise ValueError("weight evidence identifiers must be nonempty")
        if self.version < 0:
            raise ValueError("weight version must be nonnegative")
        if not self.receiver_versions:
            raise ValueError("successful WeightEvidence requires receiver results")
        if any(not receiver_id for receiver_id in self.receiver_versions):
            raise ValueError("receiver ids must be nonempty")
        if any(version != self.version for version in self.receiver_versions.values()):
            raise ValueError("every receiver must load the evidence version")
        if not self.device_complete or not self.temporary_topology_clean:
            raise ValueError("WeightEvidence exists only after device/topology completion")


@dataclass(frozen=True)
class CommitReceipt:
    header: EvidenceHeader
    owner: CommitOwner
    action: ServiceAction
    revision: int
    version: int | None
    route_epoch: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner", CommitOwner(self.owner))
        object.__setattr__(self, "action", ServiceAction(self.action))
        if self.revision < 0:
            raise ValueError("commit revision must be nonnegative")
        if self.action is ServiceAction.ADD:
            if self.version is None or self.version < 0:
                raise ValueError("ADD commit requires a nonnegative version")
        elif self.version is not None:
            raise ValueError("REMOVE commit version must be None")
        if self.owner is CommitOwner.CE:
            if self.route_epoch is not None:
                raise ValueError("CE commit must not carry route_epoch")
        else:
            if self.route_epoch is None or self.route_epoch < 0:
                raise ValueError("LB commit requires route_epoch")


@dataclass(frozen=True)
class ContinuationRecord:
    ctx: OperationContext
    key: ReplicaKey
    drain_id: str
    old_attempt_id: str
    client_session: str
    prefix_digest: str
    old_terminal: object
    old_release_acked: bool
    new_acceptance: object | None
    completed_turn: object | None

    def __post_init__(self) -> None:
        if self.key.task_session != self.ctx.task_session:
            raise ValueError("continuation key must match operation task_session")
        for name in ("drain_id", "old_attempt_id", "client_session", "prefix_digest"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        if self.old_terminal is None or not self.old_release_acked:
            raise ValueError("continuation requires terminal and release acknowledgement")
        if getattr(self.old_terminal, "device_finished", True) is not True:
            raise ValueError("old attempt device work must be finished")
        if (self.new_acceptance is None) == (self.completed_turn is None):
            raise ValueError("exactly one of new_acceptance/completed_turn must be present")


@dataclass(frozen=True)
class ExitEvidence:
    header: EvidenceHeader
    drain_id: str
    recall_mode: RecallMode
    inflight: int
    admitting: int
    queued: int
    running: int
    pending_admissions: int
    closed_admission: bool
    all_backends_confirmed: bool
    lb_revision: int
    observed_age_ms: int
    engine_digest: str
    continuations: Tuple[ContinuationRecord, ...]
    unresolved_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "recall_mode", RecallMode(self.recall_mode))
        if not self.drain_id or not self.engine_digest:
            raise ValueError("drain_id and engine_digest must be nonempty")
        counts = (
            self.inflight,
            self.admitting,
            self.queued,
            self.running,
            self.pending_admissions,
            self.unresolved_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("exit counters must be nonnegative")
        if self.lb_revision < 0 or self.observed_age_ms < 0:
            raise ValueError("lb_revision and observed_age_ms must be nonnegative")
        if any(value != 0 for value in counts):
            raise ValueError("successful ExitEvidence requires all request counters to be zero")
        if not self.closed_admission or not self.all_backends_confirmed:
            raise ValueError("successful ExitEvidence requires closed admission and backend confirmation")
        if self.recall_mode is RecallMode.NATURAL and self.continuations:
            raise ValueError("NATURAL ExitEvidence must not contain continuations")
        if self.recall_mode is RecallMode.FORCE_VERIFIED:
            for record in self.continuations:
                if record.ctx != self.header.ctx or record.key != self.header.key:
                    raise ValueError("continuation identity must match exit evidence")
                if record.drain_id != self.drain_id:
                    raise ValueError("continuation drain_id must match exit evidence")


@dataclass(frozen=True)
class ServiceEvidence:
    header: EvidenceHeader
    action: ServiceAction
    version: int | None
    ce_revision: int
    lb_revision: int
    route_epoch: int
    capacity_revision: int
    manager_revision: int
    ce_commit_digest: str
    lb_commit_digest: str
    prerequisite_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", ServiceAction(self.action))
        for value in (
            self.ce_revision,
            self.lb_revision,
            self.route_epoch,
            self.capacity_revision,
            self.manager_revision,
        ):
            if value < 0:
                raise ValueError("service revisions/epoch must be nonnegative")
        for name in ("ce_commit_digest", "lb_commit_digest", "prerequisite_digest"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        if self.action is ServiceAction.ADD:
            if self.version is None or self.version < 0:
                raise ValueError("ADD ServiceEvidence requires version")
        elif self.version is not None:
            raise ValueError("REMOVE ServiceEvidence requires version=None")


@dataclass(frozen=True)
class GPURelease:
    gpu_uuid: str
    free_hbm_bytes: int
    residual_hbm_bytes: int
    owned_processes: Tuple[ProcessIdentity, ...]
    unknown_processes: Tuple[ProcessIdentity, ...]
    device_work_complete: bool
    meets_release_budget: bool

    def __post_init__(self) -> None:
        if not self.gpu_uuid:
            raise ValueError("gpu_uuid must be nonempty")
        if self.free_hbm_bytes < 0 or self.residual_hbm_bytes < 0:
            raise ValueError("HBM byte counts must be nonnegative")
        if self.unknown_processes:
            raise ValueError("successful GPURelease cannot contain unknown processes")
        if not self.device_work_complete or not self.meets_release_budget:
            raise ValueError("successful GPURelease requires completed device work and budget")


@dataclass(frozen=True)
class ReleaseEvidence:
    header: EvidenceHeader
    release_kind: ReleaseKind
    permit_digest: str
    inventory_digest: str
    per_gpu: Tuple[GPURelease, ...]
    all_backends_confirmed: bool
    observation_interval_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "release_kind", ReleaseKind(self.release_kind))
        if not self.permit_digest or not self.inventory_digest:
            raise ValueError("permit_digest and inventory_digest must be nonempty")
        if not self.per_gpu:
            raise ValueError("ReleaseEvidence requires per-GPU proof")
        if not self.all_backends_confirmed:
            raise ValueError("ReleaseEvidence requires all backend confirmations")
        if self.observation_interval_ms < 0:
            raise ValueError("observation_interval_ms must be nonnegative")
        if self.release_kind is ReleaseKind.BORROWER_RUNTIME_DESTROYED:
            if any(gpu.owned_processes for gpu in self.per_gpu):
                raise ValueError("borrowed destroy cannot leave owned processes")


@dataclass(frozen=True)
class AdmissionSnapshot:
    scope: str
    epoch: int
    key: ReplicaKey | None
    closed: bool
    admitting: int
    all_backends_confirmed: bool

    def __post_init__(self) -> None:
        if self.scope not in {"REPLICA_DRAIN", "TASK_SYNC"}:
            raise ValueError("invalid admission scope")
        if self.epoch < 0 or self.admitting < 0:
            raise ValueError("admission epoch/count must be nonnegative")
        if self.scope == "REPLICA_DRAIN" and self.key is None:
            raise ValueError("replica drain snapshot requires a key")


@dataclass(frozen=True)
class Ack:
    accepted: bool
    revision: int
    error: OperationError | None = None

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("Ack revision must be nonnegative")
        if self.accepted and self.error is not None:
            raise ValueError("accepted Ack must not carry an error")


@dataclass(frozen=True)
class NeverPublishedProof:
    header: EvidenceHeader
    no_route_commit: bool
    no_ce_membership: bool
    fenced: bool
    inventory: CleanupInventory

    def __post_init__(self) -> None:
        if not (self.no_route_commit and self.no_ce_membership and self.fenced):
            raise ValueError("NeverPublishedProof requires queried absence and a commit fence")


CleanupPermit: TypeAlias = ServiceEvidence | NeverPublishedProof
PhaseResult: TypeAlias = (
    PreparedReplica
    | DrainTicket
    | WeightEvidence
    | ExitEvidence
    | ServiceEvidence
    | ReleaseEvidence
    | CommitReceipt
    | AdmissionSnapshot
    | Ack
    | NeverPublishedProof
)
