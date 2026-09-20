"""Cross-component value contracts for the current simplified fusion design.

Only the current wire shapes live here. Detailed execution observations belong
with their owner (Manager/CE/LB/Rollouter/RuntimeBackend) and are referenced by
stable digests instead of being copied into public lifecycle objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Literal, TYPE_CHECKING, Tuple, TypeVar

from .operation_journal import OperationKind, OperationStatus, Outcome, Phase

if TYPE_CHECKING:
    from .production_window import ProductionWindow
    from .receipts import ReleaseEvidence, ServiceEvidence
    from .replica_record import ReplicaRecord, ReplicaState

_T = TypeVar("_T")


class RecallMode(str, Enum):
    NATURAL = "NATURAL"
    FORCE_VERIFIED = "FORCE_VERIFIED"


class ServiceAction(str, Enum):
    ADD = "ADD"
    REMOVE = "REMOVE"


class RouteState(str, Enum):
    HIDDEN = "HIDDEN"
    ROUTABLE = "ROUTABLE"
    DRAINING = "DRAINING"
    REMOVED = "REMOVED"
    QUARANTINED = "QUARANTINED"


class SyncHealth(str, Enum):
    HEALTHY = "HEALTHY"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class ReplicaKey:
    task_session: str
    replica_id: str
    runtime_epoch: int

    def __post_init__(self) -> None:
        if not self.task_session or not self.replica_id:
            raise ValueError("ReplicaKey task_session and replica_id must be nonempty")
        if self.runtime_epoch < 0:
            raise ValueError("runtime_epoch must be nonnegative")


@dataclass(frozen=True)
class OperationContext:
    protocol_version: int
    gs_epoch: str
    task_id: str
    task_session: str
    operation_id: str
    lease_id: str
    lease_epoch: int
    command_seq: int
    expected_revision: int = 0

    def __post_init__(self) -> None:
        if type(self.protocol_version) is not int or self.protocol_version < 0:
            raise ValueError("protocol_version must be a nonnegative integer")
        if not isinstance(self.gs_epoch, str) or not self.gs_epoch:
            raise ValueError("gs_epoch must be a nonempty string")
        for name in ("task_id", "task_session", "operation_id", "lease_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if self.lease_epoch < 0 or self.command_seq < 0 or self.expected_revision < 0:
            raise ValueError("lease_epoch, command_seq and expected_revision must be nonnegative")

    @property
    def identity(self) -> tuple:
        return (
            self.protocol_version,
            self.gs_epoch,
            self.task_id,
            self.task_session,
            self.operation_id,
            self.lease_id,
            self.lease_epoch,
            self.command_seq,
            self.expected_revision,
        )


@dataclass(frozen=True)
class NodePlacement:
    node_id: str
    gpu_uuids: Tuple[str, ...]
    physical_gpu_ids: Tuple[int, ...]
    global_ranks: Tuple[int, ...]
    local_ranks: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be nonempty")
        size = len(self.gpu_uuids)
        if size == 0:
            raise ValueError("NodePlacement requires at least one GPU")
        if not (
            len(self.physical_gpu_ids)
            == len(self.global_ranks)
            == len(self.local_ranks)
            == size
        ):
            raise ValueError("NodePlacement GPU/rank arrays must have the same length")
        if any(not value for value in self.gpu_uuids):
            raise ValueError("gpu_uuids must be nonempty strings")
        if len(set(self.gpu_uuids)) != size:
            raise ValueError("gpu_uuids must be unique within one placement")
        if len(set(self.global_ranks)) != size or len(set(self.local_ranks)) != size:
            raise ValueError("global_ranks and local_ranks must be unique")
        if any(value < 0 for value in self.physical_gpu_ids + self.global_ranks + self.local_ranks):
            raise ValueError("GPU ids and ranks must be nonnegative")


@dataclass(frozen=True)
class PlacementSpec:
    node: NodePlacement
    tp: int
    dp: int
    pp: int
    model_signature: str
    placement_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.node, NodePlacement):
            raise TypeError("PlacementSpec.node must be a NodePlacement")
        if any(type(value) is not int or value <= 0 for value in (self.tp, self.dp, self.pp)):
            raise ValueError("tp/dp/pp must be positive integers")
        if not self.model_signature or not self.placement_digest:
            raise ValueError("model_signature and placement_digest must be nonempty")
        if self.dp != 1 or self.pp != 1:
            raise ValueError("first-release PlacementSpec requires dp=1 and pp=1")
        if len(self.node.gpu_uuids) != self.tp:
            raise ValueError("first-release PlacementSpec requires world_size == tp")


@dataclass(frozen=True)
class LeaseAuthorization:
    lease_id: str
    gs_epoch: str
    donor_session: str
    borrower_session: str
    placement_digest: str
    lease_epoch: int
    purpose: OperationKind
    prior_release_digest: str | None
    authorization_seq: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "purpose", OperationKind(self.purpose))
        for name in (
            "lease_id",
            "gs_epoch",
            "donor_session",
            "borrower_session",
            "placement_digest",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        if self.lease_epoch < 0 or self.authorization_seq < 0:
            raise ValueError("lease_epoch and authorization_seq must be nonnegative")
        if self.purpose in {OperationKind.ADD, OperationKind.RESTORE} and not self.prior_release_digest:
            raise ValueError("ADD/RESTORE authorization requires prior_release_digest")


@dataclass(frozen=True)
class IdleCandidate:
    """Lightweight GS selection token; detailed observations stay owner-local."""

    key: ReplicaKey
    production_epoch: int
    reason: Literal["CLOSED_STALENESS", "CLOSED_BACKPRESSURE", "EXHAUSTED"]
    evidence_digest: str

    def __post_init__(self) -> None:
        if self.production_epoch < 0:
            raise ValueError("production_epoch must be nonnegative")
        if self.reason not in {"CLOSED_STALENESS", "CLOSED_BACKPRESSURE", "EXHAUSTED"}:
            raise ValueError("invalid idle-candidate reason")
        if not self.evidence_digest:
            raise ValueError("evidence_digest must be nonempty")


@dataclass(frozen=True)
class IdleCandidateReport:
    task_session: str
    gs_epoch: str
    lb_session: str
    source_seq: int
    production_revision: int
    valid_for_ms: int
    candidates: Tuple[IdleCandidate, ...]

    def __post_init__(self) -> None:
        for name in ("task_session", "gs_epoch", "lb_session"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        if self.source_seq < 0 or self.production_revision < 0:
            raise ValueError("source_seq and production_revision must be nonnegative")
        if self.valid_for_ms <= 0:
            raise ValueError("valid_for_ms must be positive")
        if any(candidate.key.task_session != self.task_session for candidate in self.candidates):
            raise ValueError("candidate task_session must match report task_session")


@dataclass(frozen=True)
class RouteEntry:
    key: ReplicaKey
    head_server: object
    state: RouteState
    replica_route_epoch: int
    sync_epoch: int
    serving_version: int
    commit_operation_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", RouteState(self.state))
        if self.head_server is None:
            raise ValueError("RouteEntry head_server must be present")
        if self.replica_route_epoch < 0 or self.sync_epoch < 0 or self.serving_version < 0:
            raise ValueError("route/sync epochs and serving_version must be nonnegative")
        if not self.commit_operation_id:
            raise ValueError("commit_operation_id must be nonempty")


@dataclass(frozen=True)
class CapacityRecord:
    active_ids: frozenset[ReplicaKey]
    revision: int
    per_replica_limit: int
    last_commit_operation_id: str

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("capacity revision must be nonnegative")
        if self.per_replica_limit <= 0:
            raise ValueError("per_replica_limit must be positive")
        if not self.last_commit_operation_id:
            raise ValueError("last_commit_operation_id must be nonempty")

    @property
    def max_concurrent_samples(self) -> int:
        return len(self.active_ids) * self.per_replica_limit


@dataclass(frozen=True)
class ReceiverRef:
    receiver_id: str
    node_id: str
    gpu_uuid: str
    global_rank: int
    worker: object

    def __post_init__(self) -> None:
        if not self.receiver_id or not self.node_id or not self.gpu_uuid:
            raise ValueError("receiver identity fields must be nonempty")
        if self.global_rank < 0:
            raise ValueError("global_rank must be nonnegative")
        if self.worker is None:
            raise ValueError("worker must be a receiver ActorRef")


@dataclass(frozen=True)
class PreparedReplica:
    """Hidden-created runtime handles plus a digest of placement/revision facts."""

    key: ReplicaKey
    model_signature: str
    receivers: Tuple[ReceiverRef, ...]
    head_server: object
    prepared_digest: str

    def __post_init__(self) -> None:
        if not self.model_signature or not self.prepared_digest:
            raise ValueError("model_signature and prepared_digest must be nonempty")
        if not self.receivers:
            raise ValueError("PreparedReplica requires at least one receiver")
        if self.head_server is None:
            raise ValueError("head_server must be present")


@dataclass(frozen=True)
class PublishedWeightSnapshot:
    snapshot_id: str
    manifest_digest: str
    model_signature: str
    version: int
    sender: object

    def __post_init__(self) -> None:
        if not self.snapshot_id or not self.manifest_digest or not self.model_signature:
            raise ValueError("snapshot_id, manifest_digest and model_signature must be nonempty")
        if self.version < 0:
            raise ValueError("version must be nonnegative")
        if self.sender is None:
            raise ValueError("sender must reference the published snapshot backend")


@dataclass(frozen=True)
class RuntimeReady:
    key: ReplicaKey
    stage: Literal["RECEIVER_READY", "SERVING_READY"]
    loaded_version: int | None

    def __post_init__(self) -> None:
        if self.stage not in {"RECEIVER_READY", "SERVING_READY"}:
            raise ValueError("invalid RuntimeReady stage")
        if self.loaded_version is not None and self.loaded_version < 0:
            raise ValueError("loaded_version must be nonnegative when present")
        if self.stage == "SERVING_READY" and self.loaded_version is None:
            raise ValueError("SERVING_READY requires loaded_version")


@dataclass(frozen=True)
class SyncToken:
    task_session: str
    sync_epoch: int
    target_version: int

    def __post_init__(self) -> None:
        if not self.task_session:
            raise ValueError("task_session must be nonempty")
        if self.sync_epoch < 0 or self.target_version < 0:
            raise ValueError("sync_epoch and target_version must be nonnegative")


@dataclass(frozen=True)
class CapabilityProof:
    name: str
    backend_version: str
    model_signature: str
    placement_digest: str
    validation_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "backend_version",
            "model_signature",
            "placement_digest",
            "validation_id",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be nonempty")


@dataclass(frozen=True)
class OperationError:
    code: str
    detail: str
    phase: Phase
    outcome: Outcome
    retryable: bool
    cleanup_state: Literal["NOT_REQUIRED", "COMPLETE", "IN_PROGRESS", "UNKNOWN"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", Phase(self.phase))
        object.__setattr__(self, "outcome", Outcome(self.outcome))
        if not self.code:
            raise ValueError("error code must be nonempty")
        if self.cleanup_state not in {"NOT_REQUIRED", "COMPLETE", "IN_PROGRESS", "UNKNOWN"}:
            raise ValueError("invalid cleanup_state")


@dataclass(frozen=True)
class OperationCommand:
    ctx: OperationContext
    kind: OperationKind
    target: ReplicaKey
    authorization: LeaseAuthorization
    payload_digest: str
    remaining_budget_ms: int
    placement: PlacementSpec | None = None
    candidate: IdleCandidate | None = None
    recall_mode: RecallMode | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", OperationKind(self.kind))
        if self.recall_mode is not None:
            object.__setattr__(self, "recall_mode", RecallMode(self.recall_mode))
        if not self.payload_digest:
            raise ValueError("payload_digest must be nonempty")
        if self.remaining_budget_ms < 0:
            raise ValueError("remaining_budget_ms must be nonnegative")
        if self.target.task_session != self.ctx.task_session:
            raise ValueError("target task_session must match operation context")
        if self.authorization.lease_id != self.ctx.lease_id:
            raise ValueError("authorization lease_id must match operation context")
        if self.authorization.lease_epoch != self.ctx.lease_epoch:
            raise ValueError("authorization lease_epoch must match operation context")
        if self.authorization.gs_epoch != self.ctx.gs_epoch:
            raise ValueError("authorization gs_epoch must match operation context")
        if self.authorization.purpose is not self.kind:
            raise ValueError("authorization purpose must match operation kind")
        if self.kind in {OperationKind.ADD, OperationKind.RESTORE} and self.recall_mode is not None:
            raise ValueError("ADD/RESTORE must not carry recall_mode")
        if self.kind is not OperationKind.REMOVE and self.recall_mode is RecallMode.FORCE_VERIFIED:
            raise ValueError("FORCE_VERIFIED is valid only for REMOVE")
        if self.kind is OperationKind.ADD:
            if self.placement is None:
                raise ValueError("ADD requires placement")
            if self.placement.placement_digest != self.authorization.placement_digest:
                raise ValueError("ADD placement must match authorization")
        if self.kind is OperationKind.DONATE:
            if self.candidate is None:
                raise ValueError("DONATE requires an IdleCandidate")
            if self.candidate.key != self.target:
                raise ValueError("DONATE candidate must match target")


@dataclass(frozen=True)
class OperationResult:
    ctx: OperationContext
    target: ReplicaKey
    status: OperationStatus
    phase: Phase
    phase_revision: int
    replica_state: ReplicaState | None = None
    service: ServiceEvidence | None = None
    release: ReleaseEvidence | None = None
    error: OperationError | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", OperationStatus(self.status))
        object.__setattr__(self, "phase", Phase(self.phase))
        if self.target.task_session != self.ctx.task_session:
            raise ValueError("result target task_session must match operation context")
        if self.phase_revision < 0:
            raise ValueError("phase_revision must be nonnegative")
        if self.phase is Phase.DONE and self.status not in {
            OperationStatus.SUCCEEDED,
            OperationStatus.FAILED,
            OperationStatus.UNKNOWN,
        }:
            raise ValueError("DONE requires SUCCEEDED, FAILED, or UNKNOWN")
        if self.phase is not Phase.DONE and self.status in {
            OperationStatus.SUCCEEDED,
            OperationStatus.FAILED,
        }:
            raise ValueError("SUCCEEDED/FAILED require DONE")
        if self.status is OperationStatus.UNKNOWN and self.phase not in {
            Phase.RECONCILE,
            Phase.DONE,
        }:
            raise ValueError("UNKNOWN is only valid while reconciling or at DONE")


@dataclass(frozen=True)
class QueryResult(Generic[_T]):
    found: bool
    value: _T | None
    outcome: Outcome

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", Outcome(self.outcome))
        if self.found != (self.value is not None):
            raise ValueError("found must match whether value is present")


@dataclass(frozen=True)
class TaskSnapshot:
    task_session: str
    production: ProductionWindow
    replica_records: Tuple[ReplicaRecord, ...]
    ce_revision: int
    published_version: int
    lb_revision: int
    capacity: CapacityRecord
    sync_health: SyncHealth
    current_operation_id: str | None
    consistency: Literal["STABLE", "IN_PROGRESS", "UNKNOWN"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sync_health", SyncHealth(self.sync_health))
        if not self.task_session:
            raise ValueError("task_session must be nonempty")
        if self.ce_revision < 0 or self.published_version < 0 or self.lb_revision < 0:
            raise ValueError("CE/published/LB revisions must be nonnegative")
        if getattr(self.production, "task_session", None) != self.task_session:
            raise ValueError("production window must belong to TaskSnapshot task_session")
        if any(record.key.task_session != self.task_session for record in self.replica_records):
            raise ValueError("all replica records must belong to TaskSnapshot task_session")
        if any(key.task_session != self.task_session for key in self.capacity.active_ids):
            raise ValueError("all capacity active_ids must belong to TaskSnapshot task_session")
        if self.current_operation_id is not None and not self.current_operation_id:
            raise ValueError("current_operation_id must be nonempty when present")
        if self.consistency not in {"STABLE", "IN_PROGRESS", "UNKNOWN"}:
            raise ValueError("invalid TaskSnapshot consistency")
