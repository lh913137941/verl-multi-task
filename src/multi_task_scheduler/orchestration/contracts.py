"""Cross-component value contracts aligned with simplified fusion design §6.

This module exposes only the current contract. Superseded aliases and wire
shapes are intentionally not retained: callers must migrate as one protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TYPE_CHECKING, Tuple, TypeVar

from .operation_journal import OperationKind, OperationStatus, Outcome, Phase

if TYPE_CHECKING:
    from .receipts import ReleaseEvidence, ServiceEvidence
    from .replica_record import ReplicaState

_T = TypeVar("_T")


class RecallMode(str, Enum):
    NATURAL = "NATURAL"
    FORCE_VERIFIED = "FORCE_VERIFIED"


class ReleaseKind(str, Enum):
    DONOR_SLEEP_RELEASED = "DONOR_SLEEP_RELEASED"
    BORROWER_RUNTIME_DESTROYED = "BORROWER_RUNTIME_DESTROYED"


class ServiceAction(str, Enum):
    ADD = "ADD"
    REMOVE = "REMOVE"


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
    key: ReplicaKey
    production_epoch: int
    source_seq: int
    manager_revision: int
    lb_revision: int
    engine_digest: str
    reason: str
    stable_idle_ms: int
    observed_age_ms: int
    gpu_count: int
    evidence_digest: str
    placement_digest: str

    def __post_init__(self) -> None:
        if self.reason not in {"CLOSED_STALENESS", "CLOSED_BACKPRESSURE", "EXHAUSTED"}:
            raise ValueError("invalid idle-candidate reason")
        numeric = (
            self.production_epoch,
            self.source_seq,
            self.manager_revision,
            self.lb_revision,
            self.stable_idle_ms,
            self.observed_age_ms,
        )
        if any(value < 0 for value in numeric) or self.gpu_count <= 0:
            raise ValueError("idle candidate revisions/ages must be nonnegative and gpu_count positive")
        for name in ("engine_digest", "evidence_digest", "placement_digest"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")


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
    key: ReplicaKey
    ctx: OperationContext
    placement_digest: str
    model_signature: str
    receivers: Tuple[ReceiverRef, ...]
    head_server: object
    manager_revision: int

    def __post_init__(self) -> None:
        if self.key.task_session != self.ctx.task_session:
            raise ValueError("prepared runtime task_session must match operation context")
        if not self.placement_digest or not self.model_signature:
            raise ValueError("placement_digest and model_signature must be nonempty")
        if not self.receivers:
            raise ValueError("PreparedReplica requires at least one receiver")
        if self.head_server is None:
            raise ValueError("head_server must be present")
        if self.manager_revision < 0:
            raise ValueError("manager_revision must be nonnegative")


@dataclass(frozen=True)
class PublishedWeightSnapshot:
    snapshot_id: str
    manifest_digest: str
    model_signature: str
    version: int
    byte_size: int
    sender: object

    def __post_init__(self) -> None:
        if not self.snapshot_id or not self.manifest_digest or not self.model_signature:
            raise ValueError("snapshot_id, manifest_digest and model_signature must be nonempty")
        if self.version < 0 or self.byte_size < 0:
            raise ValueError("version and byte_size must be nonnegative")
        if self.sender is None:
            raise ValueError("sender must reference the published snapshot backend")


@dataclass(frozen=True)
class ProcessIdentity:
    node_id: str
    process_start_id: str
    role: str
    pid: int

    def __post_init__(self) -> None:
        if not self.node_id or not self.process_start_id or not self.role:
            raise ValueError("process identity fields must be nonempty")
        if self.pid < 0:
            raise ValueError("pid must be nonnegative")


@dataclass(frozen=True)
class CleanupInventory:
    processes: Tuple[ProcessIdentity, ...] = ()
    actor_ids: Tuple[str, ...] = ()
    endpoint_ids: Tuple[str, ...] = ()
    ipc_paths: Tuple[str, ...] = ()
    transfer_ids: Tuple[str, ...] = ()
    ports: Tuple[int, ...] = ()
    inventory_revision: int = 0

    def __post_init__(self) -> None:
        if self.inventory_revision < 0 or any(port < 0 for port in self.ports):
            raise ValueError("inventory_revision and ports must be nonnegative")


@dataclass(frozen=True)
class RuntimeReady:
    key: ReplicaKey
    receiver_ready: bool
    serving_ready: bool
    loaded_version: int | None
    inventory_revision: int

    def __post_init__(self) -> None:
        if self.loaded_version is not None and self.loaded_version < 0:
            raise ValueError("loaded_version must be nonnegative")
        if self.inventory_revision < 0:
            raise ValueError("inventory_revision must be nonnegative")


@dataclass(frozen=True)
class OperationError:
    code: str
    detail: str
    phase: Phase
    outcome: Outcome
    retryable: bool
    cleanup_state: str

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
    recall_mode: RecallMode = RecallMode.NATURAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", OperationKind(self.kind))
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
            if self.candidate.placement_digest != self.authorization.placement_digest:
                raise ValueError("DONATE candidate placement must match authorization")


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
class RuntimeCapabilities:
    placement: str
    sleep: bool = False
    full_weight_replay: bool = False
    target_abort_resume: bool = False
    transport_rebuild: bool = False
    applicable_versions: Tuple[str, ...] = ()
    topology: str = "standalone"

    def requires(self, capability: str) -> bool:
        value = getattr(self, capability, False)
        return value if isinstance(value, bool) else False
