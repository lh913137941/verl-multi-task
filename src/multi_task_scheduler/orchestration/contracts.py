"""Cross-process/cross-component data contracts for multi-task orchestration.

These value objects carry identity, evidence and authority.  Runtime handles stay
inside one task and GPU identifiers are stable physical identities, never local
``cuda:N`` aliases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence, Tuple

from .operation_journal import OperationType


class ReleaseKind(str, Enum):
    DONOR_SLEEP = "DONOR_SLEEP"
    BORROWER_RELEASE = "BORROWER_RELEASE"


class TransferKind(str, Enum):
    NATIVE = "NATIVE"
    BOOTSTRAP = "BOOTSTRAP"


class RecallMode(str, Enum):
    NATURAL = "NATURAL"
    FORCE_VERIFIED = "FORCE_VERIFIED"


@dataclass(frozen=True)
class OperationContext:
    protocol_version: str
    gs_epoch: int
    task_id: str
    task_session: str
    operation_id: str
    lease_id: str
    lease_epoch: int
    command_seq: int
    expected_revision: int = 0

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
        )


@dataclass(frozen=True)
class GpuPlacement:
    gpu_uuid: str
    physical_id: int
    global_rank: int
    local_rank: int


@dataclass(frozen=True)
class NodeBlock:
    node_id: str
    gpus: Tuple[GpuPlacement, ...]


@dataclass(frozen=True)
class PlacementSpec:
    node_blocks: Tuple[NodeBlock, ...]
    tp: int = 1
    dp: int = 1
    pp: int = 1
    model_signature: str = ""

    @property
    def world_size(self) -> int:
        return sum(len(block.gpus) for block in self.node_blocks)

    @property
    def gpu_uuids(self) -> Tuple[str, ...]:
        return tuple(gpu.gpu_uuid for block in self.node_blocks for gpu in block.gpus)


@dataclass(frozen=True)
class PreparedReplica:
    """A hidden runtime that has passed placement checks but is not routable."""

    replica_id: str
    runtime_epoch: int
    operation_id: str
    placement_digest: str
    receiver_ids: Tuple[str, ...] = ()
    receiver_descriptors: Tuple[object, ...] = ()
    head_server_descriptor: object = None
    health: str = "pending"

    def with_receivers(
        self, receiver_ids: Sequence[str], receivers: Sequence[object], head_server: object
    ) -> "PreparedReplica":
        return PreparedReplica(
            replica_id=self.replica_id,
            runtime_epoch=self.runtime_epoch,
            operation_id=self.operation_id,
            placement_digest=self.placement_digest,
            receiver_ids=tuple(receiver_ids),
            receiver_descriptors=tuple(receivers),
            head_server_descriptor=head_server,
            health=self.health,
        )


@dataclass(frozen=True)
class TransferReceipt:
    transfer_id: str
    kind: TransferKind
    target_version: int
    manifest_digest: str
    expected_receiver_ids: Tuple[str, ...]
    receiver_states: Mapping[str, str] = field(default_factory=dict)
    cleanup_state: str = "pending"

    @property
    def all_receivers_complete(self) -> bool:
        if not self.expected_receiver_ids:
            return False
        return all(self.receiver_states.get(r) == "complete" for r in self.expected_receiver_ids)


@dataclass(frozen=True)
class ReleaseReceipt:
    """Evidence that a runtime no longer consumes the leased device resources."""

    replica_id: str
    runtime_epoch: int
    lease_epoch: int
    lb_excluded: bool
    ce_excluded: bool
    no_inflight_transfer: bool
    process_cleared_or_slept: bool
    per_gpu_hbm_free: Mapping[str, int] = field(default_factory=dict)
    reserved_residual: Mapping[str, int] = field(default_factory=dict)
    release_kind: ReleaseKind = ReleaseKind.BORROWER_RELEASE

    @property
    def complete(self) -> bool:
        return (
            self.lb_excluded
            and self.ce_excluded
            and self.no_inflight_transfer
            and self.process_cleared_or_slept
            and bool(self.per_gpu_hbm_free)
        )


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


@dataclass(frozen=True)
class Command:
    protocol_version: str
    gs_epoch: int
    target_task_id: str
    target_task_session: str
    operation_id: str
    payload_digest: str
    kind: OperationType
    lease_id: str
    lease_epoch: int
    command_seq: int
    replica_id: str
    expected_runtime_epoch: int = 0
    expected_revision: int = 0
    placement: PlacementSpec | None = None
    candidate_epoch: int | None = None
    recall_mode: str | None = None
    remaining_budget_ms: int | None = None


@dataclass(frozen=True)
class OperationResult:
    identity_fields: OperationContext
    phase: str
    phase_revision: int
    state: str
    actual_replica_state: str
    routing_epoch: int | None = None
    ce_revision: int | None = None
    serving_version: int | None = None
    release_receipt: ReleaseReceipt | None = None
    error: str | None = None


@dataclass(frozen=True)
class PublishedWeightSnapshot:
    """Immutable reference to the exact published weight contents.

    A version number is metadata only.  ADD/RESTORE must pin a real published
    snapshot with a stable identity and non-empty manifest digest before any
    target bootstrap starts.
    """

    snapshot_id: str
    manifest_digest: str
    model_signature: str
    version: int
    byte_size: int
    sender: object

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            raise ValueError("snapshot_id must be nonempty")
        if not self.manifest_digest:
            raise ValueError("manifest_digest must be nonempty")
        if not self.model_signature:
            raise ValueError("model_signature must be nonempty")
        if self.version < 0:
            raise ValueError("version must be nonnegative")
        if self.byte_size < 0:
            raise ValueError("byte_size must be nonnegative")
        if self.sender is None:
            raise ValueError("sender must reference the published snapshot backend")

    @property
    def serving_version(self) -> int:
        """Compatibility alias for the earlier core naming."""
        return self.version
