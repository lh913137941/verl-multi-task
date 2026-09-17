"""Cross-component value contracts aligned with simplified design §6.

The canonical public names in this module follow the simplified fusion design.
A few aliases are kept only to avoid an immediate import break for callers that
have not migrated yet; repository code should use the canonical names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence, Tuple

from .operation_journal import OperationKind, OperationStatus, Phase


class ReleaseKind(str, Enum):
    DONOR_SLEEP_RELEASED = "DONOR_SLEEP_RELEASED"
    BORROWER_RUNTIME_DESTROYED = "BORROWER_RUNTIME_DESTROYED"


class TransferKind(str, Enum):
    NATIVE = "NATIVE"
    BOOTSTRAP = "BOOTSTRAP"


class RecallMode(str, Enum):
    NATURAL = "NATURAL"
    FORCE_VERIFIED = "FORCE_VERIFIED"


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
        """Immutable command identity used for replay/result fencing."""
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
class GpuPlacement:
    """Compatibility leaf used only when adapting older node-block callers."""

    gpu_uuid: str
    physical_id: int
    global_rank: int
    local_rank: int


@dataclass(frozen=True)
class NodePlacement:
    """Single-node GPU/rank placement used by first-release orchestration."""

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

    @property
    def gpus(self) -> Tuple[GpuPlacement, ...]:
        """Read-only compatibility projection; not the canonical wire shape."""
        return tuple(
            GpuPlacement(uuid, physical, global_rank, local_rank)
            for uuid, physical, global_rank, local_rank in zip(
                self.gpu_uuids,
                self.physical_gpu_ids,
                self.global_ranks,
                self.local_ranks,
            )
        )


# Compatibility import name.  The shape is now exactly NodePlacement; callers
# must no longer construct a list of node blocks for first-release placement.
NodeBlock = NodePlacement


@dataclass(frozen=True)
class PlacementSpec:
    node: NodePlacement
    model_signature: str
    placement_digest: str
    tp: int = 1
    dp: int = 1
    pp: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.node, NodePlacement):
            raise TypeError("PlacementSpec.node must be a NodePlacement")
        if any(type(v) is not int or v <= 0 for v in (self.tp, self.dp, self.pp)):
            raise ValueError("tp/dp/pp must be positive integers")
        if not self.model_signature:
            raise ValueError("model_signature must be nonempty")
        if not self.placement_digest:
            raise ValueError("placement_digest must be nonempty")
        if self.dp != 1 or self.pp != 1:
            raise ValueError("first-release PlacementSpec requires dp=1 and pp=1")
        if self.world_size != self.tp:
            raise ValueError("first-release PlacementSpec requires world_size == tp")

    @property
    def world_size(self) -> int:
        return len(self.node.gpu_uuids)

    @property
    def gpu_uuids(self) -> Tuple[str, ...]:
        return self.node.gpu_uuids

    @property
    def node_blocks(self) -> Tuple[NodePlacement, ...]:
        """Compatibility read-only view; canonical shape is the single ``node``."""
        return (self.node,)


@dataclass(frozen=True)
class PreparedReplica:
    """A hidden runtime that passed placement checks but is not routable."""

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
class ReleaseEvidence:
    """Verified physical-release evidence used by the current runtime adapter.

    The full device/process detail remains a backend responsibility.  This core
    value refuses to call a release complete without per-GPU observations and
    all routing/CE/transfer/process fences being satisfied.
    """

    replica_id: str
    runtime_epoch: int
    lease_epoch: int
    lb_excluded: bool
    ce_excluded: bool
    no_inflight_transfer: bool
    process_cleared_or_slept: bool
    per_gpu_hbm_free: Mapping[str, int] = field(default_factory=dict)
    reserved_residual: Mapping[str, int] = field(default_factory=dict)
    release_kind: ReleaseKind = ReleaseKind.BORROWER_RUNTIME_DESTROYED

    def __post_init__(self) -> None:
        object.__setattr__(self, "release_kind", ReleaseKind(self.release_kind))

    @property
    def complete(self) -> bool:
        return (
            self.lb_excluded
            and self.ce_excluded
            and self.no_inflight_transfer
            and self.process_cleared_or_slept
            and bool(self.per_gpu_hbm_free)
            and all(value >= 0 for value in self.per_gpu_hbm_free.values())
        )


# Compatibility alias only; canonical design name is ReleaseEvidence.
ReleaseReceipt = ReleaseEvidence


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
class OperationCommand:
    """Current flat transport form of the public operation command."""

    protocol_version: int
    gs_epoch: str
    target_task_id: str
    target_task_session: str
    operation_id: str
    payload_digest: str
    kind: OperationKind
    lease_id: str
    lease_epoch: int
    command_seq: int
    replica_id: str
    expected_runtime_epoch: int = 0
    expected_revision: int = 0
    placement: PlacementSpec | None = None
    candidate_epoch: int | None = None
    recall_mode: RecallMode | None = None
    remaining_budget_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", OperationKind(self.kind))
        if self.recall_mode is not None:
            object.__setattr__(self, "recall_mode", RecallMode(self.recall_mode))
        if self.kind is not OperationKind.REMOVE and self.recall_mode is RecallMode.FORCE_VERIFIED:
            raise ValueError("FORCE_VERIFIED is valid only for REMOVE")
        if self.lease_epoch < 0 or self.command_seq < 0 or self.expected_revision < 0:
            raise ValueError("lease_epoch, command_seq and expected_revision must be nonnegative")
        if self.remaining_budget_ms is not None and self.remaining_budget_ms < 0:
            raise ValueError("remaining_budget_ms must be nonnegative")

    @property
    def context(self) -> OperationContext:
        return OperationContext(
            protocol_version=self.protocol_version,
            gs_epoch=self.gs_epoch,
            task_id=self.target_task_id,
            task_session=self.target_task_session,
            operation_id=self.operation_id,
            lease_id=self.lease_id,
            lease_epoch=self.lease_epoch,
            command_seq=self.command_seq,
            expected_revision=self.expected_revision,
        )


# Compatibility import name used by existing binding code.
Command = OperationCommand


@dataclass(frozen=True)
class OperationResult:
    identity_fields: OperationContext
    phase: Phase
    phase_revision: int
    state: OperationStatus
    actual_replica_state: str | None
    routing_epoch: int | None = None
    ce_revision: int | None = None
    serving_version: int | None = None
    release_receipt: ReleaseEvidence | None = None
    error: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", Phase(self.phase))
        object.__setattr__(self, "state", OperationStatus(self.state))
        if self.phase_revision < 0:
            raise ValueError("phase_revision must be nonnegative")
        if self.phase is Phase.DONE and self.state not in {
            OperationStatus.SUCCEEDED,
            OperationStatus.FAILED,
            OperationStatus.UNKNOWN,
        }:
            raise ValueError("DONE requires a terminal operation status")

    @property
    def status(self) -> OperationStatus:
        return self.state

    @property
    def replica_state(self) -> str | None:
        return self.actual_replica_state

    @property
    def release(self) -> ReleaseEvidence | None:
        return self.release_receipt


@dataclass(frozen=True)
class PublishedWeightSnapshot:
    """Immutable reference to the exact published weight contents."""

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
