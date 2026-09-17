"""Cross-process/cross-component data contracts (fusion design section 3).

Every type here is a frozen, pickle-safe value object: it carries identity,
evidence, and authority fields but never a donor runtime handle or a raw
``cuda:0`` device id. GPU ids are physical identities from a lease, not
in-process indices.

``epoch`` marks logical isolation, ``revision`` marks ordering (section 3.2):
``gs_epoch`` bounds the GS lifetime, ``task_session`` bounds a task startup,
``lease_epoch`` bounds an authorization generation, and ``runtime_epoch``
distinguishes different runtime instances of one Replica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence, Tuple

from .operation_journal import OperationType


class ReleaseKind(str, Enum):
    """Who releases a replica and whether it may be shared for sleep."""

    DONOR_SLEEP = "DONOR_SLEEP"
    BORROWER_RELEASE = "BORROWER_RELEASE"


class TransferKind(str, Enum):
    NATIVE = "NATIVE"
    BOOTSTRAP = "BOOTSTRAP"


# --------------------------------------------------------------------------- #
# 3.1 task-to-task execution contracts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OperationContext:
    """Identity that every side-effecting step must re-verify (section 3.1)."""

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
    """One physical GPU inside a placement block."""

    gpu_uuid: str
    physical_id: int
    global_rank: int
    local_rank: int


@dataclass(frozen=True)
class NodeBlock:
    """Immutable placement for one node (section 3.1)."""

    node_id: str
    gpus: Tuple[GpuPlacement, ...]


@dataclass(frozen=True)
class PlacementSpec:
    """Ordered placement for a whole replica, with parallel layout and model."""

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
        return tuple(
            gpu.gpu_uuid for block in self.node_blocks for gpu in block.gpus
        )


@dataclass(frozen=True)
class PreparedReplica:
    """HIDDEN runtime that is materialized but not yet published (section 3.1).

    ``receiver_descriptors`` and ``head_server_descriptor`` may hold the
    borrower's own ActorHandles (task-internal only). They never reach the GS
    and never carry a donor runtime.
    """

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
    ) -> PreparedReplica:
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
    """Evidence that receivers finished loading weights and CUDA work.

    Receiver "complete" must mean weights loaded and CUDA work done, never just
    "queued" (section 3.1). ``cleanup_state`` records communication-group /
    bucket / socket teardown without dropping the published weight cache.
    """

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
        return all(
            self.receiver_states.get(r) == "complete"
            for r in self.expected_receiver_ids
        )


@dataclass(frozen=True)
class ReleaseReceipt:
    """Evidence that a replica stopped holding its GPUs (section 3.1).

    ``release_kind`` distinguishes a donor's shareable sleep from a borrower's
    full release; per-GPU HBM and residual records are evidence, not a boolean.
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
    release_kind: ReleaseKind = ReleaseKind.BORROWER_RELEASE


@dataclass(frozen=True)
class RuntimeCapabilities:
    """What this task's runtime can actually do (section 3.1).

    A missing or unverified capability must be refused, never silently degraded
    to fake success.
    """

    placement: str
    sleep: bool = False
    full_weight_replay: bool = False
    target_abort_resume: bool = False
    transport_rebuild: bool = False
    applicable_versions: Tuple[str, ...] = ()
    topology: str = "standalone"

    def requires(self, capability: str) -> bool:
        value = getattr(self, capability, False)
        if isinstance(value, bool):
            return value
        return False


# --------------------------------------------------------------------------- #
# 3.2 command and result
# --------------------------------------------------------------------------- #


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
    state: str  # OperationState value
    actual_replica_state: str
    routing_epoch: int | None = None
    ce_revision: int | None = None
    serving_version: int | None = None
    release_receipt: ReleaseReceipt | None = None
    error: str | None = None


# --------------------------------------------------------------------------- #
# 4.3 communication topology / published snapshot (narrow placeholder)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PublishedWeightSnapshot:
    """Immutable pin of the published serving weights (Vpub snapshot).

    The full NCCL transfer plan and replay live in the checkpoint binding
    (Slice 5). This minimal value carries the version + digest that a
    ``bootstrap_target`` transfer must reproduce onto a target replica.
    """

    serving_version: int
    manifest_digest: str
    receiver_ids: Tuple[str, ...] = ()
