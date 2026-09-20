"""Canonical lifecycle evidence for the current simplified orchestration contract.

Lifecycle evidence is deliberately compact: detailed receiver, request, engine,
and GPU cleanup observations remain with their authoritative owner. Public
objects bind those already-verified facts by stable digests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

from .contracts import (
    OperationContext,
    OperationError,
    RecallMode,
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
    """Proof that the complete target receiver set loaded one published version."""

    header: EvidenceHeader
    version: int
    manifest_digest: str
    receivers_digest: str

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("weight version must be nonnegative")
        if not self.manifest_digest or not self.receivers_digest:
            raise ValueError("manifest_digest and receivers_digest must be nonempty")


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
class ExitEvidence:
    """Proof that one drain is quiet and every old attempt has a legal outcome."""

    header: EvidenceHeader
    drain_id: str
    recall_mode: RecallMode
    quiescence_digest: str
    attempts_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "recall_mode", RecallMode(self.recall_mode))
        if not self.drain_id:
            raise ValueError("drain_id must be nonempty")
        if not self.quiescence_digest or not self.attempts_digest:
            raise ValueError("quiescence_digest and attempts_digest must be nonempty")


@dataclass(frozen=True)
class ServiceEvidence:
    """Combined proof that E/R/C/M committed one ADD or REMOVE service change."""

    header: EvidenceHeader
    action: ServiceAction
    prerequisite_digest: str
    service_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", ServiceAction(self.action))
        if not self.prerequisite_digest or not self.service_digest:
            raise ValueError("prerequisite_digest and service_digest must be nonempty")


@dataclass(frozen=True)
class ReleaseEvidence:
    """Proof that every GPU in the lease placement passed backend release checks."""

    header: EvidenceHeader
    release_kind: Literal["DONOR_SLEEP_RELEASED", "BORROWER_RUNTIME_DESTROYED"]
    permit_digest: str
    inventory_digest: str
    released_gpu_uuids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.release_kind not in {
            "DONOR_SLEEP_RELEASED",
            "BORROWER_RUNTIME_DESTROYED",
        }:
            raise ValueError("invalid release_kind")
        if not self.permit_digest or not self.inventory_digest:
            raise ValueError("permit_digest and inventory_digest must be nonempty")
        if not self.released_gpu_uuids:
            raise ValueError("ReleaseEvidence requires released_gpu_uuids")
        if any(not gpu_uuid for gpu_uuid in self.released_gpu_uuids):
            raise ValueError("released_gpu_uuids must be nonempty strings")
        if len(set(self.released_gpu_uuids)) != len(self.released_gpu_uuids):
            raise ValueError("released_gpu_uuids must not contain duplicates")


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
    """Permit for destroying a failed ADD after CE/LB absence is fenced."""

    header: EvidenceHeader
    publication_fence_digest: str

    def __post_init__(self) -> None:
        if not self.publication_fence_digest:
            raise ValueError("publication_fence_digest must be nonempty")


CleanupPermit: TypeAlias = ServiceEvidence | NeverPublishedProof
