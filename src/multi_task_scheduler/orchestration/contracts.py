"""Minimal cross-component contracts for the 092203 simplified fusion design.

Owner-local runtime, lifecycle and request details stay with Manager/CE/LB/
Rollouter. Only values that cross those ownership boundaries live here.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class ReplicaKind(str, Enum):
    NATIVE = "NATIVE"
    BORROWED = "BORROWED"


class ReplicaState(str, Enum):
    CREATING = "CREATING"
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    DORMANT = "DORMANT"
    RELEASED = "RELEASED"
    QUARANTINED = "QUARANTINED"


class AttemptState(str, Enum):
    ADMITTED = "ADMITTED"
    TERMINATED = "TERMINATED"
    SETTLED = "SETTLED"


class OperationKind(str, Enum):
    ADD = "ADD"
    DONATE = "DONATE"
    REMOVE = "REMOVE"
    RESTORE = "RESTORE"


class OperationStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class EvidenceType(str, Enum):
    WEIGHT_READY = "WEIGHT_READY"
    EXIT_READY = "EXIT_READY"
    SERVICE_COMMITTED = "SERVICE_COMMITTED"
    RELEASED = "RELEASED"


@dataclass(frozen=True)
class ReplicaKey:
    task_session: str
    replica_id: str
    runtime_epoch: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.task_session, str) or not self.task_session:
            raise ValueError("task_session must be a nonempty string")
        if not isinstance(self.replica_id, str) or not self.replica_id:
            raise ValueError("replica_id must be a nonempty string")
        if type(self.runtime_epoch) is not int or self.runtime_epoch < 0:
            raise ValueError("runtime_epoch must be a nonnegative integer")


@dataclass(frozen=True)
class OperationCommand:
    operation_id: str
    kind: OperationKind
    target: ReplicaKey
    lease_id: str
    force: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", OperationKind(self.kind))
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise ValueError("operation_id must be a nonempty string")
        if not isinstance(self.target, ReplicaKey):
            raise TypeError("target must be ReplicaKey")
        if not isinstance(self.lease_id, str) or not self.lease_id:
            raise ValueError("lease_id must be a nonempty string")
        if self.force is not None and type(self.force) is not bool:
            raise TypeError("force must be bool or None")
        if self.kind is not OperationKind.REMOVE and self.force:
            raise ValueError("force is valid only for REMOVE")


@dataclass
class OperationRecord:
    operation_id: str
    status: OperationStatus = OperationStatus.ACCEPTED
    result: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise ValueError("operation_id must be a nonempty string")
        self.status = OperationStatus(self.status)
        if self.result is not None and not isinstance(self.result, str):
            raise TypeError("result must be str or None")


@dataclass(frozen=True)
class OperationEvidence:
    operation_id: str
    type: EvidenceType
    timestamp: int
    released_gpu_uuids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", EvidenceType(self.type))
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise ValueError("operation_id must be a nonempty string")
        if type(self.timestamp) is not int or self.timestamp < 0:
            raise ValueError("timestamp must be a nonnegative integer")
        if len(set(self.released_gpu_uuids)) != len(self.released_gpu_uuids):
            raise ValueError("released_gpu_uuids must not contain duplicates")
        if any(not isinstance(value, str) or not value for value in self.released_gpu_uuids):
            raise ValueError("released_gpu_uuids must contain nonempty strings")
        if self.type is EvidenceType.RELEASED:
            if not self.released_gpu_uuids:
                raise ValueError("RELEASED evidence requires explicit GPU uuids")
        elif self.released_gpu_uuids:
            raise ValueError("released_gpu_uuids are valid only for RELEASED evidence")

    @classmethod
    def now(cls, operation_id: str, type: EvidenceType, *, released_gpu_uuids: tuple[str, ...] = ()) -> "OperationEvidence":
        return cls(
            operation_id=operation_id,
            type=type,
            timestamp=time.time_ns(),
            released_gpu_uuids=released_gpu_uuids,
        )


@dataclass(frozen=True)
class Lease:
    lease_id: str
    claims: tuple[Mapping[str, Any], ...]
    expires_at: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.lease_id, str) or not self.lease_id:
            raise ValueError("lease_id must be a nonempty string")
        claims = tuple(dict(claim) for claim in self.claims)
        if not claims:
            raise ValueError("Lease requires at least one claim")
        for claim in claims:
            if not claim:
                raise ValueError("lease claims must be nonempty")
            for field_name in ("pg_id", "node_id", "gpu_uuid"):
                value = claim.get(field_name)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"each claim requires a nonempty {field_name}")
            bundle_index = claim.get("bundle_index")
            if type(bundle_index) is not int or bundle_index < 0:
                raise ValueError("each claim requires a nonnegative integer bundle_index")
            gpu_fraction = claim.get("gpu_fraction", 1.0)
            if type(gpu_fraction) not in (int, float) or float(gpu_fraction) != 1.0:
                raise ValueError("first release requires whole-GPU claims")
            cpu_request = claim.get("cpu_request", 0.0)
            if type(cpu_request) not in (int, float) or float(cpu_request) < 0:
                raise ValueError("cpu_request must be nonnegative")
        uuids = [claim["gpu_uuid"] for claim in claims]
        if len(set(uuids)) != len(uuids):
            raise ValueError("lease claims must not repeat gpu_uuid")
        if type(self.expires_at) not in (int, float) or not math.isfinite(float(self.expires_at)):
            raise ValueError("expires_at must be finite")
        if self.expires_at < 0:
            raise ValueError("expires_at must be >= 0")
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "expires_at", float(self.expires_at))

    @property
    def gpu_uuids(self) -> tuple[str, ...]:
        return tuple(str(claim["gpu_uuid"]) for claim in self.claims)
