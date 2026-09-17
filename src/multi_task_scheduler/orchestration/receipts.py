"""Task-internal evidence passed between orchestration owners.

Evidence values describe facts already established by their owner. They are not
boolean success shortcuts and they do not replace the M/E/R/C source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from .contracts import RecallMode


class EngineReadiness(str, Enum):
    WEIGHTS_READY = "WEIGHTS_READY"
    SERVING_READY = "SERVING_READY"


@dataclass(frozen=True)
class DrainReceipt:
    """Compatibility form of the LB drain fact."""

    replica_id: str
    routing_epoch: int
    operation_id: str
    drained: bool = True


@dataclass(frozen=True)
class ExitEvidence:
    """Unified exit preparation result used by DONATE and REMOVE."""

    replica_id: str
    routing_epoch: int
    operation_id: str
    recall_mode: RecallMode = RecallMode.NATURAL
    attempts_drained: bool = True
    continuation_confirmed: bool = True

    @property
    def safe_to_leave_service(self) -> bool:
        return self.attempts_drained and (
            self.recall_mode is RecallMode.NATURAL or self.continuation_confirmed
        )


@dataclass(frozen=True)
class ReadyReceipt:
    """ADD target is CE-effective and LB-routable."""

    replica_id: str
    operation_id: str
    routing_epoch: int
    ce_revision: int
    serving_version: int


@dataclass(frozen=True)
class RemovedReceipt:
    """Service-removal evidence; physical resources may still be present."""

    replica_id: str
    operation_id: str
    ce_revision: int
    lb_excluded: bool
    capacity_released: bool

    @property
    def service_detached(self) -> bool:
        return self.lb_excluded and self.capacity_released


@dataclass(frozen=True)
class RestoredReceipt:
    replica_id: str
    operation_id: str
    routing_epoch: int
    ce_revision: int
    serving_version: int


@dataclass(frozen=True)
class WeightReadiness:
    replica_id: str
    operation_id: str
    state: EngineReadiness = EngineReadiness.WEIGHTS_READY


@dataclass(frozen=True)
class ServingReadiness:
    replica_id: str
    operation_id: str
    state: EngineReadiness = EngineReadiness.SERVING_READY


@dataclass(frozen=True)
class AbortReceipt:
    replica_id: str
    operation_id: str
    request_ids: Tuple[str, ...]
    abort_confirmed: bool


@dataclass(frozen=True)
class EvacuationReceipt:
    replica_id: str
    operation_id: str
    attempts: Tuple[str, ...]
    prefix_owner_confirmed: bool
