"""Task-internal phase receipts returned by section 4.1 methods.

These are frozen evidence values passed between Trainer, Rollouter, LB, CE
Manager, and LLMServerManager inside one task. Unlike the section 3 contracts,
they are not cross-task: they may carry task-local references (e.g. head server
descriptors) but still never a donor runtime handle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class EngineReadiness(str, Enum):
    WEIGHTS_READY = "WEIGHTS_READY"
    SERVING_READY = "SERVING_READY"


@dataclass(frozen=True)
class DrainReceipt:
    """LB atomically advanced routing_epoch and removed the target."""

    replica_id: str
    routing_epoch: int
    operation_id: str
    drained: bool = True


@dataclass(frozen=True)
class ReadyReceipt:
    """ADD/RESTORE target is CE-effective and LB-routable."""

    replica_id: str
    operation_id: str
    routing_epoch: int
    ce_revision: int
    serving_version: int


@dataclass(frozen=True)
class RemovedReceipt:
    """REMOVE/DONATE target is excluded from CE, LB, and capacity."""

    replica_id: str
    operation_id: str
    ce_revision: int
    lb_excluded: bool
    capacity_released: bool


@dataclass(frozen=True)
class RestoredReceipt:
    """RESTORE woke the donor target and republished it."""

    replica_id: str
    operation_id: str
    routing_epoch: int
    ce_revision: int
    serving_version: int


@dataclass(frozen=True)
class WeightReadiness:
    """wake_weights finished; generation is still disabled."""

    replica_id: str
    operation_id: str
    state: EngineReadiness = EngineReadiness.WEIGHTS_READY


@dataclass(frozen=True)
class ServingReadiness:
    """wake_kv_and_validate finished; LB must not route yet."""

    replica_id: str
    operation_id: str
    state: EngineReadiness = EngineReadiness.SERVING_READY


@dataclass(frozen=True)
class AbortReceipt:
    """abort_target acknowledged by every related server."""

    replica_id: str
    operation_id: str
    request_ids: Tuple[str, ...]
    abort_confirmed: bool


@dataclass(frozen=True)
class EvacuationReceipt:
    """Old attempts released and prefixes have a surviving owner."""

    replica_id: str
    operation_id: str
    attempts: Tuple[str, ...]
    prefix_owner_confirmed: bool
