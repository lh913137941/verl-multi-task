"""Communication-topology primitives (section 4.3), interface-only.

NCCL group names, ZMQ topics/addresses, and rendezvous are isolated by
``task_session`` and ``transfer_id``; changing one card or worker rebuilds
membership rather than trusting world_size. These primitives are the injection
boundary into verl and raise explicitly until a native backend is verified —
they never fall back to a fake or to whole-set native sync.

The published weight cache is preserved across finalize/abort; only the
communication groups / buckets / IPC / subscription sockets are released.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransferPlan:
    """Identifies one transfer and its isolated communication topology."""

    transfer_id: str
    task_session: str
    kind: str  # TransferKind value: NATIVE or BOOTSTRAP
    sender_receiver_ids: tuple[str, ...] = ()
    target_version: int | None = None


def prepare_transfer(ctx, kind: str, receivers: tuple, version) -> TransferPlan:
    """Reserve an isolated topology for a NATIVE or BOOTSTRAP transfer."""
    raise NotImplementedError(
        "prepare_transfer requires verified native NCCL/ZMQ backend"
    )


def run_transfer(plan: TransferPlan) -> object:
    """Execute one transfer; receivers report weight load + CUDA completion."""
    raise NotImplementedError(
        "run_transfer requires verified native NCCL/ZMQ backend"
    )


def finalize_transfer(plan: TransferPlan) -> None:
    """Release comm groups/buckets/sockets; keep the published weight cache."""
    raise NotImplementedError(
        "finalize_transfer requires verified native NCCL/ZMQ backend"
    )


def abort_transfer(plan: TransferPlan) -> None:
    """Idempotent teardown; an unknown result quarantines the topology."""
    raise NotImplementedError(
        "abort_transfer requires verified native NCCL/ZMQ backend"
    )
