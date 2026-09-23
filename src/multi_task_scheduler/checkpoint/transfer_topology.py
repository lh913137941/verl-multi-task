"""Communication-topology primitives (section 4.3), interface-only.

NCCL group names, ZMQ topics/addresses, and rendezvous are isolated by
``task_session`` and ``transfer_id``.  These primitives only track lifecycle
ownership; native transport remains an explicit integration point.

The published weight cache is preserved across finalize/abort; only the
communication groups / buckets / IPC / subscription sockets are released.
"""

from __future__ import annotations

from dataclasses import dataclass
import uuid


@dataclass(frozen=True)
class TransferPlan:
    """Identifies one transfer and its isolated communication topology."""

    transfer_id: str
    task_session: str
    kind: str  # TransferKind value: NATIVE or BOOTSTRAP
    sender_receiver_ids: tuple[str, ...] = ()
    target_version: int | None = None


def prepare_transfer(ctx, kind: str, receivers: tuple, version) -> TransferPlan:
    """Create an isolated transfer plan before native execution."""
    if not isinstance(kind, str) or not kind:
        raise ValueError("kind must be nonempty")
    if not receivers:
        raise ValueError("receivers must be nonempty")
    if version is not None and (type(version) is not int or version < 0):
        raise ValueError("version must be a nonnegative integer")

    task_session = getattr(ctx, "task_session", None)
    if not isinstance(task_session, str) or not task_session:
        raise ValueError("ctx.task_session must be nonempty")

    return TransferPlan(
        transfer_id=str(uuid.uuid4()),
        task_session=task_session,
        kind=kind,
        sender_receiver_ids=tuple(str(item) for item in receivers),
        target_version=version,
    )


def run_transfer(plan: TransferPlan) -> dict:
    """Record execution boundary; native NCCL/ZMQ backend owns transport."""
    if not isinstance(plan, TransferPlan):
        raise TypeError("plan must be TransferPlan")
    return {
        "transfer_id": plan.transfer_id,
        "status": "READY",
        "target_version": plan.target_version,
    }


def finalize_transfer(plan: TransferPlan) -> None:
    """Release transport resources while keeping published weight cache."""
    if not isinstance(plan, TransferPlan):
        raise TypeError("plan must be TransferPlan")


def abort_transfer(plan: TransferPlan) -> None:
    """Idempotent teardown boundary for failed or unknown transfers."""
    if not isinstance(plan, TransferPlan):
        raise TypeError("plan must be TransferPlan")
