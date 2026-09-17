"""CE effective-membership value object from the simplified fusion design §6.5."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ReceiverRef, ReplicaKey


@dataclass(frozen=True)
class EffectiveReplicaEntry:
    """One committed member of the CE effective synchronization set E."""

    key: ReplicaKey
    receivers: tuple[ReceiverRef, ...]
    loaded_version: int
    model_signature: str
    membership_operation_id: str

    def __post_init__(self) -> None:
        if not self.receivers:
            raise ValueError("EffectiveReplicaEntry requires receivers")
        if self.loaded_version < 0:
            raise ValueError("loaded_version must be nonnegative")
        if not self.model_signature or not self.membership_operation_id:
            raise ValueError(
                "model_signature and membership_operation_id must be nonempty"
            )
