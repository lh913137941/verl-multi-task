"""Internal completion evidence shared by the native MessageQueue extension."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompletionEvidence:
    task_session: str
    logical_sample_id: str
    payload_digest: str
    enqueue_seq: int
    dropped_oldest: bool
    original_return_value: bool

    def __post_init__(self) -> None:
        if not self.task_session or not self.logical_sample_id or not self.payload_digest:
            raise ValueError("CompletionEvidence identity/digest fields must be nonempty")
        if self.enqueue_seq < 0:
            raise ValueError("enqueue_seq must be nonnegative")


class DuplicateCompletionError(RuntimeError):
    """Same logical sample was re-submitted with a different payload digest."""
