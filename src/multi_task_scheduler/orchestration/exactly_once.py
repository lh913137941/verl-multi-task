"""Session-idempotent completed-sample enqueue for the current design.

The dedup key is deliberately internal: ``(task_session, logical_sample_id)``.
Across the queue boundary callers pass one CompletedSample and receive the
first CompletionEvidence. Same key + same digest replays that receipt; same key
+ different digest conflicts without a second enqueue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Tuple, TypeVar

_P = TypeVar("_P")


@dataclass(frozen=True)
class CompletedSample(Generic[_P]):
    """Thin wrapper around the native complete sample reference."""

    task_session: str
    logical_sample_id: str
    data_ref: _P
    payload_digest: str

    def __post_init__(self) -> None:
        if not self.task_session or not self.logical_sample_id:
            raise ValueError("CompletedSample identities must be nonempty")
        if not self.payload_digest:
            raise ValueError("payload_digest must be nonempty")
        if self.data_ref is None:
            raise ValueError("data_ref must reference the completed native sample")


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


class ExactlyOnceCompletionQueue(Generic[_P]):
    """First-write-wins logical-sample queue with replayable first evidence.

    ``max_size`` models the verified native queue's "drop oldest but report
    false" behavior: when full, the new payload is still present after the
    oldest payload is dropped, while ``original_return_value`` remains False.
    """

    def __init__(self, *, max_size: int | None = None) -> None:
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be positive when provided")
        self._max_size = max_size
        self._evidence: dict[tuple[str, str], CompletionEvidence] = {}
        self._payloads: list[tuple[tuple[str, str], _P]] = []
        self._next_enqueue_seq = 0

    @staticmethod
    def _key(sample: CompletedSample) -> tuple[str, str]:
        return (sample.task_session, sample.logical_sample_id)

    def put_sample_once(self, sample: CompletedSample[_P]) -> CompletionEvidence:
        """Atomically deduplicate, enqueue the native sample ref, and save receipt."""
        if not isinstance(sample, CompletedSample):
            raise TypeError("put_sample_once requires CompletedSample")
        key = self._key(sample)
        existing = self._evidence.get(key)
        if existing is not None:
            if existing.payload_digest != sample.payload_digest:
                raise DuplicateCompletionError(
                    f"conflicting payload digest for completion {key}: "
                    f"{existing.payload_digest!r} != {sample.payload_digest!r}"
                )
            return existing

        dropped_oldest = self._max_size is not None and len(self._payloads) >= self._max_size
        if dropped_oldest:
            self._payloads.pop(0)

        self._payloads.append((key, sample.data_ref))
        evidence = CompletionEvidence(
            task_session=sample.task_session,
            logical_sample_id=sample.logical_sample_id,
            payload_digest=sample.payload_digest,
            enqueue_seq=self._next_enqueue_seq,
            dropped_oldest=dropped_oldest,
            original_return_value=not dropped_oldest,
        )
        self._next_enqueue_seq += 1
        self._evidence[key] = evidence
        return evidence

    def contains(self, task_session: str, logical_sample_id: str) -> bool:
        return (task_session, logical_sample_id) in self._evidence

    def evidence_of(self, task_session: str, logical_sample_id: str) -> CompletionEvidence | None:
        return self._evidence.get((task_session, logical_sample_id))

    @property
    def queued(self) -> Tuple[_P, ...]:
        """Current native sample refs in queue order; dedup facts stay internal."""
        return tuple(payload for _, payload in self._payloads)

    def __len__(self) -> int:
        return len(self._payloads)
