"""Session-idempotent completed-sample enqueue aligned with design §6.7/§8.5.

A completed logical sample keeps the same ``CompletionKey`` across forced
continuation attempts.  The first enqueue result is persisted as
``CompletionEvidence`` and exact replays return that same evidence instead of a
new boolean whose meaning could be confused with the native queue's full/drop
return value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Tuple, TypeVar

_P = TypeVar("_P")


@dataclass(frozen=True)
class CompletionKey:
    task_session: str
    logical_sample_id: str

    def __post_init__(self) -> None:
        if not self.task_session or not self.logical_sample_id:
            raise ValueError("CompletionKey fields must be nonempty")


@dataclass(frozen=True)
class CompletionEvidence:
    key: CompletionKey
    payload_digest: str
    enqueue_seq: int
    dropped_oldest: bool
    original_return_value: bool

    def __post_init__(self) -> None:
        if not self.payload_digest:
            raise ValueError("payload_digest must be nonempty")
        if self.enqueue_seq < 0:
            raise ValueError("enqueue_seq must be nonnegative")


class DuplicateCompletionError(RuntimeError):
    """Same logical sample was re-submitted with a different payload digest."""


class ExactlyOnceCompletionQueue(Generic[_P]):
    """First-write-wins logical-sample queue with replayable first evidence.

    ``max_size`` models the verified native queue's "drop oldest but report
    false" behavior: when full, the new payload is still present after the
    oldest payload is dropped, while ``original_return_value`` remains False.
    The dedup result is therefore never represented by that boolean.
    """

    def __init__(self, *, max_size: int | None = None) -> None:
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be positive when provided")
        self._max_size = max_size
        self._evidence: dict[CompletionKey, CompletionEvidence] = {}
        self._payloads: list[tuple[CompletionKey, _P]] = []
        self._next_enqueue_seq = 0

    def put_sample_once(
        self, key: CompletionKey, payload_digest: str, payload: _P
    ) -> CompletionEvidence:
        """Enqueue once and return the immutable first-submit evidence.

        Same key + same digest returns the original evidence without touching the
        queue. Same key + different digest is a hard conflict.
        """
        existing = self._evidence.get(key)
        if existing is not None:
            if existing.payload_digest != payload_digest:
                raise DuplicateCompletionError(
                    f"conflicting payload digest for completion {key}: "
                    f"{existing.payload_digest!r} != {payload_digest!r}"
                )
            return existing
        if not payload_digest:
            raise ValueError("payload_digest must be nonempty")

        dropped_oldest = self._max_size is not None and len(self._payloads) >= self._max_size
        if dropped_oldest:
            self._payloads.pop(0)

        self._payloads.append((key, payload))
        evidence = CompletionEvidence(
            key=key,
            payload_digest=payload_digest,
            enqueue_seq=self._next_enqueue_seq,
            dropped_oldest=dropped_oldest,
            original_return_value=not dropped_oldest,
        )
        self._next_enqueue_seq += 1
        self._evidence[key] = evidence
        return evidence

    def contains(self, key: CompletionKey) -> bool:
        return key in self._evidence

    def digest_of(self, key: CompletionKey) -> str | None:
        evidence = self._evidence.get(key)
        return evidence.payload_digest if evidence is not None else None

    def evidence_of(self, key: CompletionKey) -> CompletionEvidence | None:
        return self._evidence.get(key)

    @property
    def queued(self) -> Tuple[_P, ...]:
        """Current payloads in native queue order; dedup facts live in evidence."""
        return tuple(payload for _, payload in self._payloads)

    def __len__(self) -> int:
        return len(self._payloads)
