"""Completion-queue exactly-once deduplication (sections 5.4, 6.2, 7).

A completed sample has a stable ``CompletionKey`` (session + sample id + turn)
so that a force-recall partial rollout resumed on another replica enqueues the
finished sample exactly once. The dedup table and the enqueue are one
synchronous step: a replay returns ``False`` (already present) rather than
appending a second copy, and a conflicting digest is a hard error rather than a
silent overwrite.

This is *not* network exactly-once; the protocol only claims session-idempotent
enqueue with a queryable result (section 7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Tuple, TypeVar

_P = TypeVar("_P")


@dataclass(frozen=True)
class CompletionKey:
    task_session: str
    sample_id: str
    turn: int


@dataclass(frozen=True)
class CompletionRecord(Generic[_P]):
    key: CompletionKey
    payload_digest: str
    payload: _P


class DuplicateCompletionError(RuntimeError):
    """Same completion key re-submitted with a different payload digest."""


class ExactlyOnceCompletionQueue(Generic[_P]):
    """First-write-wins completion queue with digest-conflict detection."""

    def __init__(self) -> None:
        self._seen: dict[CompletionKey, str] = {}
        self._records: list[CompletionRecord[_P]] = []

    def put_sample_once(
        self, key: CompletionKey, payload_digest: str, payload: _P
    ) -> bool:
        """Return True if this call enqueued the sample, False if already present.

        A ``False`` return therefore means "the sample is already queued", not
        "the write failed" — callers must not retry-write on False.
        """
        existing = self._seen.get(key)
        if existing is not None:
            if existing != payload_digest:
                raise DuplicateCompletionError(
                    f"conflicting payload digest for completion {key}: "
                    f"{existing!r} != {payload_digest!r}"
                )
            return False
        self._seen[key] = payload_digest
        self._records.append(
            CompletionRecord(key=key, payload_digest=payload_digest, payload=payload)
        )
        return True

    def contains(self, key: CompletionKey) -> bool:
        return key in self._seen

    def digest_of(self, key: CompletionKey) -> str | None:
        return self._seen.get(key)

    @property
    def queued(self) -> Tuple[CompletionRecord[_P], ...]:
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)
