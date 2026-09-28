"""Native MessageQueue extension with task-local exactly-once sample commit."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass

import ray
from verl.experimental.fully_async_policy.message_queue import MessageQueue

from verl.single_controller.ray.base import _unwrap_ray_remote
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


@ray.remote(num_cpus=2, max_concurrency=20)
class MultiTaskMessageQueue(_unwrap_ray_remote(MessageQueue)):
    """Reuse native queue behavior and add one internal completion ledger."""

    def __init__(self, config, max_queue_size: int = 1000, *, task_session: str):
        if not isinstance(task_session, str) or not task_session:
            raise ValueError("task_session must be a nonempty string")
        self.task_session = task_session
        self._completion_evidence: dict[tuple[str, str], CompletionEvidence] = {}
        self._next_completion_seq = 0
        self._completion_lock = asyncio.Lock()
        super().__init__(config, max_queue_size=max_queue_size)

    @staticmethod
    def _decode_identity(sample) -> tuple[str, str]:
        if isinstance(sample, (bytes, bytearray, memoryview)):
            payload = bytes(sample)
            decoded = ray.cloudpickle.loads(payload)
        else:
            decoded = sample
            payload = ray.cloudpickle.dumps(sample)
        logical_sample_id = getattr(decoded, "sample_id", None)
        if not isinstance(logical_sample_id, str) or not logical_sample_id:
            raise ValueError("completed rollout sample requires nonempty sample_id")
        return logical_sample_id, hashlib.sha256(payload).hexdigest()

    async def put_sample_once(self, sample) -> CompletionEvidence:
        """Deduplicate one completed native sample under the queue's existing lock."""
        if sample is None:
            raise ValueError("put_sample_once is for completed samples, not termination signals")
        logical_sample_id, payload_digest = self._decode_identity(sample)
        key = (self.task_session, logical_sample_id)

        async with self._completion_lock:
            existing = self._completion_evidence.get(key)
            if existing is not None:
                if existing.payload_digest != payload_digest:
                    raise DuplicateCompletionError(
                        f"conflicting payload digest for completion {key}: "
                        f"{existing.payload_digest!r} != {payload_digest!r}"
                    )
                return existing

            original_return_value = await super().put_sample(sample)
            evidence = CompletionEvidence(
                task_session=self.task_session,
                logical_sample_id=logical_sample_id,
                payload_digest=payload_digest,
                enqueue_seq=self._next_completion_seq,
                dropped_oldest=not original_return_value,
                original_return_value=original_return_value,
            )
            self._next_completion_seq += 1
            self._completion_evidence[key] = evidence
            return evidence

    async def put_sample(self, sample) -> bool:
        # Preserve the native termination signal behavior. Completed rollout samples
        # use exactly-once commit transparently, so existing Rollouter code remains unchanged.
        if sample is None:
            return await super().put_sample(sample)
        evidence = await self.put_sample_once(sample)
        return evidence.original_return_value
