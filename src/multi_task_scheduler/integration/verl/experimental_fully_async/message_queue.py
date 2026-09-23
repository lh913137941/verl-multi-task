"""Native MessageQueue extension with task-local exactly-once sample commit."""

from __future__ import annotations

import hashlib
import logging

import ray
from verl.experimental.fully_async_policy.message_queue import MessageQueue

from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.orchestration.exactly_once import (
    CompletionEvidence,
    DuplicateCompletionError,
)

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=2, max_concurrency=20)
class MultiTaskMessageQueue(unwrap_native_actor_class(MessageQueue)):
    """Reuse native queue behavior and add one internal completion ledger."""

    def __init__(self, config, max_queue_size: int = 1000, *, task_session: str):
        if not isinstance(task_session, str) or not task_session:
            raise ValueError("task_session must be a nonempty string")
        self.task_session = task_session
        self._completion_evidence: dict[tuple[str, str], CompletionEvidence] = {}
        self._next_completion_seq = 0
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

        async with self._lock:
            existing = self._completion_evidence.get(key)
            if existing is not None:
                if existing.payload_digest != payload_digest:
                    raise DuplicateCompletionError(
                        f"conflicting payload digest for completion {key}: "
                        f"{existing.payload_digest!r} != {payload_digest!r}"
                    )
                return existing

            dropped_oldest = len(self.queue) >= self.max_queue_size
            if dropped_oldest:
                self.queue.popleft()
                self.dropped_samples += 1
                logger.warning("Queue full, dropped sample")

            self.queue.append(sample)
            self.total_produced += 1
            self._consumer_condition.notify_all()
            evidence = CompletionEvidence(
                task_session=self.task_session,
                logical_sample_id=logical_sample_id,
                payload_digest=payload_digest,
                enqueue_seq=self._next_completion_seq,
                dropped_oldest=dropped_oldest,
                original_return_value=not dropped_oldest,
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
