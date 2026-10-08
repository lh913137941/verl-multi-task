"""Native MessageQueue extension with task-local exactly-once sample commit."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import tempfile
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

    def __post_init__(self) -> None:
        if not self.task_session or not self.logical_sample_id or not self.payload_digest:
            raise ValueError("CompletionEvidence identity/digest fields must be nonempty")
        if self.enqueue_seq < 0:
            raise ValueError("enqueue_seq must be nonnegative")


class DuplicateCompletionError(RuntimeError):
    """Same logical sample was re-submitted with a different payload digest."""


@ray.remote(num_cpus=2, max_concurrency=20)
class MultiTaskMessageQueue(_unwrap_ray_remote(MessageQueue)):
    """Reuse native queue behavior with a disk-backed exactly-once ledger.

    The previous Python dict retained one CompletionEvidence object for every
    sample for the actor lifetime.  The ledger is still exact (old retries and
    digest conflicts remain queryable) but rows now live in a local SQLite file
    instead of accumulating in actor heap memory.
    """

    def __init__(self, config, max_queue_size: int = 1000, *, task_session: str):
        if not isinstance(task_session, str) or not task_session:
            raise ValueError("task_session must be a nonempty string")
        self.task_session = task_session
        ledger_root = os.environ.get("VERL_MULTITASK_QUEUE_LEDGER_DIR")
        if ledger_root:
            os.makedirs(ledger_root, exist_ok=True)
        self._completion_tmpdir = tempfile.TemporaryDirectory(
            prefix="verl-multitask-completion-",
            dir=ledger_root or None,
        )
        ledger_path = os.path.join(
            self._completion_tmpdir.name,
            "completion-evidence.sqlite3",
        )
        self._completion_db = sqlite3.connect(ledger_path)
        self._completion_db.execute(
            """
            CREATE TABLE completion_evidence (
                task_session TEXT NOT NULL,
                logical_sample_id TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                enqueue_seq INTEGER NOT NULL,
                dropped_oldest INTEGER,
                PRIMARY KEY (task_session, logical_sample_id)
            )
            """
        )
        self._completion_db.commit()
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

    def _lookup_completion(
        self,
        logical_sample_id: str,
    ) -> CompletionEvidence | None:
        row = self._completion_db.execute(
            """
            SELECT payload_digest, enqueue_seq, dropped_oldest
            FROM completion_evidence
            WHERE task_session = ? AND logical_sample_id = ?
            """,
            (self.task_session, logical_sample_id),
        ).fetchone()
        if row is None:
            return None
        payload_digest, enqueue_seq, dropped_oldest = row
        if dropped_oldest is None:
            # A provisional row means the queue operation did not reach a
            # durable evidence commit. Never re-enqueue it blindly.
            raise RuntimeError(
                "completion ledger contains an unresolved enqueue; "
                "refusing duplicate admission"
            )
        return CompletionEvidence(
            task_session=self.task_session,
            logical_sample_id=logical_sample_id,
            payload_digest=payload_digest,
            enqueue_seq=enqueue_seq,
            dropped_oldest=bool(dropped_oldest),
        )

    async def put_sample_once(self, sample) -> CompletionEvidence:
        """Deduplicate one completed native sample without lifetime heap growth."""
        if sample is None:
            raise ValueError("put_sample_once is for completed samples, not termination signals")
        logical_sample_id, payload_digest = self._decode_identity(sample)

        async with self._completion_lock:
            existing = self._lookup_completion(logical_sample_id)
            if existing is not None:
                if existing.payload_digest != payload_digest:
                    raise DuplicateCompletionError(
                        "conflicting payload digest for completion "
                        f"{(self.task_session, logical_sample_id)}: "
                        f"{existing.payload_digest!r} != {payload_digest!r}"
                    )
                return existing

            enqueue_seq = self._next_completion_seq
            # Reserve the logical key before yielding into the native queue.
            # If the native put fails we remove the reservation; if evidence
            # finalization itself fails the unresolved row remains fail-closed
            # so a retry cannot enqueue a duplicate sample.
            self._completion_db.execute(
                """
                INSERT INTO completion_evidence (
                    task_session,
                    logical_sample_id,
                    payload_digest,
                    enqueue_seq,
                    dropped_oldest
                ) VALUES (?, ?, ?, ?, NULL)
                """,
                (
                    self.task_session,
                    logical_sample_id,
                    payload_digest,
                    enqueue_seq,
                ),
            )
            self._completion_db.commit()
            self._next_completion_seq += 1

            # Native enqueue may have mutated the queue before raising.
            # An uncertain result must retain the provisional ledger row
            # so an exact replay cannot enqueue this logical sample again.
            original_return_value = await super().put_sample(sample)

            dropped_oldest = not original_return_value
            try:
                self._completion_db.execute(
                    """
                    UPDATE completion_evidence
                    SET dropped_oldest = ?
                    WHERE task_session = ? AND logical_sample_id = ?
                    """,
                    (
                        int(dropped_oldest),
                        self.task_session,
                        logical_sample_id,
                    ),
                )
                self._completion_db.commit()
            except BaseException as exc:
                raise RuntimeError(
                    "sample was queued but completion evidence could not be "
                    "finalized; ledger remains fail-closed"
                ) from exc

            return CompletionEvidence(
                task_session=self.task_session,
                logical_sample_id=logical_sample_id,
                payload_digest=payload_digest,
                enqueue_seq=enqueue_seq,
                dropped_oldest=dropped_oldest,
            )

    async def put_sample(self, sample) -> bool:
        # Preserve the native termination signal behavior. Completed rollout
        # samples use exactly-once commit transparently, so existing Rollouter
        # code remains unchanged.
        if sample is None:
            return await super().put_sample(sample)
        evidence = await self.put_sample_once(sample)
        return not evidence.dropped_oldest

    async def shutdown(self):
        """Shutdown native queue and release the local completion ledger."""
        await super().shutdown()
        self._completion_db.close()
        self._completion_tmpdir.cleanup()
