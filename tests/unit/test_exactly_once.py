"""Completed-sample deduplication follows simplified design §6.7/§8.5."""

import pytest

from multi_task_scheduler.orchestration.exactly_once import (
    CompletionKey,
    DuplicateCompletionError,
    ExactlyOnceCompletionQueue,
)


def _key(sample_id="sample-1"):
    return CompletionKey(task_session="s1", logical_sample_id=sample_id)


def test_first_write_returns_completion_evidence():
    queue = ExactlyOnceCompletionQueue()
    evidence = queue.put_sample_once(_key(), "digest", "payload")
    assert evidence.key == _key()
    assert evidence.payload_digest == "digest"
    assert evidence.enqueue_seq == 0
    assert evidence.dropped_oldest is False
    assert evidence.original_return_value is True
    assert queue.queued == ("payload",)


def test_same_logical_sample_replay_returns_original_evidence_without_reenqueue():
    queue = ExactlyOnceCompletionQueue()
    first = queue.put_sample_once(_key(), "digest", "payload")
    second = queue.put_sample_once(_key(), "digest", "different-object-is-ignored")
    assert second is first
    assert queue.queued == ("payload",)
    assert len(queue) == 1


def test_conflicting_digest_is_a_hard_error():
    queue = ExactlyOnceCompletionQueue()
    queue.put_sample_once(_key(), "digest-a", "payload")
    with pytest.raises(DuplicateCompletionError):
        queue.put_sample_once(_key(), "digest-b", "payload")


def test_no_turn_or_attempt_dimension_exists_in_completion_key():
    key = _key()
    assert not hasattr(key, "turn")
    assert not hasattr(key, "attempt_id")


def test_different_logical_samples_are_different_keys():
    queue = ExactlyOnceCompletionQueue()
    one = queue.put_sample_once(_key("sample-1"), "d1", "p1")
    two = queue.put_sample_once(_key("sample-2"), "d2", "p2")
    assert one.enqueue_seq == 0
    assert two.enqueue_seq == 1
    assert queue.queued == ("p1", "p2")


def test_full_queue_preserves_native_false_but_still_enqueues_new_sample():
    queue = ExactlyOnceCompletionQueue(max_size=1)
    queue.put_sample_once(_key("sample-1"), "d1", "old")
    evidence = queue.put_sample_once(_key("sample-2"), "d2", "new")
    assert evidence.dropped_oldest is True
    assert evidence.original_return_value is False
    assert queue.queued == ("new",)
    # The dropped queue payload does not erase completion evidence.
    assert queue.contains(_key("sample-1"))
    assert queue.digest_of(_key("sample-1")) == "d1"


def test_contains_digest_and_evidence_lookup():
    queue = ExactlyOnceCompletionQueue()
    evidence = queue.put_sample_once(_key(), "digest", "payload")
    assert queue.contains(_key())
    assert queue.digest_of(_key()) == "digest"
    assert queue.evidence_of(_key()) is evidence
    assert queue.digest_of(_key("missing")) is None
