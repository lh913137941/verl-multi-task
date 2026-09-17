"""Exactly-once completion enqueue (sections 5.4, 6.2, 7)."""

import pytest

from multi_task_scheduler.orchestration.exactly_once import (
    CompletionKey,
    DuplicateCompletionError,
    ExactlyOnceCompletionQueue,
)


def _key(turn=0):
    return CompletionKey(task_session="s1", sample_id="sample-1", turn=turn)


def test_first_write_enqueues_and_returns_true():
    queue = ExactlyOnceCompletionQueue()
    assert queue.put_sample_once(_key(), "digest", "payload") is True
    assert len(queue) == 1


def test_replay_returns_false_and_does_not_reenqueue():
    queue = ExactlyOnceCompletionQueue()
    assert queue.put_sample_once(_key(), "digest", "payload") is True
    # Regression: a False return means "already queued", not "write failed".
    assert queue.put_sample_once(_key(), "digest", "payload") is False
    assert len(queue) == 1
    assert queue.queued[0].payload == "payload"


def test_conflicting_digest_is_a_hard_error():
    queue = ExactlyOnceCompletionQueue()
    queue.put_sample_once(_key(), "digest-a", "payload")
    with pytest.raises(DuplicateCompletionError):
        queue.put_sample_once(_key(), "digest-b", "payload")


def test_different_turns_are_different_keys():
    queue = ExactlyOnceCompletionQueue()
    assert queue.put_sample_once(_key(turn=0), "d", "p0") is True
    assert queue.put_sample_once(_key(turn=1), "d", "p1") is True
    assert len(queue) == 2


def test_contains_and_digest_of():
    queue = ExactlyOnceCompletionQueue()
    queue.put_sample_once(_key(), "digest", "payload")
    assert queue.contains(_key())
    assert queue.digest_of(_key()) == "digest"
    assert queue.digest_of(_key(turn=9)) is None
