"""Completed-sample exactly-once uses an internal queue key only."""

import pytest

from multi_task_scheduler.orchestration.exactly_once import (
    CompletedSample,
    DuplicateCompletionError,
    ExactlyOnceCompletionQueue,
)


def _sample(sample_id="sample-1", *, digest="digest", data_ref="payload"):
    return CompletedSample(
        task_session="s1",
        logical_sample_id=sample_id,
        data_ref=data_ref,
        payload_digest=digest,
    )


def test_first_write_returns_completion_evidence():
    queue = ExactlyOnceCompletionQueue()
    evidence = queue.put_sample_once(_sample())
    assert evidence.task_session == "s1"
    assert evidence.logical_sample_id == "sample-1"
    assert evidence.payload_digest == "digest"
    assert evidence.enqueue_seq == 0
    assert evidence.dropped_oldest is False
    assert evidence.original_return_value is True
    assert queue.queued == ("payload",)


def test_same_logical_sample_replay_returns_original_evidence_without_reenqueue():
    queue = ExactlyOnceCompletionQueue()
    first = queue.put_sample_once(_sample())
    second = queue.put_sample_once(_sample(data_ref="different-object-is-ignored"))
    assert second is first
    assert queue.queued == ("payload",)
    assert len(queue) == 1


def test_conflicting_digest_is_a_hard_error():
    queue = ExactlyOnceCompletionQueue()
    queue.put_sample_once(_sample(digest="digest-a"))
    with pytest.raises(DuplicateCompletionError):
        queue.put_sample_once(_sample(digest="digest-b"))


def test_no_public_completion_key_type_is_required():
    sample = _sample()
    assert sample.task_session == "s1"
    assert sample.logical_sample_id == "sample-1"
    assert not hasattr(sample, "attempt_id")


def test_different_logical_samples_are_different_internal_keys():
    queue = ExactlyOnceCompletionQueue()
    one = queue.put_sample_once(_sample("sample-1", digest="d1", data_ref="p1"))
    two = queue.put_sample_once(_sample("sample-2", digest="d2", data_ref="p2"))
    assert one.enqueue_seq == 0
    assert two.enqueue_seq == 1
    assert queue.queued == ("p1", "p2")


def test_full_queue_preserves_native_false_but_still_enqueues_new_sample():
    queue = ExactlyOnceCompletionQueue(max_size=1)
    queue.put_sample_once(_sample("sample-1", digest="d1", data_ref="old"))
    evidence = queue.put_sample_once(_sample("sample-2", digest="d2", data_ref="new"))
    assert evidence.dropped_oldest is True
    assert evidence.original_return_value is False
    assert queue.queued == ("new",)
    assert queue.contains("s1", "sample-1")
    assert queue.evidence_of("s1", "sample-1").payload_digest == "d1"


def test_evidence_lookup_uses_internal_tuple_components_not_a_public_key_object():
    queue = ExactlyOnceCompletionQueue()
    evidence = queue.put_sample_once(_sample())
    assert queue.contains("s1", "sample-1")
    assert queue.evidence_of("s1", "sample-1") is evidence
    assert queue.evidence_of("s1", "missing") is None
