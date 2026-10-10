"""Exactly-once queue semantics without importing the heavy native runtime."""

import ast
import asyncio
import hashlib
import os
import pickle
import sqlite3
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/multi_task_scheduler/integration/verl/experimental_fully_async/message_queue.py"
)


@dataclass
class Sample:
    sample_id: str
    payload: tuple[int, ...]


class Parent:
    def __init__(self, config, max_queue_size=1000):
        self.config = config
        self.max_queue_size = max_queue_size
        self.queue = deque()
        self.total_produced = 0
        self.dropped_samples = 0

    async def put_sample(self, sample):
        dropped = False
        if len(self.queue) >= self.max_queue_size:
            self.queue.popleft()
            self.dropped_samples += 1
            dropped = True
        self.queue.append(sample)
        self.total_produced += 1
        return not dropped


def queue_class():
    parsed = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = []
    for node in parsed.body:
        if isinstance(node, ast.ClassDef) and node.name in {
            "CompletionEvidence",
            "DuplicateCompletionError",
            "MultiTaskMessageQueue",
        }:
            if node.name == "MultiTaskMessageQueue":
                node.decorator_list = []
                node.bases = [ast.Name(id="Parent", ctx=ast.Load())]
            nodes.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    scope = {
        "Parent": Parent,
        "asyncio": asyncio,
        "hashlib": hashlib,
        "os": os,
        "sqlite3": sqlite3,
        "tempfile": tempfile,
        "dataclass": dataclass,
        "ray": SimpleNamespace(cloudpickle=pickle),
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)
    return (
        scope["MultiTaskMessageQueue"],
        scope["DuplicateCompletionError"],
    )


def test_same_completion_is_enqueued_once_and_returns_same_evidence():
    Queue, _ = queue_class()
    queue = Queue({}, max_queue_size=3, task_session="task-a")
    payload = pickle.dumps(Sample("sample-1", (1, 2, 3)))

    first = asyncio.run(queue.put_sample_once(payload))
    replay = asyncio.run(queue.put_sample_once(bytes(payload)))

    assert replay == first
    assert first.enqueue_seq == 0
    assert queue.total_produced == 1
    assert len(queue.queue) == 1


def test_conflicting_payload_for_same_logical_sample_is_rejected():
    Queue, DuplicateCompletionError = queue_class()
    queue = Queue({}, max_queue_size=3, task_session="task-a")
    asyncio.run(
        queue.put_sample_once(
            pickle.dumps(Sample("sample-1", (1, 2, 3)))
        )
    )

    with pytest.raises(DuplicateCompletionError, match="conflicting payload digest"):
        asyncio.run(
            queue.put_sample_once(
                pickle.dumps(Sample("sample-1", (9, 9, 9)))
            )
        )

    assert queue.total_produced == 1
    assert len(queue.queue) == 1


def test_new_completion_preserves_native_drop_oldest_fact():
    Queue, _ = queue_class()
    queue = Queue({}, max_queue_size=1, task_session="task-a")

    first = asyncio.run(
        queue.put_sample_once(
            pickle.dumps(Sample("sample-1", (1,)))
        )
    )
    second = asyncio.run(
        queue.put_sample_once(
            pickle.dumps(Sample("sample-2", (2,)))
        )
    )

    assert first.dropped_oldest is False
    assert second.dropped_oldest is True
    assert second.enqueue_seq == 1
    assert queue.total_produced == 2
    assert queue.dropped_samples == 1
    assert len(queue.queue) == 1


def test_completion_ledger_is_disk_backed_and_exact_after_many_samples():
    Queue, DuplicateCompletionError = queue_class()
    queue = Queue({}, max_queue_size=8, task_session="task-a")
    first_payload = pickle.dumps(Sample("sample-0", (0,)))

    first = asyncio.run(queue.put_sample_once(first_payload))
    for index in range(1, 128):
        asyncio.run(
            queue.put_sample_once(
                pickle.dumps(Sample(f"sample-{index}", (index,)))
            )
        )

    assert not hasattr(queue, "_completion_evidence")
    count = queue._completion_db.execute(
        "SELECT COUNT(*) FROM completion_evidence"
    ).fetchone()[0]
    assert count == 128

    replay = asyncio.run(queue.put_sample_once(bytes(first_payload)))
    assert replay == first

    with pytest.raises(DuplicateCompletionError, match="conflicting payload digest"):
        asyncio.run(
            queue.put_sample_once(
                pickle.dumps(Sample("sample-0", (999,)))
            )
        )

    assert queue.total_produced == 128


def test_native_enqueue_ambiguous_ack_remains_fail_closed(monkeypatch):
    """After a possible enqueue, a lost ACK must not admit the same sample twice."""
    Queue, _ = queue_class()
    queue = Queue({}, max_queue_size=8, task_session="task-a")
    payload = pickle.dumps(Sample("sample-ack-lost", (1, 2, 3)))
    native_put = Parent.put_sample

    async def put_then_lose_ack(self, sample):
        await native_put(self, sample)
        raise RuntimeError("native enqueue acknowledgement lost")

    monkeypatch.setattr(Parent, "put_sample", put_then_lose_ack)
    with pytest.raises(RuntimeError, match="acknowledgement lost"):
        asyncio.run(queue.put_sample_once(payload))

    row = queue._completion_db.execute(
        "SELECT dropped_oldest FROM completion_evidence "
        "WHERE task_session = ? AND logical_sample_id = ?",
        ("task-a", "sample-ack-lost"),
    ).fetchone()
    assert row == (None,)

    with pytest.raises(RuntimeError, match="unresolved enqueue"):
        asyncio.run(queue.put_sample_once(payload))
    assert queue.total_produced == 1
    assert len(queue.queue) == 1
