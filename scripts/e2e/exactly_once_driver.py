#!/usr/bin/env python3
"""Real-Ray exactly-once MessageQueue acceptance."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    sample_id: str
    payload: tuple[int, ...]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=Path, required=True)
    args = parser.parse_args()
    result = {"schema_version": 1, "scenario": "exactly_once"}

    try:
        import ray
        from multi_task_scheduler.integration.verl.experimental_fully_async.message_queue import (
            MultiTaskMessageQueue,
        )
    except Exception as exc:
        result.update(state="BLOCKED", detail=f"full Ray/VERL runtime unavailable: {exc}")
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("MULTITASK_E2E_RESULT " + json.dumps(result, sort_keys=True))
        return 2

    started_ray = False
    if not ray.is_initialized():
        ray.init(num_cpus=3, ignore_reinit_error=True, log_to_driver=True)
        started_ray = True

    queue = None
    try:
        session = f"queue-e2e-{uuid.uuid4().hex[:8]}"
        queue = MultiTaskMessageQueue.remote({}, 1, task_session=session)
        first_bytes = ray.cloudpickle.dumps(Sample("sample-1", (1, 2, 3)))
        same_bytes = bytes(first_bytes)
        conflict_bytes = ray.cloudpickle.dumps(Sample("sample-1", (9, 9, 9)))
        second_bytes = ray.cloudpickle.dumps(Sample("sample-2", (4, 5)))

        first = ray.get(queue.put_sample_once.remote(first_bytes), timeout=10)
        replay = ray.get(queue.put_sample_once.remote(same_bytes), timeout=10)
        if first != replay:
            raise AssertionError("same logical completion did not return identical evidence")
        stats = ray.get(queue.get_statistics.remote(), timeout=10)
        if stats["total_produced"] != 1 or stats["queue_size"] != 1:
            raise AssertionError(f"duplicate completion was enqueued twice: {stats}")

        try:
            ray.get(queue.put_sample_once.remote(conflict_bytes), timeout=10)
        except Exception as exc:
            if "conflicting payload digest" not in str(exc):
                raise
        else:
            raise AssertionError("conflicting completion payload was accepted")

        second = ray.get(queue.put_sample_once.remote(second_bytes), timeout=10)
        if not second.dropped_oldest:
            raise AssertionError("bounded queue did not report dropped_oldest evidence")
        stats = ray.get(queue.get_statistics.remote(), timeout=10)
        if stats["total_produced"] != 2 or stats["dropped_samples"] != 1:
            raise AssertionError(f"queue accounting is inconsistent: {stats}")

        result.update(
            state="PASSED",
            detail="same payload idempotent; conflicting payload rejected; overflow evidence preserved",
            first_evidence={
                "logical_sample_id": first.logical_sample_id,
                "payload_digest": first.payload_digest,
                "enqueue_seq": first.enqueue_seq,
            },
            statistics=stats,
        )
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("MULTITASK_E2E_RESULT " + json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        result.update(state="FAILED", detail=f"{type(exc).__name__}: {exc}")
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("MULTITASK_E2E_RESULT " + json.dumps(result, sort_keys=True))
        return 1
    finally:
        if queue is not None:
            try:
                ray.kill(queue)
            except Exception:
                pass
        if started_ray:
            ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
