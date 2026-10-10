#!/usr/bin/env python3
"""Real-Ray exactly-once MessageQueue acceptance."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
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
    queue = None
    try:
        startup_timeout = float(os.environ.get("MT_E2E_ACTOR_STARTUP_TIMEOUT_S", "180"))
        rpc_timeout = float(os.environ.get("MT_E2E_QUEUE_RPC_TIMEOUT_S", "30"))
        if not 0 < startup_timeout < float("inf") or not 0 < rpc_timeout < float("inf"):
            raise ValueError("E2E Ray timeouts must be finite positive seconds")

        def wait_for(stage, ref, timeout=rpc_timeout):
            result["last_stage"] = stage
            print(f"[MULTITASK-E2E] exactly_once {stage}: waiting (timeout={timeout:g}s)", flush=True)
            started_at = time.monotonic()
            try:
                reply = ray.get(ref, timeout=timeout)
            except ray.exceptions.GetTimeoutError as exc:
                raise TimeoutError(
                    f"exactly_once {stage} timed out after {timeout:g}s; "
                    "check Ray actor scheduling, worker startup and actor logs"
                ) from exc
            print(
                f"[MULTITASK-E2E] exactly_once {stage}: completed "
                f"in {time.monotonic() - started_at:.1f}s",
                flush=True,
            )
            return reply

        result["last_stage"] = "ray_init"
        if not ray.is_initialized():
            address = os.environ.get("RAY_ADDRESS")
            kwargs = {"ignore_reinit_error": True, "log_to_driver": True}
            if address:
                # Join the parent E2E's existing Ray cluster without trying to
                # replace its CPU resources or creating another local cluster.
                kwargs["address"] = address
                if os.environ.get("PYTHONPATH"):
                    kwargs["runtime_env"] = {
                        "env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}
                    }
            else:
                kwargs.update(address="local", num_cpus=3)
            ray.init(**kwargs)
            started_ray = True

        session = f"queue-e2e-{uuid.uuid4().hex[:8]}"
        result["last_stage"] = "queue_actor_create"
        queue = MultiTaskMessageQueue.remote({}, 1, task_session=session)
        # The first Ray RPC also waits for cold-start VERL imports and actor
        # scheduling. Do not judge that startup using a 10s steady-state budget.
        wait_for("queue_actor_ready", queue.get_queue_size.remote(), startup_timeout)
        first_bytes = ray.cloudpickle.dumps(Sample("sample-1", (1, 2, 3)))
        same_bytes = bytes(first_bytes)
        conflict_bytes = ray.cloudpickle.dumps(Sample("sample-1", (9, 9, 9)))
        second_bytes = ray.cloudpickle.dumps(Sample("sample-2", (4, 5)))

        first = wait_for("first_enqueue", queue.put_sample_once.remote(first_bytes))
        replay = wait_for("same_payload_replay", queue.put_sample_once.remote(same_bytes))
        if first != replay:
            raise AssertionError("same logical completion did not return identical evidence")
        stats = wait_for("queue_statistics", queue.get_statistics.remote())
        if stats["total_produced"] != 1 or stats["queue_size"] != 1:
            raise AssertionError(f"duplicate completion was enqueued twice: {stats}")

        try:
            wait_for("conflicting_payload_rejection", queue.put_sample_once.remote(conflict_bytes))
        except Exception as exc:
            if "conflicting payload digest" not in str(exc):
                raise
        else:
            raise AssertionError("conflicting completion payload was accepted")

        second = wait_for("overflow_enqueue", queue.put_sample_once.remote(second_bytes))
        if not second.dropped_oldest:
            raise AssertionError("bounded queue did not report dropped_oldest evidence")
        stats = wait_for("queue_statistics", queue.get_statistics.remote())
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
        traceback.print_exc(file=sys.stderr)
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
