#!/usr/bin/env python3
"""Real-Ray control-plane recovery acceptance without GPU/runtime dependencies."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--stale-wait-s", type=float, default=10.2)
    args = parser.parse_args()
    result = {"schema_version": 1, "scenario": "control_plane_recovery", "checks": []}

    try:
        import ray
        from multi_task_scheduler.orchestration.contracts import (
            EvidenceType,
            Lease,
            OperationCommand,
            OperationEvidence,
            OperationKind,
            OperationRecord,
            OperationStatus,
            ReplicaKey,
            ReplicaKind,
        )
        from multi_task_scheduler.scheduler.group_scheduler import GroupScheduler
    except Exception as exc:
        result.update(state="BLOCKED", detail=f"Ray/plugin import unavailable: {exc}")
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("MULTITASK_E2E_RESULT " + json.dumps(result, sort_keys=True))
        return 2

    started_ray = False
    try:
        if not ray.is_initialized():
            address = os.environ.get("RAY_ADDRESS")
            kwargs = {"ignore_reinit_error": True, "log_to_driver": True}
            if address:
                # In attach mode CPU resources belong to the existing cluster.
                # Ray rejects num_cpus when the driver connects to one.
                kwargs["address"] = address
                if os.environ.get("PYTHONPATH"):
                    kwargs["runtime_env"] = {
                        "env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}
                    }
            else:
                kwargs.update(address="local", num_cpus=2)
            ray.init(**kwargs)
            started_ray = True
    except Exception as exc:
        result.update(state="BLOCKED", detail=f"cannot connect control-plane Ray: {type(exc).__name__}: {exc}")
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("MULTITASK_E2E_RESULT " + json.dumps(result, sort_keys=True))
        return 2

    @ray.remote
    class DummyTaskRunner:
        def __init__(self):
            self.add_calls = 0

        def submit_operation(self, command, lease=None):
            if command.kind is OperationKind.ADD:
                self.add_calls += 1
                if self.add_calls == 1:
                    raise RuntimeError("simulated ADD reply loss after acceptance")
            return OperationRecord(
                command.operation_id,
                status=OperationStatus.SUCCEEDED,
                result="synthetic-taskrunner",
            )

        def query_operation(self, operation_id):
            return OperationRecord(
                operation_id,
                status=OperationStatus.UNKNOWN,
                result=None,
            )

        def get_add_calls(self):
            return self.add_calls

    def expect_error(label, callback, text):
        try:
            callback()
        except Exception as exc:
            message = str(exc)
            if text not in message:
                raise AssertionError(
                    f"{label}: expected {text!r}, got {message!r}"
                ) from exc
            result["checks"].append({"check": label, "state": "PASS", "detail": text})
            return
        raise AssertionError(f"{label}: expected failure")

    token = uuid.uuid4().hex[:10]
    task_session = f"e2e-task-{token}"
    gs = GroupScheduler.remote()
    runner = DummyTaskRunner.remote()

    try:
        ray.get(gs.attach_task.remote(task_session, runner), timeout=10)

        def claim(suffix="a"):
            return {
                "claim_id": f"claim-{token}-{suffix}",
                "source_lease_id": f"source-{token}",
                "donor_task_id": task_session,
                "donor_replica_rank": 0,
                "pg_id": f"pg-{token}-{suffix}",
                "node_id": f"node-{token}",
                "gpu_uuid": f"gpu-{token}-{suffix}",
                "bundle_index": 0,
                "gpu_fraction": 0.5,
                "cpu_request": 1.0,
            }

        lease = Lease(
            f"lease-{token}",
            (claim(),),
            expires_at=time.time() + 300,
        )
        ray.get(gs.open_lease.remote(lease), timeout=10)
        donor = ReplicaKey(task_session, "native-0", 0)
        borrower = ReplicaKey(task_session, f"borrowed-{token}", 1)

        expect_error(
            "unknown lease rejected",
            lambda: ray.get(
                gs.submit_operation.remote(
                    OperationCommand(
                        f"op-{token}-unknown",
                        OperationKind.ADD,
                        borrower,
                        f"missing-{token}",
                    )
                ),
                timeout=10,
            ),
            "unknown lease",
        )
        expect_error(
            "ADD before donor RELEASED rejected",
            lambda: ray.get(
                gs.submit_operation.remote(
                    OperationCommand(
                        f"op-{token}-early-add",
                        OperationKind.ADD,
                        borrower,
                        lease.lease_id,
                    )
                ),
                timeout=10,
            ),
            "requires donor RELEASED handoff",
        )
        expect_error(
            "RESTORE while claims owned rejected",
            lambda: ray.get(
                gs.submit_operation.remote(
                    OperationCommand(
                        f"op-{token}-early-restore",
                        OperationKind.RESTORE,
                        donor,
                        lease.lease_id,
                    )
                ),
                timeout=10,
            ),
            "fully returned and unclaimed",
        )

        collision = Lease(
            f"collision-{token}",
            (dict(claim(), source_lease_id=f"other-source-{token}"),),
            expires_at=time.time() + 300,
        )
        expect_error(
            "claim collision rejected",
            lambda: ray.get(gs.open_lease.remote(collision), timeout=10),
            "already belongs to lease",
        )

        expired = Lease(
            f"expired-{token}",
            (claim("expired"),),
            expires_at=time.time() - 1,
        )
        ray.get(gs.open_lease.remote(expired), timeout=10)
        expect_error(
            "expired ADD rejected before dispatch",
            lambda: ray.get(
                gs.submit_operation.remote(
                    OperationCommand(
                        f"op-{token}-expired",
                        OperationKind.ADD,
                        ReplicaKey(task_session, f"borrowed-expired-{token}", 1),
                        expired.lease_id,
                    )
                ),
                timeout=10,
            ),
            "expired lease",
        )

        ray.get(
            gs.submit_idle_report.remote(
                {
                    "task_session": task_session,
                    "candidates": [
                        {"replica_key": donor, "kind": ReplicaKind.NATIVE.value}
                    ],
                }
            ),
            timeout=10,
        )
        time.sleep(args.stale_wait_s)
        expect_error(
            "stale idle report rejected",
            lambda: ray.get(
                gs.submit_operation.remote(
                    OperationCommand(
                        f"op-{token}-stale",
                        OperationKind.DONATE,
                        donor,
                        lease.lease_id,
                    )
                ),
                timeout=10,
            ),
            "idle report is stale",
        )

        # Clear the stale advisory report without changing the lease ledger.
        ray.get(gs.detach_task.remote(task_session), timeout=10)
        ray.get(gs.attach_task.remote(task_session, runner), timeout=10)

        donate = OperationCommand(
            f"op-{token}-donate",
            OperationKind.DONATE,
            donor,
            lease.lease_id,
        )
        ray.get(gs.submit_operation.remote(donate), timeout=10)
        released = OperationEvidence.now(
            donate.operation_id,
            EvidenceType.RELEASED,
            released_gpu_uuids=lease.gpu_uuids,
        )
        first_ack = ray.get(
            gs.advance_lease.remote(lease.lease_id, released), timeout=10
        )
        second_ack = ray.get(
            gs.advance_lease.remote(lease.lease_id, released), timeout=10
        )
        if first_ack != second_ack or not first_ack.get("released"):
            raise AssertionError("identical RELEASED evidence was not idempotent")
        result["checks"].append(
            {"check": "advance_lease identical evidence idempotent", "state": "PASS"}
        )

        conflicting = OperationEvidence(
            released.operation_id,
            released.type,
            released.timestamp + 1,
            released.released_gpu_uuids,
        )
        expect_error(
            "conflicting release evidence rejected",
            lambda: ray.get(
                gs.advance_lease.remote(lease.lease_id, conflicting), timeout=10
            ),
            "conflicting lease evidence replay",
        )

        add = OperationCommand(
            f"op-{token}-add",
            OperationKind.ADD,
            borrower,
            lease.lease_id,
        )
        expect_error(
            "simulated ADD reply loss is surfaced",
            lambda: ray.get(gs.submit_operation.remote(add), timeout=10),
            "simulated ADD reply loss",
        )

        # GS must keep borrower intent after the ambiguous reply. A second ADD
        # with a new operation id therefore cannot steal the same lease.
        expect_error(
            "ambiguous submit preserves staged borrower intent",
            lambda: ray.get(
                gs.submit_operation.remote(
                    OperationCommand(
                        f"op-{token}-add-other",
                        OperationKind.ADD,
                        borrower,
                        lease.lease_id,
                    )
                ),
                timeout=10,
            ),
            "replay the original operation_id",
        )
        replay = ray.get(gs.submit_operation.remote(add), timeout=10)
        if replay.status is not OperationStatus.SUCCEEDED:
            raise AssertionError(f"exact ADD replay did not recover: {replay}")
        if ray.get(runner.get_add_calls.remote(), timeout=10) != 2:
            raise AssertionError("exact ADD replay did not reach the same TaskRunner twice")
        result["checks"].append(
            {"check": "same-operation replay recovers ambiguous submit", "state": "PASS"}
        )

        result.update(state="PASSED", detail="real Ray GS recovery contract passed")
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
        try:
            ray.kill(runner)
            ray.kill(gs)
        except Exception:
            pass
        if started_ray:
            ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
