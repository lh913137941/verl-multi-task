#!/usr/bin/env python3
"""Black-box lifecycle acceptance driver for a running MultiTask VERL job."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path


def _jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _blocked(message: str) -> int:
    print(f"MULTITASK_E2E_BLOCKED {message}", file=sys.stderr)
    return 2


def _init_ray(timeout_s: float):
    try:
        import ray
    except Exception as exc:
        raise RuntimeError(f"Ray is unavailable: {exc}") from exc
    if ray.is_initialized():
        return ray

    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            ray.init(
                address=os.environ.get("RAY_ADDRESS", "auto"),
                ignore_reinit_error=True,
                log_to_driver=True,
            )
            return ray
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(
        f"could not attach to Ray before timeout; last_error={last_error}"
    )


def _discover(ray, timeout_s: float):
    from multi_task_scheduler.scheduler.discovery import (
        GROUP_SCHEDULER_NAME,
        GROUP_SCHEDULER_NAMESPACE,
    )

    deadline = time.monotonic() + timeout_s
    donor_session = os.environ.get("MT_E2E_DONOR_SESSION")
    borrower_session = os.environ.get("MT_E2E_BORROWER_SESSION")
    if bool(donor_session) != bool(borrower_session):
        raise RuntimeError(
            "set both MT_E2E_DONOR_SESSION and MT_E2E_BORROWER_SESSION"
        )
    last_error = None
    while time.monotonic() < deadline:
        try:
            gs = ray.get_actor(
                GROUP_SCHEDULER_NAME,
                namespace=GROUP_SCHEDULER_NAMESPACE,
            )
            runners = ray.get(gs.get_task_runners.remote(), timeout=10)
            if donor_session and borrower_session:
                donor_runner = runners.get(donor_session)
                borrower_runner = runners.get(borrower_session)
                if donor_runner is not None and borrower_runner is not None:
                    if donor_session == borrower_session:
                        raise RuntimeError(
                            "donor and borrower must be different task_session values"
                        )
                    return (
                        gs,
                        donor_session,
                        donor_runner,
                        borrower_session,
                        borrower_runner,
                    )
            elif len(runners) == 2:
                raise RuntimeError(
                    "two TaskRunners are attached but their roles are ambiguous; "
                    "set MT_E2E_DONOR_SESSION and MT_E2E_BORROWER_SESSION: "
                    + ", ".join(sorted(runners))
                )
            elif len(runners) > 2:
                raise RuntimeError(
                    "multiple TaskRunners are attached; select donor/borrower with "
                    "MT_E2E_DONOR_SESSION and MT_E2E_BORROWER_SESSION: "
                    + ", ".join(sorted(runners))
                )
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(
        f"no usable donor/borrower TaskRunner pair before timeout; "
        f"last_error={last_error}"
    )


def _load_fixture(
    path: Path,
    donor_session: str,
    borrower_session: str,
    run_id: str,
):
    from multi_task_scheduler.orchestration.contracts import Lease, ReplicaKey

    raw = json.loads(path.read_text(encoding="utf-8"))
    claims = []
    for index, source in enumerate(raw.get("claims", [])):
        claim = dict(source)
        donor = claim.get("donor_task_id")
        if donor in (None, "", "__TASK_SESSION__", "$TASK_SESSION"):
            claim["donor_task_id"] = donor_session
        claim.setdefault("source_lease_id", f"e2e-source-{run_id}")
        # GS retains claim_id ownership for its entire lifetime, even after
        # RESTORE closes the previous lease. Full E2E runs the natural and
        # FORCE cycles against the same GS; give each new lease a distinct
        # logical claim id while keeping its actual physical placement fixed.
        claim_id = str(claim.get("claim_id") or f"e2e-claim-{index}")
        if not raw.get("reuse_lease_id"):
            claim_id = f"{claim_id}-{run_id}"
        claim["claim_id"] = claim_id
        claim.setdefault("gpu_fraction", 0.5)
        claim.setdefault("cpu_request", 1.0)
        claims.append(claim)
    if not claims:
        raise ValueError("lease fixture requires at least one real physical claim")

    lease_base = str(raw.get("lease_id") or "e2e-lease")
    lease_id = lease_base if raw.get("reuse_lease_id") else f"{lease_base}-{run_id}"
    expires_at = raw.get("expires_at")
    if expires_at is None:
        expires_at = time.time() + float(raw.get("expires_in_s", 600))

    lease = Lease(lease_id=lease_id, claims=tuple(claims), expires_at=float(expires_at))
    donor_rank = claims[0]["donor_replica_rank"]
    donor = ReplicaKey(donor_session, f"native-{donor_rank}", 0)
    borrower = ReplicaKey(
        borrower_session,
        str(raw.get("borrower_replica_id") or f"borrowed-e2e-{run_id}"),
        int(raw.get("runtime_epoch", time.time_ns() % 2_000_000_000)),
    )
    return lease, donor, borrower


def _wait_operation(ray, gs, runner, command, timeout_s: float, expected: str):
    from multi_task_scheduler.orchestration.contracts import OperationStatus

    events = []
    submit_error = None
    try:
        initial = ray.get(gs.submit_operation.remote(command), timeout=30)
        events.append({"submit": _jsonable(initial)})
    except Exception as exc:
        submit_error = f"{type(exc).__name__}: {exc}"
        events.append({"submit_error": submit_error})

    deadline = time.monotonic() + timeout_s
    replayed = False
    last = None
    while time.monotonic() < deadline:
        last = ray.get(runner.query_operation.remote(command.operation_id), timeout=10)
        if last.status is OperationStatus.SUCCEEDED:
            if last.result != expected:
                raise RuntimeError(
                    f"{command.kind.value} succeeded with unexpected evidence "
                    f"{last.result!r}, expected {expected!r}"
                )
            events.append({"terminal": _jsonable(last), "replayed_unknown": replayed})
            break
        if last.status is OperationStatus.FAILED:
            raise RuntimeError(
                f"{command.kind.value} resolved FAILED: {last.result}"
            )
        if last.status is OperationStatus.UNKNOWN and not replayed:
            # Same-command replay is the design's reconciliation entry. This also
            # covers a submit reply lost after the TaskRunner accepted the command.
            try:
                replay = ray.get(gs.submit_operation.remote(command), timeout=30)
                events.append({"unknown_replay_submit": _jsonable(replay)})
            except Exception as exc:
                events.append(
                    {"unknown_replay_submit_error": f"{type(exc).__name__}: {exc}"}
                )
            replayed = True
        time.sleep(0.2)
    else:
        raise TimeoutError(
            f"{command.kind.value} did not resolve before timeout; last={last}"
        )

    # Prove exact-command replay is idempotent after terminal success.
    replay = ray.get(gs.submit_operation.remote(command), timeout=30)
    terminal = ray.get(runner.query_operation.remote(command.operation_id), timeout=10)
    if terminal.status is not OperationStatus.SUCCEEDED or terminal.result != expected:
        raise RuntimeError(
            f"terminal replay changed {command.operation_id}: {terminal}"
        )
    events.append({"terminal_replay": _jsonable(replay)})
    return events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("full_cycle", "force_cycle"), required=True)
    parser.add_argument("--lease-file", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--attach-timeout-s", type=float, default=float(os.environ.get("MT_E2E_ATTACH_TIMEOUT_S", "180")))
    parser.add_argument("--operation-timeout-s", type=float, default=float(os.environ.get("MT_E2E_OPERATION_TIMEOUT_S", "300")))
    args = parser.parse_args()

    run_id = uuid.uuid4().hex[:10]
    result = {"schema_version": 1, "scenario": args.scenario, "run_id": run_id, "events": []}
    try:
        ray = _init_ray(args.attach_timeout_s)
        from multi_task_scheduler.orchestration.contracts import (
            OperationCommand,
            OperationKind,
        )

        (
            gs,
            donor_session,
            donor_runner,
            borrower_session,
            borrower_runner,
        ) = _discover(ray, args.attach_timeout_s)
        result["donor_task_session"] = donor_session
        result["borrower_task_session"] = borrower_session
        lease, donor, borrower = _load_fixture(
            args.lease_file,
            donor_session,
            borrower_session,
            run_id,
        )
        opened = ray.get(gs.open_lease.remote(lease), timeout=30)
        result["lease"] = _jsonable(opened)

        sequence = [
            ("donate", OperationKind.DONATE, donor, False, "RELEASED"),
            ("add", OperationKind.ADD, borrower, False, "SERVICE_COMMITTED"),
            (
                "force_remove" if args.scenario == "force_cycle" else "remove",
                OperationKind.REMOVE,
                borrower,
                args.scenario == "force_cycle",
                "RELEASED",
            ),
            ("restore", OperationKind.RESTORE, donor, False, "SERVICE_COMMITTED"),
        ]
        for label, kind, target, force, expected in sequence:
            operation_id = f"e2e-{run_id}-{label}"
            runner = (
                donor_runner
                if target.task_session == donor_session
                else borrower_runner
            )
            command = OperationCommand(
                operation_id=operation_id,
                kind=kind,
                target=target,
                lease_id=lease.lease_id,
                force=True if force else None,
            )
            step = {
                "label": label,
                "command": _jsonable(command),
                "expected_evidence": expected,
            }
            try:
                step["events"] = _wait_operation(
                    ray, gs, runner, command, args.operation_timeout_s, expected
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                if args.scenario == "force_cycle" and (
                    "requires another active rollout server" in message
                    or "partial_rollout=true" in message
                ):
                    result.update(state="BLOCKED", detail=message)
                    args.result_file.parent.mkdir(parents=True, exist_ok=True)
                    args.result_file.write_text(
                        json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    print("MULTITASK_E2E_RESULT " + json.dumps(result, sort_keys=True))
                    return 2
                raise
            result["events"].append(step)

        result.update(state="PASSED", detail="lifecycle sequence and exact-command replay succeeded")
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("MULTITASK_E2E_RESULT " + json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        result.update(state="FAILED", detail=f"{type(exc).__name__}: {exc}")
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("MULTITASK_E2E_RESULT " + json.dumps(result, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
