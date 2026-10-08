#!/usr/bin/env python3
"""Launch two real VERL Fully Async MultiTask jobs and run shared-GS E2E.

Runs *native* VERL entrypoint twice; delegates acceptance to the repository's
scripts/e2e/run_all.sh. Does not create fake jobs, fake leases, or synthetic
lifecycle success evidence.

Input: a real donor Lease fixture plus native Fully Async Hydra overrides
(one override per line in a file, or pass overrides after --). For newly
started jobs, use --interactive-lease to fill the PG ID after registration.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd(), help="verl-multi-task checkout root")
    p.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS", "auto"))
    p.add_argument("--namespace", default="multitask-jobs", help="same namespace for both VERL jobs and donor PG")
    p.add_argument("--lease", type=Path, required=True, help="real donor Lease JSON (may be populated after job startup with --interactive-lease)")
    p.add_argument("--interactive-lease", action="store_true",
                   help="pause after both tasks start, print PGs and let you fill the real Lease before E2E")
    p.add_argument("--native-args", type=Path,
                   help="optional native Fully Async Hydra overrides file (one argument per line)")
    p.add_argument("--donor-args", type=Path, help="optional donor-only Hydra overrides")
    p.add_argument("--borrower-args", type=Path, help="optional borrower-only Hydra overrides")
    p.add_argument("--trainer-gpus", type=int, default=1, help="trainer GPUs per node per job")
    p.add_argument("--rollout-gpus", type=int, default=1, help="native standalone rollout GPUs per node per job")
    p.add_argument("--startup-timeout", type=int, default=900)
    p.add_argument("--e2e-timeout", type=int, default=5400)
    p.add_argument("--scenarios", default="control_plane exactly_once recovery lifecycle force",
                   help="space-separated names among control_plane exactly_once recovery lifecycle force")
    p.add_argument("--require-inflight-force", action="store_true",
                   help="also require a matching positive in-flight FORCE handoff receipt in the borrower log")
    p.add_argument("--keep-running", action="store_true", help="do not stop drivers after verification")
    p.add_argument("--logs", type=Path, default=Path.cwd() / "logs" / "two_real_jobs")
    p.add_argument("native_overrides", nargs="*", help="Hydra overrides after --; same for both jobs")
    args = p.parse_args()
    if args.native_overrides and args.native_overrides[0] == "--":
        args.native_overrides.pop(0)
    return args


def read_args(path):
    if path is None:
        return []
    if not path.is_file():
        raise ValueError(f"missing Hydra argument file: {path}")
    args = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not ("=" in line and not line.startswith("--")):
            raise ValueError(f"{path}:{n}: expected one key=value Hydra override per line")
        args.append(line)
    return args


def log(message):
    print("[REAL-VERL-E2E] " + message, flush=True)


def record(path, **details):
    path.write_text(json.dumps(details, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def validate_lease(path, donor_session, ray, namespace):
    """Validate fixture vs Ray PG; physical GPU identity is proven at runtime."""
    from multi_task_scheduler.orchestration.contracts import Lease

    payload = json.loads(path.read_text(encoding="utf-8"))
    claims = [dict(item) for item in payload.get("claims", [])]
    if len(claims) != 1:
        raise ValueError("first-release physical lifecycle requires exactly 1 donor claim")
    c = claims[0]
    if c.get("donor_task_id") in (None, "", "__TASK_SESSION__", "$TASK_SESSION"):
        c["donor_task_id"] = donor_session
    if c["donor_task_id"] != donor_session:
        raise ValueError("Lease donor_task_id does not match the newly started donor session")
    if "expires_at" in payload and float(payload["expires_at"]) <= time.time():
        raise ValueError("Lease expires_at is already in the past")
    if "expires_at" not in payload and float(payload.get("expires_in_s", 600)) <= 0:
        raise ValueError("Lease expires_in_s must be positive")
    for key in ("pg_id", "node_id", "gpu_uuid"):
        if "replace-" in str(c.get(key, "")) or "实际" in str(c.get(key, "")):
            raise ValueError(f"replace the sample placeholder for {key}")
    if c.get("pg_namespace", namespace) != namespace:
        raise ValueError("Lease pg_namespace must equal the training Ray namespace")
    Lease(payload.get("lease_id", "e2e-lease"), (c,), time.time() + 600)

    pg_id = c["pg_id"]
    table = ray.util.placement_group_table()
    info = table.get(pg_id)
    if info is None:
        raise RuntimeError(f"donor placement group is absent: {pg_id}")
    if info.get("state") != "CREATED":
        raise RuntimeError(f"donor placement group not CREATED: {info.get('state')}")
    idx = c["bundle_index"]
    node_ids = info.get("bundles_to_node_id") or {}
    real_node = node_ids.get(idx, node_ids.get(str(idx)))
    if real_node != c["node_id"]:
        raise RuntimeError(f"donor PG bundle {idx} is on node {real_node!r}, not {c['node_id']!r}")
    bundles = info.get("bundles") or {}
    bundle = bundles.get(idx, bundles.get(str(idx)))
    if not isinstance(bundle, dict):
        raise RuntimeError(f"donor PG has no bundle {idx}")
    pg_name = info.get("name")
    if not pg_name:
        raise RuntimeError("donor PG has no name; borrowed actor requires named PG recovery")
    resolved = ray.util.get_placement_group(pg_name)
    if resolved.id.hex() != pg_id:
        raise RuntimeError("donor named PG points to another PG id")
    # Do not guess a physical accelerator from this PG: the E2E backend validates
    # the CE actor's actual GPU UUID/NPU id against the claim.
    log(f"Verified real PG id={pg_id} name={pg_name} bundle={idx} node={real_node} resources={bundle}")
    return payload


def main():
    a = arguments()
    repo = a.repo.resolve()
    logs = (a.logs / time.strftime("%Y%m%d-%H%M%S") ).resolve()
    logs.mkdir(parents=True, exist_ok=True)
    e2e_root = logs / "scenarios"
    summary_file = logs / "orchestration_summary.json"
    procs = []
    opened_logs = []
    state = {"state": "BLOCKED", "detail": "not started", "log_dir": str(logs),
             "ray_address": a.ray_address, "namespace": a.namespace, "scenarios": a.scenarios}
    result_code = 2
    try:
        if not (repo / "scripts/e2e/run_all.sh").is_file():
            raise ValueError(f"not a verl-multi-task checkout: {repo}")
        if not a.interactive_lease and not a.lease.is_file():
            raise ValueError(f"Lease fixture not found: {a.lease}; use --interactive-lease to fill it after startup")
        if not (a.native_args or a.donor_args or a.borrower_args or a.native_overrides):
            raise ValueError("provide native VERL Hydra args after -- or via --native-args (model/data/etc)")
        if a.trainer_gpus <= 0 or a.rollout_gpus <= 0:
            raise ValueError("trainer-gpus and rollout-gpus must be positive")
        chosen = a.scenarios.split()
        known = {"control_plane", "exactly_once", "recovery", "lifecycle", "force"}
        if not chosen or any(x not in known for x in chosen):
            raise ValueError(f"unknown/empty scenario list: {a.scenarios!r}")
        if a.require_inflight_force and "force" not in chosen:
            raise ValueError("--require-inflight-force requires scenario force")

        import ray
        import verl.experimental.fully_async_policy.fully_async_main as entry
        from multi_task_scheduler.scheduler.discovery import get_or_create_group_scheduler

        if not hasattr(entry, "_resolve_task_runner_class"):
            raise ValueError("VERL Fully Async entry has no MultiTask bridge; apply compatible entry patch")
        shared_args = read_args(a.native_args) + list(a.native_overrides)
        for item in a.native_overrides:
            if "=" not in item:
                raise ValueError(f"expected Hydra key=value after --, got: {item!r}")
        local_args = {"donor": read_args(a.donor_args), "borrower": read_args(a.borrower_args)}
        env = dict(os.environ)
        env.update(RAY_ADDRESS=a.ray_address, PYTHONUNBUFFERED="1",
                   MT_E2E_LOG_ROOT=str(e2e_root), MT_E2E_ATTACH_ONLY="1",
                   MT_E2E_REQUIRE_COMPLETE="1")
        env["PYTHONPATH"] = str(repo / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        # Avoid inherited FORCE proof flag: in existing attach-mode script it
        # only supports launcher logs; our optional proof checker scans task logs.
        env.pop("MT_E2E_REQUIRE_INFLIGHT_FORCE", None)

        log(f"Connect Ray {a.ray_address}, namespace={a.namespace}")
        ray.init(address=a.ray_address, namespace=a.namespace, ignore_reinit_error=False,
                 log_to_driver=False)
        gs = get_or_create_group_scheduler()
        gs_actor_id = str(gs._actor_id)  # audit-only identity, not placement authority
        baseline = set(ray.get(gs.get_task_runners.remote(), timeout=15))
        log(f"Shared GroupScheduler ActorId={gs_actor_id}; baseline sessions={len(baseline)}")
        state["group_scheduler_actor_id"] = gs_actor_id
        state["baseline_sessions"] = sorted(baseline)
        log(f"Ray cluster resources: {ray.cluster_resources()}")

        # User's native configuration comes first, then enforce MultiTask's
        # supported runtime contract and a shared Ray address/namespace.
        fixed = [
            "multitask.enabled=true",
            "multitask.runtime.profile=experimental_fully_async_standalone",
            f"++ray_kwargs.ray_init.address={a.ray_address}",
            f"++ray_kwargs.ray_init.namespace={a.namespace}",
            "trainer.project_name=multitask_e2e",
            "trainer.resume_mode=disable",
            "trainer.logger=[console]",
            "trainer.nnodes=1",
            f"trainer.n_gpus_per_node={a.trainer_gpus}",
            "rollout.nnodes=1",
            f"rollout.n_gpus_per_node={a.rollout_gpus}",
            "actor_rollout_ref.hybrid_engine=false",
            "actor_rollout_ref.rollout.name=vllm",
            "actor_rollout_ref.rollout.mode=async",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "actor_rollout_ref.rollout.data_parallel_size=1",
            "actor_rollout_ref.rollout.pipeline_model_parallel_size=1",
            "++actor_rollout_ref.rollout.enable_sleep_mode=true",
            "actor_rollout_ref.rollout.free_cache_engine=true",
            "actor_rollout_ref.rollout.calculate_log_probs=true",
            "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
            "++actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group=true",
            "async_training.use_trainer_do_validate=false",
            "async_training.use_dynamic_resource_scheduling=false",
            "async_training.partial_rollout=true",
            "data.train_batch_size=0",
            "data.gen_batch_size=1",
        ]
        # The native args file can override various model, trainer, dataset and
        # algorithm settings, but cannot silently disable MultiTask/Ray wiring.
        def start(role, token):
            overrides = shared_args + local_args[role] + fixed + [f"trainer.experiment_name=two_real_{role}_{token}"]
            cmd = [sys.executable, "-m", "verl.experimental.fully_async_policy.fully_async_main", *overrides]
            (logs / f"{role}_command.json").write_text(
                json.dumps(cmd, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            fout = (logs / f"{role}.log").open("w", encoding="utf-8")
            opened_logs.append(fout)
            proc = subprocess.Popen(cmd, env=env, cwd=str(repo), stdin=subprocess.DEVNULL,
                                    stdout=fout, stderr=subprocess.STDOUT, start_new_session=True)
            procs.append(proc)
            log(f"Started real {role}: pid={proc.pid}, log={logs / (role + '.log')}")
            return proc

        def await_new_session(role, previous, proc):
            deadline = time.monotonic() + a.startup_timeout
            last_error = None
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"{role} native VERL exited early with rc={proc.returncode}; check {logs / (role + '.log')}")
                try:
                    registered = ray.get(gs.get_task_runners.remote(), timeout=10)
                    fresh = set(registered) - previous
                    if len(fresh) == 1:
                        session = fresh.pop()
                        log(f"Real {role} attached: task_session={session}")
                        return session
                    if len(fresh) > 1:
                        raise RuntimeError("multiple unrecognized new sessions; cluster is not isolated for this run")
                except RuntimeError:
                    raise
                except Exception as exc:
                    last_error = repr(exc)
                time.sleep(2)
            raise TimeoutError(f"{role} did not register within startup timeout; last={last_error}")

        tag = uuid.uuid4().hex[:8]
        donor_proc = start("donor", tag)
        donor = await_new_session("donor", baseline, donor_proc)
        borrower_proc = start("borrower", tag)
        borrower = await_new_session("borrower", baseline | {donor}, borrower_proc)
        if donor == borrower:
            raise RuntimeError("donor and borrower registered the same task session")
        state["donor_task_session"] = donor
        state["borrower_task_session"] = borrower

        registered = ray.get(gs.get_task_runners.remote(), timeout=15)
        if donor not in registered or borrower not in registered:
            raise RuntimeError("one of the new TaskRunners disappeared from the shared GS")
        gs2 = ray.get_actor("verl-multi-task-group-scheduler", namespace="verl-multi-task")
        if gs2._actor_id != gs._actor_id:
            raise RuntimeError("named GroupScheduler identity changed during startup")
        log("PASS: two real tasks registered under ONE shared GroupScheduler")
        state["registration"] = "PASS"

        if a.interactive_lease:
            candidate_pgs = {}
            for pg_id, info in ray.util.placement_group_table().items():
                if info.get("state") == "CREATED":
                    candidate_pgs[pg_id] = {
                        "name": info.get("name"), "bundles": info.get("bundles"),
                        "bundles_to_node_id": info.get("bundles_to_node_id"),
                    }
            (logs / "placement_groups.json").write_text(
                json.dumps(candidate_pgs, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
            )
            log(f"Ray PG snapshot: {logs / 'placement_groups.json'}")
            log("Use the donor CE worker placement to identify exact pg_id, node_id, GPU UUID/NPU ID, bundle_index.")
            log(f"Write/replace the Lease file: {a.lease}")
            log("Neither rank=0 nor placement group name alone proves physical accelerator ownership.")
            input("Press Enter after you have saved the correct Lease JSON (Ctrl+C to cancel): ")
        validate_lease(a.lease, donor, ray, a.namespace)
        state["lease_preflight"] = "PASS"
        for proc in procs:
            if proc.poll() is not None:
                raise RuntimeError(f"training driver exited before E2E: pid={proc.pid}, rc={proc.returncode}")

        # Execute the repository's existing actual Ray/evidence validators.
        # This intentionally does NOT set MT_E2E_REQUIRE_INFLIGHT_FORCE=1, as
        # validate_force_remove.sh expects launcher-mode runtime.log.
        env.update(MT_E2E_DONOR_SESSION=donor, MT_E2E_BORROWER_SESSION=borrower,
                   MT_E2E_SCENARIOS=" ".join(chosen), RAY_ADDRESS=a.ray_address,
                   MT_E2E_ATTACH_ONLY="1")
        log(f"Run actual E2E scenarios: {' '.join(chosen)}")
        fout = (logs / "e2e.log").open("w", encoding="utf-8")
        opened_logs.append(fout)
        try:
            completed = subprocess.run(["bash", str(repo / "scripts/e2e/run_all.sh")],
                                       env=env, cwd=str(repo), stdout=fout,
                                       stderr=subprocess.STDOUT, timeout=a.e2e_timeout)
            result_code = completed.returncode
        except subprocess.TimeoutExpired:
            result_code = 2
            state["detail"] = "E2E run exceeded configured timeout; outcomes are unverified"
        log(f"E2E runner return code: {result_code} (0=PASS, 1=FAIL, 2=BLOCKED)")
        summaries = sorted(e2e_root.glob("comprehensive/*/summary.json"))
        if summaries:
            source_summary = summaries[-1]
            state["scenario_summary"] = str(source_summary)
            parsed = json.loads(source_summary.read_text(encoding="utf-8"))
            state["scenario_results"] = parsed
            for row in parsed.get("results", []):
                log(f"  {row['scenario']}: {row['state']} ({row['detail']})")
        else:
            state["scenario_results"] = None
            result_code = 1 if result_code == 0 else result_code

        if a.require_inflight_force and result_code == 0:
            force_results = sorted(e2e_root.glob("force_cycle/*/result.json"))
            if not force_results:
                raise RuntimeError("force cycle result.json is absent")
            force_result = json.loads(force_results[-1].read_text(encoding="utf-8"))
            commands = [step["command"] for step in force_result.get("events", [])
                        if step.get("command", {}).get("kind") == "REMOVE"
                        and step["command"].get("force") is True]
            if len(commands) != 1:
                raise RuntimeError("could not uniquely identify FORCE REMOVE operation")
            op_id = commands[0]["operation_id"]
            pattern = re.compile(r"MULTITASK_FORCE_HANDOFF\s+(\{.*\})")
            receipts = []
            # Actor logs may not be forwarded into the driver log; absence is BLOCKED.
            for ln in (logs / "borrower.log").read_text(encoding="utf-8", errors="replace").splitlines():
                m = pattern.search(ln)
                if m:
                    try:
                        item = json.loads(m.group(1))
                    except json.JSONDecodeError:
                        continue
                    if item.get("operation_id") == op_id:
                        receipts.append(item)
            valid = []
            for item in receipts:
                ad, conf, ab, known = (item.get("admitted_count"), item.get("confirmed_count"),
                                       item.get("aborted_count"), item.get("abort_ack_known"))
                if type(ad) is not int or type(conf) is not int or type(known) is not bool or not 0 <= conf <= ad:
                    result_code = 1
                    break
                if known:
                    if type(ab) is not int or not 0 <= ab <= conf:
                        result_code = 1
                        break
                    if ab > 0:
                        valid.append(item)
                else:
                    if ab is not None or conf != ad:
                        result_code = 1
                        break
                    if ad > 0:
                        valid.append(item)
            if result_code == 0 and not valid:
                result_code = 2
                state["detail"] = "no matching positive in-flight FORCE proof in borrower forwarded logs"
            if result_code == 0:
                state["inflight_force_proof"] = valid[-1]
                log(f"PASS: in-flight FORCE handoff evidence: {valid[-1]}")

        if result_code == 0:
            for proc in procs:
                if proc.poll() is not None:
                    result_code = 1
                    state["detail"] = f"training driver exited during E2E pid={proc.pid} rc={proc.returncode}"
                    break
        state["state"] = {0:"PASS",1:"FAIL",2:"BLOCKED"}.get(result_code,"FAIL")
        if state["detail"] == "not started":
            state["detail"] = "two real jobs and requested acceptance scenarios completed" if result_code == 0 else "see e2e.log and scenario summaries"
    except (FileNotFoundError, ValueError) as exc:
        result_code = 1
        state.update(state="FAIL", detail=f"{type(exc).__name__}: {exc}")
        log("FAIL: " + state["detail"])
    except (TimeoutError, RuntimeError) as exc:
        result_code = 2
        state.update(state="BLOCKED", detail=f"{type(exc).__name__}: {exc}")
        log("BLOCKED: " + state["detail"])
    except Exception as exc:
        result_code = 1
        state.update(state="FAIL", detail=f"{type(exc).__name__}: {exc}")
        log("FAIL: " + state["detail"])
    finally:
        for f in opened_logs:
            f.flush()
            f.close()
        state["driver_pids"] = [p.pid for p in procs]
        state["drivers_running_at_end"] = [p.poll() is None for p in procs]
        if not a.keep_running:
            for proc in reversed(procs):
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for proc in reversed(procs):
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
        else:
            log("--keep-running: leaving real VERL drivers alive; manage with PIDs above")
        record(summary_file, **state)
        log(f"STATE={state['state']} ({state['detail']})")
        log(f"Summary: {summary_file}")
        log(f"Logs: {logs}")
    return result_code if result_code in (0, 1, 2) else 1


if __name__ == "__main__":
    sys.exit(main())
