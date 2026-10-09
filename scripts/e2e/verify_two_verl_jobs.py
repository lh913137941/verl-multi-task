#!/usr/bin/env python3
"""Launch two real VERL Fully Async MultiTask jobs and run shared-GS E2E.

Runs *native* VERL entrypoint twice; delegates acceptance to the repository's
scripts/e2e/run_all.sh. Does not create fake jobs, fake leases, or synthetic
lifecycle success evidence.

Input: native Fully Async Hydra overrides (one per line or after --).
Without --lease, discovers the real donor CE placement and automatically
writes a verified Lease fixture for the E2E lifecycle driver.
--lease can supply a previously verified manual fixture for diagnostics.
"""
from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid


def configure_multitask_import(repo):
    """Expose this checkout to the driver, spawned VERL jobs and Ray workers.

    Adding PYTHONPATH to the subprocess env alone is too late: this driver
    imports GroupScheduler before any child VERL process is launched.
    """
    source = (repo / "src").resolve()
    package = source / "multi_task_scheduler" / "__init__.py"
    if not package.is_file():
        raise ValueError(f"MultiTask source checkout is missing: {package}")

    # Avoid duplicate entries while giving the requested checkout precedence.
    original = os.environ.get("PYTHONPATH", "")
    components = [p for p in original.split(os.pathsep) if p and p != str(source)]
    new_pythonpath = os.pathsep.join([str(source), *components])
    os.environ["PYTHONPATH"] = new_pythonpath
    if str(source) in sys.path:
        sys.path.remove(str(source))
    sys.path.insert(0, str(source))

    # Guard against silently using another version from site-packages.
    imported = sys.modules.get("multi_task_scheduler")
    if imported is not None:
        actual = Path(getattr(imported, "__file__", "") or "").resolve()
        if not actual.is_relative_to(source):
            raise RuntimeError(
                f"multi_task_scheduler already imported from {actual}, expected {source}"
            )
    spec = importlib.util.find_spec("multi_task_scheduler")
    if spec is None or not spec.origin or not Path(spec.origin).resolve().is_relative_to(source):
        raise RuntimeError(
            f"multi_task_scheduler cannot be imported from current checkout {source}"
        )
    return new_pythonpath


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd(), help="verl-multi-task checkout root")
    p.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS", "auto"))
    p.add_argument("--start-local-ray", action="store_true",
                   help="if auto cannot find Ray, start a temporary single-node cluster for this E2E only")
    p.add_argument("--namespace", default="multitask-jobs", help="same namespace for both VERL jobs and donor PG")
    p.add_argument("--lease", type=Path, help="optional manual donor Lease JSON; omit for automatic lease discovery")
    p.add_argument("--donor-replica-rank", type=int, default=0,
                   help="native donor replica rank selected for automatic Lease (default: 0)")
    p.add_argument("--native-args", type=Path,
                   help="optional native Fully Async Hydra overrides file (one argument per line)")
    p.add_argument("--model-path", help="actual local HF model directory or Hugging Face repo ID; overrides fixture placeholder")
    p.add_argument("--train-files", help="optional real training Parquet path overriding native-args")
    p.add_argument("--val-files", help="optional real validation Parquet path overriding native-args")
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
    p.add_argument("--no-auto-bridge", action="store_true",
                   help="verify VERL entry read-only; do not patch imported VERL sources automatically")
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


def _effective_override(overrides, name):
    """Hydra applies repeated key=value overrides in argument order; last wins."""
    result = None
    for item in overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key.lstrip("+") == name:
            result = value.strip().strip("'\\\"")
    return result


def checkpoint_backend_overrides(device):
    """Use native NCCL on CUDA and opt-in HCCL teardown on Ascend NPU."""
    if device == "npu":
        return [
            "actor_rollout_ref.rollout.checkpoint_engine.backend=multitask_hccl",
            "++actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.multitask_hccl.rebuild_group=true",
            "++actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=multi_task_scheduler.checkpoint.hccl_checkpoint_engine",
        ]
    if device == "cuda":
        return [
            "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
            "++actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group=true",
        ]
    raise ValueError(f"Unsupported checkpoint backend device: {device!r}")


def validate_training_inputs(overrides, *, role):
    """Fail early on template paths; never launch Ray just to find missing files."""
    model = _effective_override(overrides, "actor_rollout_ref.model.path")
    if not model:
        raise ValueError(
            f"{role}: actor_rollout_ref.model.path is missing. "
            "Pass --model-path /absolute/path/to/model or set it in --native-args."
        )
    if "REPLACE_WITH" in model or "/path/to/" in model:
        raise ValueError(
            f"{role}: placeholder model path {model!r}. "
            "Pass --model-path /real/model/directory; a local model needs config.json and tokenizer files."
        )
    if model.startswith(("/", "./", "../", "~")):
        local = Path(model).expanduser()
        if not local.is_dir() or not (local / "config.json").is_file():
            raise ValueError(
                f"{role}: local model directory is missing or lacks config.json: {model!r}. "
                "Set --model-path to a real Hugging Face model directory."
            )

    for key, flag in (("data.train_files", "--train-files"), ("data.val_files", "--val-files")):
        value = _effective_override(overrides, key)
        if not value:
            raise ValueError(f"{role}: {key} is missing; pass {flag} /real/file.parquet")
        if "REPLACE_WITH" in value or "/path/to/" in value:
            raise ValueError(f"{role}: {key} still contains a placeholder: {value!r}")
        # Check ordinary absolute local-file overrides; Hydra lists, globs and
        # remote dataset URIs are handled by native VERL instead of guessed here.
        if value.startswith("/") and not any(c in value for c in "[],*?") and not Path(value).is_file():
            raise ValueError(
                f"{role}: {key} points to a nonexistent local file: {value!r}. "
                f"Set {flag} to a real parquet file."
            )


def log(message):
    print("[REAL-VERL-E2E] " + message, flush=True)


def record(path, **details):
    path.write_text(json.dumps(details, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def driver_log_tail(path, max_lines=100):
    """Return a bounded failure excerpt without loading a potentially huge driver log."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            return "".join(deque(stream, maxlen=max_lines)).rstrip()
    except OSError as exc:
        return f"<could not read driver log: {exc}>"


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


def build_auto_lease(donor_session, candidates, donor_rank, token, ttl_s, namespace):
    """Build a first-release Lease solely from donor-owned CE/PG evidence."""
    matches = [c for c in candidates if c.get("donor_replica_rank") == donor_rank]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one verified ACTIVE native rank {donor_rank}; "
            f"found {len(matches)} (candidates={candidates!r})"
        )
    c = matches[0]
    required = ("pg_id", "node_id", "gpu_uuid", "bundle_index", "resource_name")
    if any(not isinstance(c.get(k), (str, int)) or c[k] == "" for k in required):
        raise RuntimeError("native candidate lacks verified CE and Ray PG placement facts")
    if c["resource_name"] not in ("GPU", "NPU") or c["bundle_index"] != 0:
        raise RuntimeError("first release supports only one whole GPU/NPU in bundle zero")
    return {
        "lease_id": f"auto-real-e2e-{token}",
        "expires_in_s": ttl_s,
        "borrower_replica_id": f"borrowed-e2e-{token}",
        "claims": [{
            "claim_id": f"native-{donor_session}-{donor_rank}-{token}",
            "source_lease_id": f"native-{donor_session}-{donor_rank}",
            "donor_task_id": donor_session,
            "donor_replica_rank": donor_rank,
            "pg_id": c["pg_id"],
            "node_id": c["node_id"],
            "gpu_uuid": c["gpu_uuid"],
            "pg_namespace": namespace,
            "bundle_index": c["bundle_index"],
            "gpu_fraction": 0.5,
            "cpu_request": 1.0,
        }],
    }


def connect_ray(ray, *, address, namespace, start_local=False, pythonpath=None):
    """Connect existing Ray or opt-in to temporary, single-node Ray.

    Auto never silently starts a different Ray cluster in production.
    Children must receive the concrete GCS address, never 'local'.
    """
    if start_local and address != "auto":
        raise ValueError("--start-local-ray requires --ray-address auto")
    init_kwargs = dict(namespace=namespace, ignore_reinit_error=False, log_to_driver=False)
    # The driver itself and all Ray workers must resolve the same checkout,
    # including the detached GroupScheduler that is created on first startup.
    if pythonpath is not None:
        init_kwargs["runtime_env"] = {"env_vars": {"PYTHONPATH": pythonpath}}
    try:
        context = ray.init(address=address, **init_kwargs)
        return address, False
    except ConnectionError as exc:
        if address != "auto":
            raise RuntimeError(
                f"Cannot connect to Ray at {address!r}; check head address and network"
            ) from exc
        if not start_local:
            raise RuntimeError(
                "No running Ray cluster found. For a single-node run, add "
                "--start-local-ray; alternatively run 'ray start --head' once "
                "or provide --ray-address <head-ip>:<port> for an existing cluster."
            ) from exc

    context = ray.init(address="local", **init_kwargs)
    # Ray 'local' is an instruction to CREATE a cluster, not a connectable
    # address. Use the real GCS endpoint for both external VERL processes.
    runtime_context = ray.get_runtime_context()
    resolved = getattr(runtime_context, "gcs_address", None)
    if not resolved:
        resolved = getattr(context, "address_info", {}).get("address")
    if not isinstance(resolved, str) or not resolved or resolved in ("auto", "local"):
        raise RuntimeError("temporary Ray started but no concrete GCS address is available")
    resources = ray.cluster_resources()
    if not (float(resources.get("GPU", 0)) > 0 or float(resources.get("NPU", 0)) > 0):
        raise RuntimeError(
            "temporary Ray has no GPU/NPU resources; check Ascend Ray NPU "
            "discovery and start the node with a properly configured NPU resource"
        )
    return resolved, True


def validate_inflight_force_proof(force_result: dict, borrower_log: str) -> tuple[int, dict | None]:
    """Validate one actual FORCE operation from this run.

    Return (exit_status, positive_receipt): PASS=0, FAIL=1, BLOCKED=2.
    All receipts for the matching operation must be internally consistent.
    Missing actual in-flight interruption is BLOCKED, never a synthetic PASS.
    """
    if force_result.get("state") != "PASSED" or force_result.get("scenario") != "force_cycle":
        return 1, None
    commands = [
        step.get("command", {})
        for step in force_result.get("events", [])
        if step.get("command", {}).get("kind") == "REMOVE"
        and step["command"].get("force") is True
    ]
    if len(commands) != 1:
        return 1, None
    operation_id = commands[0].get("operation_id")
    if not isinstance(operation_id, str) or not operation_id:
        return 1, None

    records = []
    for line in borrower_log.splitlines():
        match = re.search(r"MULTITASK_FORCE_HANDOFF\s+(\{.*\})", line)
        if not match:
            continue
        try:
            receipt = json.loads(match.group(1))
        except json.JSONDecodeError:
            return 1, None
        if not isinstance(receipt, dict):
            return 1, None
        if receipt.get("operation_id") == operation_id:
            records.append(receipt)
    if not records:
        return 2, None

    proven = []
    for item in records:
        admitted, confirmed = item.get("admitted_count"), item.get("confirmed_count")
        aborted, known = item.get("aborted_count"), item.get("abort_ack_known")
        if (
            type(admitted) is not int or type(confirmed) is not int
            or type(known) is not bool or not 0 <= confirmed <= admitted
        ):
            return 1, None
        if known:
            if type(aborted) is not int or not 0 <= aborted <= confirmed:
                return 1, None
            if aborted > 0:
                proven.append(item)
        else:
            if aborted is not None or confirmed != admitted:
                return 1, None
            if admitted > 0:
                proven.append(item)
    if not proven:
        return 2, None
    return 0, proven[-1]


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
    temporary_ray = False
    try:
        if not (repo / "scripts/e2e/run_all.sh").is_file():
            raise ValueError(f"not a verl-multi-task checkout: {repo}")
        pythonpath = configure_multitask_import(repo)
        log(f"MultiTask source: {repo / 'src'}")
        auto_lease = a.lease is None
        lease_path = logs / "auto_lease.json" if auto_lease else a.lease.resolve()
        if not auto_lease and not lease_path.is_file():
            raise ValueError(f"manual Lease fixture not found: {lease_path}")
        if a.donor_replica_rank < 0:
            raise ValueError("--donor-replica-rank must be >= 0")
        if a.start_local_ray and a.keep_running:
            raise ValueError(
                "--keep-running requires a persistent Ray cluster; "
                "start Ray with 'ray start --head' instead"
            )
        state["lease_mode"] = "auto" if auto_lease else "manual"
        state["lease_file"] = str(lease_path)
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

        # Fix the specific VERL source imported by this Python, not a guessed
        # checkout. Default: standard repo patch, then AST-guided safe fallback;
        # --no-auto-bridge keeps the preflight strictly read-only.
        from ensure_verl_multitask_bridge import ensure_current_verl_bridge
        ensure_current_verl_bridge(check_only=a.no_auto_bridge)

        import ray
        import verl.experimental.fully_async_policy.fully_async_main as entry
        from multi_task_scheduler.scheduler.discovery import get_or_create_group_scheduler

        if not callable(getattr(entry, "_resolve_task_runner_class", None)):
            entry_path = Path(getattr(entry, "__file__", "<unavailable>")).resolve()
            raise ValueError(
                "Imported VERL Fully Async entry has no MultiTask TaskRunner bridge: "
                f"entry={entry_path}, python={sys.executable}. "
                "Apply patches/verl-v0.10-fully-async-multitask-entry.patch to "
                "the VERL source checkout used by THIS interpreter, or install "
                "that patched VERL checkout with python -m pip install -e <VERL_ROOT>; "
                "do not bypass this preflight check."
            )
        shared_args = read_args(a.native_args) + list(a.native_overrides)
        for item in a.native_overrides:
            if "=" not in item:
                raise ValueError(f"expected Hydra key=value after --, got: {item!r}")
        local_args = {"donor": read_args(a.donor_args), "borrower": read_args(a.borrower_args)}
        input_overrides = [
            f"{key}={value}" for key, value in (
                ("actor_rollout_ref.model.path", a.model_path),
                ("data.train_files", a.train_files),
                ("data.val_files", a.val_files),
            ) if value is not None
        ]
        for role in ("donor", "borrower"):
            validate_training_inputs(
                shared_args + local_args[role] + input_overrides, role=role
            )
        log("Validated donor/borrower model and local dataset paths before Ray startup")
        devices = {
            role: _effective_override(shared_args + local_args[role], "trainer.device")
            for role in ("donor", "borrower")
        }
        if devices["donor"] != devices["borrower"]:
            raise ValueError(
                f"both jobs must use the same trainer.device for shared checkpoint backend: {devices}"
            )
        if devices["donor"] not in ("npu", "cuda"):
            raise ValueError(
                f"set trainer.device=npu or trainer.device=cuda in --native-args, got {devices}"
            )
        env = dict(os.environ)
        env.update(RAY_ADDRESS=a.ray_address, PYTHONUNBUFFERED="1",
                   MT_E2E_LOG_ROOT=str(e2e_root), MT_E2E_ATTACH_ONLY="1",
                   MT_E2E_REQUIRE_COMPLETE="1")
        env["PYTHONPATH"] = pythonpath

        log(f"Connect Ray {a.ray_address}, namespace={a.namespace}")
        connected_address, temporary_ray = connect_ray(
            ray, address=a.ray_address, namespace=a.namespace,
            start_local=a.start_local_ray, pythonpath=pythonpath,
        )
        if temporary_ray:
            log(f"Started isolated local Ray: {connected_address}")
        a.ray_address = connected_address
        state["ray_address"] = connected_address
        state["temporary_ray"] = temporary_ray
        env["RAY_ADDRESS"] = connected_address
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
            # The native 'nccl' registry entry maps to HCCL on NPU, whose
            # finalize() uses PyHcclCommunicator.destroyComm (not implemented).
            # Both transfer endpoints load the compatible opt-in plugin on NPU.
            *checkpoint_backend_overrides(devices["donor"]),
            "async_training.use_trainer_do_validate=false",
            "async_training.use_dynamic_resource_scheduling=false",
            "async_training.partial_rollout=true",
            "data.train_batch_size=0",
            "data.gen_batch_size=1",
        ]
        log(
            "Checkpoint backend: "
            + ("multitask_hccl (NPU-compatible communicator finalize)"
               if devices["donor"] == "npu" else "nccl (CUDA native)")
        )
        # The native args file can override various model, trainer, dataset and
        # algorithm settings, but cannot silently disable MultiTask/Ray wiring.
        def start(role, token):
            overrides = shared_args + local_args[role] + input_overrides + fixed + [f"trainer.experiment_name=two_real_{role}_{token}"]
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
                    driver_log = logs / (role + ".log")
                    tail = driver_log_tail(driver_log)
                    state["failed_driver"] = role
                    state["failed_driver_exit_code"] = proc.returncode
                    state["failed_driver_log"] = str(driver_log)
                    state["failed_driver_log_tail"] = tail
                    log(f"----- {role} native VERL error log (last 100 lines) -----")
                    for line in tail.splitlines():
                        print(line, flush=True)
                    log(f"----- End {role} log; full file: {driver_log} -----")
                    raise RuntimeError(
                        f"{role} native VERL exited early with rc={proc.returncode}; "
                        f"see {driver_log} and the log excerpt above"
                    )
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

        if auto_lease:
            try:
                candidates = ray.get(
                    registered[donor].native_placement_candidates.remote(), timeout=45
                )
            except Exception as exc:
                raise RuntimeError(
                    "donor CE placement could not be proven; refusing to invent pg_id/gpu_uuid: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            lease = build_auto_lease(
                donor, candidates, a.donor_replica_rank, tag,
                max(7200, a.e2e_timeout + 600), a.namespace,
            )
            record(lease_path, **lease)
            state["donor_placement"] = lease["claims"][0]
            log(
                "Auto Lease from actual donor CE + named PG: "
                f"rank={a.donor_replica_rank}, "
                f"PG={lease['claims'][0]['pg_id']}, "
                f"device={lease['claims'][0]['gpu_uuid']}"
            )
            log(f"Generated Lease fixture: {lease_path}")
        validate_lease(lease_path, donor, ray, a.namespace)
        state["lease_preflight"] = "PASS"
        env["MT_E2E_LEASE_FILE"] = str(lease_path)
        for proc in procs:
            if proc.poll() is not None:
                raise RuntimeError(f"training driver exited before E2E: pid={proc.pid}, rc={proc.returncode}")

        # Run the five scenario validators on the same real Ray cluster.
        # Strict FORCE receipts are verified here from the borrower actor log.
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
            result_code, receipt = validate_inflight_force_proof(
                force_result,
                (logs / "borrower.log").read_text(encoding="utf-8", errors="replace"),
            )
            if result_code == 0:
                state["inflight_force_proof"] = receipt
                log(f"PASS: in-flight FORCE handoff evidence: {receipt}")
            elif result_code == 2:
                state["detail"] = "no positive in-flight FORCE proof for this operation in borrower forwarded logs"
            else:
                state["detail"] = "invalid or conflicting in-flight FORCE proof for this operation"

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
        if temporary_ray:
            # ray.init(address="local") owns the cluster and shuts it down with
            # this parent process; do not leave it running after the drivers stop.
            try:
                ray.shutdown()
            except Exception as exc:
                log(f"Local Ray shutdown warning: {exc}")
        record(summary_file, **state)
        log(f"STATE={state['state']} ({state['detail']})")
        log(f"Summary: {summary_file}")
        log(f"Logs: {logs}")
    return result_code if result_code in (0, 1, 2) else 1


if __name__ == "__main__":
    sys.exit(main())
