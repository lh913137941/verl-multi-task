"""Static acceptance-runner validation for CI without Ray/VERL."""

import ast
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
E2E = ROOT / "scripts" / "e2e"


def test_e2e_shell_scripts_parse_with_bash():
    scripts = sorted(E2E.glob("*.sh"))
    assert scripts
    for path in scripts:
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{path}: {result.stderr}"


def test_e2e_python_drivers_parse_without_importing_runtime_dependencies():
    drivers = sorted(E2E.glob("*.py"))
    assert drivers
    for path in drivers:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _load_live_fixture_under_test():
    """Isolate the real driver parser; do not import its Ray CLI entrypoint."""
    import json
    import time

    path = E2E / "lifecycle_driver.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    func = next(
        node for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_fixture"
    )
    scope = {"json": json, "time": time, "Path": Path}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        func,
                    ],
                    type_ignores=[],
                )
            ),
            str(path),
            "exec",
        ),
        scope,
    )
    return scope["_load_fixture"]


def test_live_e2e_claim_ids_are_unique_for_consecutive_lease_cycles(tmp_path):
    import json

    sample = {
        "lease_id": "e2e-lifecycle",
        "claims": [
            {
                "claim_id": "fixed-claim-from-example",
                "source_lease_id": "source-0",
                "donor_task_id": "__TASK_SESSION__",
                "donor_replica_rank": 0,
                "pg_id": "same-physical-pg",
                "bundle_index": 0,
                "node_id": "node-0",
                "gpu_uuid": "NPU:node-0:0",
                "gpu_fraction": 0.5,
                "cpu_request": 1.0,
            }
        ],
    }
    path = tmp_path / "lease.json"
    path.write_text(json.dumps(sample), encoding="utf-8")
    load = _load_live_fixture_under_test()
    first, _, _ = load(path, "donor", "borrower", "run-1")
    second, _, _ = load(path, "donor", "borrower", "run-2")

    assert first.lease_id != second.lease_id
    assert first.claim_ids != second.claim_ids
    assert first.claim_ids == ("fixed-claim-from-example-run-1",)
    assert second.claim_ids == ("fixed-claim-from-example-run-2",)
    # Only the logical claim IDs change; do not fabricate physical placement.
    assert first.gpu_uuids == second.gpu_uuids
    assert first.bundle_keys == second.bundle_keys
    assert first.claims[0]["donor_task_id"] == "donor"


def test_live_e2e_explicit_lease_replay_preserves_original_claim_id(tmp_path):
    import json

    sample = {
        "lease_id": "replay-lease",
        "reuse_lease_id": True,
        "claims": [
            {
                "claim_id": "replay-claim",
                "source_lease_id": "source-0",
                "donor_task_id": "donor",
                "donor_replica_rank": 0,
                "pg_id": "pg-0",
                "bundle_index": 0,
                "node_id": "node-0",
                "gpu_uuid": "NPU:node-0:0",
                "gpu_fraction": 0.5,
                "cpu_request": 1.0,
            }
        ],
    }
    path = tmp_path / "lease-replay.json"
    path.write_text(json.dumps(sample), encoding="utf-8")
    load = _load_live_fixture_under_test()
    first, _, _ = load(path, "donor", "borrower", "run-1")
    second, _, _ = load(path, "donor", "borrower", "run-2")

    assert first.lease_id == second.lease_id == "replay-lease"
    assert first.claim_ids == second.claim_ids == ("replay-claim",)
