"""Guard the live two-job E2E dispatch after removing legacy launchers.

No Ray/VERL/NPU required: enforce the script call graph and fail-closed
behavior rather than relying on tests that import the full training runtime.
"""

import ast
import os
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "e2e"


def test_only_one_supported_two_job_entrypoint():
    for obsolete in ("verify_cluster.sh", "verify_quick.sh", "list_tasks.py"):
        assert not (SCRIPTS / obsolete).exists()

    runner = (SCRIPTS / "verify_two_verl_jobs.py").read_text(encoding="utf-8")
    assert 'scripts/e2e/run_all.sh' in runner
    assert '--scenarios' in runner
    assert '--require-inflight-force' in runner
    # Keep manual proven fixture support, but no interactive Ray-PG editing.
    assert '--lease' in runner
    assert '--auto-lease' not in runner
    assert '--interactive-lease' not in runner
    assert 'input("Press Enter' not in runner


def test_all_five_validate_scripts_still_reachable_from_real_two_job_runner():
    dispatch = (SCRIPTS / "run_all.sh").read_text(encoding="utf-8")
    expected = {
        "control_plane": ("validate_control_plane.sh", "control_plane_recovery.py"),
        "exactly_once": ("validate_exactly_once.sh", "exactly_once_driver.py"),
        "recovery": ("validate_recovery_faults.sh", "test_taskrunner_wiring.py"),
        "lifecycle": ("validate_lifecycle_cycle.sh", "lifecycle_driver.py"),
        "force": ("validate_force_remove.sh", "lifecycle_driver.py"),
    }
    common = (SCRIPTS / "common.sh").read_text(encoding="utf-8")
    for name, (wrapper, target) in expected.items():
        assert re.search(
            rf"\b{re.escape(name)}\) script=.*{re.escape(wrapper)}",
            dispatch,
        ), (name, wrapper)
        assert (SCRIPTS / wrapper).is_file()
        assert target in (
            (SCRIPTS / wrapper).read_text(encoding="utf-8") + common
        )
    assert 'MT_E2E_SCENARIOS' in dispatch
    assert 'MT_E2E_REQUIRE_COMPLETE' in dispatch

    # This scenario invokes named unit tests rather than the entire suite:
    # moving a test must update its selector, not silently lose recovery proof.
    recovery = (SCRIPTS / "validate_recovery_faults.sh").read_text(encoding="utf-8")
    selectors = re.findall(
        r'"(tests/unit/test_[^"]+\.py)::(test_\w+)"', recovery
    )
    assert len(selectors) == 9
    for relative_path, test_name in selectors:
        test_file = ROOT / relative_path
        assert test_file.is_file(), relative_path
        module = ast.parse(test_file.read_text(encoding="utf-8"))
        assert any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == test_name
            for node in module.body
        ), f"stale recovery selector: {relative_path}::{test_name}"


@pytest.mark.parametrize(
    "script", ["validate_lifecycle_cycle.sh", "validate_force_remove.sh"]
)
def test_real_lifecycle_rejects_standalone_without_running_donor_and_borrower(script):
    env = dict(os.environ)
    env.pop("MT_E2E_ATTACH_ONLY", None)
    env.pop("MT_E2E_LEASE_FILE", None)
    env.pop("MULTITASK_LAUNCH_SCRIPT", None)
    result = subprocess.run(
        ["bash", str(SCRIPTS / script)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert "BLOCKED" in result.stderr
    assert "verify_two_verl_jobs.py" in result.stderr


def test_strict_force_receipt_validation_has_one_owner():
    wrapper = (SCRIPTS / "validate_force_remove.sh").read_text(encoding="utf-8")
    runner = (SCRIPTS / "verify_two_verl_jobs.py").read_text(encoding="utf-8")
    assert "MULTITASK_FORCE_HANDOFF" not in wrapper
    assert "MT_E2E_REQUIRE_INFLIGHT_FORCE" not in wrapper
    assert "MULTITASK_FORCE_HANDOFF" in runner
    assert "force_cycle" in runner
    assert "aborted_count" in runner
    assert "confirmed_count" in runner
    assert "MULTITASK_LAUNCH_SCRIPT" not in (
        (SCRIPTS / "common.sh").read_text(encoding="utf-8")
    )
