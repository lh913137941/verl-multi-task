"""Execute the FORCE runner's receipt checker without Ray or a live launcher."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/e2e/validate_force_remove.sh"


def _receipt(operation_id="force-current", **changes):
    return {
        "operation_id": operation_id,
        "admitted_count": 2,
        "abort_ack_known": True,
        "aborted_count": 1,
        "confirmed_count": 1,
        **changes,
    }


def _check(tmp_path, receipts):
    runtime_log = tmp_path / "runtime.log"
    runtime_log.write_text("\n".join(
        "MULTITASK_FORCE_HANDOFF " + json.dumps(item) for item in receipts
    ), encoding="utf-8")
    result_file = tmp_path / "result.json"
    result_file.write_text(json.dumps({
        "state": "PASSED", "scenario": "force_cycle",
        "events": [{"command": {
            "kind": "REMOVE", "force": True, "operation_id": "force-current",
        }}],
    }), encoding="utf-8")
    code = SCRIPT.read_text(encoding="utf-8").split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
    return subprocess.run(
        [sys.executable, "-c", code, str(runtime_log), str(result_file)],
        capture_output=True, text=True,
    ).returncode


@pytest.mark.parametrize("receipts,expected", [
    ([_receipt()], 0),
    ([_receipt("other-operation")], 2),
    ([_receipt(aborted_count=0, confirmed_count=0)], 2),
    ([_receipt(confirmed_count=0), _receipt("other", admitted_count=0,
                                         aborted_count=0, confirmed_count=0)], 1),
    ([_receipt(abort_ack_known=False, aborted_count=None, confirmed_count=2)], 0),
    ([_receipt(abort_ack_known=False, aborted_count=None, confirmed_count=1)], 1),
    ([_receipt(confirmed_count=3)], 1),
    ([_receipt(aborted_count=True)], 1),
])
def test_force_proof_belongs_to_this_operation_and_one_complete_receipt(tmp_path, receipts, expected):
    assert _check(tmp_path, receipts) == expected
