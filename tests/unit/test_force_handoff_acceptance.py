"""Exercise the actual two-job FORCE proof checker without Ray or NPU."""

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/e2e/verify_two_verl_jobs.py"
spec = importlib.util.spec_from_file_location("e2e_two_jobs_proof_under_test", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


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
    force_result = {
        "state": "PASSED",
        "scenario": "force_cycle",
        "events": [{"command": {
            "kind": "REMOVE", "force": True, "operation_id": "force-current",
        }}],
    }
    borrower_log = "\n".join(
        "MULTITASK_FORCE_HANDOFF " + json.dumps(item)
        for item in receipts
    )
    status, proof = runner.validate_inflight_force_proof(force_result, borrower_log)
    if status == 0:
        assert proof in receipts
        assert proof["operation_id"] == "force-current"
    else:
        assert proof is None
    return status

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


def test_force_proof_rejects_wrong_scenario_or_nonterminal_result():
    receipt = _receipt()
    log = "MULTITASK_FORCE_HANDOFF " + json.dumps(receipt)
    for result in (
        {"state": "FAILED", "scenario": "force_cycle", "events": []},
        {"state": "PASSED", "scenario": "full_cycle", "events": []},
        {"state": "PASSED", "scenario": "force_cycle", "events": []},
    ):
        assert runner.validate_inflight_force_proof(result, log) == (1, None)


def test_force_proof_rejects_malformed_receipt():
    result = {
        "state": "PASSED", "scenario": "force_cycle",
        "events": [{"command": {
            "kind": "REMOVE", "force": True, "operation_id": "force-current",
        }}],
    }
    assert runner.validate_inflight_force_proof(
        result, "MULTITASK_FORCE_HANDOFF {bad-json}"
    ) == (1, None)
