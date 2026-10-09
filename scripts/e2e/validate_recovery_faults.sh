#!/usr/bin/env bash
set -eu
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common.sh"

e2e_require_python || exit $?
"${PYTHON_BIN}" -m pytest --version >/dev/null 2>&1 || {
    e2e_blocked "pytest is required for deterministic fault-injection acceptance"
    exit $?
}
e2e_new_run_dir recovery_faults
log_file="${E2E_RUN_DIR}/pytest.log"

tests=(
  "tests/unit/test_taskrunner_wiring.py::test_taskrunner_replays_identical_lease_evidence_after_ack_loss"
  "tests/unit/test_scheduler_wiring.py::test_group_scheduler_preserves_staged_add_state_when_taskrunner_rejects_unknown"
  "tests/unit/test_trainer_wiring.py::test_trainer_blocked_add_replay_reconciles_committed_route_and_clears_g"
  "tests/unit/test_trainer_wiring.py::test_trainer_blocked_exit_replay_reconciles_service_commit_and_clears_g"
  "tests/unit/test_trainer_wiring.py::test_trainer_restore_unverified_bootstrap_quarantines_m_projection_and_blocks_g"
  "tests/unit/test_rollouter_wiring.py::test_rollouter_restore_ack_loss_reconciles_committed_route_without_sleep"
  "tests/unit/test_rollouter_wiring.py::test_rollouter_natural_drain_timeout_stays_draining_and_same_op_resumes"
  "tests/unit/test_rollouter_wiring.py::test_rollouter_force_timeout_same_op_resumes_without_repeating_abort"
  "tests/unit/test_rollouter_wiring.py::test_rollouter_force_abort_ack_loss_recovers_from_full_continuation_proof_without_reabort"
  "tests/unit/test_orchestration.py"
  "tests/unit/test_message_queue_exactly_once.py"
)

cd "${REPO_ROOT}"
set +e
"${PYTHON_BIN}" -m pytest -q "${tests[@]}" 2>&1 | tee "${log_file}"
codes=( "${PIPESTATUS[@]}" )
set -e
if [ "${codes[0]}" -ne 0 ] || [ "${codes[1]}" -ne 0 ]; then
    echo "[MULTITASK-E2E] deterministic recovery acceptance FAILED" >&2
    exit 1
fi
echo "MULTITASK_RECOVERY_ACCEPTANCE {\"state\":\"PASSED\",\"log\":\"${log_file}\"}"
