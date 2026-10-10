#!/usr/bin/env bash
set -eu

E2E_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${E2E_DIR}/../.." && pwd)"
# Python drivers and Ray workers must import the current checkout.
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
E2E_LOG_ROOT="${MT_E2E_LOG_ROOT:-${REPO_ROOT}/logs/multitask_e2e}"

e2e_blocked() {
    echo "[MULTITASK-E2E] BLOCKED: $*" >&2
    return 2
}

e2e_new_run_dir() {
    local scenario="$1"
    local run_id
    run_id="$(date +%Y%m%d%H%M%S)-$$"
    E2E_RUN_DIR="${E2E_LOG_ROOT}/${scenario}/${run_id}"
    mkdir -p "${E2E_RUN_DIR}"
    export E2E_RUN_DIR
}

e2e_require_python() {
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || e2e_blocked "Python not found: ${PYTHON_BIN}"
}

e2e_run_live_driver() {
    local scenario="$1"
    e2e_require_python || return $?
    if [ "${MT_E2E_ATTACH_ONLY:-0}" != "1" ]; then
        e2e_blocked "real lifecycle requires running donor/borrower jobs; use verify_two_verl_jobs.py"
        return $?
    fi
    local lease_file="${MT_E2E_LEASE_FILE:-}"
    if [ -z "${lease_file}" ] || [ ! -f "${lease_file}" ]; then
        e2e_blocked "MT_E2E_LEASE_FILE must reference a verified donor claim"
        return $?
    fi

    e2e_new_run_dir "${scenario}"
    "${PYTHON_BIN}" "${E2E_DIR}/lifecycle_driver.py" \
      --scenario "${scenario}" \
      --lease-file "${lease_file}" \
      --result-file "${E2E_RUN_DIR}/result.json"
}
