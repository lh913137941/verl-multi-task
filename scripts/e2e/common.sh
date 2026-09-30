#!/usr/bin/env bash
set -eu

E2E_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${E2E_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
E2E_LOG_ROOT="${MT_E2E_LOG_ROOT:-${REPO_ROOT}/logs/multitask_e2e}"

e2e_blocked() {
    echo "[MULTITASK-E2E] BLOCKED: $*" >&2
    return 2
}

e2e_new_run_dir() {
    scenario="$1"
    run_id="$(date +%Y%m%d%H%M%S)-$$"
    E2E_RUN_DIR="${E2E_LOG_ROOT}/${scenario}/${run_id}"
    mkdir -p "${E2E_RUN_DIR}"
    export E2E_RUN_DIR
}

e2e_require_python() {
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || e2e_blocked "Python not found: ${PYTHON_BIN}"
}

e2e_run_live_driver() {
    scenario="$1"
    shift
    e2e_require_python || return $?
    lease_file="${MT_E2E_LEASE_FILE:-}"
    [ -n "${lease_file}" ] && [ -f "${lease_file}" ] || {
        e2e_blocked "set MT_E2E_LEASE_FILE to a real donor claim fixture"
        return $?
    }

    e2e_new_run_dir "${scenario}"
    E2E_LAST_RESULT_FILE="${E2E_RUN_DIR}/result.json"
    E2E_LAST_RUNTIME_LOG=""
    export E2E_LAST_RESULT_FILE E2E_LAST_RUNTIME_LOG

    if [ "${MT_E2E_ATTACH_ONLY:-0}" = "1" ]; then
        set +e
        "${PYTHON_BIN}" "${E2E_DIR}/lifecycle_driver.py"             --scenario "${scenario}"             --lease-file "${lease_file}"             --result-file "${E2E_LAST_RESULT_FILE}"
        driver_status=$?
        set -e
        return "${driver_status}"
    fi

    launcher="${MULTITASK_LAUNCH_SCRIPT:-}"
    [ -n "${launcher}" ] && [ -f "${launcher}" ] || {
        e2e_blocked "set MULTITASK_LAUNCH_SCRIPT or use MT_E2E_ATTACH_ONLY=1"
        return $?
    }
    runtime_timeout="${MT_E2E_RUNTIME_TIMEOUT_S:-900}"
    E2E_LAST_RUNTIME_LOG="${E2E_RUN_DIR}/runtime.log"
    export E2E_LAST_RUNTIME_LOG
    launcher_status_file="${E2E_RUN_DIR}/launcher.status"

    (
        set +e
        if command -v timeout >/dev/null 2>&1; then
            timeout --signal=TERM "${runtime_timeout}s" bash "${launcher}" "$@"                 2>&1 | tee "${E2E_LAST_RUNTIME_LOG}"
        else
            bash "${launcher}" "$@" 2>&1 | tee "${E2E_LAST_RUNTIME_LOG}"
        fi
        codes=( "${PIPESTATUS[@]}" )
        printf '%s\n' "${codes[0]}" > "${launcher_status_file}"
        exit 0
    ) &
    launcher_job=$!

    set +e
    "${PYTHON_BIN}" "${E2E_DIR}/lifecycle_driver.py"         --scenario "${scenario}"         --lease-file "${lease_file}"         --result-file "${E2E_LAST_RESULT_FILE}"
    driver_status=$?
    set -e

    if [ "${MT_E2E_TERMINATE_AFTER_DRIVER:-0}" = "1" ] && kill -0 "${launcher_job}" 2>/dev/null; then
        kill "${launcher_job}" 2>/dev/null || true
    fi
    wait "${launcher_job}" || true
    launcher_status="$(cat "${launcher_status_file}" 2>/dev/null || echo 1)"

    echo "[MULTITASK-E2E] runtime log: ${E2E_LAST_RUNTIME_LOG}"
    echo "[MULTITASK-E2E] result: ${E2E_LAST_RESULT_FILE}"

    [ "${driver_status}" -eq 0 ] || return "${driver_status}"
    if [ "${MT_E2E_TERMINATE_AFTER_DRIVER:-0}" != "1" ] && [ "${launcher_status}" -ne 0 ]; then
        echo "[MULTITASK-E2E] launcher failed: exit=${launcher_status}" >&2
        return 1
    fi
    return 0
}
