#!/usr/bin/env bash
set -eu
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common.sh"

e2e_require_python || exit $?
e2e_new_run_dir control_plane
set +e
"${PYTHON_BIN}" "${SCRIPT_DIR}/control_plane_recovery.py"     --result-file "${E2E_RUN_DIR}/result.json"     --stale-wait-s "${MT_E2E_STALE_WAIT_S:-10.2}"
status=$?
set -e
echo "[MULTITASK-E2E] result: ${E2E_RUN_DIR}/result.json"
exit "${status}"
