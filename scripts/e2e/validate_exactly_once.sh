#!/usr/bin/env bash
set -eu
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common.sh"

e2e_require_python || exit $?
e2e_new_run_dir exactly_once
set +e
"${PYTHON_BIN}" "${SCRIPT_DIR}/exactly_once_driver.py"     --result-file "${E2E_RUN_DIR}/result.json"
status=$?
set -e
echo "[MULTITASK-E2E] result: ${E2E_RUN_DIR}/result.json"
exit "${status}"
