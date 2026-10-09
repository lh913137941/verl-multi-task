#!/usr/bin/env bash
set -eu
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common.sh"

echo "[MULTITASK-E2E] FORCE lifecycle: DONATE -> ADD -> FORCE REMOVE -> RESTORE"
# Strict in-flight proof is checked by verify_two_verl_jobs.py against
# the exact operation id and real forwarded Rollouter receipts.
e2e_run_live_driver force_cycle
