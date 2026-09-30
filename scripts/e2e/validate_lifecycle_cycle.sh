#!/usr/bin/env bash
set -eu
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
. "${SCRIPT_DIR}/common.sh"

echo "[MULTITASK-E2E] full lifecycle: DONATE -> ADD -> REMOVE -> RESTORE"
e2e_run_live_driver full_cycle "$@"
