#!/usr/bin/env bash
set -eu
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
. "${SCRIPT_DIR}/common.sh"

echo "[MULTITASK-E2E] FORCE lifecycle: DONATE -> ADD -> FORCE REMOVE -> RESTORE"
set +e
e2e_run_live_driver force_cycle "$@"
status=$?
set -e
[ "${status}" -eq 0 ] || exit "${status}"

if [ "${MT_E2E_REQUIRE_INFLIGHT_FORCE:-0}" = "1" ]; then
    [ -n "${E2E_LAST_RUNTIME_LOG:-}" ] && [ -f "${E2E_LAST_RUNTIME_LOG}" ] || {
        e2e_blocked "inflight FORCE proof requires launcher mode so actor logs are captured"
        exit $?
    }
    set +e
    "${PYTHON_BIN}" - "${E2E_LAST_RUNTIME_LOG}" <<'PY'
import json
import re
import sys

records = []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    match = re.search(r"MULTITASK_FORCE_HANDOFF\s+(\{.*\})", line)
    if match:
        records.append(json.loads(match.group(1)))
if not records:
    raise SystemExit(2)
if max(int(item.get("admitted_count", 0)) for item in records) <= 0:
    raise SystemExit(2)
if not any(
    item.get("abort_ack_known")
    and int(item.get("confirmed_count", 0)) >= int(item.get("aborted_count", 0))
    for item in records
):
    raise SystemExit(1)
print(json.dumps(records[-1], sort_keys=True))
PY
    proof_status=$?
    set -e
    if [ "${proof_status}" -eq 2 ]; then
        e2e_blocked "FORCE completed, but this run did not prove an in-flight continuation handoff"
        exit $?
    fi
    [ "${proof_status}" -eq 0 ] || exit 1
fi
