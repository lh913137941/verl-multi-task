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
    "${PYTHON_BIN}" - "${E2E_LAST_RUNTIME_LOG}" "${E2E_LAST_RESULT_FILE}" <<'PY'
import json
import re
import sys

with open(sys.argv[2], encoding="utf-8") as stream:
    result = json.load(stream)
if result.get("state") != "PASSED" or result.get("scenario") != "force_cycle":
    raise SystemExit(1)
commands = [
    step["command"] for step in result.get("events", [])
    if step.get("command", {}).get("kind") == "REMOVE"
    and step["command"].get("force") is True
]
if len(commands) != 1 or not commands[0].get("operation_id"):
    raise SystemExit(1)
operation_id = commands[0]["operation_id"]

records = []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    match = re.search(r"MULTITASK_FORCE_HANDOFF\s+(\{.*\})", line)
    if match:
        receipt = json.loads(match.group(1))
        if receipt.get("operation_id") == operation_id:
            records.append(receipt)
if not records:
    raise SystemExit(2)
# Exact-op recovery can emit the same completed receipt more than once.
# Every candidate must be internally valid; never combine counts across rows.
proven = []
for item in records:
    admitted = item.get("admitted_count")
    confirmed = item.get("confirmed_count")
    aborted = item.get("aborted_count")
    ack_known = item.get("abort_ack_known")
    if (type(admitted) is not int or type(confirmed) is not int
            or not 0 <= confirmed <= admitted or type(ack_known) is not bool):
        raise SystemExit(1)
    if ack_known:
        if type(aborted) is not int or not 0 <= aborted <= confirmed:
            raise SystemExit(1)
        if aborted > 0:
            proven.append(item)
    else:
        # Match Rollouter's lost-abort-ACK recovery: every boundary request
        # must have continuation proof before the operation can finish.
        if aborted is not None or confirmed != admitted:
            raise SystemExit(1)
        if admitted > 0:
            proven.append(item)
if not proven:
    raise SystemExit(2)
print(json.dumps(proven[-1], sort_keys=True))
PY
    proof_status=$?
    set -e
    if [ "${proof_status}" -eq 2 ]; then
        e2e_blocked "FORCE completed, but this run did not prove an in-flight continuation handoff"
        exit $?
    fi
    [ "${proof_status}" -eq 0 ] || exit 1
fi
