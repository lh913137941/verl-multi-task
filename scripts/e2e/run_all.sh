#!/usr/bin/env bash
set -eu
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common.sh"

e2e_new_run_dir comprehensive
results="${E2E_RUN_DIR}/results.tsv"
summary="${E2E_RUN_DIR}/summary.json"
: > "${results}"
scenarios="${MT_E2E_SCENARIOS:-control_plane exactly_once recovery lifecycle force}"

for scenario in ${scenarios}; do
    case "${scenario}" in
        control_plane) script="${SCRIPT_DIR}/validate_control_plane.sh" ;;
        exactly_once) script="${SCRIPT_DIR}/validate_exactly_once.sh" ;;
        recovery) script="${SCRIPT_DIR}/validate_recovery_faults.sh" ;;
        lifecycle) script="${SCRIPT_DIR}/validate_lifecycle_cycle.sh" ;;
        force) script="${SCRIPT_DIR}/validate_force_remove.sh" ;;
        *)
            printf '%s\tBLOCKED\tunknown scenario\n' "${scenario}" >> "${results}"
            continue
            ;;
    esac
    echo "[MULTITASK-E2E] running ${scenario}"
    set +e
    bash "${script}" "$@"
    rc=$?
    set -e
    case "${rc}" in
        0) state=PASS ;;
        2) state=BLOCKED ;;
        *) state=FAIL ;;
    esac
    printf '%s\t%s\texit=%s\n' "${scenario}" "${state}" "${rc}" >> "${results}"
done

export MT_E2E_RESULTS="${results}" MT_E2E_SUMMARY="${summary}"
"${PYTHON_BIN}" - <<'PY'
import json
import os

rows = []
for line in open(os.environ["MT_E2E_RESULTS"], encoding="utf-8"):
    scenario, state, detail = line.rstrip("\n").split("\t", 2)
    rows.append({"scenario": scenario, "state": state, "detail": detail})
summary = {
    "results": rows,
    "passed": sum(x["state"] == "PASS" for x in rows),
    "failed": sum(x["state"] == "FAIL" for x in rows),
    "blocked": sum(x["state"] == "BLOCKED" for x in rows),
}
with open(os.environ["MT_E2E_SUMMARY"], "w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[MULTITASK-E2E] summary: ${summary}"
grep -q $'\tFAIL\t' "${results}" && exit 1
if [ "${MT_E2E_REQUIRE_COMPLETE:-1}" = "1" ] && grep -q $'\tBLOCKED\t' "${results}"; then
    exit 2
fi
exit 0
