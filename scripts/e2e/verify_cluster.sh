#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/e2e/verify_cluster.sh --launcher /path/to/launcher.sh --lease /path/to/lease.json [--quick|--full] [-- <launcher overrides...>]
  bash scripts/e2e/verify_cluster.sh --attach --lease /path/to/lease.json [--quick|--full]

Modes:
  --quick   control-plane + exactly-once + deterministic recovery only
  --full    quick checks + real lifecycle + FORCE lifecycle (default)

Exit codes:
  0 PASS
  1 FAIL
  2 BLOCKED (environment/topology insufficient)
EOF
}

mode="full"
attach=0
launcher=""
lease=""
require_complete=1
launcher_args=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --launcher)
            [ "$#" -ge 2 ] || { usage >&2; exit 2; }
            launcher="$2"
            shift 2
            ;;
        --lease)
            [ "$#" -ge 2 ] || { usage >&2; exit 2; }
            lease="$2"
            shift 2
            ;;
        --attach)
            attach=1
            shift
            ;;
        --quick)
            mode="quick"
            shift
            ;;
        --full)
            mode="full"
            shift
            ;;
        --allow-blocked)
            require_complete=0
            shift
            ;;
        --)
            shift
            launcher_args=( "$@" )
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[ -n "${lease}" ] || {
    echo "--lease is required" >&2
    usage >&2
    exit 2
}
[ -f "${lease}" ] || {
    echo "Lease file does not exist: ${lease}" >&2
    exit 2
}

export MT_E2E_LEASE_FILE="${lease}"
export MT_E2E_REQUIRE_COMPLETE="${require_complete}"

if [ "${mode}" = "quick" ]; then
    export MT_E2E_SCENARIOS="control_plane exactly_once recovery"
else
    export MT_E2E_SCENARIOS="control_plane exactly_once recovery lifecycle force"
fi

if [ "${attach}" -eq 1 ]; then
    export MT_E2E_ATTACH_ONLY=1
    export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
else
    [ -n "${launcher}" ] || {
        echo "--launcher is required unless --attach is used" >&2
        usage >&2
        exit 2
    }
    [ -f "${launcher}" ] || {
        echo "Launcher does not exist: ${launcher}" >&2
        exit 2
    }
    export MULTITASK_LAUNCH_SCRIPT="${launcher}"
    unset MT_E2E_ATTACH_ONLY || true
fi

echo "[MULTITASK-E2E] mode=${mode}"
echo "[MULTITASK-E2E] lease=${MT_E2E_LEASE_FILE}"
if [ "${attach}" -eq 1 ]; then
    echo "[MULTITASK-E2E] attach existing Ray job: ${RAY_ADDRESS}"
else
    echo "[MULTITASK-E2E] launcher=${MULTITASK_LAUNCH_SCRIPT}"
fi

exec bash "${SCRIPT_DIR}/run_all.sh" "${launcher_args[@]}"
