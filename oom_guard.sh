#!/bin/bash
# Preemptive OOM guard for a training process: poll free system RAM and ALL
# visible GPUs, warn under a soft threshold, and SIGTERM the training process
# under a hard emergency threshold -- so it dies BEFORE the OS/driver hits an
# unrecoverable OOM state, not after.
#
# Supports AMD GPUs via `rocm-smi`. If rocm-smi is not available, the guard
# falls back to system-RAM-only monitoring.
#
# Usage: nohup bash oom_guard.sh <training_pid> [warn_free_mb] [emergency_free_mb] \
#                                   [poll_sec] [vram_warn_mb] [vram_emergency_mb] \
#                                   > oom_guard.log 2>&1 &
# Stop:  kill the guard's own PID (printed at start), or pkill -f oom_guard.sh

set -uo pipefail
TRAIN_PID="${1:?usage: oom_guard.sh <training_pid> [warn_free_mb] [emergency_free_mb] [poll_sec] [vram_warn_mb] [vram_emergency_mb]}"
WARN_FREE_MB="${2:-4000}"
EMERGENCY_FREE_MB="${3:-1500}"
POLL_SEC="${4:-30}"
VRAM_WARN_MB="${5:-2048}"
VRAM_EMERGENCY_MB="${6:-512}"

echo "[oom_guard] watching PID $TRAIN_PID, poll ${POLL_SEC}s"
echo "[oom_guard] system-RAM: warn<${WARN_FREE_MB}MB, emergency<${EMERGENCY_FREE_MB}MB (via /proc/meminfo)"

read_available_mb() {
    # meminfo path is overridable via $OOM_GUARD_MEMINFO purely so the
    # system-RAM parsing can be tested against real captured meminfo files
    # (theres no portable way to redirect /proc/meminfo itself on Linux, and
    # no /proc at all on macOS). defaults to the real thing -- production
    # behavior is unchanged.
    local meminfo="${OOM_GUARD_MEMINFO:-/proc/meminfo}"
    local kb
    kb=$(awk '/^MemAvailable:/ {print $2; found=1} END {if (!found) print ""}' "$meminfo" 2>/dev/null)
    if [ -z "$kb" ]; then
        kb=$(awk '/^MemFree:/ {print $2}' "$meminfo" 2>/dev/null)
    fi
    if [ -z "$kb" ]; then
        echo ""
        return
    fi
    echo $((kb / 1024))
}

# ---------------------------------------------------------------------------
# GPU backend detection and per-GPU free-VRAM polling.
# Try KFD sysfs first (fast: ~1ms, no subprocess), then fall back to rocm-smi
# (~50-100ms per poll, competes for the SMI lock).
#
# $OOM_GUARD_ROCM_SMI overrides which rocm-smi binary the guard polls,
# same spirit as $OOM_GUARD_MEMINFO above. on a box with a real GPU you
# just leave it unset and the rocm-smi on PATH is used; the override
# exists so the rocm-smi code path can be driven against real captured
# output on machines with no AMD hardware. the parsing itself lives in
# vram_parse.py now so the exact code the guard runs is directly testable
# and the bash side stays a thin call.
#
# timeout guards the rocm-smi calls becuase a hung SMI read would
# otherwise wedge the guard's whole poll loop. systems without the
# coreutil (macOS) just run the call bare rather than silently disabling
# VRAM monitoring entirely.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_SMI_CMD="${OOM_GUARD_ROCM_SMI:-rocm-smi}"
TIMEOUT_BIN="$(command -v timeout 2>/dev/null || true)"

_smi_call() {
    # $1 = timeout seconds, rest = command and its args
    local secs="$1"; shift
    if [ -n "$TIMEOUT_BIN" ]; then
        "$TIMEOUT_BIN" "$secs" "$@"
    else
        "$@"
    fi
}

GPU_BACKEND="none"
if [ -d /sys/class/kfd/topology/nodes ]; then
    GPU_BACKEND="kfd"
    echo "[oom_guard] GPU backend: KFD sysfs (/sys/class/kfd)"
elif [ -n "${OOM_GUARD_ROCM_SMI:-}" ] || command -v rocm-smi >/dev/null 2>&1; then
    GPU_BACKEND="rocm"
    echo "[oom_guard] GPU backend: ROCm (rocm-smi)"
else
    echo "[oom_guard] no KFD sysfs or rocm-smi found -- GPU-VRAM checks skipped (system-RAM only)."
fi

read_all_vram_free_mb() {
    # Prints one line per GPU: "<index> <free_mb>". Empty output means unavailable.
    if [ "$GPU_BACKEND" = "kfd" ]; then
        _read_kfd_vram
    elif [ "$GPU_BACKEND" = "rocm" ]; then
        _read_rocm_vram
    fi
}

_read_kfd_vram() {
    # Read per-GPU free VRAM from KFD sysfs topology. Each GPU node has a
    # mem_banks subdir with properties files containing "Size in bytes" and
    # "Used bytes". This is ~100x faster than spawning rocm-smi (no subprocess)
    # and doesn't compete for the SMI lock.
    local idx=0
    for node in /sys/class/kfd/topology/nodes/*/; do
        [ -d "$node" ] || continue
        # Skip CPU-only nodes (no mem_banks).
        local mem_dir="${node}mem_banks/"
        [ -d "$mem_dir" ] || { idx=$((idx + 1)); continue; }
        for bank in "$mem_dir"*/; do
            [ -d "$bank" ] || continue
            local props="${bank}properties"
            [ -f "$props" ] || continue
            local size_bytes=0 used_bytes=0
            while IFS=' ' read -r key val; do
                case "$key" in
                    "Size") size_bytes="$val" ;;
                    "Used") used_bytes="$val" ;;
                esac
            done < "$props"
            if [ "$size_bytes" -gt 0 ] 2>/dev/null; then
                local free_bytes=$((size_bytes - used_bytes))
                local free_mb=$((free_bytes / 1024 / 1024))
                echo "$idx $free_mb"
                break  # one bank per GPU (VRAM)
            fi
        done
        idx=$((idx + 1))
    done
}

_read_rocm_vram() {
    # Try JSON first, then fall back to text. Aggregate free memory per GPU.
    #
    # output goes to vram_parse.py as an argv argument, not over stdin. a
    # `python3 - <<'PY'` heredoc on the same command as a pipe redirects
    # stdin too and the heredoc wins, so the parser would alwasy see an
    # empty string -- an earlier inline-heredoc version of this did exactly
    # that, and its unconditional return meant VRAM monitoring was silently
    # a permanent no-op on every real ROCm box. the JSON branch now also
    # falls through to the text fallback when it produces no rows.
    #
    # vram_parse.py computes free as Total - Used: rocm-smi has never
    # emitted a "VRAM Free Memory (B)" field in any release, an earlier
    # parser here looked for one which kept this whole function a silent
    # no-op against real rocm-smi output.
    local json_out
    json_out=$(_smi_call 10 "$ROCM_SMI_CMD" --showmeminfo vram --json 2>/dev/null)
    if [ -n "$json_out" ]; then
        local json_result
        json_result=$(python3 "$SCRIPT_DIR/vram_parse.py" rocm-json "$json_out" 2>/dev/null)
        if [ -n "$json_result" ]; then
            echo "$json_result"
            return
        fi
        # Fall through to the text fallback if JSON parsed to zero rows.
    fi

    local raw
    raw=$(_smi_call 10 "$ROCM_SMI_CMD" --showmeminfo vram 2>/dev/null)
    if [ -z "$raw" ]; then
        return
    fi
    python3 "$SCRIPT_DIR/vram_parse.py" rocm-text "$raw" 2>/dev/null
}

lowest_vram_free_mb() {
    local min_free=""
    while read -r idx free_mb; do
        if [ -z "$min_free" ] || [ "$free_mb" -lt "$min_free" ]; then
            min_free="$free_mb"
        fi
    done < <(read_all_vram_free_mb)
    echo "$min_free"
}

# ---------------------------------------------------------------------------
# Main poll loop.
# ---------------------------------------------------------------------------
while true; do
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "[oom_guard] $(date '+%H:%M:%S') training PID $TRAIN_PID no longer exists -- exiting guard."
        exit 0
    fi

    free_mb=$(read_available_mb)
    if [ -z "$free_mb" ]; then
        echo "[oom_guard] $(date '+%H:%M:%S') could not read /proc/meminfo -- skipping this poll"
        sleep "$POLL_SEC"
        continue
    fi

    if [ "$free_mb" -lt "$EMERGENCY_FREE_MB" ]; then
        echo "[oom_guard] $(date '+%H:%M:%S') EMERGENCY: only ${free_mb}MB system RAM available -- sending SIGTERM to $TRAIN_PID."
        kill -TERM "$TRAIN_PID" 2>/dev/null
    elif [ "$free_mb" -lt "$WARN_FREE_MB" ]; then
        echo "[oom_guard] $(date '+%H:%M:%S') WARNING: ${free_mb}MB system RAM available -- getting tight."
    fi

    if [ "$GPU_BACKEND" != "none" ]; then
        vram_free_mb=$(lowest_vram_free_mb)
        if [ -n "$vram_free_mb" ]; then
            if [ "$vram_free_mb" -lt "$VRAM_EMERGENCY_MB" ]; then
                echo "[oom_guard] $(date '+%H:%M:%S') EMERGENCY: only ${vram_free_mb}MB VRAM free across GPUs -- sending SIGTERM to $TRAIN_PID."
                kill -TERM "$TRAIN_PID" 2>/dev/null
            elif [ "$vram_free_mb" -lt "$VRAM_WARN_MB" ]; then
                echo "[oom_guard] $(date '+%H:%M:%S') WARNING: ${vram_free_mb}MB VRAM free across GPUs -- getting tight."
            fi
        fi
    fi

    sleep "$POLL_SEC"
done
