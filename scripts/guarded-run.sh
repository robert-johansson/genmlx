#!/bin/bash
# Live memory-guarded runner for GPU workloads on unified-memory hosts
# (genmlx-h3p5). Polls MemAvailable every 3s DURING the run and SIGKILLs the
# workload's process group if it drops below the floor — the box survives,
# and the MemAvailable curve is logged for post-mortem.
#
# WHY: on Tegra-class unified memory, GPU allocations are INVISIBLE to kernel
# LRU accounting and process RSS ("dark pages"). A runaway workload shows a
# healthy-looking system right up until the global OOM cascade, whose killer
# can only shoot small adj-boosted services — the box spirals to a reboot
# instead of recovering (three reboots on the Thor: 2026-07-07 x2,
# 2026-07-10). MemAvailable is the one signal that moves; watch it live.
#
# Usage: scripts/guarded-run.sh <name> <command...>
# Logs:  $GENMLX_GUARD_LOGDIR (default ~/genmlx-battery-logs — NOT /tmp,
#        which is cleared at boot; reboot-safe logs are the point):
#          guarded-<name>.txt      (workload output)
#          guarded-<name>.mem.csv  (epoch,MemAvailable_MB)
# Env:   FLOOR_MB (default 25000) — the kill floor.
#
# Rules of the box (genmlx-h3p5, all three reboots):
#   1. ONE GPU process at a time (reboot #3 was two concurrent).
#   2. EVERY heavy (35B/80B-class) model run goes through this guard —
#      including agent/provider runs, not just test suites.
#   3. After killing in-flight GPU work, cool down before the next heavy
#      load; prefer letting runs finish.
#   4. Unattended/autonomous sessions never run unguarded GPU work at all.
set -u
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}
export GLIBC_TUNABLES=${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=8192}

NAME="$1"; shift
LOGDIR="${GENMLX_GUARD_LOGDIR:-$HOME/genmlx-battery-logs}"; mkdir -p "$LOGDIR"
OUT="$LOGDIR/guarded-$NAME.txt"; MEMCSV="$LOGDIR/guarded-$NAME.mem.csv"
FLOOR_MB=${FLOOR_MB:-25000}

avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }

a=$(avail_mb)
if [ "$a" -lt "$FLOOR_MB" ]; then
  echo "[guard] refusing to start: MemAvailable=${a}MB < ${FLOOR_MB}MB floor" | tee "$OUT"
  exit 42
fi

setsid "$@" > "$OUT" 2>&1 &
PID=$!
echo "epoch,mem_available_mb" > "$MEMCSV"
while kill -0 "$PID" 2>/dev/null; do
  a=$(avail_mb)
  echo "$(date +%s),$a" >> "$MEMCSV"
  if [ "$a" -lt "$FLOOR_MB" ]; then
    echo "[guard] KILL: MemAvailable=${a}MB < ${FLOOR_MB}MB floor" | tee -a "$OUT"
    kill -9 -- -"$PID" 2>/dev/null
    wait "$PID" 2>/dev/null
    exit 43
  fi
  sleep 3
done
wait "$PID"; EC=$?
echo "[guard] done: exit=$EC MemAvailable=$(avail_mb)MB" >> "$OUT"
exit $EC
