#!/usr/bin/env bash
# R2 bake-off launcher (genmlx-xrps): bring up the resident mlx-lm worker, wait for READY,
# run scripts/r2_bakeoff.cljs against it, then ALWAYS tear the worker down (trap on EXIT —
# no orphaned 19GB worker). The policy LLM lives OUT-OF-PROCESS; the native-free synthesis
# loop + exact-evidence oracle curl it. Mirrors scripts/run_llm_probe.sh.
#
# Usage:  scripts/run_r2_bakeoff.sh <model> <tier> [PORT=8765]
#   e.g.  scripts/run_r2_bakeoff.sh Qwen3.6-35B-A3B-4bit 35b
#         scripts/run_r2_bakeoff.sh qwen3.5-0.8b-cljs-sft600 sft-0.8b
# Bake-off knobs are env vars passed through: ROUND INSTANCES FAMILIES EVAL_FAMILIES
#   TASKS_LIMIT K_STEP MAX_STEPS REVISE TEMP MAX_TOKENS NP SEED BOOT SYSTEM OUT
set -uo pipefail
MODEL="${1:?usage: run_r2_bakeoff.sh <model> <tier> [PORT]}"
TIER="${2:?usage: run_r2_bakeoff.sh <model> <tier> [PORT]}"
PORT="${3:-${PORT:-8765}}"
ADAPTER="${ADAPTER:-}"
URL="http://127.0.0.1:${PORT}"
LOG="$(mktemp -t llm_server.XXXXXX)"

# No bash arrays — macOS bash 3.2 errors on an empty-array expansion under `set -u`.
echo "[r2] launching worker: $MODEL ${ADAPTER:+(+LoRA $ADAPTER)} on :$PORT (log $LOG)"
if [ -n "$ADAPTER" ]; then
  python3 scripts/llm_server.py --model "$MODEL" --port "$PORT" --adapter "$ADAPTER" >"$LOG" 2>&1 &
else
  python3 scripts/llm_server.py --model "$MODEL" --port "$PORT" >"$LOG" 2>&1 &
fi
SRV=$!
trap 'echo "[r2] tearing down worker $SRV"; kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null' EXIT

echo "[r2] waiting for READY (35B load can take minutes)..."
for i in $(seq 1 1200); do
  if curl -s "${URL}/health" >/dev/null 2>&1; then echo "[r2] worker READY after ${i}s"; break; fi
  if ! kill -0 "$SRV" 2>/dev/null; then echo "[r2] worker DIED. log:"; cat "$LOG"; exit 1; fi
  sleep 1
done

SERVER_URL="$URL" TIER="$TIER" bun run --bun nbb scripts/r2_bakeoff.cljs
RC=$?
echo "[r2] bake-off exit $RC. worker log tail:"; tail -6 "$LOG"
exit $RC
