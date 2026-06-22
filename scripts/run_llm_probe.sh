#!/usr/bin/env bash
# Launch the resident mlx-lm worker (scripts/llm_server.py), wait for READY, run the
# real-LLM experiment-B probe (scripts/synth_llm_probe.cljs), then tear the worker down.
# The policy LLM lives OUT-OF-PROCESS; the native-free synthesis loop curls it.
#
# Usage:  scripts/run_llm_probe.sh <model> <tier> [PORT=8765]
#   e.g.  scripts/run_llm_probe.sh qwen3.5-0.8b-cljs-sft600 fast
#         scripts/run_llm_probe.sh Qwen3.6-35B-A3B-4bit    slow
# Probe knobs are env vars passed through: K_ONESHOT K_STEP BEAM_WIDTH MAX_STEPS SEEDS NP TEMP TASKS
set -uo pipefail
MODEL="${1:?usage: run_llm_probe.sh <model> <tier> [PORT]}"
TIER="${2:?usage: run_llm_probe.sh <model> <tier> [PORT]}"
PORT="${3:-${PORT:-8765}}"
URL="http://127.0.0.1:${PORT}"
LOG="$(mktemp -t llm_server.XXXXXX).log"

echo "[run] launching worker: $MODEL on :$PORT (log $LOG)"
python3 scripts/llm_server.py --model "$MODEL" --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
trap 'echo "[run] tearing down worker $SRV"; kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null' EXIT

echo "[run] waiting for READY (model load can take minutes for the 35B)..."
for i in $(seq 1 900); do
  if curl -s "${URL}/health" >/dev/null 2>&1; then echo "[run] worker READY after ${i}s"; break; fi
  if ! kill -0 "$SRV" 2>/dev/null; then echo "[run] worker DIED. log:"; cat "$LOG"; exit 1; fi
  sleep 1
done

SERVER_URL="$URL" TIER="$TIER" bun run --bun nbb scripts/synth_llm_probe.cljs
RC=$?
echo "[run] probe exit $RC. worker log tail:"; tail -5 "$LOG"
exit $RC
