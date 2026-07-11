#!/usr/bin/env bash
# T1 bake-off launcher (genmlx-8lm2): for each requested arm, run the GENERATION
# phase (scripts/t1_bakeoff.cljs, model resident once) under the live
# memory-guarded runner — STRICTLY SEQUENTIAL, ONE GPU process at a time (Thor
# discipline, genmlx-h3p5) — then run the model-free SCORING phase
# (scripts/t1_score.cljs) unguarded (CPU + tiny oracle graphs, no checkpoint).
# With exactly two arms, finishes with the side-by-side COMPARE table.
#
# Usage:  scripts/run_t1_bakeoff.sh [arm ...]          (default: a b)
#   Each arm needs its checkpoint via MODEL_DIR_<ARM> (uppercased), e.g.
#     MODEL_DIR_A=~/.cache/.../Ornith-1.0-35B-4bit/snapshots/<hash> \
#     MODEL_DIR_B=~/.cache/.../Ornith-1.0-9B-bf16/snapshots/<hash> \
#       scripts/run_t1_bakeoff.sh a b
#   (a bare MODEL_DIR is the fallback for any arm without its own var).
#
# Generation knobs pass through: K SEED TEMPERATURE MAX_TOKENS TASKS GREEDY_FIRST
# Scoring knobs pass through:    NP BOOT TIMEOUT_MS
# Guard knob:                    FLOOR_MB (guarded runner MemAvailable floor)
#
# Generation output streams to ~/genmlx-battery-logs/guarded-t1-<arm>.txt (the
# guarded runner owns stdout); JSONL/meta/score land in results/t1-bakeoff/.
set -uo pipefail
cd "$(dirname "$0")/.."

# Thor/CUDA env (same incantation the guarded runner exports — replicated here
# for the UNGUARDED scoring phase, which still loads the native addon; see
# scripts/distill_teacher.cljs and scripts/grpo_student.cljs headers).
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8192
export PATH="$HOME/.local/node/bin:$HOME/.bun/bin:$PATH"

GUARD="$HOME/genmlx-guarded-run.sh"
[ -x "$GUARD" ] || { echo "[t1] missing guarded runner: $GUARD"; exit 1; }

ARMS="${*:-a b}"

# --- generation: guarded, strictly sequential (ONE GPU process at a time) ---
for arm in $ARMS; do
  var="MODEL_DIR_$(echo "$arm" | tr '[:lower:]' '[:upper:]')"
  MD="${!var:-${MODEL_DIR:-}}"
  if [ -z "$MD" ]; then
    echo "[t1] arm $arm: set $var (or MODEL_DIR) to its checkpoint dir"; exit 1
  fi
  echo "[t1] === arm $arm: guarded generation ($MD) ==="
  echo "[t1]     log: ~/genmlx-battery-logs/guarded-t1-$arm.txt"
  ARM="$arm" MODEL_DIR="$MD" \
  K="${K:-4}" SEED="${SEED:-42}" TEMPERATURE="${TEMPERATURE:-0.8}" \
  MAX_TOKENS="${MAX_TOKENS:-512}" TASKS="${TASKS:-}" GREEDY_FIRST="${GREEDY_FIRST:-1}" \
    "$GUARD" "t1-$arm" bunx --bun nbb@1.4.208 scripts/t1_bakeoff.cljs \
    || { echo "[t1] arm $arm generation FAILED (exit $?)"; exit 1; }
done

# --- scoring: model-free, unguarded ---
for arm in $ARMS; do
  echo "[t1] === arm $arm: scoring (model-free) ==="
  ARM="$arm" NP="${NP:-50}" SEED="${SEED:-42}" BOOT="${BOOT:-2000}" \
  TIMEOUT_MS="${TIMEOUT_MS:-60000}" \
    bunx --bun nbb@1.4.208 scripts/t1_score.cljs \
    || { echo "[t1] arm $arm scoring FAILED"; exit 1; }
done

# --- side-by-side when exactly two arms ---
set -- $ARMS
if [ $# -eq 2 ]; then
  echo "[t1] === compare $1 vs $2 ==="
  COMPARE="$1,$2" bunx --bun nbb@1.4.208 scripts/t1_score.cljs
fi
