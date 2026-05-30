#!/usr/bin/env bash
# ============================================================================
# GenMLX — distinctive-features demo suite runner
# ============================================================================
# Runs each demo with nbb (via Bun). Pure-GenMLX demos need no weights; the
# LLM demos (03/04/05) load a local model from ~/.cache/models.
#
#   ./run_all.sh          # run the whole suite
#   ./run_all.sh core     # only the pure-GenMLX demos (no model weights)
#   ./run_all.sh 06       # run a single demo by number
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."        # repo root (examples/distinctive -> ../../)

CORE=(01_model_is_a_value
      02_compilation_ladder_equivalence
      06_auto_analytical
      07_shape_vectorization
      08_value_semantics_gpu)
LLM=(03_llm_is_a_distribution
     04_grammar_conditioning
     05_program_synthesis)

run_one () {
  local d="$1"
  echo ""
  echo "############################################################"
  echo "#  $d"
  echo "############################################################"
  bun run --bun nbb "examples/distinctive/${d}.cljs" 2>&1 \
    | grep -vE "SIGTRAP|Trace/breakpoint"   # harmless Bun/JSC teardown noise
}

case "${1:-all}" in
  core) for d in "${CORE[@]}"; do run_one "$d"; done ;;
  llm)  for d in "${LLM[@]}";  do run_one "$d"; done ;;
  all)  for d in "${CORE[@]}" "${LLM[@]}"; do run_one "$d"; done ;;
  *)    # single demo by number prefix, e.g. "06"
        match=$(printf '%s\n' "${CORE[@]}" "${LLM[@]}" | grep "^${1}_" | head -1)
        if [ -n "$match" ]; then run_one "$match"; else echo "no demo matching '$1'"; exit 1; fi ;;
esac
