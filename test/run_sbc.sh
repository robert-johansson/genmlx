#!/usr/bin/env bash
# Per-combo isolated-process SBC runner (genmlx-18q9).
#
# Runs ONE bun+nbb process per model x algorithm combo, strictly serially
# (never parallel GPU — Metal wedge risk), writing one crash-safe JSON
# fragment per combo to results/sbc/. A wedge or crash in one combo cannot
# take down the others, and process exit resets all GPU/JS memory between
# combos. Fragments merge into results/sbc_results.json at the end; the
# merge only trusts fragments whose own summary says complete?: true, so a
# partially-written fragment from a killed process is ignored and the combo
# reported as missing.
#
# WHERE RESULTS GO (genmlx-ty5u). results/sbc/ and results/sbc_results.json are
# git-TRACKED frozen evidence. A run only writes there with SBC_FREEZE=1; by
# default fragments and the merge land in a scratch dir outside the checkout.
# The default used to be the tracked paths, so a smoke or timed-out run
# silently overwrote the freeze — on 2026-07-07 a 448-line partial CUDA run
# replaced 25,876 lines of Mac evidence (recovered only by git checkout).
#
# Usage:
#   bash test/run_sbc.sh                         # full run (~6h at N=500) → scratch
#   SBC_N=20 SBC_L=50 bash test/run_sbc.sh       # smoke run (minutes) → scratch
#   SBC_ONLY=single-gaussian:cmh bash test/run_sbc.sh   # one combo
#   SBC_FREEZE=1 bash test/run_sbc.sh            # UPDATE the frozen evidence
#   SBC_RUN_DIR=path bash test/run_sbc.sh        # pick the scratch dir explicitly
#
# Notes:
# - A combo that exceeds SBC_COMBO_TIMEOUT (default 7200s) is abandoned; if
#   the process entered the Metal uninterruptible-sleep state it may linger
#   until reboot, but the runner moves on (see memory: Metal-wedge runs).
# - Re-running skips combos whose fragment ran to a PASSED verdict, so an
#   interrupted sweep resumes where it left off; combos that ran to a
#   FAILED verdict (sim-failure budget) are re-run. Delete the fragment dir
#   for a fresh run.
# - A freeze run refuses to shrink the record: merge_sbc_results.py exits
#   non-zero if the new merge has fewer passed combos than the file it would
#   overwrite (override with --force, deliberately).
set -u
cd "$(dirname "$0")/.."

NBB="bun run --bun nbb"
TIMEOUT_S="${SBC_COMBO_TIMEOUT:-7200}"

case "${SBC_FREEZE:-}" in
  1|true|yes)
    FREEZE=1
    FRAG_DIR="results/sbc"
    OUT_JSON="results/sbc_results.json" ;;
  *)
    FREEZE=0
    FRAG_DIR="${SBC_RUN_DIR:-${TMPDIR:-/tmp}/genmlx-sbc/run}"
    OUT_JSON="$FRAG_DIR/sbc_results.json" ;;
esac
mkdir -p "$FRAG_DIR"

echo "── SBC per-combo runner (serial, isolated processes) ──"
if [ "$FREEZE" = "1" ]; then
  echo "   SBC_FREEZE=1 — writing the git-TRACKED frozen evidence in results/"
else
  echo "   scratch run → $FRAG_DIR  (SBC_FREEZE=1 to update the frozen evidence)"
fi
combos=$(SBC_LIST=1 $NBB test/genmlx/sbc_test.cljs 2>/dev/null | grep '^COMBO ' | sed 's/^COMBO //')
if [ -z "$combos" ]; then
  echo "ERROR: could not enumerate combos (SBC_LIST run failed)"; exit 1
fi

if [ -n "${SBC_ONLY:-}" ]; then
  combos="$SBC_ONLY"
fi

total=$(echo "$combos" | wc -l | tr -d ' ')
echo "   $total combos, timeout ${TIMEOUT_S}s/combo, N=${SBC_N:-500} L=${SBC_L:-200}"

i=0
failed=0
for combo in $combos; do
  i=$((i+1))
  frag="$FRAG_DIR/$(echo "$combo" | tr ':' '_').json"
  if [ -f "$frag" ] && python3 scripts/merge_sbc_results.py --skip-ok "$frag" "$combo" 2>/dev/null; then
    echo "[$i/$total] $combo — fragment passed, skipping"
    continue
  fi
  echo "[$i/$total] $combo"
  SBC_ONLY="$combo" SBC_OUT="$frag" timeout "$TIMEOUT_S" $NBB test/genmlx/sbc_test.cljs
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "   combo exited rc=$rc (timeout=124)"
    failed=$((failed+1))
  fi
done

echo "── merging fragments ──"
python3 scripts/merge_sbc_results.py "$FRAG_DIR" "$OUT_JSON"
rc=$?
echo "   merged → $OUT_JSON"
echo "── done ($failed combo process failures) ──"
exit $rc
