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
# Usage:
#   bash test/run_sbc.sh                         # full run (~6h at N=500) — overnight
#   SBC_N=20 SBC_L=50 bash test/run_sbc.sh       # smoke run (minutes)
#   SBC_ONLY=single-gaussian:cmh bash test/run_sbc.sh   # one combo
#
# Notes:
# - A combo that exceeds SBC_COMBO_TIMEOUT (default 7200s) is abandoned; if
#   the process entered the Metal uninterruptible-sleep state it may linger
#   until reboot, but the runner moves on (see memory: Metal-wedge runs).
# - Re-running skips combos whose fragment ran to a PASSED verdict, so an
#   interrupted sweep resumes where it left off; combos that ran to a
#   FAILED verdict (sim-failure budget) are re-run. Delete results/sbc/
#   for a fresh run.
set -u
cd "$(dirname "$0")/.."

NBB="bun run --bun nbb"
FRAG_DIR="results/sbc"
TIMEOUT_S="${SBC_COMBO_TIMEOUT:-7200}"
mkdir -p "$FRAG_DIR"

echo "── SBC per-combo runner (serial, isolated processes) ──"
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
python3 scripts/merge_sbc_results.py "$FRAG_DIR" results/sbc_results.json
rc=$?
echo "── done ($failed combo process failures) ──"
exit $rc
