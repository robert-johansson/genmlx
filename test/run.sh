#!/usr/bin/env bash
# GenMLX tiered test runner.
#
#   test/run.sh core            # per-change smoke loop (~30 high-signal fast files, < ~90s)
#   test/run.sh fast            # the full fast tier (pure/cheap tests, parallel)
#   test/run.sh medium          # GPU inference tests (parallel)
#   test/run.sh slow            # SBC / convergence / stress / agentmodels / LLM (serial)
#   test/run.sh bench           # benchmarks (opt-in, serial; no assertions)
#   test/run.sh all             # fast + medium + slow  (the pre-merge gate)
#   test/run.sh fast medium     # any explicit combination
#   test/run.sh check           # classification gate: every test file is classified exactly once
#
# `core` is a 3rd-column marker on `fast` lines (core ⊆ fast), not a separate tier —
# the manifest stays the single, complete classification that `check` guards.
#
# WHY each file runs in its own process: MLX/Metal segfaults under sustained
# single-process GPU load (see CLAUDE.md / genmlx-5ucd). Isolation is mandatory.
#
# HONESTY CONTRACT (the thing run_all.sh got wrong): a Metal SIGTRAP/SIGSEGV
# (CRASH) or a TIMEOUT is a FAIL, never a silent "skip". Exit is non-zero if any
# file does not cleanly PASS. CI green must mean CI green.
#
# Tunables: TEST_JOBS — parallel degree for fast/medium tiers (default 4).
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
MANIFEST="test/tiers.txt"

# Always run nbb on the BUN runtime via `bun run --bun nbb`. Two hard lessons from
# benchmarking this on Apple Silicon:
#  - The direct `nbb` binary runs on NODE — far slower for GenMLX's compute, and it
#    wedged heavy tests into uninterruptible Metal sleep (unkillable by `timeout`,
#    exhausted RAM, load spiked to 67). NEVER use node-nbb.
#  - Concurrent runs race two ways: `bunx` collides on a shared link ("EEXIST"), and
#    concurrent GPU load wedges Metal (bean genmlx-5ucd). So default SERIAL. Once the
#    membrane buffer-count fix lands, parallelism can be revisited via TEST_JOBS.
NBB_CMD="bun run --bun nbb"
export NBB_CMD
JOBS="${TEST_JOBS:-1}"

[ -f "$MANIFEST" ] || { echo "FATAL: $MANIFEST not found"; exit 2; }

tier_timeout() {            # per-tier per-file wall-clock cap (seconds)
  case "$1" in
    core|fast) echo 45  ;;
    medium)    echo 150 ;;
    slow)      echo 600 ;;
    bench)     echo 900 ;;
    *)         echo 120 ;;
  esac
}
tier_jobs() {               # fast/medium parallel; slow/bench serial (GPU/RAM contention)
  case "$1" in
    core|fast|medium) echo "$JOBS" ;;
    *)                echo 1 ;;
  esac
}

# ---- manifest helpers -------------------------------------------------------
# `core` is not a tier — it is a 3rd-column marker on `fast` lines selecting a
# small high-signal subset (the per-change smoke loop). core ⊆ fast.
manifest_files_for_tier() {
  if [ "$1" = core ]; then
    awk '!/^#/ && $1=="fast" && $3=="core" {print $2}' "$MANIFEST"
  else
    awk -v t="$1" '!/^#/ && NF>=2 && $1==t {print $2}' "$MANIFEST"
  fi
}
manifest_all_files()      { awk '!/^#/ && NF>=2 {print $2}' "$MANIFEST"; }
disk_all_files()          { find test -name '*.cljs' -type f | sort; }

# ---- classification gate (test:check) --------------------------------------
do_check() {
  local fail=0
  # 1. duplicate manifest entries
  local dups
  dups="$(manifest_all_files | sort | uniq -d)"
  if [ -n "$dups" ]; then echo "DUPLICATE manifest entries:"; echo "$dups" | sed 's/^/  /'; fail=1; fi
  # 2. files on disk missing from the manifest  (the anti-rot guarantee)
  local missing
  missing="$(comm -23 <(disk_all_files) <(manifest_all_files | sort -u))"
  if [ -n "$missing" ]; then
    echo "UNCLASSIFIED — on disk but not in $MANIFEST (add a tier line):"
    echo "$missing" | sed 's/^/  /'; fail=1
  fi
  # 3. manifest entries whose file no longer exists
  local stale
  stale="$(comm -13 <(disk_all_files) <(manifest_all_files | sort -u))"
  if [ -n "$stale" ]; then echo "STALE — in $MANIFEST but not on disk:"; echo "$stale" | sed 's/^/  /'; fail=1; fi
  # 4. no excluded helper may contain a real test (would be silently never-run)
  local f
  while IFS= read -r f; do
    [ -f "$f" ] && grep -q '(deftest' "$f" && { echo "EXCLUDED file contains (deftest — it would never run: $f"; fail=1; }
  done < <(manifest_files_for_tier exclude)
  # 5. unknown tier names
  local badt
  badt="$(awk '!/^#/ && NF>=2 && $1!~/^(fast|medium|slow|bench|exclude)$/ {print $1" "$2}' "$MANIFEST")"
  if [ -n "$badt" ]; then echo "UNKNOWN tier name(s):"; echo "$badt" | sed 's/^/  /'; fail=1; fi
  # 6. the `core` marker (3rd column) is only valid on `fast` lines, and must be
  #    the literal word `core` — anything else is a typo that would silently
  #    drop the file from the per-change loop.
  local badcore
  badcore="$(awk '!/^#/ && NF>=3 && !($1=="fast" && $3=="core") {print $0}' "$MANIFEST")"
  if [ -n "$badcore" ]; then
    echo "INVALID 3rd column — only 'core' on a 'fast' line is allowed:"
    echo "$badcore" | sed 's/^/  /'; fail=1
  fi

  if [ "$fail" -eq 0 ]; then
    local ncore; ncore="$(manifest_files_for_tier core | grep -c .)"
    echo "check: OK — $(disk_all_files | wc -l | tr -d ' ') test files, all classified exactly once ($ncore in fast-core)."
  fi
  return "$fail"
}

# ---- single-file worker (internal; invoked per file, possibly in parallel) --
# usage: run.sh __one <tier> <resultdir> <file>
do_one() {
  local tier="$1" rdir="$2" file="$3"
  local to; to="$(tier_timeout "$tier")"
  local key; key="$(echo "$file" | tr '/.' '__')"
  local log="$rdir/$key.log"
  local start=$SECONDS status code dur
  if timeout "$to" $NBB_CMD "$file" > "$log" 2>&1; then code=0; else code=$?; fi
  dur=$((SECONDS - start))
  if   [ "$code" -eq 124 ]; then status="TIMEOUT"
  elif [ "$code" -ge 128 ]; then status="CRASH($code)"
  elif [ "$code" -ne 0 ];   then status="FAIL($code)"
  # exit code is the reliable signal (cljs.test AND legacy files exit non-zero on
  # failure). For a clean exit, only the machine-stable cljs.test summary can still
  # indicate failure; do NOT loosely grep 'FAIL' (legacy prints 'FAILED: 0' on pass).
  elif grep -qE '[1-9][0-9]* (failures|errors)' "$log"; then status="FAIL(asserts)"
  elif grep -qE 'Ran 0 tests' "$log";                    then status="FAIL(0 tests)"
  else status="PASS"; fi
  # bench files have no assertions: a clean exit is success regardless of FAIL-word noise
  if [ "$tier" = bench ] && [ "$code" -eq 0 ]; then status="PASS"; fi
  printf '%s\t%s\t%s\n' "$status" "${dur}s" "$file" > "$rdir/$key.result"
}

# ---- tier runner ------------------------------------------------------------
rdir=""   # global so the EXIT trap can see it under `set -u`
run_tiers() {
  local tiers=("$@") overall=0
  rdir="$(mktemp -d "${TMPDIR:-/tmp}/genmlx_tests.XXXXXX")"
  trap 'rm -rf "${rdir:-}"' EXIT
  export -f do_one tier_timeout

  echo "nbb: '$NBB_CMD'  jobs(fast/medium): $JOBS"
  local grand_pass=0 grand_fail=0
  for tier in "${tiers[@]}"; do
    local files; files="$(manifest_files_for_tier "$tier")"
    local n; n="$(printf '%s\n' "$files" | grep -c . )"
    [ "$n" -eq 0 ] && { echo "── $tier: (no files)"; continue; }
    local j; j="$(tier_jobs "$tier")"
    echo "── $tier: $n files, ${j}-way, $(tier_timeout "$tier")s cap ──"
    # dispatch (xargs -P preserves isolation: one process per file)
    printf '%s\n' "$files" | grep . | \
      xargs -P "$j" -I {} bash -c 'do_one "$0" "$1" "$2"' "$tier" "$rdir" {} 2>/dev/null
    # aggregate this tier
    local r st dur fn tpass=0 tfail=0
    while IFS=$'\t' read -r st dur fn; do
      if [ "$st" = PASS ]; then tpass=$((tpass+1));
      else
        tfail=$((tfail+1))
        printf '  %-14s %6s  %s\n' "$st" "$dur" "$(basename "$fn")"
        # show a short tail of the failing log
        local key; key="$(echo "$fn" | tr '/.' '__')"
        grep -hE '(^| )FAIL|ERROR|Exception|[1-9][0-9]* (failures|errors)|SIGTRAP|Resource limit' "$rdir/$key.log" 2>/dev/null | head -3 | sed 's/^/      | /'
      fi
    done < <(cat "$rdir"/*.result 2>/dev/null | sort -t$'\t' -k3)
    echo "   $tier: $tpass passed, $tfail not-passed"
    grand_pass=$((grand_pass+tpass)); grand_fail=$((grand_fail+tfail))
    rm -f "$rdir"/*.result "$rdir"/*.log
  done

  echo "═══════════════════════════════════════════"
  echo "TOTAL: $grand_pass passed, $grand_fail not-passed"
  [ "$grand_fail" -eq 0 ] || overall=1
  if [ "$overall" -eq 0 ]; then echo "RESULT: PASS"; else echo "RESULT: FAIL"; fi
  return "$overall"
}

# ---- main -------------------------------------------------------------------
[ $# -ge 1 ] || { echo "usage: test/run.sh {core|fast|medium|slow|bench|all|check} [tier...]"; exit 2; }

# internal worker dispatch
if [ "$1" = "__one" ]; then shift; do_one "$@"; exit $?; fi

declare -a TIERS=()
for arg in "$@"; do
  case "$arg" in
    check) do_check; exit $? ;;
    all)   TIERS+=(fast medium slow) ;;
    core|fast|medium|slow|bench) TIERS+=("$arg") ;;
    *) echo "unknown tier: $arg"; exit 2 ;;
  esac
done
run_tiers "${TIERS[@]}"
