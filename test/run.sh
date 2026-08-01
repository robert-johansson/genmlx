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
#   test/run.sh tags [--write]  # regenerate test/tiers.txt from the in-file @tier tags
#
# CLASSIFICATION SOURCE OF TRUTH: a ';; @tier <fast|medium|slow|bench|exclude> [core]'
# line near the top of each test file. test/tiers.txt is a GENERATED cache the runner
# reads (fast awk lookup); `check` FAILS if it drifts from the tags. To change a file's
# tier, edit its @tier line and run `test/run.sh tags --write`.
#
# `core` is a 3rd-column marker on `fast` lines (core ⊆ fast), not a separate tier —
# the manifest stays the single, complete classification that `check` guards.
#
# WHY each file runs in its own process: originally because MLX/Metal aborted
# under sustained single-process GPU load (genmlx-5ucd). That crash class is
# now catchable + proactively swept (5ucd Layer 1/2) and proven under N-way
# parallel load (genmlx-7yam), so isolation is no longer REQUIRED for safety —
# it is kept as a measurement/cleanliness choice: per-file walls and honest
# per-file tallies, leaked state cannot cross files, and one file's band flake
# can never poison another's GPU context.
#
# HONESTY CONTRACT (the thing run_all.sh got wrong): a Metal SIGTRAP/SIGSEGV
# (CRASH) or a TIMEOUT is a FAIL, never a silent "skip". Exit is non-zero if any
# file does not cleanly PASS. CI green must mean CI green.
#
# Tunables: TEST_JOBS — parallel degree for fast/medium tiers (default 4,
# validated 2026-06-10 on the full medium tier: per-file process isolation +
# the genmlx-5ucd buffer-count mitigation make concurrent GPU load safe, and
# do_one retries once on the known parallel-bunx launcher race; genmlx-q69j).
# TEST_JOBS=1 restores fully-serial runs. Bench stays strictly serial
# (perf numbers under contention are noise). TEST_JOBS_SLOW (default 4) sets
# the slow tier's degree: its long files are mostly small-model convergence
# tests that barely load the GPU, and the CUDA boxes have the VRAM for the
# few that do (35B resident = 31.5 GiB in 96 GiB on sm_120). The old
# serial-slow-on-Mac precaution (Metal wedge history) is RETIRED: the Metal
# buffer-count wall is per-process and both crash defenses hold at 8-way
# parallel (genmlx-7yam, validated M2 Max 2026-07-28: 2 full batteries at
# TEST_JOBS=8 TEST_JOBS_SLOW=4, zero crashes/timeouts, zero parallel-only
# failures; slow tier ~15 min vs ~71 serial). TEST_JOBS_SLOW=1 is the
# escape hatch.
#
# TEST_TIME_SCALE — host-speed scale (positive integer, default 1) multiplying
# every per-tier wall-clock cap, so slower-than-Apple hosts don't need retagged
# tiers (the @tier tags are shared with Apple Silicon; genmlx-9ox0). It is
# exported to the test processes, and perf-assert tests (fused_mcmc_test) scale
# their absolute-ms assertions by the same knob. Thor/CUDA (aarch64 Tegra):
#   TEST_TIME_SCALE=8 test/run.sh all
# (8, not the 4-6 first estimated: measured solo times llm_token_mcmc 264s vs
# the 45s fast cap and vi_property 672s vs the 150s medium cap need >=6, and 8
# keeps headroom for TEST_JOBS=1 serial runs where the J-way cap relief is off.)
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
MANIFEST="test/tiers.txt"

# Always run nbb on the BUN runtime via `bun run --bun nbb`. Two hard lessons from
# benchmarking this on Apple Silicon:
#  - The direct `nbb` binary runs on NODE — far slower for GenMLX's compute, and it
#    wedged heavy tests into uninterruptible Metal sleep (unkillable by `timeout`,
#    exhausted RAM, load spiked to 67). NEVER use node-nbb.
#  - Concurrent runs used to race two ways: `bunx` collides on a shared link
#    ("EEXIST" / "could not determine executable" — do_one now retries that one
#    identifiable launcher race once), and concurrent GPU load wedged Metal before
#    the genmlx-5ucd buffer-count mitigation. With both addressed, fast/medium run
#    TEST_JOBS-way parallel (validated at 4; genmlx-q69j). Slow/bench stay serial.
NBB_CMD="bun run --bun nbb"
export NBB_CMD
JOBS="${TEST_JOBS:-4}"
JOBS_SLOW="${TEST_JOBS_SLOW:-4}"

# GNU timeout is the honesty-contract enforcer (a hung file must FAIL, not hang
# the battery). macOS does not ship it; Homebrew coreutils installs it as
# `gtimeout`. Resolve whichever exists — and hard-fail if neither, because
# running without per-file caps would silently break the contract.
if   command -v timeout  >/dev/null 2>&1; then TIMEOUT_BIN=timeout
elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_BIN=gtimeout
else echo "FATAL: neither 'timeout' nor 'gtimeout' found (brew install coreutils)"; exit 2
fi
export TIMEOUT_BIN

[ -f "$MANIFEST" ] || { echo "FATAL: $MANIFEST not found"; exit 2; }

# Host-speed scale (see header). Validated once here; exported so both the
# per-file workers (tier_timeout below) and the test processes themselves
# (perf asserts reading js/process.env.TEST_TIME_SCALE) see the same value.
SCALE="${TEST_TIME_SCALE:-1}"
case "$SCALE" in
  ''|*[!0-9]*) echo "FATAL: TEST_TIME_SCALE must be a positive integer (got '$SCALE')"; exit 2 ;;
esac
[ "$SCALE" -ge 1 ] || { echo "FATAL: TEST_TIME_SCALE must be >= 1 (got '$SCALE')"; exit 2; }
export TEST_TIME_SCALE="$SCALE"

tier_timeout() {            # per-tier per-file wall-clock cap (seconds), x host-speed scale
  local base
  case "$1" in
    core|fast) base=45  ;;
    medium)    base=150 ;;
    slow)      base=600 ;;
    bench)     base=900 ;;
    *)         base=120 ;;
  esac
  echo $(( base * ${TEST_TIME_SCALE:-1} ))
}
tier_jobs() {               # fast/medium parallel; slow via TEST_JOBS_SLOW (default 4); bench always serial
  case "$1" in
    core|fast|medium) echo "$JOBS" ;;
    slow)             echo "$JOBS_SLOW" ;;
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
# LC_ALL=C: the manifest is a cross-host artifact — collation must not depend
# on the generating host's locale (en_US ignores '_' in ordering, C does not;
# a locale mismatch makes `check` diff-fail on a byte-identical tag set).
disk_all_files()          { find test -name '*.cljs' -type f | LC_ALL=C sort; }

# ---- in-file @tier tags (the source of truth) -------------------------------
# Each test file carries  ';; @tier <tier> [core]'  near the top. tiers.txt is a
# generated cache; `check` guards that the two never drift, and `tags --write`
# regenerates the cache from the tags.
file_tag() {                # echo "<tier>" or "<tier> core"; empty if missing/invalid
  awk '
    NR>8 { exit }
    /^[[:space:]]*;;[[:space:]]*@tier[[:space:]]/ {
      for (i=1;i<=NF;i++) if ($i=="@tier") {
        t=$(i+1); c=$(i+2)
        if (t ~ /^(fast|medium|slow|bench|exclude)$/)
          print (c=="core" ? t" core" : t)
        exit
      }
    }' "$1"
}
gen_tiers() {               # canonical manifest derived from the tags (path-sorted)
  printf '%s\n' \
    '# GenMLX test tier manifest — GENERATED from the in-file ";; @tier" tags.' \
    '# SOURCE OF TRUTH is the @tier line in each test file; do NOT hand-edit here.' \
    '# Change a file tier by editing its tag, then run: test/run.sh tags --write' \
    '# `test/run.sh check` FAILS if this cache drifts. Format: <tier> <path> [core]'
  local f tag
  while IFS= read -r f; do
    tag="$(file_tag "$f")"
    [ -z "$tag" ] && continue
    set -- $tag
    if [ "${2:-}" = core ]; then printf '%s %s core\n' "$1" "$f"
    else printf '%s %s\n' "$1" "$f"; fi
  done < <(disk_all_files)
}

# ---- process-tree teardown (bean genmlx-tkbs) -------------------------------
# A test is `timeout -> bun -> bunx -> nbb -> node`. Killing the bash parent
# orphans the grandchildren (they reparent to init), so a Ctrl-C / kill / harness
# stop used to leave GPU procs running until a manual pkill. macOS has no setsid,
# so we get a killable process GROUP per file via bash job control (`set -m`) in
# do_one, and reap everything here on interrupt by walking the descendant tree.
kill_tree() {               # SIGKILL $1 and all its descendants, children-first
  local p="$1" c
  for c in $(pgrep -P "$p" 2>/dev/null); do kill_tree "$c"; done
  kill -KILL "$p" 2>/dev/null
}
reap_children() {           # kill every descendant of THIS script (not the script)
  local c
  for c in $(pgrep -P $$ 2>/dev/null); do kill_tree "$c"; done
}
cleanup() {                 # reap the whole run, then the tmpdir (idempotent)
  trap - INT TERM EXIT
  reap_children
  [ -n "${rdir:-}" ] && rm -rf "$rdir"
}
on_signal() { cleanup; exit 143; }   # INT/TERM: reap and stop the run now

# `run.sh clean` — manual escape hatch for orphans from an older, hard-killed run.
do_clean() {
  local n; n="$(pgrep -lf 'bun run --bun nbb|bunx nbb@|nbb_main\.js' 2>/dev/null \
                | grep -v "^$$ " | grep -c . )"
  pkill -KILL -f 'bun run --bun nbb' 2>/dev/null
  pkill -KILL -f 'bunx nbb@'         2>/dev/null
  pkill -KILL -f 'nbb_main\.js'      2>/dev/null
  find "${TMPDIR:-/tmp}" -maxdepth 1 -type d -name 'genmlx_tests.*' -exec rm -rf {} + 2>/dev/null
  echo "clean: SIGKILLed orphaned genmlx test procs (~$n) and removed runner temp dirs."
}

# ---- classification gate (test:check) --------------------------------------
do_check() {
  local fail=0
  # 1. duplicate manifest entries
  local dups
  dups="$(manifest_all_files | LC_ALL=C sort | uniq -d)"
  if [ -n "$dups" ]; then echo "DUPLICATE manifest entries:"; echo "$dups" | sed 's/^/  /'; fail=1; fi
  # 2. files on disk missing from the manifest  (the anti-rot guarantee)
  local missing
  missing="$(comm -23 <(disk_all_files) <(manifest_all_files | LC_ALL=C sort -u))"
  if [ -n "$missing" ]; then
    echo "UNCLASSIFIED — on disk but not in $MANIFEST (add a tier line):"
    echo "$missing" | sed 's/^/  /'; fail=1
  fi
  # 3. manifest entries whose file no longer exists
  local stale
  stale="$(comm -13 <(disk_all_files) <(manifest_all_files | LC_ALL=C sort -u))"
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
  # 7. every file must carry a valid in-file @tier tag (the source of truth) ...
  local untagged f
  untagged="$(while IFS= read -r f; do [ -z "$(file_tag "$f")" ] && echo "$f"; done < <(disk_all_files))"
  if [ -n "$untagged" ]; then
    echo "MISSING/INVALID ';; @tier' tag (add one near the top of each file):"
    echo "$untagged" | sed 's/^/  /'; fail=1
  fi
  # 8. ... and the generated tiers.txt cache must match the tags (no drift)
  if ! diff -q <(gen_tiers) "$MANIFEST" >/dev/null 2>&1; then
    echo "$MANIFEST is STALE vs the in-file @tier tags — run: test/run.sh tags --write"
    fail=1
  fi
  # 9. every asserting file must be ABLE to fail. do_one classifies on the exit
  #    code plus one anchored grep for the cljs.test summary; a hand-rolled
  #    harness that counts failures in an atom, prints "  FAIL: ..." and then
  #    exits 0 is recorded PASS and its log is deleted at the end of the tier.
  #    36 tiered files were in exactly that state on 2026-08-01 — including 40
  #    of the 49 under test/genmlx/llm/, i.e. the whole LLM layer was invisible
  #    to the battery (bean genmlx-n061). A file passes this check by either
  #    using cljs.test (whose run-tests exits non-zero on a failing `is`) or
  #    carrying an explicit nonzero-exit gate on its own failure counter.
  #    `bench` is exempt by design: bench files assert nothing and do_one
  #    already treats a clean bench exit as PASS. `exclude` never runs.
  #    Two honest shapes have no local gate and declare so with a `;; @gate`
  #    line near the top, so the exemption is explicit and greppable:
  #      ;; @gate delegated — the gate lives in the ns this file requires
  #                           (e.g. agentmodels.harness/report!, which exits 1)
  #      ;; @gate exception — no assertions at all; the contract is only that
  #                           these operations run without throwing, and nbb
  #                           already exits non-zero on an unhandled error
  #    Both require a trailing reason so the claim is auditable, not a rubber
  #    stamp. A file that asserts nothing and is not a smoke test is a bench.
  local ungated
  ungated="$(for t in fast medium slow; do manifest_files_for_tier "$t"; done | while IFS= read -r f; do
      [ -f "$f" ] || continue
      grep -qE 'run-tests|js/process\.exit 1|\.exit js/process 1|\(set! \(\.-exitCode js/process\) 1\)|js/process\.exit \(if|\.exit js/process \(if|^;; @gate (delegated|exception) — .+' "$f" || echo "$f"
    done)"
  if [ -n "$ungated" ]; then
    echo "NO FAILURE GATE — these would be recorded PASS even if every assertion failed."
    echo "  Add:  (when (pos? @fail-count) (set! (.-exitCode js/process) 1))"
    echo "  In a promesa-based file the gate MUST go inside the final p/let body,"
    echo "  after the summary println — at end-of-file it runs before the chain resolves."
    echo "  A file that asserts nothing belongs in the bench tier instead."
    echo "$ungated" | sed 's/^/  /'; fail=1
  fi
  # 10. ... and none of them may hardcode an UNCONDITIONAL zero exit, which
  #     defeats the gate no matter what the counter says. Three files did
  #     (genmlx-n061). A documented capability/checkpoint SKIP legitimately
  #     exits 0 — it prints "SKIP ..." first, so a zero exit within three lines
  #     of a SKIP print is allowed and anything else is not. A deliberate early
  #     exit that is NOT a skip must carry the counter:
  #     (.exit js/process (if (pos? @fail) 1 0)).
  local zeroexit
  zeroexit="$(for t in fast medium slow; do manifest_files_for_tier "$t"; done | while IFS= read -r f; do
      [ -f "$f" ] || continue
      awk '/SKIP/ {skip=NR}
           /\(\.exit js\/process 0\)|\(js\/process\.exit 0\)/ {
             if (NR - skip > 3) { print FILENAME; exit }
           }' "$f"
    done)"
  if [ -n "$zeroexit" ]; then
    echo "UNCONDITIONAL ZERO EXIT — the failure gate can never fire in:"
    echo "  (a checkpoint/capability SKIP is exempt: print \"SKIP ...\" immediately before it)"
    echo "$zeroexit" | sed 's/^/  /'; fail=1
  fi
  # 11. Rule 9 proves a gate EXISTS, not that it can RUN. In a promesa-based
  #     file the top level returns a promise immediately, so a gate written as
  #     the last TOP-LEVEL form executes before any assertion has landed and
  #     always reads a zero counter — the exact mistake rule 9 was written to
  #     prevent, which its grep would happily accept. An in-chain gate is
  #     always indented (it sits inside a p/let or pr/handle body); a gate at
  #     column 0 in an async file is a red flag worth a human look.
  local misplaced
  misplaced="$(for t in fast medium slow; do manifest_files_for_tier "$t"; done | while IFS= read -r f; do
      [ -f "$f" ] || continue
      grep -qE 'promesa|p/let|pr/let|pr/handle' "$f" || continue
      grep -qE '^\((when|if) \(pos\? @' "$f" || continue
      # A column-0 gate is fine when the file ALSO carries an indented one:
      # that is the belt-and-braces shape (in-chain gate covers the async
      # tail, column-0 gate covers the synchronous sections) and is correct
      # under either await semantics. Only a LONE column-0 gate is suspect.
      grep -qE '^[[:space:]]+\((when|if) \(pos\? @|^[[:space:]]+\(set! \(\.-exitCode' "$f" || echo "$f"
    done)"
  if [ -n "$misplaced" ]; then
    echo "ASYNC FILE WITH ONLY A COLUMN-0 GATE — it may run before the chain resolves:"
    echo "  (move it inside the final p/let / pr/handle body, after the summary println,"
    echo "   or keep both: an indented in-chain gate for the async tail plus the"
    echo "   column-0 one for the synchronous sections — that combination is accepted)"
    echo "$misplaced" | sed 's/^/  /'; fail=1
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
  # Under J-way parallelism GPU-bound files share the device, inflating
  # wall-clock up to J-fold — scale the per-file cap accordingly so a busy
  # GPU is not misreported as a hang. TIMEOUT remains a FAIL (honesty
  # contract); only the cap is contention-aware.
  local j; j="$(tier_jobs "$tier")"
  [ "$j" -gt 1 ] && to=$((to * j))
  local key; key="$(echo "$file" | tr '/.' '__')"
  local log="$rdir/$key.log"
  local start=$SECONDS status code dur pid
  # Run in its OWN process group so the whole bun->bunx->nbb->node tree is reaped
  # atomically (bean genmlx-tkbs). `set -m` makes the backgrounded job a group
  # leader (PGID == pid); `kill -KILL -$pid` then kills the entire group, so a
  # timeout or an interrupt can never orphan the GPU child procs. `-k 10` bounds
  # the worker itself: if the test ignores SIGTERM, timeout SIGKILLs it after 10s
  # rather than hanging the worker forever.
  set -m
  local attempt
  for attempt in 1 2 3; do
    # TEST_PAR: the tier's parallel degree, for in-test absolute-ms assertions
    # (test_helpers wall-scale) — the same J-fold contention allowance this
    # worker already applies to its cap above (genmlx-7yam).
    TEST_PAR="$j" "$TIMEOUT_BIN" -k 10 "$to" $NBB_CMD "$file" > "$log" 2>&1 &
    pid=$!
    trap 'kill -KILL -'"$pid"' 2>/dev/null' TERM INT
    wait "$pid"; code=$?
    trap - TERM INT
    kill -KILL -"$pid" 2>/dev/null   # sweep any stragglers left alive in the group
    # Parallel bun/bunx launches race on shared state (install link EEXIST ->
    # "could not determine executable to run for package nbb"; bun.lock write
    # contention -> instant SIGKILL). Every observed flavor dies BEFORE any
    # test code runs, so retry once IFF the log shows the test never began
    # executing (no test/error output at all) — a real test failure or load
    # error always leaves output, so the retry cannot mask one.
    # Test-output markers ONLY (not generic 'error' — the launcher failures
    # themselves print 'error:'). A deterministic load error lacks these too
    # and gets harmless identical retries; it still reports as FAIL.
    # Up to TWO retries with a jittered pause: at a tier start the whole
    # first J-wave races the shared bunx state at once, so an immediate
    # identical retry often re-collides with the same cohort (2-9 casualties
    # per battery measured on 2026-07-27; genmlx-lr9c).
    if [ "$code" -ne 0 ] && [ "$attempt" -lt 3 ] && \
       ! grep -qE 'Testing |Ran [0-9]+ tests|[0-9]+ failures|PASS|FAIL' "$log"; then
      sleep $(( (RANDOM % 3) + attempt ))
      continue
    fi
    break
  done
  set +m
  dur=$((SECONDS - start))
  if   [ "$code" -eq 124 ]; then status="TIMEOUT"
  elif [ "$code" -ge 128 ]; then status="CRASH($code)"
  elif [ "$code" -ne 0 ];   then status="FAIL($code)"
  # exit code is the reliable signal (cljs.test AND legacy files exit non-zero on
  # failure). For a clean exit, only the machine-stable cljs.test summary can still
  # indicate failure; do NOT loosely grep 'FAIL' (legacy prints 'FAILED: 0' on pass).
  # Anchor to the cljs.test summary line ("N failures, M errors." at line start).
  # An unanchored match also fires on a PASSING assertion whose MESSAGE mentions
  # failures — e.g. llm/codegen_test prints "PASS: bad fn: 2 failures (expected 2,
  # got 2)" and was reported FAIL(asserts) while genuinely 91/91.
  elif grep -qE '^[[:space:]]*([1-9][0-9]* failures?,|[0-9]+ failures?, [1-9][0-9]* errors)' "$log"; then status="FAIL(asserts)"
  elif grep -qE 'Ran 0 tests' "$log";                    then status="FAIL(0 tests)"
  else status="PASS"; fi
  # bench files have no assertions: a clean exit is success regardless of FAIL-word noise
  if [ "$tier" = bench ] && [ "$code" -eq 0 ]; then status="PASS"; fi
  printf '%s\t%s\t%s\n' "$status" "${dur}s" "$file" > "$rdir/$key.result"
}

# ---- tier runner ------------------------------------------------------------
rdir=""   # global so the EXIT trap can see it under `set -u`
keepdir="" # created lazily on the first not-clean tier; survives cleanup's rm -rf
run_tiers() {
  local tiers=("$@") overall=0
  rdir="$(mktemp -d "${TMPDIR:-/tmp}/genmlx_tests.XXXXXX")"
  # reap the whole process tree on interrupt. INT/TERM go through on_signal (which
  # exits); a clean finish goes through the EXIT trap. Both call reap_children.
  trap on_signal INT TERM
  trap cleanup EXIT
  export -f do_one tier_timeout tier_jobs
  export JOBS JOBS_SLOW

  echo "nbb: '$NBB_CMD'  jobs(fast/medium): $JOBS  jobs(slow): $JOBS_SLOW  time-scale: ${TEST_TIME_SCALE}x"
  # Pre-warm the bunx nbb environment ONCE before any parallel fan-out: the
  # first J workers of a tier otherwise race the shared bunx cache (EEXIST /
  # "could not determine executable" / instant SIGKILL on its lockfile write),
  # costing 2-9 alphabetically-first files per battery even with do_one's
  # retry (measured across three 2026-07-27 Metal batteries; genmlx-lr9c).
  $NBB_CMD -e nil >/dev/null 2>&1 || true
  local grand_pass=0 grand_fail=0
  for tier in "${tiers[@]}"; do
    local files; files="$(manifest_files_for_tier "$tier")"
    local n; n="$(printf '%s\n' "$files" | grep -c . )"
    [ "$n" -eq 0 ] && { echo "── $tier: (no files)"; continue; }
    local j; j="$(tier_jobs "$tier")"
    echo "── $tier: $n files, ${j}-way, $(tier_timeout "$tier")s cap ──"
    # dispatch (xargs -P preserves isolation: one process per file). Run it
    # BACKGROUNDED and `wait` on it: bash defers a trap while blocked in a
    # FOREGROUND child, but the `wait` builtin returns immediately on a trapped
    # signal — so on_signal fires at once and reaps the tree (bean genmlx-tkbs).
    printf '%s\n' "$files" | grep . | \
      xargs -P "$j" -I {} bash -c 'do_one "$0" "$1" "$2"' "$tier" "$rdir" {} 2>/dev/null &
    wait $!
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
    # Accounting check: every dispatched file must have produced a verdict. A
    # worker killed before do_one wrote its .result (OOM killer, a reaped
    # process tree, an xargs launch failure) would otherwise just VANISH from
    # the tally — "48 passed, 0 not-passed" out of 50 dispatched reads as a
    # clean tier. Missing verdicts are failures, not absences (genmlx-n061).
    local accounted=$((tpass + tfail))
    if [ "$accounted" -ne "$n" ]; then
      echo "  ACCOUNTING  $accounted verdicts for $n dispatched files — no result from:"
      local mf
      while IFS= read -r mf; do
        [ -z "$mf" ] && continue
        [ -f "$rdir/$(echo "$mf" | tr '/.' '__').result" ] || { echo "      $mf"; tfail=$((tfail+1)); }
      done < <(printf '%s\n' "$files" | grep .)
    fi
    echo "   $tier: $tpass passed, $tfail not-passed"
    grand_pass=$((grand_pass+tpass)); grand_fail=$((grand_fail+tfail))
    # Preserve the evidence when the tier was not clean. do_one's tail above
    # shows at most three lines, and the EXIT trap rm -rf's $rdir — so without
    # this the full log of a failure is gone by the time anyone reads the
    # summary. Copy OUT of $rdir into a dir cleanup does not touch.
    if [ "$tfail" -gt 0 ]; then
      [ -z "$keepdir" ] && keepdir="$(mktemp -d "${TMPDIR:-/tmp}/genmlx_failures.XXXXXX")"
      cp "$rdir"/*.log "$keepdir/" 2>/dev/null || true
    fi
    rm -f "$rdir"/*.result "$rdir"/*.log
  done

  echo "═══════════════════════════════════════════"
  echo "TOTAL: $grand_pass passed, $grand_fail not-passed"
  [ -n "$keepdir" ] && echo "full logs for the not-passed files: $keepdir"
  [ "$grand_fail" -eq 0 ] || overall=1
  if [ "$overall" -eq 0 ]; then echo "RESULT: PASS"; else echo "RESULT: FAIL"; fi
  return "$overall"
}

# ---- main -------------------------------------------------------------------
[ $# -ge 1 ] || { echo "usage: test/run.sh {core|fast|medium|slow|bench|all|check|clean|tags} [tier...]"; exit 2; }

# internal worker dispatch
if [ "$1" = "__one" ]; then shift; do_one "$@"; exit $?; fi

declare -a TIERS=()
for arg in "$@"; do
  case "$arg" in
    check) do_check; exit $? ;;
    clean) do_clean; exit $? ;;
    tags)  if [ "${2:-}" = --write ]; then
             gen_tiers > "$MANIFEST"
             echo "wrote $MANIFEST from in-file @tier tags ($(disk_all_files | wc -l | tr -d ' ') files)."
           else gen_tiers; fi
           exit $? ;;
    all)   TIERS+=(fast medium slow) ;;
    core|fast|medium|slow|bench) TIERS+=("$arg") ;;
    *) echo "unknown tier: $arg"; exit 2 ;;
  esac
done
run_tiers "${TIERS[@]}"
