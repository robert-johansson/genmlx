# GPU run discipline on unified-memory hosts (Thor)

> **Tracking:** bean `genmlx-h3p5`. This is the ops documentation that bean
> was re-scoped to for release-v0: the failure mechanism, the standing rules,
> and the in-tree tooling. The underlying kernel-level dark-pages question
> stays open in the bean; the DISCIPLINE below is what keeps the box alive.

## The failure mechanism (three reboots' worth of evidence)

On Tegra-class unified memory (Jetson AGX Thor, 128GB), GPU allocations do
not appear in process RSS and are invisible to kernel LRU accounting. Under
runaway GPU allocation the kernel sees ~115GB of "dark" driver-side pages it
cannot reclaim (`all_unreclaimable=yes`), the OOM killer can only shoot small
adj-boosted services (pipewire/wireplumber), and the box spirals to a hard
reboot instead of recovering.

Observed triggers, each confirmed by journal + guarded-rerun forensics:

1. **Two concurrent GPU processes** (reboot #3, 2026-07-10): a 35B owned
   forward in a background shell while a dense test ran in the foreground.
   Each ran clean solo minutes earlier.
2. **Lazy-graph transient pileup in sync token loops** (isolated by the
   guarded rerun of the same workload): 32 uncached 35B forwards each
   dropping a ~3-4GB lazy transient graph JS GC cannot size → 95GB→20GB in
   ~12s. Fixed in-code (`run-filter` per-round `mx/force-gc!`, `:gc-every`;
   `llm-mh-chain` per-step force-gc; genmlx-7f93) — but the class is generic:
   any new hot loop over big forwards needs a collection trigger.
3. **Killing in-flight GPU work, then immediately launching a heavy load**
   (reboot #4, 2026-07-17): killed mid-tier workers followed by an UNGUARDED
   35B agent run. MemAvailable read 117GB just before launch — consistent
   with dark pages accumulated across the day's sessions.

`MemAvailable` (from `/proc/meminfo`) is the one signal that moves before
the collapse. Process RSS is blind — do not use it.

## The standing rules

1. **ONE GPU process at a time.** No background benchmark while a test runs.
   The tier runner's serial slow tier exists for this reason.
2. **Every heavy (35B/80B-class) model run goes through
   `scripts/guarded-run.sh`** — including `mlx agent` / provider probes /
   one-off REPL experiments, not just test suites. The guard refuses to start
   below the floor, SIGKILLs the process group if MemAvailable crosses it,
   and logs the curve for post-mortem.
3. **After killing any in-flight GPU workload, cool down** before the next
   heavy load; prefer letting runs finish over killing them.
4. **Unattended/autonomous sessions never run unguarded GPU work at all.**
5. Logs live in `~/genmlx-battery-logs/` (override with
   `GENMLX_GUARD_LOGDIR`) — **not `/tmp`**, which is cleared at boot; the
   first battery's logs were lost exactly that way.

## The tooling

```bash
# Wrap ANY GPU workload (FLOOR_MB defaults to 25000):
scripts/guarded-run.sh my-run bunx --bun nbb@1.4.208 test/genmlx/foo_test.cljs

# Outputs:
#   ~/genmlx-battery-logs/guarded-my-run.txt      workload stdout/stderr
#   ~/genmlx-battery-logs/guarded-my-run.mem.csv  epoch,MemAvailable_MB curve
# Exit codes: 42 = refused to start (below floor), 43 = floor-killed,
#             otherwise the workload's own exit code.
```

For multi-phase batteries, run each phase as its own process through the
guard (the per-phase process boundary is itself a defense: model teardown
returns memory between phases, and a phase crash cannot take the battery's
logs with it).

## Related

- `docs/cuda-test-triage.md` — the per-file suite ledger this discipline
  protects.
- Exit-teardown aborts after green asserts are a separate, known wart
  (`genmlx-gr51`, README "Known platform warts") — a guard-killed run (exit
  43) and a teardown abort (nonzero after `ALL PASS` output) are different
  signatures; read the log.
