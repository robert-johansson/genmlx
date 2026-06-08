# Running the GenMLX test suite

GenMLX has ~340 test files. Each **must** run in its own process — MLX/Metal segfaults
under sustained single-process GPU load (see `CLAUDE.md` / bean `genmlx-5ucd`). The runner
(`test/run.sh`) enforces that isolation and runs cheap tests in parallel.

## Tiers

Every test file is assigned exactly one tier in **`test/tiers.txt`** (the source of truth):

| Tier      | What                                                        | When you run it            |
|-----------|------------------------------------------------------------|----------------------------|
| `fast`    | pure/cheap, parallel                                        | before merge (`test:all`)  |
| `medium`  | GPU inference, bounded                                      | before merge (`test:all`)  |
| `slow`    | SBC / convergence / stress / agentmodels / LLM             | on demand                  |
| `bench`   | benchmarks, no pass/fail assertions                        | opt-in                     |
| `exclude` | shared helpers/runners — never run standalone              | —                          |

The `fast`/`medium` boundary is **empirically calibrated** (a file is `fast` only if it
actually runs in a few seconds), not guessed from its imports.

### fast-core — the per-change loop

`fast` is the full pure/cheap tier (~130 files, a few minutes serial). For the tight
per-change loop you want something that finishes in seconds, so a curated **fast-core**
subset (~30 high-signal files: data structures, schema, dist log-prob, handler purity,
the membrane contract guards, GFI contracts/ops, core combinators) is marked with an
optional `core` 3rd column on its `fast` manifest lines. `core ⊆ fast`, so the manifest
stays one complete classification. **`test:fast` runs fast-core** (~17s serial); the full
tier is `test:fast-all`.

## Commands

```bash
bun run test:fast      # fast-core — the per-change smoke loop (~30 files, ~17s)
bun run test:fast-all  # the full fast tier
bun run test:medium    # GPU inference tier
bun run test:slow      # SBC/convergence/stress/agentmodels/LLM (serial)
bun run test:all       # fast + medium + slow — the pre-merge gate
bun run test:bench     # benchmarks (opt-in)
bun run test:check     # classification gate (no tests run)

# or directly, with any combination / a custom parallel degree:
bash test/run.sh core
TEST_JOBS=8 bash test/run.sh fast medium
```

## The honesty contract

A Metal **CRASH** (SIGTRAP/SIGSEGV) or a **TIMEOUT** is a **FAIL**, never a silent "skip".
`run.sh` exits non-zero if any file does not cleanly PASS. (The old `run_all.sh` counted
crashes as passes — that is exactly what eroded confidence in the suite.)

## Adding a test

Create `test/genmlx/<name>_test.cljs` as usual, then **add one line to `test/tiers.txt`**.
If you forget, `bun run test:check` fails loudly with `UNCLASSIFIED — on disk but not in
test/tiers.txt`. That is the anti-rot guarantee: a new test can never silently fall out of
coverage. Put it in `fast` only if it has no GPU inference and runs in a couple of seconds;
otherwise `medium` (or `slow` for SBC/convergence/LLM).
