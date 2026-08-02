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

### Every asserting file MUST be able to fail

`run.sh` classifies on the child's **exit code** plus one anchored grep for the cljs.test
summary line. A hand-rolled harness that counts failures in an atom, prints `FAIL: ...`
and then exits 0 is recorded **PASS**, and its log is deleted at the end of the tier.
On 2026-08-01, **36 tiered files were in exactly that state** — including 40 of the 49
under `test/genmlx/llm/`, i.e. the whole LLM layer was invisible to the battery
(bean `genmlx-n061`). `test/run.sh check` now fails if any `fast`/`medium`/`slow` file
has no failure gate, so the class cannot come back.

A file satisfies the gate in one of four ways:

| shape | how it gates |
|---|---|
| `cljs.test` | `(cljs.test/run-tests)` exits non-zero on a failing `is` — nothing to add |
| hand-rolled harness | `(when (pos? @fail-count) (set! (.-exitCode js/process) 1))` |
| wrapper around an example | `;; @gate delegated — <the ns it requires, and how that ns exits non-zero>` |
| pure smoke test, no assertions | `;; @gate exception — <why "does not throw" is the whole contract>` |

Both `;; @gate` markers **require a trailing reason**, so the exemption is auditable
rather than a rubber stamp. A file that asserts nothing and is not a smoke test belongs
in the `bench` tier, where a clean exit is PASS by design.

Prefer `(set! (.-exitCode js/process) 1)` over `(js/process.exit 1)`: `process.exit`
truncates pending stdout (losing the very summary line the failure report needs) and
kills in-flight async work.

**Placement matters in async files.** Most `test/genmlx/llm/*` tests are promesa-based.
Their top level returns a promise immediately, so a gate appended at end-of-file runs
*before any assertion has executed* and always reads 0. In those files the gate belongs
**inside the final `p/let` body, right after the summary `println`**.

A related trap: an unconditional `(js/process.exit 0)` defeats the gate no matter what
the counter says. `check` rejects one unless it is a documented skip — print
`SKIP <reason>` immediately before it, as the checkpoint-absent paths do.

### Skipping a file on a host that cannot run it

A `cljs.test` file that skips still prints `Ran 0 tests`, and zero tests is normally a
FAIL — correctly so, for a file that failed to *load*. To skip deliberately, print
`SKIP <reason>` **at line start, within three lines above the `Ran 0 tests` summary**
(a `println` at the top level of the skip branch does this naturally). The runner then
scores the file `SKIP`: listed in the tier output so it stays visible, tallied in its own
`skipped` column, and excluded from the exit code.

The marker is required rather than inferred, and the distance rule is the point: a file
that *silently* stops registering tests prints no `SKIP` line and still scores
`FAIL(0 tests)`, so this cannot be used to make a broken file green. This is the same
marker and proximity rule `check` uses to exempt a documented skip from the failure-gate
audit.

Before this existed, a correct per-arch skip (`vgenerate_compiled_test` on Metal) counted
as not-passed and drove the whole battery to exit 1 — and a permanently-red battery
teaches everyone to discount reds, which is the same corrosion the honesty contract exists
to prevent (`genmlx-wg95`).

## Adding a test

Create `test/genmlx/<name>_test.cljs` as usual, then **add one line to `test/tiers.txt`**.
If you forget, `bun run test:check` fails loudly with `UNCLASSIFIED — on disk but not in
test/tiers.txt`. That is the anti-rot guarantee: a new test can never silently fall out of
coverage. Put it in `fast` only if it has no GPU inference and runs in a couple of seconds;
otherwise `medium` (or `slow` for SBC/convergence/LLM).
