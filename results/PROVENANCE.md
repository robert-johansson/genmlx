# Provenance of the committed `results/` artifacts

**Read this before citing any number in this directory.**

Recorded 2026-08-01 by the health-audit sweep (bean `genmlx-gt0q`, milestone
`genmlx-wkdl`). Nothing here is regenerated automatically — there is no CI —
so this file is the only drift signal the directory has.

## What is actually committed

47 experiment directories, 29 metadata files carrying a `git_sha`:

| `git_sha` | count | when | platform |
|---|---|---|---|
| `c9253bc` | 25 | 2026-06-13 | `darwin` (arm64) |
| `de4911c` | 2 | later | mixed |
| `58b0afd` | 1 | later | mixed |

Platform field across all metadata: **25 `darwin`, 7 `linux`, 6 `macOS`**.
Run timestamps span 2026-03-13 … 2026-07-11.

## The staleness, stated plainly

- The dominant freeze commit `c9253bc` is **477 commits behind `main`** as of
  2026-08-01.
- It is dated **2026-06-13** — seven weeks before that count was taken.
- It was produced on **darwin/arm64 (Metal)**. The current primary dev and
  validation host is **x86_64 / CUDA sm_120**, and CLAUDE.md records numeric
  tolerances as **per-ARCH, not per-backend** — the same law measured
  Metal ~0 / sm_110 0.003 / sm_120 0.199. A Metal-produced number is therefore
  not interchangeable with an sm_120 one.
- The repo's own change detector reports **24 of 26 `source_hash` values no
  longer match** the sources they were computed from.

**Consequence: these artifacts are a dated snapshot, not a current run.** They
may still be correct; nothing here asserts they are wrong. What is asserted is
that *no evidence in this repository currently establishes that they are
current*, and nothing re-checks them.

## Known correctness caveat

`c9253bc` is the merge of *"fix/smc-rejuvenation-mxss"*. The 2026-08-01 audit
found (`genmlx-0zr4`) that seeded SMC/cSMC rejuvenation never threaded a
**proposal** key, so all K MH moves within a timestep proposed the identical
value — and that the SBC calibration validating that very fix was collected on
the **unseeded** path, which the defect does not affect. Any artifact here that
depends on seeded rejuvenation should be re-run after `genmlx-0zr4` lands.

Separately, `genmlx-5ytq` (conjugacy firing when the observation's noise
references the eliminated latent) produced a **wrong marginal likelihood
published as `:exact`**. Any experiment whose model has a latent-dependent
scale parameter must be re-run; see that bean's blast-radius checklist.

## How to check drift yourself

```bash
bun run --bun nbb run_experiments.cljs --changed --dry-run
```

## What to do about it

The decision has not been made. The two honest options are:

1. **Re-run the frozen set on a named host** and re-commit with fresh
   provenance — after `genmlx-yy8u`, `genmlx-5ytq` and `genmlx-0zr4` land, since
   all three can move published numbers.
2. **Mark `results/` explicitly historical** and point the paper at a new run.

Until one is chosen, cite these numbers only with their date and platform
attached.
