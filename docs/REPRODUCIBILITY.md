# Reproducing the GenMLX evidence results

Two experiment suites back the results-bearing claims. Both are
one-command, crash-safe, and record provenance into their outputs.

## Requirements

- macOS on Apple Silicon (Metal GPU; MLX has no other backend here)
- [Bun](https://bun.sh) ≥ 1.3 (`bun run --bun nbb` resolves the
  project-pinned nbb from `package.json` — do not use a globally
  installed nbb)
- `npm install` (pulls `@mlx-node/core` / `@mlx-node/lm`; the `mlx-node`
  git submodule pins the exact native source the binary is built from)
- Nothing else GPU-heavy running: the suites are strictly serial on the
  GPU by design (sustained parallel Metal load risks the
  uninterruptible-sleep wedge documented in `test/run_sbc.sh`)

## 1. Simulation-based calibration (SBC)

```bash
bash test/run_sbc.sh
```

- Runs all 27 model × algorithm combos (N=500 sims, L=200 posterior
  draws each) as one isolated `bun`+`nbb` process per combo, strictly
  serially. Expect ~12–15 h.
- Each combo writes a crash-safe fragment to `results/sbc/`; re-running
  the command resumes exactly where it stopped (combos with a PASSED
  fragment are skipped; failed-verdict combos re-run).
- Fragments auto-merge into `results/sbc_results.json` at the end
  (`scripts/merge_sbc_results.py` — three-way passed/failed/missing
  classification; an executed-and-failed combo is surfaced, never
  silently dropped).

**Success criteria** in `results/sbc_results.json`:

- `summary.complete?` is `true` (full registry coverage AND no
  failed-verdict combos)
- `summary.fail` is `0` (every per-param chi²+KS test passed at
  Bonferroni-corrected α = 0.01)
- `summary.provenance` has exactly **one** entry — all fragments were
  produced on a single (genmlx commit, mlx-node commit) pair. More than
  one entry means the run spans code states. (The frozen artifact
  committed in this repo is a documented two-pair exception — see
  "Provenance of the frozen artifact" below. A fresh full run from the
  freeze commit produces a single pair.)

## 2. Evidence experiments

```bash
bun run --bun nbb run_experiments.cljs
```

- Runs the benchmark/verification experiments defined in
  `experiments.edn`, one result directory per experiment under
  `results/<name>/data.json`, each with reproducibility metadata.
- Summary written to `results/_last_run.json`.

**Success criterion:** every entry in `results/_last_run.json`
`experiments[].success?` is `true`.

## Provenance of the frozen artifact

The committed `results/sbc_results.json` carries **two** provenance
pairs, by explicit decision (2026-06-12):

- 26 of 27 combos come from the pristine freeze sweep on
  {genmlx `6d09405`, mlx-node `a7e5f04`}.
- `conjugate-linreg-elim:smc` initially failed chi² on both params in
  that sweep (KS passed — the structured-ranks signature; tracked as
  genmlx-mxss). The diagnosis was sampler adequacy — too few MH
  rejuvenation steps to restore particle diversity after resampling on
  a correlated 2D posterior — not weight math. The fix raised
  `:rejuvenation-steps` in that combo's `smc-opts` in
  `test/genmlx/sbc_test.cljs`: a sampler-tuning line in the test spec,
  the same ruling class as the cmh thinning fix (genmlx-t757), not a
  weakening of the experiment. The combo was then re-run at full
  N=500 on the spec-fix commit {genmlx `c9253bc`, mlx-node `a7e5f04`},
  passing both tests (slope χ²=18.12, intercept χ²=21.04; critical
  21.67).

The full diff between the two genmlx commits is two configuration
lines plus their comments: the SBC `:rejuvenation-steps` spec fix
(`test/genmlx/sbc_test.cljs`, genmlx-mxss) and a benchmark-runner
timeout raised for the heavier 4-chain funnel experiment
(`experiments.edn`, genmlx-2zxo) — `git diff 6d09405 c9253bc` touches
only those two files. No engine, inference, or distribution code
differs, and the mlx-node native binary is identical across all 27
fragments. The alternative — an ~18 h uniform re-run of the whole
registry on the spec-fix commit — was declined as pure GPU cost with
no evidential value.

## Provenance and seeds

Every SBC fragment records `config.meta`: the genmlx commit, the
mlx-node commit (native binary source), Bun version, Node-compat
version, GPU device, and a start timestamp. The merge surfaces the
distinct commit pairs across all ingested fragments as
`summary.provenance`.

PRNG seeds are intentionally **not** pinned: SBC's statistical claim
(rank uniformity) is seed-free by construction, and each simulation
draws fresh entropy. Bit-for-bit replay of a full run is therefore not
expected; statistical agreement of the rank tests is. (Known caveat:
`csmc` consumes unkeyed entropy even under an explicit `:key` —
tracked as genmlx-g5ys.)
