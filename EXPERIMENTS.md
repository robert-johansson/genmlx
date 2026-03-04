# Reproducing Paper Experiments

Instructions for re-running all experiments from the TOPML system paper.
All benchmarks write JSON output directly to `results/`.

**Hardware:** Apple Silicon Mac (M1/M2/M3/M4)
**Requirements:** Bun, nbb, Python 3 with matplotlib/numpy/scipy

---

## Step 0: Verification (Exp 5)

Confirm all tests pass before benchmarking.

```bash
bun run --bun nbb test/genmlx/gen_clj_compat_test.cljs      # 165/165
bun run --bun nbb test/genmlx/genjax_compat_test.cljs        # 73/73
bun run --bun nbb test/genmlx/gfi_contract_test.cljs         # 64 checks
```

Output: stdout (pass/fail counts). `verification_summary.json` is manually
updated after confirming all pass.

## Step 1: Vectorization Speedup (Exp 1 — Section 6.2)

Measures particle-scaling speedup of shape-based vectorization.

```bash
bun run --bun nbb test/genmlx/paper_bench_vectorization.cljs
```

Writes:
- `results/exp1_vectorization/particle_scaling.json` — N=1..1000 scaling
- `results/exp1_vectorization/method_speedup.json` — per-method speedup at N=1000
- `results/exp1_vectorization/SUMMARY.md`

Produces **Figure 1** (particle scaling) and **Figure 2** (method speedup bars).

## Step 2: FFI Bottleneck (Exp 2 — Section 6.3)

Compares per-site vs `gaussian-vec` across model dimensions D=10..200.

```bash
bun run --bun nbb test/genmlx/perfbench_large.cljs
```

Writes:
- `results/exp2_ffi_bottleneck/is_D*_n10000.json` — per-site variant
- `results/exp2_ffi_bottleneck/is_fast_D*_n10000.json` — gaussian-vec variant

GenJAX comparison data (`genjax_is_D*_n10000.json`) is static — collected from
GenJAX's own benchmark suite and does not need re-running.

Produces **Figure 3** (FFI scaling) and **Figure 4** (speedup table).

## Step 3: Canonical Models (Exp 3 — Section 6.4)

Three models with correctness verification against ground truth.

```bash
bun run --bun nbb test/genmlx/paper_bench_linreg.cljs    # Bayesian linear regression
bun run --bun nbb test/genmlx/paper_bench_hmm.cljs        # Hidden Markov model
bun run --bun nbb test/genmlx/paper_bench_gmm.cljs        # Gaussian mixture model
```

Writes:
- `results/exp3_canonical_models/linreg_results.json` — 5 algorithms vs analytic posterior
- `results/exp3_canonical_models/hmm_results.json` — IS vs SMC (forward algorithm ground truth)
- `results/exp3_canonical_models/gmm_results.json` — IS vs Gibbs (enumeration ground truth)
- `SUMMARY.md`, `SUMMARY_hmm.md`, `SUMMARY_gmm.md`

Produces **Figure 5** (posterior densities), **Figure 6** (linreg table),
**Figure 7** (HMM log-ML), **Figure 8** (GMM comparison).

## Step 4: Compilation Speedup (Exp 6 — Section 6.3)

Measures loop compilation speedup for MCMC.

```bash
bun run --bun nbb test/genmlx/compiled_benchmark.cljs
```

Writes:
- `results/exp6_compilation/compiled_speedup.json` — 4 benchmarks (MH, score-fn, HMC, vectorized MH)
- `results/exp6_compilation/SUMMARY.md`

## Step 5: System Comparison (Exp 4 — Section 6.7)

GenMLX IS(1000) timings for cross-system comparison.

```bash
bun run --bun nbb scripts/exp4_genmlx_is1000.cljs
python3 scripts/exp4_extract_genmlx.py
```

Writes:
- `results/exp4_system_comparison/genmlx_is1000.json` — IS timings for 3 models
- `results/exp4_system_comparison/genmlx.json` — aggregated with MH/SMC from Exp 3

Gen.jl and GenJAX comparison runs are on separate codebases and don't need
re-running unless those systems change:

```bash
# Optional — only if re-running reference implementations:
julia scripts/exp4_genjl_benchmarks.jl                 # → genjl.json
python3 scripts/exp4_genjax_benchmarks.py              # → genjax.json (needs pyenv genjax-05)
```

Produces **Figure 9** (system comparison bars) and **Figure 10** (system table).

## Step 6: Generate Figures

```bash
cd paper/viz && python generate_figures.py
```

Reads all JSON from `results/` and writes publication figures to `paper/figs/`:
- `fig1_particle_scaling.pdf` — vectorization scaling
- `fig2_method_speedup.pdf` — per-method speedup bars
- `fig3_ffi_scaling.pdf` — FFI bottleneck (star figure)
- `fig4_speedup_table.tex` — speedup table
- `fig5_linreg_posteriors.pdf` — posterior density comparison
- `fig6_linreg_table.tex` — algorithm comparison table
- `fig7_hmm_logml.pdf` — HMM log-ML accuracy
- `fig8_gmm_logml.pdf` — GMM IS vs Gibbs
- `fig9_system_bars.pdf` — cross-system comparison
- `fig10_system_table.tex` — system comparison table
- `fig11_verification_ladder.tex` — verification summary
- `fig12_contract_heatmap.pdf` — contract pass matrix

## Step 7: Sync to Overleaf

```bash
./scripts/sync_to_overleaf.sh "Update results from latest code"
```

Copies figures and tables from `results/` to `paper/TOPML_system/` (Overleaf
submodule) and pushes.

---

## Quick Reference

| Experiment | Command | Time | Output |
|-----------|---------|------|--------|
| Verification | 3 test commands | ~1 min | stdout |
| Exp 1: Vectorization | `paper_bench_vectorization.cljs` | ~2 min | `exp1_vectorization/` |
| Exp 2: FFI bottleneck | `perfbench_large.cljs` | ~5 min | `exp2_ffi_bottleneck/` |
| Exp 3A: LinReg | `paper_bench_linreg.cljs` | ~5 min | `exp3_canonical_models/` |
| Exp 3B: HMM | `paper_bench_hmm.cljs` | ~5 min | `exp3_canonical_models/` |
| Exp 3C: GMM | `paper_bench_gmm.cljs` | ~3 min | `exp3_canonical_models/` |
| Exp 4: System comparison | `exp4_genmlx_is1000.cljs` + extract | ~3 min | `exp4_system_comparison/` |
| Exp 6: Compilation | `compiled_benchmark.cljs` | ~3 min | `exp6_compilation/` |
| Figures | `generate_figures.py` | ~10 sec | `paper/figs/` |

Total wall-clock time: ~25–30 minutes.

## Timing Protocol

All benchmarks follow the same methodology:

- **Warmup:** 5 runs discarded before timing
- **Nested loops:** min of K inner repetitions, mean ± std of R outer repetitions
- **Timer:** `performance.now()` (milliseconds)
- **Memory:** `mx/tidy` wraps each benchmark iteration
- **Reproducibility:** Fixed PRNG seeds for data generation, random seeds for inference
