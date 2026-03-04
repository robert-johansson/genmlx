# Experiment 6: Loop Compilation Speedup

**Date:** 2026-03-03
**Platform:** macOS, Apple Silicon, MLX GPU via @frost-beta/mlx, Bun + nbb
**Benchmark file:** `test/genmlx/compiled_benchmark.cljs`
**Methodology:** performance.now(), warmup + nested loop (min-of-inner, mean+std-of-outer)

## Results

| Benchmark | Baseline (ms) | Compiled/Batched (ms) | Speedup |
|-----------|--------------|----------------------|--------|
| GFI MH vs Compiled MH (500 samples) | 1338.1 +/- 32.9 | 122.4 +/- 1.1 | 10.9x |
| Uncompiled vs Compiled score-fn (50 evals) | 67.0 +/- 0.4 | 9.3 +/- 0.8 | 7.2x |
| GenMLX HMC vs Handcoded HMC (200 samples) | 441.5 +/- 8.8 | 438.6 +/- 1.9 | 1.01x overhead |
| Serial 10x MH vs Vectorized 10-chain MH | 741.7 | 152.3 +/- 2.9 | 4.9x |

## Key Findings

1. **Compiled MH achieves 10.9x speedup over GFI MH.**

2. **Score function compilation gives 7.2x speedup.**

3. **GenMLX HMC overhead vs handcoded: 1.01x.**
   Both use fused leapfrog (L+1 gradient evals for L steps).

4. **Vectorized 10-chain MH gives 4.9x speedup.**
