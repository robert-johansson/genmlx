# Experiment 6: Loop Compilation Speedup

**Date:** 2026-03-03
**Platform:** macOS, Apple Silicon, MLX GPU via @frost-beta/mlx, Bun + nbb
**Benchmark file:** `test/genmlx/compiled_benchmark.cljs`
**Methodology:** performance.now(), warmup + nested loop (min-of-inner, mean+std-of-outer)

## Results

| Benchmark | Baseline (ms) | Compiled/Batched (ms) | Speedup |
|-----------|--------------|----------------------|--------|
| GFI MH vs Compiled MH (500 samples) | 4741.7 +/- 1472.4 | 1557.4 +/- 205.3 | 3.0x |
| Uncompiled vs Compiled score-fn (50 evals) | 390.5 +/- 36.3 | 9.6 +/- 0.9 | 40.6x |
| GenMLX HMC vs Handcoded HMC (200 samples) | 7769.1 +/- 503.5 | 2402.8 +/- 118.4 | 3.23x overhead |
| Serial 10x MH vs Vectorized 10-chain MH | 8761.5 | 619.8 +/- 82.5 | 14.1x |

## Key Findings

1. **Compiled MH achieves 3.0x speedup over GFI MH.**

2. **Score function compilation gives 40.6x speedup.**

3. **GenMLX HMC overhead vs handcoded: 3.23x.**
   Both use fused leapfrog (L+1 gradient evals for L steps).

4. **Vectorized 10-chain MH gives 14.1x speedup.**
