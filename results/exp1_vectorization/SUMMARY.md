# Experiment 1: Vectorized Inference Speedup

**Date:** 2026-03-03
**Platform:** macOS, Apple Silicon, MLX GPU via @frost-beta/mlx, Bun + nbb
**Benchmark file:** `test/genmlx/paper_bench_vectorization.cljs`
**Methodology:** performance.now(), warmup, min-of-inner, mean+std-of-outer

## Benchmark 1a: Particle Scaling

| N | Sequential (ms) | Batched (ms) | Speedup |
|---|----------------|--------------|--------|
| 1 | 0.8 +/- 0.1 | 0.76 +/- 0.02 | 1.1x |
| 10 | 6.9 +/- 0.1 | 0.67 +/- 0.02 | 10.4x |
| 100 | 110.7 +/- 0.5 | 0.82 +/- 0.03 | 134.9x |
| 500 | 657.2 +/- 1.2 | 0.87 +/- 0.03 | 757.7x |
| 1000 | 1399.6 +/- 2.4 | 0.87 +/- 0.04 | 1603.2x |

## Benchmark 1b: Method Speedup (N=1000)

| Method | Sequential (ms) | Batched (ms) | Speedup |
|--------|----------------|--------------|--------|
| dist-sample-n | 251.1 +/- 3.6 | 0.25 +/- 0.01 | 1018.3x |
| importance_sampling | 1618.3 +/- 2.3 | 0.80 +/- 0.02 | 2014.5x |
| smc_init | 1182.7 +/- 3.5 | 0.77 +/- 0.03 | 1539.4x |
