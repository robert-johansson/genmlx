# Experiment 1: Vectorized Inference Speedup

**Date:** 2026-03-03
**Platform:** macOS, Apple Silicon, MLX GPU via @frost-beta/mlx, Bun + nbb
**Benchmark file:** `test/genmlx/paper_bench_vectorization.cljs`
**Methodology:** performance.now(), warmup, min-of-inner, mean+std-of-outer

## Benchmark 1a: Particle Scaling

| N | Sequential (ms) | Batched (ms) | Speedup |
|---|----------------|--------------|--------|
| 1 | 0.8 +/- 0.2 | 0.64 +/- 0.06 | 1.3x |
| 10 | 6.5 +/- 0.5 | 0.57 +/- 0.05 | 11.3x |
| 100 | 188.7 +/- 32.0 | 1.55 +/- 0.34 | 122.1x |
| 500 | 4350.3 +/- 1572.1 | 2.99 +/- 2.02 | 1453.1x |
| 1000 | 11976.5 +/- 712.9 | 3.12 +/- 1.11 | 3843.9x |

## Benchmark 1b: Method Speedup (N=1000)

| Method | Sequential (ms) | Batched (ms) | Speedup |
|--------|----------------|--------------|--------|
| dist-sample-n | 566.8 +/- 33.2 | 0.24 +/- 0.01 | 2391.5x |
| importance_sampling | 11542.6 +/- 885.4 | 1.48 +/- 0.59 | 7812.9x |
| smc_init | 4917.4 +/- 523.4 | 2.07 +/- 0.86 | 2372.8x |
