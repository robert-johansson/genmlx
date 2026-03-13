# Experiment 1: Vectorized Inference Speedup

**Date:** 2026-03-03
**Platform:** macOS, Apple Silicon, MLX GPU via @frost-beta/mlx, Bun + nbb
**Benchmark file:** `test/genmlx/paper_bench_vectorization.cljs`
**Methodology:** performance.now(), warmup, min-of-inner, mean+std-of-outer

## Benchmark 1a: Particle Scaling

| N | Sequential (ms) | Batched (ms) | Speedup |
|---|----------------|--------------|--------|
| 1 | 0.7 +/- 0.1 | 0.62 +/- 0.03 | 1.2x |
| 10 | 6.4 +/- 0.3 | 0.56 +/- 0.02 | 11.4x |
| 100 | 114.4 +/- 2.5 | 0.66 +/- 0.00 | 172.3x |
| 500 | 616.0 +/- 4.3 | 0.69 +/- 0.02 | 898.2x |
| 1000 | 1280.7 +/- 2.8 | 0.68 +/- 0.04 | 1875.0x |

## Benchmark 1b: Method Speedup (N=1000)

| Method | Sequential (ms) | Batched (ms) | Speedup |
|--------|----------------|--------------|--------|
| dist-sample-n | 260.4 +/- 7.8 | 0.23 +/- 0.00 | 1116.5x |
| importance_sampling | 1493.6 +/- 2.5 | 0.64 +/- 0.02 | 2350.3x |
| smc_init | 1288.2 +/- 2.0 | 0.62 +/- 0.03 | 2089.2x |
