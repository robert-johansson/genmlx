# Experiment 3B: HMM — IS vs SMC

**Date:** 2026-03-04
**Model:** 2-state Gaussian-emission HMM, T=50 timesteps
**Transition:** A = [[0.9, 0.1], [0.1, 0.9]] (sticky)
**Emission:** y_t | z_t=k ~ N(mu_k, 1.0), mu = [-2, 2]
**Exact log P(y):** -86.7223 (forward algorithm)

## Methods

- **IS:** GenMLX GFI `p/generate` on Unfold combinator (all T observations at once)
- **Vectorized IS:** Shape-based batched `vgenerate` on flat model — single model call
- **SMC:** Unfold combinator + `smc-unfold` (sequential, one observation per step)
- **Batched SMC:** `batched-smc-unfold` (one vgenerate per timestep for all particles)

## Results (10 runs each)

| Method | log-ML (mean +/- std) | |Error| (mean +/- std) | ESS (mean) | Time (ms) |
|--------|----------------------|----------------------|------------|----------|
| IS (N=1000) | -95.67 +/- 3.94 | 8.95 +/- 3.94 | 2.5 | 20079 |
| Vec IS (N=1000) | -94.03 +/- 4.22 | 7.30 +/- 4.22 | 1.8 | 32 |
| SMC (N=100) | -87.82 +/- 1.08 | 1.28 +/- 0.86 | 100.0 | 5852 |
| SMC (N=250) | -87.92 +/- 0.51 | 1.20 +/- 0.51 | 250.0 | 14652 |
| Batched SMC (N=100) | -86.95 +/- 0.71 | 0.66 +/- 0.35 | 100.0 | 100 |
| Batched SMC (N=250) | -86.42 +/- 0.20 | 0.33 +/- 0.16 | 250.0 | 108 |
| Batched SMC (N=1000) | -86.69 +/- 0.15 | 0.12 +/- 0.08 | 1000.0 | 147 |

## Interpretation

IS with prior proposals suffers exponential weight degeneracy for sequential models (T=50 timesteps). Even with 1000 particles, ESS collapses near 1 and log-ML estimates are highly variable.

Vectorized IS runs the flat model body ONCE for all 1000 particles using shape-based batching, achieving massive speedup over sequential IS. Note: ESS is still ~1 due to inherent weight degeneracy — vectorization speeds up computation, not statistical efficiency.

SMC via the Unfold combinator exploits the sequential structure: at each timestep, only one new observation is incorporated, and resampling prevents catastrophic weight collapse. With just 100 particles, SMC achieves dramatically lower log-ML error than IS with 1000 particles.

This demonstrates that combinators enable efficient SMC — algorithmic structure matters far more than brute-force particle count for sequential models.
