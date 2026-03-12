# Experiment 3B: HMM — IS vs SMC

**Date:** 2026-03-04
**Model:** 2-state Gaussian-emission HMM, T=50 timesteps
**Transition:** A = [[0.9, 0.1], [0.1, 0.9]] (sticky)
**Emission:** y_t | z_t=k ~ N(mu_k, 1.0), mu = [-2, 2]
**Exact log P(y):** -77.1351 (forward algorithm)

## Methods

- **IS:** GenMLX GFI `p/generate` on Unfold combinator (all T observations at once)
- **Vectorized IS:** Shape-based batched `vgenerate` on flat model — single model call
- **SMC:** Unfold combinator + `smc-unfold` (sequential, one observation per step)
- **Batched SMC:** `batched-smc-unfold` (one vgenerate per timestep for all particles)

## Results (10 runs each)

| Method | log-ML (mean +/- std) | |Error| (mean +/- std) | ESS (mean) | Time (ms) |
|--------|----------------------|----------------------|------------|----------|
| IS (N=1000) | -96.87 +/- 7.45 | 19.74 +/- 7.45 | 1.2 | 63413 |
| Vec IS (N=1000) | -99.14 +/- 8.47 | 22.00 +/- 8.47 | 1.1 | 84 |
| SMC (N=100) | -77.07 +/- 0.58 | 0.45 +/- 0.37 | 100.0 | 22419 |
| SMC (N=250) | -77.33 +/- 0.53 | 0.41 +/- 0.39 | 250.0 | 49337 |
| Batched SMC (N=100) | -77.29 +/- 0.55 | 0.45 +/- 0.35 | 100.0 | 296 |
| Batched SMC (N=250) | -77.08 +/- 0.47 | 0.41 +/- 0.24 | 250.0 | 318 |
| Batched SMC (N=1000) | -77.22 +/- 0.14 | 0.12 +/- 0.11 | 1000.0 | 319 |

## Interpretation

IS with prior proposals suffers exponential weight degeneracy for sequential models (T=50 timesteps). Even with 1000 particles, ESS collapses near 1 and log-ML estimates are highly variable.

Vectorized IS runs the flat model body ONCE for all 1000 particles using shape-based batching, achieving massive speedup over sequential IS. Note: ESS is still ~1 due to inherent weight degeneracy — vectorization speeds up computation, not statistical efficiency.

SMC via the Unfold combinator exploits the sequential structure: at each timestep, only one new observation is incorporated, and resampling prevents catastrophic weight collapse. With just 100 particles, SMC achieves dramatically lower log-ML error than IS with 1000 particles.

This demonstrates that combinators enable efficient SMC — algorithmic structure matters far more than brute-force particle count for sequential models.
