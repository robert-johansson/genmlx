# Experiment 3B: HMM — IS vs SMC

**Date:** 2026-03-04
**Model:** 2-state Gaussian-emission HMM, T=50 timesteps
**Transition:** A = [[0.9, 0.1], [0.1, 0.9]] (sticky)
**Emission:** y_t | z_t=k ~ N(mu_k, 1.0), mu = [-2, 2]
**Exact log P(y):** -80.6073 (forward algorithm)

## Methods

- **IS:** GenMLX GFI `p/generate` on Unfold combinator (all T observations at once)
- **Vectorized IS:** Shape-based batched `vgenerate` on flat model — single model call
- **SMC:** Unfold combinator + `smc-unfold` (sequential, one observation per step)
- **Batched SMC:** `batched-smc-unfold` (one vgenerate per timestep for all particles)

## Results (10 runs each)

| Method | log-ML (mean +/- std) | |Error| (mean +/- std) | ESS (mean) | Time (ms) |
|--------|----------------------|----------------------|------------|----------|
| IS (N=1000) | -112.68 +/- 9.61 | 32.07 +/- 9.61 | 1.1 | 25789 |
| Vec IS (N=1000) | -109.13 +/- 6.86 | 28.52 +/- 6.86 | 1.2 | 49 |
| SMC (N=100) | -80.65 +/- 0.30 | 0.24 +/- 0.19 | 100.0 | 7131 |
| SMC (N=250) | -80.66 +/- 0.36 | 0.28 +/- 0.23 | 250.0 | 15191 |
| Batched SMC (N=100) | -80.93 +/- 0.53 | 0.50 +/- 0.37 | 100.0 | 130 |
| Batched SMC (N=250) | -80.92 +/- 0.40 | 0.38 +/- 0.34 | 250.0 | 132 |
| Batched SMC (N=1000) | -80.65 +/- 0.16 | 0.13 +/- 0.09 | 1000.0 | 150 |

## Interpretation

IS with prior proposals suffers exponential weight degeneracy for sequential models (T=50 timesteps). Even with 1000 particles, ESS collapses near 1 and log-ML estimates are highly variable.

Vectorized IS runs the flat model body ONCE for all 1000 particles using shape-based batching, achieving massive speedup over sequential IS. Note: ESS is still ~1 due to inherent weight degeneracy — vectorization speeds up computation, not statistical efficiency.

SMC via the Unfold combinator exploits the sequential structure: at each timestep, only one new observation is incorporated, and resampling prevents catastrophic weight collapse. With just 100 particles, SMC achieves dramatically lower log-ML error than IS with 1000 particles.

This demonstrates that combinators enable efficient SMC — algorithmic structure matters far more than brute-force particle count for sequential models.
