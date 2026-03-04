# Experiment 3B: HMM — IS vs SMC

**Date:** 2026-03-03
**Model:** 2-state Gaussian-emission HMM, T=50 timesteps
**Transition:** A = [[0.9, 0.1], [0.1, 0.9]] (sticky)
**Emission:** y_t | z_t=k ~ N(mu_k, 1.0), mu = [-2, 2]
**Exact log P(y):** -85.9017 (forward algorithm)

## Methods

- **IS:** GenMLX GFI `p/generate` on Unfold combinator (all T observations at once)
- **SMC:** Unfold combinator + `smc-unfold` (sequential, one observation per step)
- **Batched SMC:** `batched-smc-unfold` (one vgenerate per timestep for all particles)

## Results (10 runs each)

| Method | log-ML (mean +/- std) | |Error| (mean +/- std) | ESS (mean) | Time (ms) |
|--------|----------------------|----------------------|------------|----------|
| IS (N=1000) | -105.94 +/- 11.27 | 20.04 +/- 11.27 | 1.0 | 20104 |
| SMC (N=100) | -86.54 +/- 0.97 | 0.81 +/- 0.84 | 100.0 | 5869 |
| SMC (N=250) | -86.52 +/- 0.54 | 0.65 +/- 0.51 | 250.0 | 14697 |
| Batched SMC (N=100) | -86.26 +/- 0.85 | 0.67 +/- 0.64 | 100.0 | 100 |
| Batched SMC (N=250) | -85.86 +/- 0.34 | 0.30 +/- 0.18 | 250.0 | 106 |
| Batched SMC (N=1000) | -85.98 +/- 0.17 | 0.15 +/- 0.12 | 1000.0 | 147 |

## Interpretation

IS with prior proposals suffers exponential weight degeneracy for sequential models (T=50 timesteps). Even with 1000 particles, ESS collapses near 1 and log-ML estimates are highly variable.

SMC via the Unfold combinator exploits the sequential structure: at each timestep, only one new observation is incorporated, and resampling prevents catastrophic weight collapse. With just 100 particles, SMC achieves dramatically lower log-ML error than IS with 1000 particles.

This demonstrates that combinators enable efficient SMC — algorithmic structure matters far more than brute-force particle count for sequential models.
