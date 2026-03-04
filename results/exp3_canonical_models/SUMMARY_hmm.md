# Experiment 3B: HMM — IS vs SMC

**Date:** 2026-03-03
**Model:** 2-state Gaussian-emission HMM, T=50 timesteps
**Transition:** A = [[0.9, 0.1], [0.1, 0.9]] (sticky)
**Emission:** y_t | z_t=k ~ N(mu_k, 1.0), mu = [-2, 2]
**Exact log P(y):** -84.7858 (forward algorithm)

## Methods

- **IS:** GenMLX GFI `p/generate` on Unfold combinator (all T observations at once)
- **SMC:** Unfold combinator + `smc-unfold` (sequential, one observation per step)

## Results (10 runs each)

| Method | log-ML (mean +/- std) | |Error| (mean +/- std) | ESS (mean) | Time (ms) |
|--------|----------------------|----------------------|------------|----------|
| IS (N=1000) | -124.92 +/- 10.17 | 40.14 +/- 10.17 | 1.2 | 19819 |
| SMC (N=100) | -86.98 +/- 2.94 | 2.19 +/- 2.94 | 100.0 | 5846 |
| SMC (N=250) | -85.55 +/- 0.37 | 0.77 +/- 0.37 | 250.0 | 14711 |

## Interpretation

IS with prior proposals suffers exponential weight degeneracy for sequential models (T=50 timesteps). Even with 1000 particles, ESS collapses near 1 and log-ML estimates are highly variable.

SMC via the Unfold combinator exploits the sequential structure: at each timestep, only one new observation is incorporated, and resampling prevents catastrophic weight collapse. With just 100 particles, SMC achieves dramatically lower log-ML error than IS with 1000 particles.

This demonstrates that combinators enable efficient SMC — algorithmic structure matters far more than brute-force particle count for sequential models.
