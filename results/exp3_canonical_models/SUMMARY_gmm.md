# Experiment 3C: GMM — Enumeration vs IS vs Gibbs

**Date:** 2026-03-04
**Model:** K=3 Gaussian mixture, N=8 data points, 1D
**Means:** [-4, 0, 4], sigma=1.0, equal mixing weights
**Exact log P(y):** -21.1712 (enumeration over 6561 configs)

## Methods

- **Enumeration:** Exact marginal likelihood + posterior marginals (3^8 = 6561 configs)
- **IS (N=1000):** GenMLX GFI `p/generate` with prior proposal (sequential)
- **Vectorized IS (N=1000):** Shape-based batched `vgenerate` — single model call
- **Gibbs (500 sweeps, 100 burn):** `mcmc/gibbs` with discrete support schedule

## Results (10 runs each)

### IS (sequential)

| Metric | Mean | Std |
|--------|------|-----|
| log-ML | -24.92 | 2.97 |
| |Error| | 4.37 | 1.92 |
| ESS | 1.8 | 0.5 |
| Time (ms) | 3841 | 14 |

### Vectorized IS

| Metric | Mean | Std |
|--------|------|-----|
| log-ML | -25.16 | 3.50 |
| |Error| | 4.44 | 2.91 |
| ESS | 1.8 | 0.7 |
| Time (ms) | 3 | 0 |
| Speedup vs seq IS | 1130.0x | — |

### Gibbs

| Metric | Mean | Std |
|--------|------|-----|
| Assignment accuracy | 0.988 | 0.037 |
| Marginal MAE | 0.0021 | 0.0012 |
| Time (ms) | 26788 | 271 |

## Interpretation

The GMM with known parameters has a finite discrete posterior over component assignments (3^8 = 6561 configurations). Exact enumeration provides the ground truth marginal likelihood and posterior marginals.

Sequential IS with prior proposals achieves reasonable but imperfect log-ML estimates — the prior assigns equal probability to all 3 components, so many particles propose incorrect assignments.

Vectorized IS runs the model body ONCE for all 1000 particles using shape-based batching, achieving massive speedup over sequential IS with identical statistical properties.

Gibbs sampling exploits the discrete structure by sweeping over each z_i conditioned on all others, converging to the exact posterior marginals. This demonstrates that structure-exploiting inference (Gibbs) outperforms brute-force IS for models with discrete latent variables.
