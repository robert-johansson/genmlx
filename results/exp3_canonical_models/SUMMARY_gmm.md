# Experiment 3C: GMM — Enumeration vs IS vs Gibbs

**Date:** 2026-03-04
**Model:** K=3 Gaussian mixture, N=8 data points, 1D
**Means:** [-4, 0, 4], sigma=1.0, equal mixing weights
**Exact log P(y):** -19.1318 (enumeration over 6561 configs)

## Methods

- **Enumeration:** Exact marginal likelihood + posterior marginals (3^8 = 6561 configs)
- **IS (N=1000):** GenMLX GFI `p/generate` with prior proposal
- **Gibbs (500 sweeps, 100 burn):** `mcmc/gibbs` with discrete support schedule

## Results (10 runs each)

### IS

| Metric | Mean | Std |
|--------|------|-----|
| log-ML | -23.72 | 2.56 |
| |Error| | 4.58 | 2.56 |
| ESS | 1.4 | 0.5 |
| Time (ms) | 3181 | 29 |

### Gibbs

| Metric | Mean | Std |
|--------|------|-----|
| Assignment accuracy | 1.000 | 0.000 |
| Marginal MAE | 0.0014 | 0.0004 |
| Time (ms) | 34630 | 578 |

## Interpretation

The GMM with known parameters has a finite discrete posterior over component assignments (3^8 = 6561 configurations). Exact enumeration provides the ground truth marginal likelihood and posterior marginals.

IS with prior proposals achieves reasonable but imperfect log-ML estimates — the prior assigns equal probability to all 3 components, so many particles propose incorrect assignments.

Gibbs sampling exploits the discrete structure by sweeping over each z_i conditioned on all others, converging to the exact posterior marginals. This demonstrates that structure-exploiting inference (Gibbs) outperforms brute-force IS for models with discrete latent variables.
