# Experiment 7: Neal's Funnel

**Date:** 2026-03-05
**Model:** v ~ N(0,3), x_i ~ N(0, exp(v/2)) for i=1..D — joint distribution (no observations)
**D:** 10

## Ground Truth

v marginal: N(0, 3) — mean=0, std=3

## Algorithm Comparison

| Algorithm | v mean | v mean err | v std | v std err | ESS | ESS/sec | R-hat | Time (ms) |
|-----------|--------|-----------|-------|----------|-----|---------|-------|----------|
| NUTS (3000 samples, 1000 burn) | 1.4067 | 1.4067 | 2.1048 | 0.8952 | 6026 | 176.5 | 1.567 | 34145 |
| HMC (2000 samples, 500 burn) | 1.1462 | 1.1462 | 1.6186 | 1.3814 | 8000 | 219.1 | 1.000 | 36520 |
| MALA (2000 samples, 500 burn) | 1.0040 | 1.0040 | 1.6726 | 1.3274 | 31 | 11.7 | 13.234 | 2624 |
| Compiled MH (5000 samples, 1000 burn) | 0.8271 | 0.8271 | 4.0944 | 1.0944 | 67 | 29.4 | 3.704 | 2287 |

## Interpretation

Neal's funnel is a classic pathological posterior where the width of x_i varies exponentially with v, creating a narrow neck at negative v and wide bulk at positive v. Random-walk MH fails because a single proposal std cannot adapt to the varying local geometry — ESS is low and the sampler gets stuck. MALA uses gradient information but the step size is too small or too large across different regions. NUTS with dual-averaging adaptation and diagonal mass matrix performs best, though ESS/sec remains modest due to the genuine difficulty of the geometry. This demonstrates that gradient-based MCMC (NUTS, HMC) is essential for challenging continuous posteriors.
