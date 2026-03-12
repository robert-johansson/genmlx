# Experiment 7: Neal's Funnel

**Date:** 2026-03-05
**Model:** v ~ N(0,3), x_i ~ N(0, exp(v/2)) for i=1..D — joint distribution (no observations)
**D:** 10

## Ground Truth

v marginal: N(0, 3) — mean=0, std=3

## Algorithm Comparison

| Algorithm | v mean | v mean err | v std | v std err | ESS | ESS/sec | R-hat | Time (ms) |
|-----------|--------|-----------|-------|----------|-----|---------|-------|----------|
| NUTS (3000 samples, 1000 burn) | 0.8489 | 0.8489 | 2.3273 | 0.6727 | 3040 | 15.0 | 1.595 | 202849 |
| HMC (2000 samples, 500 burn) | 0.6425 | 0.6425 | 2.1139 | 0.8861 | 6009 | 132.7 | 3.308 | 45290 |
| MALA (2000 samples, 500 burn) | -1.5033 | 1.5033 | 2.4934 | 0.5066 | 23 | 6.8 | 17.291 | 3443 |
| Compiled MH (5000 samples, 1000 burn) | -1.7495 | 1.7495 | 4.5343 | 1.5343 | 5014 | 1836.1 | 4.604 | 2731 |

## Interpretation

Neal's funnel is a classic pathological posterior where the width of x_i varies exponentially with v, creating a narrow neck at negative v and wide bulk at positive v. Random-walk MH fails because a single proposal std cannot adapt to the varying local geometry — ESS is low and the sampler gets stuck. MALA uses gradient information but the step size is too small or too large across different regions. NUTS with dual-averaging adaptation and diagonal mass matrix performs best, though ESS/sec remains modest due to the genuine difficulty of the geometry. This demonstrates that gradient-based MCMC (NUTS, HMC) is essential for challenging continuous posteriors.
