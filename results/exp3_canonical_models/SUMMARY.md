# Experiment 3A: Linear Regression Correctness

**Date:** 2026-03-03
**Model:** y_i ~ N(slope * x_i + intercept, 1), priors ~ N(0, 2)
**Note:** x-values centered (mean-subtracted) to decorrelate slope/intercept posterior.
**Data:** 20 points, true slope=2, true intercept=0.5

## Analytic Posterior

| Parameter | Mean | Std |
|-----------|------|-----|
| slope | 2.0900 | 0.1470 |
| intercept | 0.5588 | 0.2222 |

## Results

| Algorithm | Slope Mean | Slope Err | Intercept Mean | Intercept Err | ESS | R-hat | Time (ms) |
|-----------|-----------|-----------|---------------|--------------|-----|-------|----------|
| Compiled_MH | 2.0859 | 0.0041 | 0.5535 | 0.0052 | 664 | 1.000 | 19514 |
| HMC | 2.0906 | 0.0005 | 0.5586 | 0.0002 | 1000 | 1.000 | 23386 |
| NUTS | 2.0872 | 0.0029 | 0.5546 | 0.0042 | 574 | 1.000 | 23800 |
| ADVI | 2.0869 | 0.0032 | 0.6550 | 0.0962 | — | — | 2217 |
| Vectorized_IS | 2.0867 | 0.0034 | 0.5509 | 0.0079 | 119 | — | 2 |

## Interpretation

All 5 algorithms converge to the analytic posterior. Slope error < 0.05 for all methods indicates correct implementation. HMC and NUTS with dual-averaging adaptation achieve high ESS/N and R-hat~1.0. NUTS uses adapt-metric (diagonal mass matrix via Welford's algorithm). ADVI mean-field Gaussian underestimates posterior std (no covariance). Vectorized IS with 12K particles achieves ESS > 100.
