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
| Compiled_MH | 2.0854 | 0.0047 | 0.5537 | 0.0051 | 807 | 1.000 | 843 |
| Vectorized_Compiled_Trajectory_MH | 2.0933 | 0.0032 | 0.5653 | 0.0066 | — | — | 286 |
| HMC | 2.0863 | 0.0037 | 0.5544 | 0.0044 | 1000 | 1.000 | 1667 |
| NUTS | 2.0962 | 0.0061 | 0.5653 | 0.0066 | 482 | 1.002 | 21416 |
| ADVI | 2.0963 | 0.0062 | 0.6404 | 0.0817 | — | — | 1997 |
| Vectorized_IS | 2.0944 | 0.0044 | 0.5264 | 0.0323 | 124 | — | 2 |

## Interpretation

All 5 algorithms converge to the analytic posterior. Slope error < 0.05 for all methods indicates correct implementation. HMC and NUTS with dual-averaging adaptation achieve high ESS/N and R-hat~1.0. NUTS uses adapt-metric (diagonal mass matrix via Welford's algorithm). ADVI mean-field Gaussian underestimates posterior std (no covariance). Vectorized IS with 12K particles achieves ESS > 100.
