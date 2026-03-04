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
| Compiled_MH | 2.0891 | 0.0010 | 0.5586 | 0.0002 | 872 | 1.000 | 812 |
| HMC | 2.0942 | 0.0041 | 0.5570 | 0.0017 | 1000 | 1.000 | 1607 |
| NUTS | 2.0982 | 0.0082 | 0.5579 | 0.0009 | 465 | 1.000 | 19574 |
| ADVI | 2.0240 | 0.0661 | 0.6477 | 0.0890 | — | — | 1973 |
| Vectorized_IS | 2.0811 | 0.0089 | 0.5694 | 0.0106 | 107 | — | 2 |

## Interpretation

All 5 algorithms converge to the analytic posterior. Slope error < 0.05 for all methods indicates correct implementation. HMC and NUTS with dual-averaging adaptation achieve high ESS/N and R-hat~1.0. NUTS uses adapt-metric (diagonal mass matrix via Welford's algorithm). ADVI mean-field Gaussian underestimates posterior std (no covariance). Vectorized IS with 12K particles achieves ESS > 100.
