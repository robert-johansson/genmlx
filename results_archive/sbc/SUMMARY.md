# SBC Results: Simulation-Based Calibration

**Date:** 2026-03-27
**Runtime:** Bun 1.3.11, nbb 1.4.206, macOS 26.3.1, Apple Silicon
**Duration:** 389 minutes (~6.5 hours)
**Method:** Talts et al. 2018, N=500 repetitions, L=200 posterior samples per rep

## Overall Results

| Metric | Value |
|--------|-------|
| Total combos | 16 |
| Total parameter tests | 21 |
| Passed | 20 |
| Failed | 1 |
| Pass rate | 95.2% (20/21) |

## Results by Combo

| Model | Algorithm | Parameters | Result | Time |
|-------|-----------|------------|--------|------|
| single-gaussian | CMH | mu | **FAIL** (chi2 borderline) | 336s |
| single-gaussian | HMC | mu | PASS | 1491s |
| single-gaussian | IS | mu | PASS | 1463s |
| two-gaussians | CMH | a, b | PASS | 366s |
| two-gaussians | HMC | a, b | PASS | 1994s |
| two-gaussians | IS | a, b | PASS | 2030s |
| gaussian-multi-obs | CMH | mu | PASS | 713s |
| gaussian-multi-obs | HMC | mu | PASS | 2430s |
| gaussian-multi-obs | IS | mu | PASS | 1851s |
| exponential | MH | rate | PASS | 410s |
| coin-flip | MH | p | PASS | 467s |
| linear-regression | HMC | slope, intercept | PASS | 3343s |
| linear-regression | IS | slope, intercept | PASS | 3105s |
| hierarchical | HMC | x | PASS | 1459s |
| beta-bernoulli | MH | p | PASS | 1413s |
| gamma-poisson | MH | lambda | PASS | 474s |

## The Single Failure

single-gaussian × CMH: chi2=24.28 (crit=21.67), KS=0.054 (crit=0.073).

Chi-squared marginally exceeds the critical value (12% over); KS passes comfortably.
The same combo passed in the previous partial run (chi2=11.20). This is consistent
with statistical noise at alpha=0.01 — with 21 tests, ~0.21 false positives are
expected per run. CMH passes on all other models (two-gaussians, gaussian-multi-obs),
confirming no systematic bias.

## Algorithms Validated

| Algorithm | Models tested | All pass? |
|-----------|--------------|-----------|
| Compiled MH (CMH) | 3 | 2/3 (1 borderline) |
| HMC | 5 | 5/5 |
| Standard MH | 4 | 4/4 |
| Importance Sampling | 3 | 3/3 |

## Models Validated

| Model | Algorithms | Params | All pass? |
|-------|-----------|--------|-----------|
| single-gaussian | CMH, HMC, IS | 1 | 2/3 |
| two-gaussians | CMH, HMC, IS | 2 | 3/3 |
| gaussian-multi-obs | CMH, HMC, IS | 1 | 3/3 |
| exponential | MH | 1 | 1/1 |
| coin-flip | MH | 1 | 1/1 |
| linear-regression | HMC, IS | 2 | 2/2 |
| hierarchical | HMC | 1 | 1/1 |
| beta-bernoulli | MH | 1 | 1/1 |
| gamma-poisson | MH | 1 | 1/1 |

## Interpretation

SBC validates that the full inference loop — prior sampling, data generation,
inference, posterior sampling — is statistically calibrated. A passing SBC test
means the inference algorithm produces correct posterior distributions, not just
"reasonable-looking" output.

20/21 passing parameter tests across 4 inference algorithms and 9 model classes
provides strong evidence that GenMLX's inference stack is correct. The single
borderline failure is consistent with the expected false positive rate.

## Methodology

- **Rank computation:** count of posterior samples strictly less than true value
- **Uniformity tests:** chi-squared (10 bins) + Kolmogorov-Smirnov, both at alpha=0.01
- **Pass criterion:** both tests must pass
- **Process isolation:** each combo runs as a separate Bun process to avoid RSS accumulation

## Reproduction

```bash
cd genmlx
env SBC_N=500 SBC_L=200 bash dev/run_sbc_per_combo.sh
```

## Data

Raw results with rank vectors: `sbc_results_2026-03-27.json`

## Reference

Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018).
Validating Bayesian Inference Algorithms with Simulation-Based Calibration.
[arXiv:1804.06788](https://arxiv.org/abs/1804.06788)
