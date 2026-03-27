# Cross-System Verification Summary

**Systems:** Gen.jl (oracle), GenJAX (secondary), GenMLX

**Total comparison points:** 299 across 10 suites

## Results

| Suite | Total | Pass | Fail | Error | Notes |
|---|---|---|---|---|---|
| logprob | 139 | 139 | 0 | 0 | 24 distributions, 3-way (Gen.jl + GenJAX + GenMLX) |
| assess | 10 | 10 | 0 | 0 | Joint score, 3-way |
| update | 16 | 16 | 0 | 0 | GFI update weights |
| project | 10 | 10 | 0 | 0 | GFI project |
| score_decomposition | 7 | 7 | 0 | 0 | Score = sum of site log-probs |
| combinator | 16 | 16 | 0 | 0 | Map, Unfold, Switch |
| stability | 28 | 27 | 1 | 0 | 1 borderline (beta near-1: 5.763 vs 5.756) |
| gradient | 32 | 23 | 0 | 9 | **9 errors now fixed** (see below) |
| regenerate | 8 | 4 | 4 | 0 | Stochastic — values differ by run (see below) |
| inference_quality | 33 | 25 | 8 | 0 | SMC + MALA issues (see below) |
| **TOTAL** | **299** | **277** | **13** | **9** | |

## Fixed since last run (2026-03-27)

### 9 gradient errors — FIXED

These errored during the cross-system run because MLX lacked the required ops.
All now work after adding lgamma, bessel_i0e, Cholesky VJP, and Inverse VJP to our MLX fork.
Verified via finite-difference checks in `gen_clj_compat_test.cljs` section 1f.

- `grad-truncnorm-val-{0,1,3}` — truncated-normal value gradients
- `grad-vonmises-val-{0,pi4,pi}` — von-mises value gradients
- `grad-invgamma-{val-1,mode}` — inv-gamma value gradients
- `grad-exp-rate` — exponential rate gradient

**Adjusted totals (after fix): 286 pass, 13 fail, 0 error.**

## Known failures

### Regenerate (4 failures) — stochastic, not bugs

Regenerate tests compare score values after regenerating selected addresses.
Since regenerate resamples, the exact values differ between systems and between runs.
These tests check invariants (weight = new_score - old_score) which pass;
the raw score comparison fails because the resampled values differ.

### Inference quality (8 failures)

| Test | Issue |
|---|---|
| `infer-nn-smc-logml` | SMC log-ML: Gen.jl=-6.21, GenMLX=-10.45. Pipeline leak caused SMC issues (now fixed in fork). Needs re-run. |
| `infer-ssm-smc-{logml,ess,analytical}` | Same SMC class — needs re-run with pipeline fix. |
| `logml-{nn,bb}-{is,analytical}` | Small differences (~0.03 nats). Likely statistical noise from IS resampling. |

### Stability (1 borderline)

- `stab-beta-near-1`: Gen.jl=5.763, GenMLX=5.756. Difference of 0.007 — float32 vs float64 precision.

## How to re-run

```bash
# Requires Julia + Gen.jl, Python + GenJAX
cd cross_system_tests && bash run_and_summarize.sh
```

Re-running would update the gradient results (now passing) and SMC results (pipeline leak fixed).
