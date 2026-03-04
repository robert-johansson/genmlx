# Experiment 5: Verification and Correctness

**Date:** 2026-03-03
**Platform:** macOS, Apple Silicon, Bun + nbb

## Results

| Level | What | Count | Pass rate | Validates |
|-------|------|-------|-----------|-----------|
| Runtime contracts | GFI invariants (10 contracts x 5 models) | 64 checks | 100% | Measure-theoretic correctness |
| Gen.clj compat | Regression suite | 165 tests | 100% | Cross-implementation (Gen.clj) |
| GenJAX compat | Regression suite | 73 tests | 100% | Cross-implementation (GenJAX) |

**Total: 302 checks, 0 failures.**

## GFI Contract Details

10 contract types tested across 5 model architectures:

**Contracts:**
1. `simulate` — produces valid trace with choices, gen-fn, and finite score
2. `generate(all)` — weight equals score when all choices constrained
3. `generate(empty)` — weight is 0 when no choices constrained
4. `assess` — weight matches generate score
5. `update(same)` — weight is 0 when replacing with same values
6. `update` — sets new values and supports round-trip recovery
7. `regenerate(none)` — weight is 0 and preserves existing values
8. `propose` — weight is finite
9. `project(all)` — projecting all choices equals score
10. `project(none)` — projecting no choices equals 0

**Models:**
1. `single-site` — one Gaussian latent
2. `multi-site` — two dependent Gaussians
3. `linreg` — linear regression with 3 observations
4. `splice` — nested generative function call
5. `mixed` — combination of trace sites and splice

## Compatibility Suite Details

**Gen.clj (165 tests):** Port of the Gen.clj test suite covering choicemaps,
traces, selections, distributions (log-pdf, sampling, parameter ranges),
generative function operations, and end-to-end model execution.

**GenJAX (73 tests):** Port of GenJAX compatibility tests covering distribution
correctness, multivariate distributions, hierarchical models, generate/update
consistency, score consistency, and combinator behavior.
