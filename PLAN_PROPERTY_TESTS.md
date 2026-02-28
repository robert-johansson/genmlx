# Property-Based Testing Expansion Plan

## Status: COMPLETE

Expanded from ~80 properties (5 files) to ~162 properties (9 files) across 4 new test files.

## Results

### File 1: `test/genmlx/inference_property_test.cljs` — 24/24 PASS
- [x] Importance Sampling (4): log-weights finite, log-ml finite, normalized weights sum to 1, empty constraints weight ~0
- [x] MH / Accept Decision (3): accept-mh?(0) true, accept-mh?(-100) rarely true, regenerate(sel/none) weight ~0
- [x] Kernel Composition (5): chain valid, repeat valid, seed valid, run-kernel count, cycle finite
- [x] Score Function Utilities (3): extract/make round-trip, mode > far, make-score-fn finite
- [x] Weight Utilities (4): normalize sums to 1, ESS in (0,N], uniform ESS=N, materialize finite
- [x] Diagnostics (3): ESS <= N, r-hat identical ~1.0, r-hat >= 1.0
- [x] SMC Resampling (2): systematic indices in [0,N), uniform weight valid

### File 2: `test/genmlx/vectorized_property_test.cljs` — 18/18 PASS
- [x] vsimulate Shape (4): score [N], choices [N], n-particles match, scores finite
- [x] vgenerate Shape (4): weight [N] (dep model), constrained scalar, unconstrained [N], empty weight scalar 0
- [x] Statistical Equivalence (2): mean score match, log-ML finite
- [x] VectorizedTrace Operations (4): ESS in (0,N], log-ml finite, resample weights, resample n-particles
- [x] Batched vs Scalar (4): N=1 shapes, all-N finite, resample shape

### File 3: `test/genmlx/combinator_extra_property_test.cljs` — 20/20 PASS
- [x] Mix Combinator (6): component-idx valid, score finite, generate/update/regenerate
- [x] Recurse Combinator (5): simulate finite, generate/update/regenerate/project
- [x] Contramap (5): identity score match, retval match, generate/update, structural
- [x] Dimap (4): score match, retval transformed, generate/update

### File 4: `test/genmlx/gradient_learning_property_test.cljs` — 20/20 PASS
- [x] Choice Gradients (4): all addrs, finite, shape scalar, mode ~0
- [x] Score Function Gradients (4): finite, shape, correct gaussian grad, autodiff ≈ numerical
- [x] Parameter Store (4): get/set, param-names, array round-trip, empty
- [x] Optimizers (4): SGD direction, magnitude, Adam loss decrease, state
- [x] Training Loop (4): history length, loss decrease, params move, SGD works

### Existing suite (no regressions)
- [x] choicemap_property_test: 11/11
- [x] selection_property_test: 11/11
- [x] gfi_property_test: 17/17
- [x] dist_property_test: 17/17
- [x] combinator_property_test: 15/15

## Notes

- `mx/compile-fn` wrapping `mx/grad` or `mx/value-and-grad` produces zero gradients for handler-based models (CLJS side effects invisible to MLX tracer). Used `make-score-fn` + `mx/grad` directly instead.
- `vgenerate` weight shape depends on model structure: only `[N]` when constrained sites depend on unconstrained ones (broadcasting). Used dependent model (`y ~ gaussian(x, 1)`) for vgenerate tests.
- Recurse combinator tested without `dyn/splice` to avoid `regenerate` complexity with nested traces.
