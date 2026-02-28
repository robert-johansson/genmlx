# Property-Based Testing Expansion Plan

## Status: PHASE 1 COMPLETE — PHASE 2 NOT STARTED

Phase 1 expanded from ~80 properties (5 files) to ~162 properties (9 files).
Current coverage is ~30% of public API surface. Significant gaps remain.

---

## Phase 1 Results (COMPLETE)

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

---

## Phase 2 Gap Analysis

### Critical Gaps (0% property coverage)

#### MCMC Algorithms — `inference/mcmc.cljs`
Untested: `mh-step`, `mh-custom-step`, `mh-custom`, `compiled-mh`, `vectorized-compiled-mh`
Gradient-based: MALA, HMC, NUTS (most sophisticated MCMC)
Discrete: Gibbs, involutive MH
Specialized: Elliptical Slice, MAP optimization
**Impact:** ~40% of inference LOC, 0% property coverage.

#### Variational Inference — `inference/vi.cljs`
Untested: `vi` (ADVI), `compiled-vi`, `vectorized-compiled-vi`, `vi-from-model`,
`elbo-objective`, `iwelbo-objective`, `pwake-objective`, `qwake-objective`,
`reinforce-estimator`, `vimco-objective`, `vimco`, `programmable-vi`
**Impact:** 13 public functions, second-most important algorithm class, 0% coverage.

#### SMCP3 — `inference/smcp3.cljs`
Untested: `smcp3-init`, `smcp3-step`, `smcp3`, `smcp3-with-guidance`
**Impact:** "Most powerful inference algorithm" per design docs. 0% coverage.

#### Edit Interface — `edit.cljs`
Untested: `edit-dispatch`, `->ConstraintEdit`, `->SelectionEdit`, `->ProposalEdit`,
`constraint-edit`, `selection-edit`, `proposal-edit`
**Impact:** GenJAX's distinguishing feature. Backward-request semantics unverified.

#### Diff Tracking — `diff.cljs`
Untested: `no-change`, `unknown-change`, `value-change`, `vector-diff`, `map-diff`,
`no-change?`, `compute-diff`, `compute-vector-diff`, `compute-map-diff`
**Impact:** Used by `update-with-diffs` for incremental MCMC. Correctness unverified.

#### Handler System — `handler.cljs`
Untested: `run-handler`, all batched handler transitions, `update-with-diffs` impl
**Impact:** Heart of GenMLX. Batched execution variants only tested indirectly.

### High Priority Gaps

#### SMC Pipeline — `inference/smc.cljs`
Only resampling indices tested. Missing: `smc-init-step`, `smc-rejuvenate`, `smc-step`,
`smc`, `conditional-smc`, residual/stratified resampling variants.

#### ADEV — `inference/adev.cljs`
Untested: `has-reparam?`, `adev-transition`, `adev-execute`, `adev-loss`, `adev-optimize`

#### Combinator assess — `combinators.cljs`
Mix, Recurse, Contramap, Dimap all missing `assess` and some `project` properties.

### Medium Priority Gaps

#### NN Integration — `nn.cljs`
Layer constructors, `NeuralNetGF` record, gradient integration untested.

#### Verify/Contracts — `verify.cljs`, `contracts.cljs`
`validate-gen-fn`, contract registry, violation checkers untested.

### Low Priority Gaps

#### PRNG — `mlx/random.cljs`
`split`/`split-n` key splitting tested only indirectly.

#### Multivariate distributions
Only scalar distributions tested. Any multivariate variants untested.

---

## Phase 2 Proposed Files (if pursued)

### Tier 1 — Most tractable, highest value
1. `edit_diff_property_test.cljs` (~15 properties) — edit dispatch, diff computation, round-trips
2. `smc_property_test.cljs` (~12 properties) — SMC init/step, residual/stratified resampling, rejuvenation
3. `mcmc_property_test.cljs` (~15 properties) — mh-step, mh-custom, Gibbs, Elliptical Slice basic contracts

### Tier 2 — Important but harder (stochastic convergence)
4. `vi_property_test.cljs` (~12 properties) — ADVI, ELBO finite, programmable VI
5. `smcp3_property_test.cljs` (~10 properties) — init/step/pipeline, backward proposals
6. `gradient_mcmc_property_test.cljs` (~10 properties) — MALA/HMC/NUTS produce finite scores, correct shapes

### Tier 3 — Completeness
7. `handler_property_test.cljs` (~8 properties) — batched transitions, device management
8. `adev_property_test.cljs` (~6 properties) — reparam detection, surrogate losses

---

## Phase 1 Notes

- `mx/compile-fn` wrapping `mx/grad` or `mx/value-and-grad` produces zero gradients for handler-based models (CLJS side effects invisible to MLX tracer). Used `make-score-fn` + `mx/grad` directly instead.
- `vgenerate` weight shape depends on model structure: only `[N]` when constrained sites depend on unconstrained ones (broadcasting). Used dependent model (`y ~ gaussian(x, 1)`) for vgenerate tests.
- Recurse combinator tested without `dyn/splice` to avoid `regenerate` complexity with nested traces.
