# GenMLX Testing Strategy

> Systematic plan for comprehensive correctness verification.
> Goal: find bugs through mathematical properties, not just happy-path checks.

---

## Strategy 1: Property-Based / Statistical Tests

*Sample many times, verify empirical statistics match theoretical predictions.
These catch numerical bugs, off-by-one errors in log-prob, and wrong
parameterizations.*

### Distribution statistics
- For every distribution: sample 10,000 times, verify E[X] and Var[X]
  within tolerance of known analytical values
- Discrete distributions: verify PMF sums to 1 (enumerate small supports)
- Continuous distributions: verify CDF at known quantiles via empirical CDF
- Log-prob spot checks: known (value, params) → known log-density

### GFI weight correctness
- `generate` weight should equal `log p(obs)` for models with known
  marginal likelihood (conjugate models, single-site models)
- `update` weight: should equal `score(new_trace) - score(old_trace)` for
  the changed addresses (verify on models where this is analytically tractable)
- `regenerate` weight: run long MH chains using regenerate, verify ergodicity
  by checking that the stationary distribution matches the model posterior
- `assess` weight: for fully-constrained models, should equal total log-joint

### Broadcasting equivalence
- `vsimulate` with N=1000 should produce marginal statistics (mean, variance
  per address) matching 1000 sequential `simulate` calls
- `vgenerate` weights should have same distribution as sequential `generate` weights
- Per-address log-probs in batched mode should match scalar mode

---

## Strategy 2: Round-Trip / Identity Tests

*Operations that should be no-ops or self-inverses. These catch state
corruption, incorrect discard computation, and weight accounting errors.*

### Update identity
- `update(trace, empty_constraints)` → same trace, weight 0
- `update(trace, all_choices_from_trace)` → same trace, weight 0
- `update(trace, constraints)` then `update(new_trace, discard)` → original trace

### Regenerate identity
- `regenerate(trace, empty_selection)` → same trace, weight 0
- `regenerate(trace, all_selection)` → different trace (unless deterministic)

### Edit duality
- `edit(trace, ConstraintEdit(c))` then `edit(trace', backward_req)` →
  recovers original trace, weights sum to 0
- `edit(trace, SelectionEdit(s))` → backward request is same selection
- `edit(trace, ProposalEdit(f, b))` → backward request is `ProposalEdit(b, f)`

### Project / score consistency
- `project(trace, select_all)` should equal `score(trace)`
- `project(trace, empty_selection)` should equal 0

### Simulate / generate consistency
- `simulate` then `update` with same choices → weight 0, identical trace
- `simulate` then `generate` with all choices → weight equals score

---

## Strategy 3: Cross-Validation Between GFI Operations

*Different operations computing the same quantity should agree. These catch
bugs where one operation is correct but another has a sign error, missing
term, or wrong dispatch.*

### assess vs generate
- `assess(model, args, choices).weight` should equal
  `score` of `generate(model, args, choices).trace`
- Both should equal the log-joint evaluated at those choices

### propose vs simulate
- `propose(model, args).choices` fed back to `generate` should give
  `generate_weight = 0` (proposal matches model prior)
- `propose` weight should equal `simulate` score

### project vs score decomposition
- Sum of `project(trace, sel_i)` over a partition of all addresses
  should equal `score(trace)` (when partition covers all addresses)
- `project` on single address should equal that address's log-prob

### update vs fresh generate
- `update(trace, constraints)` weight + `score(old_trace)` should equal
  `score(new_trace)` (for the standard case without proposal corrections)
- `generate(model, args, full_choices)` should produce same trace as
  `update(simulate_trace, full_choices)`

### regenerate vs update equivalence
- `regenerate(trace, selection)` should be equivalent to:
  sample new values for selected addresses, then `update` with those as constraints
  (modulo the proposal correction term)

---

## Strategy 4: Combinator Compositionality

*Every combinator must preserve all GFI contracts. Combinatorial testing:
each combinator × each GFI operation.*

### Degenerate cases (combinator reduces to kernel)
- `Map(kernel, n=1)` should behave identically to kernel itself
  (same trace structure nested under address 1, same score, same weight)
- `Unfold(kernel, n=1)` should behave like a single kernel step
- `Switch(g1, g2)` with constant index should equal the selected branch
- `Mask(kernel, true)` should equal kernel; `Mask(kernel, false)` should
  produce empty trace with score 0
- `Recurse(kernel)` with zero children should equal single kernel application

### Nested combinators
- `Map(Switch(...))` — verify trace structure, update across branch changes
- `Unfold(Map(...))` — temporal sequence of parallel models
- `Switch(Map(...), Unfold(...))` — heterogeneous branches with combinators
- `Scan(kernel)` with carry — verify carry propagation through update

### Score additivity
- `Map`: total score = sum of per-element scores
- `Unfold`: total score = sum of per-step scores
- `Switch`: total score = selected branch score
- `Mix`: total score = categorical log-prob + component score

### Update correctness per combinator
- `Map` + VectorDiff: only changed elements re-executed
- `Unfold` + prefix skip: unchanged prefix reuses cached scores
- `Switch` branch change: old branch discarded, new branch generated
- `Scan` carry propagation: change at step t forces re-execution from t onward

---

## Strategy 5: Differential Testing Against Gen.jl

*Same model, same observations → same mathematical results. Catches
semantic differences between GenMLX and the reference implementation.*

### Existing coverage
- 165 Gen.clj compatibility tests
- 73 GenJAX compatibility tests

### Systematic expansion
- For every distribution: verify log-prob matches Gen.jl's `logpdf` at
  5+ test points (including boundary values, zero, negative, large)
- For every GFI operation × every distribution: generate with known
  constraints, compare weight
- Edge cases: empty choice maps, single-address models, models with
  only deterministic computation, models with splice

### Cross-implementation weight comparison
- Define 5 canonical models in both GenMLX and Gen.jl
- For each: fix random choices, compute generate/update/regenerate weights
- Verify agreement to numerical precision (~1e-6)

---

## Strategy 6: Inference Algorithm Convergence Tests

*Run inference on models with known posteriors. Verify the algorithm
converges to the correct answer. Catches bugs in weight computation,
acceptance logic, and resampling.*

### Conjugate pair models (analytical posterior)

**Beta-Bernoulli:**
- Prior: Beta(α, β), Likelihood: Bernoulli(p)
- Posterior: Beta(α + k, β + n - k) where k = number of successes
- Test: IS, MH, HMC, SMC, VI → posterior mean within tolerance of (α+k)/(α+β+n)

**Normal-Normal (known variance):**
- Prior: N(μ₀, σ₀²), Likelihood: N(μ, σ²) with n observations
- Posterior: N(μ_post, σ_post²) with known formulas
- Test: all algorithms → posterior mean and variance within tolerance

**Gamma-Poisson:**
- Prior: Gamma(α, β), Likelihood: Poisson(λ)
- Posterior: Gamma(α + Σx_i, β + n)
- Test: IS, MH, SMC → posterior mean within tolerance

### Algorithm-specific checks

**MH:** acceptance rate should be between 0.1 and 0.9 for well-tuned proposals
**HMC:** acceptance rate should be > 0.6 with good step size
**NUTS:** no divergent transitions on simple models
**SMC:** log-ML estimate should be close to analytical value for conjugate models
**VI:** ELBO should converge and final KL divergence should be small
**Gibbs:** each conditional should match the full conditional distribution

### Mass matrix effectiveness (14.1)
- Anisotropic Gaussian: HMC with correct mass matrix should have higher
  acceptance rate and lower autocorrelation than identity metric
- Compare ESS/second between identity, diagonal, and dense metrics

### Resampling method comparison (14.3, 14.4)
- All three methods (systematic, residual, stratified) should produce
  correct posterior on conjugate models
- Stratified and residual should have lower variance in log-ML estimates
  than systematic (verify on 100+ independent SMC runs)

---

## Implementation Plan

### Recommended order (maximum coverage, minimum code)

**Phase A: Cross-validation harness (~100 lines)**

Write a single test harness function that takes any `(model, args, observations)`
triple and runs ~15 cross-validation checks across all GFI operations:

```
(defn verify-gfi-contract [model args observations]
  ;; 1. simulate → trace
  ;; 2. generate with all choices → weight = score
  ;; 3. assess with all choices → weight = score
  ;; 4. update with empty → same trace, weight 0
  ;; 5. update with all choices → same trace, weight 0
  ;; 6. regenerate with empty → same trace, weight 0
  ;; 7. project(all) = score
  ;; 8. project(empty) = 0
  ;; 9. propose → generate with choices → weight 0
  ;; 10. update round-trip via discard
  ;; 11. edit(constraint) round-trip
  ;; 12. edit(selection) backward = same selection
  ;; 13. score decomposition via project
  ;; 14. generate weight ≈ analytical (if known)
  ;; 15. broadcasting equivalence (if applicable)
  )
```

**Phase B: Canonical model suite (~80 lines)**

Define 10-15 models covering all features:

1. Single Gaussian (1 address)
2. Multi-address (5 independent Gaussians)
3. Dependent addresses (linear regression)
4. Discrete model (Bernoulli)
5. Mixed discrete/continuous (mixture)
6. Nested splice (sub-GF)
7. Map combinator
8. Unfold combinator
9. Switch combinator
10. Scan combinator
11. Mask combinator
12. Mix combinator
13. Recurse combinator
14. Deep nesting (Map(Switch(...)))
15. Model with `param`

**Phase C: Run harness on every model**

Apply the harness to all 15 models → 15 × 15 = 225+ test assertions
from ~180 lines of code.

**Phase D: Distribution exhaustive tests (~150 lines)**

For all 27 distributions:
- Sample statistics (mean, variance)
- Log-prob spot checks (5 points each)
- Boundary values
- Batch sampling equivalence (`dist-sample-n` vs sequential)

**Phase E: Inference convergence tests (~120 lines)**

3 conjugate models × 6 algorithms = 18 convergence tests.
Each verifies posterior mean within tolerance of analytical value.

**Phase F: Combinator-specific tests (~100 lines)**

Degenerate cases, nested combinators, score additivity, diff-aware update.

### Expected coverage

| Phase | Tests | Lines | Catches |
|-------|-------|-------|---------|
| A+B+C | ~225 | ~180 | GFI contract violations, weight bugs, state corruption |
| D | ~135 | ~150 | Distribution parameterization, log-prob errors |
| E | ~18 | ~120 | Inference algorithm bugs, acceptance logic |
| F | ~40 | ~100 | Combinator compositionality, diff tracking |
| **Total** | **~418** | **~550** | — |

This would bring total test assertions from ~640 to ~1060.
