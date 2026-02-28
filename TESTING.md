# GenMLX Testing Strategy

> Systematic correctness verification through mathematical properties,
> cross-validation, and contract checking — not just happy-path checks.
>
> **Current state (Feb 2026):** ~750-800 unit test assertions + 575 GFI contract
> checks + 162 property tests = ~1,500+ total checks across 77 test files.

---

## Strategy 1: Property-Based / Statistical Tests — DONE

*Sample many times, verify empirical statistics match theoretical predictions.
Catches numerical bugs, off-by-one errors in log-prob, and wrong
parameterizations.*

### Distribution statistics — `dist_statistics_test.cljs`, `dist_property_test.cljs`

- [x] E[X] and Var[X] for 16 distributions (10,000 samples each): gaussian,
  uniform, bernoulli, beta, gamma, exponential, poisson, laplace, log-normal,
  geometric, neg-binomial, binomial, discrete-uniform, truncated-normal,
  student-t, inv-gamma + MVN and dirichlet component-wise means (~40 assertions)
- [x] Discrete PMF sums to 1 for 8 distributions: bernoulli×2, categorical,
  binomial, discrete-uniform, geometric, poisson, neg-binomial (~8 assertions)
- [x] Log-prob spot checks via property tests (17 distribution properties)
- [ ] Continuous CDF at known quantiles via empirical CDF (not yet implemented)

### GFI weight correctness — `gfi_contract_test.cljs`, `contracts.cljs`

- [x] `generate` weight = score when fully constrained (11 contracts × 13 models)
- [x] `update` weight = new_score - old_score (contract #3)
- [x] `assess` weight = total log-joint for fully-constrained models (contract #8)
- [x] `regenerate` with empty selection = identity, weight 0 (contract #5)

### Broadcasting equivalence — `contracts.cljs`, `vectorized_property_test.cljs`

- [x] `vsimulate(N)` statistically matches N `simulate` calls (contract #11)
- [x] Batched vs scalar weight distribution equivalence (18 vectorized properties)
- [x] Per-address log-probs in batched mode match scalar mode

---

## Strategy 2: Round-Trip / Identity Tests — DONE

*Operations that should be no-ops or self-inverses. Catches state corruption,
incorrect discard computation, and weight accounting errors.*

### Update identity — `gfi_contract_test.cljs`, `contracts.cljs`

- [x] `update(trace, empty_constraints)` → same trace, weight 0 (contract #2)
- [x] `update(trace, constraints)` then `update(trace', discard)` → original trace
  (contract #4, tested on 5 canonical models)

### Regenerate identity — `contracts.cljs`

- [x] `regenerate(trace, empty_selection)` → same trace, weight 0 (contract #5)

### Edit duality — `combinator_contract_test.cljs`

- [x] `edit(trace, ConstraintEdit(c))` round-trip recovers original trace
- [x] `edit(trace, SelectionEdit(s))` → backward request type check
- [x] `edit(trace, ProposalEdit(f, b))` → weight finiteness

### Project / score consistency — `contracts.cljs`

- [x] `project(trace, select_all)` = `score(trace)` (contract #6)
- [x] `project(trace, empty_selection)` = 0 (contract #7)

### Simulate / generate consistency — `contracts.cljs`

- [x] `simulate` then `generate` with all choices → weight = score (contract #1)

---

## Strategy 3: Cross-Validation Between GFI Operations — DONE

*Different operations computing the same quantity should agree. Catches bugs
where one operation is correct but another has a sign error, missing term,
or wrong dispatch.*

### assess vs generate — `gfi_contract_test.cljs`, `contracts.cljs`

- [x] `assess(model, args, choices).weight` = `generate` score (contract #8,
  tested on 5 canonical models: single-site, multi-site, linreg, splice, mixed)

### propose vs simulate — `gfi_contract_test.cljs`, `contracts.cljs`

- [x] `propose` choices fed back to `generate` → finite weight (contract #9)
- [x] Round-trip weight consistency

### project vs score decomposition — `contracts.cljs`

- [x] Sum of `project(trace, sel_i)` over address partition = `score(trace)`
  (contract #10)

### update vs fresh generate — `gfi_property_test.cljs`

- [x] Update weight + old score ≈ new score (17 GFI properties)

---

## Strategy 4: Combinator Compositionality — DONE

*Every combinator preserves all GFI contracts. Tested via combinator-specific
tests and the contract verification framework.*

### Degenerate cases — `combinator_contract_test.cljs`

- [x] `Map(kernel, n=1)` behaves identically to kernel
- [x] `Unfold(kernel, n=1)` behaves like single step
- [x] `Switch(g1, g2)` with constant index = selected branch
- [x] `Mask(kernel, true)` = kernel; `Mask(kernel, false)` = empty trace, score 0
- [x] `Scan(kernel, n=1)` = single step

### Nested combinators — `combinator_contract_test.cljs`

- [x] `Map(Switch(...))` — trace structure, update across branch changes
- [x] `Unfold(Mask(...))` — temporal sequence with masking
- [x] `Switch(Map, Map)` — heterogeneous branches

### Score additivity — `combinator_contract_test.cljs`

- [x] Map element-scores sum to total score
- [x] Unfold step-scores sum to total score
- [x] Switch branch score = total score
- [x] Scan step-scores sum to total score

### GFI contracts on combinator models — `contract_verification_test.cljs`

- [x] Map, Unfold, Switch, Scan, Mask, Recurse models all pass GFI contract
  verification (subset of 11 contracts appropriate to each combinator type)

---

## Strategy 5: Differential Testing Against Gen.jl — PARTIAL

*Same model, same observations → same mathematical results.*

### Existing coverage

- [x] 165 Gen.clj compatibility tests (`gen_clj_compat_test.cljs`)
- [x] 73 GenJAX compatibility tests (`genjax_compat_test.cljs`)

### Remaining (TODO Phase 23 in TODO.md)

- [ ] Julia script to generate `gen_jl_reference.json` with deterministic
  log-prob values and GFI operation outputs for 10 canonical models
- [ ] GenMLX test loader that compares against reference JSON (~185 assertions)
- [ ] Cross-process Gen.jl oracle for dynamic queries (future, if needed)

---

## Strategy 6: Inference Algorithm Convergence Tests — DONE

*Run inference on models with known posteriors. Verify convergence to the
correct answer.*

### Conjugate pair models — `inference_convergence_test.cljs`

- [x] **Gamma-Poisson:** IS (100 particles) and MH (500 samples) → posterior
  mean ≈ 3.0 for Gamma(3,1)/Poisson data
- [x] **Normal-Normal:** HMC and NUTS → posterior mean ≈ analytical, acceptance
  rate > 0.3, no NaN values
- [x] **Beta-Bernoulli:** tested via `conjugate_bb_test.cljs`, `conjugate_posterior_test.cljs`

### Algorithm-specific checks — `inference_property_test.cljs`, various test files

- [x] MH acceptance rate in valid range (24 inference properties)
- [x] HMC acceptance rate > 0.6 with good step size (`adaptive_hmc_test.cljs`)
- [x] NUTS no divergent transitions on simple models (`adaptive_nuts_test.cljs`)
- [x] SMC weight finiteness and resampling correctness
- [x] VI ELBO convergence (`adev_test.cljs`, `vimco_test.cljs`)

### Mass matrix and adaptive tuning — `adaptive_hmc_test.cljs`, `adaptive_nuts_test.cljs`

- [x] Dual averaging step-size adaptation (Algorithm 4+5, Hoffman & Gelman 2014)
- [x] Welford diagonal mass matrix estimation
- [x] HMC target-accept 0.65, NUTS target-accept 0.8

### Resampling methods — `hmc_mass_resample_test.cljs`

- [x] Systematic, residual, and stratified resampling produce correct posteriors

---

## Verification Frameworks

### GFI Contract Registry — `src/genmlx/contracts.cljs`

A data-driven contract registry where each contract is a measure-theoretic
theorem expressed as an executable predicate. 11 core contracts:

1. `generate` weight = score when fully constrained
2. `update` with empty constraints = identity (weight 0)
3. `update` weight = new_score - old_score
4. `update` round-trip via discard recovers original trace
5. `regenerate` with empty selection = identity (weight 0)
6. `project(all)` = score
7. `project(none)` = 0
8. `assess` weight = `generate` score for same choices
9. `propose` → `generate` round-trip = finite weight
10. Score decomposition: sum of `project` over partition = score
11. Broadcasting equivalence: `vsimulate(N)` matches N `simulate` calls

**Verified on 13 canonical models:** single-site, multi-site, linreg, splice,
mixed discrete/continuous, deep-nesting (3-level splice), vec-compatible,
Map, Unfold, Switch, Scan, Mask, Recurse. **575 checks, 0 failures.**

### Static Validator — `src/genmlx/verify.cljs`

`validate-gen-fn` performs static analysis: execution errors, address
uniqueness (via validation handler), score finiteness, empty models,
materialization in body, multi-trial conditional duplicate detection.

### Property-Based Testing — Phase 1 complete (162 properties, 9 files)

| File | Properties | Covers |
|------|-----------|--------|
| `inference_property_test.cljs` | 24 | IS, MH, SMC, kernels |
| `vectorized_property_test.cljs` | 18 | vsimulate, vgenerate, batched ops |
| `combinator_extra_property_test.cljs` | 20 | Map, Unfold, Switch, Scan, Mask, Mix |
| `gradient_learning_property_test.cljs` | 20 | gradients, Adam, wake-sleep |
| `combinator_property_test.cljs` | 15 | combinator GFI contracts |
| `gfi_property_test.cljs` | 17 | simulate, generate, update, regenerate |
| `dist_property_test.cljs` | 17 | distribution sampling, log-prob |
| `choicemap_property_test.cljs` | 11 | choicemap algebra |
| `selection_property_test.cljs` | 11 | selection algebra |

**Phase 2 remaining (TODO 21.14-21.21 in TODO.md):** ~88 more properties
covering edit/diff, SMC, MCMC, VI, SMCP3, gradient MCMC, handler internals,
and ADEV.

---

## Test File Inventory

### Core unit tests (run with `bun run --bun nbb test/genmlx/<file>`)

| File | Focus |
|------|-------|
| `choicemap_test.cljs` | ChoiceMap data structure |
| `trace_test.cljs` | Trace record |
| `selection_test.cljs` | Selection algebra |
| `handler_test.cljs` | Handler state transitions |
| `dist_test.cljs` | Distribution sampling and log-prob |
| `gen_test.cljs` | `gen` macro, DynamicGF |
| `combinators_test.cljs` | All combinators |
| `inference_test.cljs` | Core inference algorithms |

### Compatibility suites (must always pass)

| File | Assertions | Source |
|------|-----------|--------|
| `gen_clj_compat_test.cljs` | 165/165 | Gen.jl reference |
| `genjax_compat_test.cljs` | 73/73 | GenJAX reference |

### Feature-specific tests

| File | Focus |
|------|-------|
| `adev_test.cljs` | ADEV gradient estimation |
| `adaptive_hmc_test.cljs` | Adaptive HMC (5 tests) |
| `adaptive_nuts_test.cljs` | Adaptive NUTS (6 tests) |
| `amortized_test.cljs` | Amortized inference |
| `assess_propose_test.cljs` | assess/propose operations |
| `batch_sample_n_test.cljs` | Batch sampling |
| `choicemap_algebra_test.cljs` | ChoiceMap merge/diff |
| `compile_fn_test.cljs` | mx/compile-fn |
| `correctness_test.cljs` | Cross-cutting correctness |
| `custom_gradient_test.cljs` | CustomGradientGF |
| `elliptical_slice_test.cljs` | Elliptical slice sampling |
| `error_message_test.cljs` | Error message quality |
| `gamma_batch_test.cljs` | Gamma batch sampling |
| `hmc_mass_resample_test.cljs` | Mass matrix + resampling |
| `kernel_combinator_test.cljs` | Kernel algebra |
| `kernel_dsl_test.cljs` | Trace kernel DSL |
| `lanczos_test.cljs` | Lanczos log-gamma |
| `lazy_mcmc_test.cljs` | Lazy MCMC chains |
| `loop_compilation_test.cljs` | Loop compilation |
| `loop_compiled_hmc_test.cljs` | Compiled HMC |
| `loop_compiled_mala_test.cljs` | Compiled MALA |
| `map_test.cljs` | Map combinator |
| `map_dist_test.cljs` | map->dist bridge |
| `memory_test.cljs` | Memory management |
| `neg_binomial_test.cljs` | Negative binomial |
| `new_dist_test.cljs` | New distributions |
| `nn_test.cljs` | Neural network GFs |
| `prng_key_test.cljs` | Functional PRNG |
| `project_test.cljs` | Project operation |
| `proposal_edit_test.cljs` | ProposalEdit |
| `recurse_test.cljs` | Recurse combinator |
| `remaining_fixes_test.cljs` | Bug fix verification |
| `resource_test.cljs` | Resource management |
| `sbc_test.cljs` | Simulation-based calibration |
| `smcp3_kernel_test.cljs` | SMCP3 |
| `stress_test.cljs` | Long-chain stress tests |
| `untested_features_test.cljs` | Edge cases |
| `validation_test.cljs` | Input validation |
| `vectorized_grad_test.cljs` | Vectorized gradients |
| `vectorized_mcmc_fix_test.cljs` | Vectorized MCMC |
| `vectorized_test.cljs` | Vectorized inference |
| `verify_test.cljs` | Static validator |
| `vimco_test.cljs` | VIMCO |
| `vmap_test.cljs` | vmap-gf combinator |
| `vupdate_test.cljs` | Batched update |

### Strategy test suites (from this document)

| File | Assertions | Strategy |
|------|-----------|----------|
| `dist_statistics_test.cljs` | ~48 | Strategy 1 |
| `gfi_contract_test.cljs` | ~65 | Strategies 2-3 |
| `combinator_contract_test.cljs` | ~23 | Strategy 4 |
| `inference_convergence_test.cljs` | ~10 | Strategy 6 |
| `contract_verification_test.cljs` | 575 | Contract framework |
| `conjugate_bb_test.cljs` | ~8 | Strategy 6 |
| `conjugate_gp_test.cljs` | ~8 | Strategy 6 |
| `conjugate_posterior_test.cljs` | ~8 | Strategy 6 |

### Property test suites

| File | Properties |
|------|-----------|
| `inference_property_test.cljs` | 24 |
| `vectorized_property_test.cljs` | 18 |
| `combinator_extra_property_test.cljs` | 20 |
| `gradient_learning_property_test.cljs` | 20 |
| `combinator_property_test.cljs` | 15 |
| `gfi_property_test.cljs` | 17 |
| `dist_property_test.cljs` | 17 |
| `choicemap_property_test.cljs` | 11 |
| `selection_property_test.cljs` | 11 |

---

## Remaining Work

Tracked in TODO.md:

- **Phase 21.14-21.21:** Property test Phase 2 (~88 properties across 8 files)
  — edit/diff, SMC, MCMC, VI, SMCP3, gradient MCMC, handler internals, ADEV
- **Phase 23.1-23.2:** Gen.jl differential testing — Julia reference script +
  GenMLX comparison loader (~185 assertions)
