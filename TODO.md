# GenMLX Master TODO

> Single source of truth for all remaining work. Consolidated from GAPS.md,
> AUDIT.md, and LAMBDA_MLX.md.
>
> **Goals**: Feature parity with Gen.jl (minimum). Speed parity with GenJAX.
> 100% idiomatic, purely functional ClojureScript. Formally grounded in λ_MLX.
>
> Current state: ~10,800 lines across 27 source files, 640+ test assertions,
> 27 distributions, 9 combinators, 15+ inference algorithms.

---

## Phase 1: Functional Purity & Code Quality

*Eliminate all unnecessary mutation. Make every boundary between pure and
impure code explicit. Align the implementation with λ_MLX's handler state
types so the formal correspondence is obvious.*

### Handler architecture

- [x] **1.1** Document handler state schemas and λ_MLX correspondence
  - Documented all five handler state shapes (simulate, generate, update,
    regenerate, project) plus batched variants in `handler.cljs` ns docstring
  - Typed records rejected: CLJS records offer no compile-time checking and
    would add 9 record types for zero safety gain. Plain maps with documented
    schemas are idiomatic and sufficient.
  - *Files*: `handler.cljs`

- [x] **1.2** Audit all uses of `volatile!` — confirm they are confined to `run-handler`
  - Confirmed: only created in `run-handler`. All `vreset!/vswap!` are in
    handler functions + `trace-gf!` (handler.cljs) + `adev-handler` (adev.cljs).
  - Documented the mutable boundary in `handler.cljs` ns docstring.
  - *Files*: `handler.cljs`

- [x] **1.3** Audit dynamic vars (`*handler*`, `*state*`, `*param-store*`)
  - Confirmed: these three are the only dynamic scope in the system.
  - `*handler*` and `*state*` bound exclusively by `run-handler`.
  - `*param-store*` bound by `learning.cljs` and `inference/adev.cljs`.
  - Documented in `handler.cljs` ns docstring and inline comment.
  - *Files*: `handler.cljs`

### ChoiceMap algebra

- [x] **1.4** Add property tests for the ChoiceMap algebra
  - [x] `merge(EMPTY, cm) = cm` (left identity)
  - [x] `merge(cm, EMPTY) = cm` (right identity)
  - [x] `get(set(cm, a, v), a) = v` (get-set)
  - [x] `get(set(cm, a, v), b) = get(cm, b)` when `a ≠ b` (set-get)
  - [x] Trace type concatenation ⊕ is associative with `{}` as unit
  - *Files*: `choicemap.cljs`, `test/genmlx/choicemap_algebra_test.cljs`

### Effect boundary

- [x] **1.5** Document the three effect operations (`trace`, `splice`, `param`) as
  the complete set of impure operations within `gen` bodies — everything else
  is pure ClojureScript
  - Added comment block in `dynamic.cljs` above the effect operations
  - *Files*: `dynamic.cljs`

---

## Phase 2: Missing GFI Operations

*Complete the Generative Function Interface so that every operation in the
λ_MLX calculus has a corresponding implementation.*

- [x] **2.1** Implement `project` — `(project trace selection) → log-weight`
  - Computes the log-probability of selected choices under the model
  - `IProject` protocol on DynamicGF and all combinators
  - *Files*: `protocols.cljs`, `dynamic.cljs`, `combinators.cljs`

- [x] **2.2** Add `IProject` to Distribution record
  - *Files*: `dist/core.cljs`

---

## Phase 3: Missing Distributions (Gen.jl Parity)

*Bring the distribution library to full Gen.jl parity and beyond.*

### Already done (27)

gaussian, uniform, bernoulli, beta, gamma, exponential, categorical, poisson,
laplace, student-t, log-normal, multivariate-normal, dirichlet, delta, cauchy,
inv-gamma, geometric, neg-binomial, binomial, discrete-uniform, truncated-normal,
mixture, piecewise-uniform, beta-uniform-mixture, wishart, inv-wishart,
broadcasted-normal

### Remaining

- [x] **3.1** Piecewise uniform — `(piecewise-uniform bounds probs)` (~30 lines)
- [x] **3.2** Beta-uniform mixture — `(beta-uniform theta alpha beta)` (~20 lines)
- [x] **3.3** Wishart — `(wishart df scale-matrix)` (~50 lines, requires matrix ops)
- [x] **3.4** Inverse Wishart — `(inv-wishart df scale-matrix)` (~50 lines)
- [x] **3.5** Broadcasted normal — `(broadcasted-normal mu sigma)` for batch params (~20 lines)

### Distribution infrastructure

- [x] **3.6** External distribution compatibility bridge — `map->dist`
  ```clojure
  (defn map->dist [{:keys [sample log-prob reparam support]}] ...)
  ```
  Allows users to define distributions as plain maps without `defdist`
  (~30 lines in `dist/core.cljs`)

- [x] **3.7** Add tests for neg-binomial (log-prob, sample stats, generate weight)
  - *File*: `test/genmlx/neg_binomial_test.cljs`
- [x] **3.8** Add native `dist-sample-n` for remaining batchable distributions
  - 10 of 27 now have native batch sampling: gaussian, uniform, bernoulli,
    exponential, laplace, log-normal, delta, cauchy, truncated-normal,
    broadcasted-normal
  - Remaining distributions without batch sampling use sequential fallback
    (beta, gamma, categorical, poisson, student-t, dirichlet, geometric,
    neg-binomial, binomial, discrete-uniform, multivariate-normal,
    piecewise-uniform) — most require rejection sampling or complex logic
  - Further batch sampling expansion tracked in 7.1

---

## Phase 4: Missing Inference Algorithms (Gen.jl Parity)

*Match Gen.jl's complete inference toolkit.*

- [x] **4.1** Elliptical slice sampling — `(elliptical-slice-step trace selection)`
  - Specialized MCMC kernel for models with multivariate Gaussian priors
  - ~40 lines, purely functional
  - *File*: `inference/mcmc.cljs`

- [x] **4.2** MAP optimization — `(map-optimize opts model args observations)`
  - Gradient ascent via `mx/value-and-grad`, Adam/SGD optimizers
  - ~50 lines
  - *Files*: `inference/mcmc.cljs`, `test/genmlx/map_test.cljs`

- [x] **4.3** VIMCO — `(vimco model guide args observations)`
  - Variational inference with multi-sample objective
  - ~80 lines
  - *File*: `inference/vi.cljs`

- [x] **4.4** ADEV gradient estimation
  - Sound automatic differentiation of expected values
  - Integrates with `mx/grad` / `mx/value-and-grad`
  - ~100 lines
  - *File*: `inference/adev.cljs`

---

## Phase 5: Missing Combinators

- [x] **5.1** Recurse combinator — recursive model structures
  - Corresponds to `fix(f)` in λ_MLX
  - Full GFI: simulate, generate, update, regenerate, edit
  - ~100 lines
  - *File*: `combinators.cljs`

- [x] **5.2** Full GFI on Mask combinator
  - Added update, regenerate, update-with-diffs
  - Edit already works via `edit-dispatch` default

- [x] **5.3** Full GFI on Contramap / Dimap
  - Already had update + regenerate
  - Added update-with-diffs
  - Edit works via `edit-dispatch` default

---

## Phase 6: Testing Gaps (from AUDIT.md)

*Cover every untested path identified in the audit.*

- [x] **6.1** ProposalEdit in the edit interface
  - Test with actual forward/backward generative functions
  - Verify backward request swaps correctly
  - Verify weight computation: `w_update + w_backward - w_forward`

- [x] **6.2** SMCP3 with actual forward/backward kernels
  - End-to-end test: model + custom proposal + SMCP3 inference
  - Verify incremental weight computation

- [x] **6.3** Kernel combinators beyond `repeat-kernel` and `mh-kernel`
  - [x] `chain` — sequential composition of 2+ kernels
  - [x] `cycle-kernels` — round-robin cycling
  - [x] `mix-kernels` — random mixture of kernels
  - [x] `seed` — fixed PRNG key

- [x] **6.4** Batched update path (`vupdate`)
  - Shape correctness tests
  - Statistical equivalence with sequential update

- [x] **6.5** Comprehensive neg-binomial distribution tests (same as 3.7)
  - Log-prob spot checks
  - Sample mean/variance
  - Generate weight test

---

## Phase 7: Vectorization & Performance (GenJAX Speed Parity)

*Achieve GenJAX-level performance through compiled inference loops and
broader batch sampling support.*

### Batch sampling coverage

- [x] **7.1** Native `dist-sample-n` for more distributions
  - [x] laplace, cauchy, log-normal, truncated-normal (already implemented)
  - [x] discrete-uniform, geometric, categorical, multivariate-normal, binomial, student-t
  - [x] gamma (vectorized Marsaglia-Tsang with Ahrens-Dieter for α < 1)
  - [x] inv-gamma, beta, dirichlet (built on batched gamma)
  - Remaining without batch sampling: poisson (Knuth's algorithm, inherently sequential)

### Vectorized inference completeness

- [x] **7.2** Vectorized splice support in batched mode
  - DynamicGF sub-GFs run under batched handlers (simulate, generate, update, regenerate)
  - Nested splice (3+ levels) supported
  - Non-DynamicGF splice (combinators) still throws with clear error

- [x] **7.3** Vectorized MCMC (batched MH chains)
  - Run N independent MH chains in parallel via broadcasting
  - Enables parallel tempering and R-hat diagnostics from a single call

### Compiled inference loops

- [x] **7.4** Compile entire MH chain into a single Metal program
  ```clojure
  (defn compiled-mh-chain [model args obs selection n-samples]
    (mx/compile-fn (fn [key] ...)))
  ```
  ~100 lines

- [x] **7.5** Vectorized SMC sweep — multi-step batched particle filtering
  - vsmc runs model body ONCE per timestep for all N particles
  - 24.5x speedup over sequential SMC (50 particles, 5 steps)
  ~100 lines

- [x] **7.6** Compile VI optimization loop
  ~50 lines

### Benchmarking

- [x] **7.7** Benchmark suite comparing GenMLX vs handcoded MLX
  - Importance sampling (N=100, 1000, 10000)
  - HMC (chain length 100, 500, 1000)
  - Match the paper's Fig. 17 methodology

- [ ] **7.8** Benchmark suite comparing GenMLX vs GenJAX (where possible)
  - Same models, same algorithms, different hardware
  - Document Apple Silicon vs NVIDIA GPU tradeoffs

---

## Phase 8: Gradient & Differentiable Programming

- [x] **8.1** Custom gradient generative functions — `CustomGradientGF`
  - User-supplied forward and backward passes
  - ~60 lines
  - *File*: `custom_gradient.cljs`

- [x] **8.2** Argument gradient annotations — `has-argument-grads`
  - `IHasArgumentGrads` protocol on generative functions indicating which arguments are differentiable
  - Used by gradient-based inference to know what to differentiate
  - *Files*: `protocols.cljs`, `custom_gradient.cljs`, `dynamic.cljs`, `dist/core.cljs`

---

## Phase 9: Incremental Computation

- [x] **9.1** Per-step optimization for Unfold/Scan
  - Store `::step-scores` metadata in Unfold traces (and `::step-carries` for Scan)
  - During `update`, detect earliest constrained step and skip unchanged prefix
  - Replay old choices/scores from metadata, only re-execute from first changed step
  - `update-with-diffs` strips metadata when args change to prevent invalid prefix-skip
  - *Files*: `combinators.cljs`, `test/genmlx/combinators_test.cljs`

- [ ] **9.2** Handler-level diff awareness for DynamicGF *(low priority)*
  - Currently DynamicGF `update-with-diffs` re-executes the full body
  - Optimization: skip unchanged trace sites during body re-execution
  - Requires tracking which addresses were visited and what distributions they used
  - Deprioritized: MLX's lazy graph evaluation and broadcasting-based vectorization
    already cover most performance needs. Revisit if a specific use case demands it

---

## Phase 10: Formal Foundation (λ_MLX)

*Ground GenMLX in a formal lambda calculus, extending λ_GEN from the
POPL 2026 paper with handler types, full GFI operations, broadcasting
correctness, and algebraic effect semantics.*

### Prerequisites (implementation alignment)

- [x] **10.1** Complete Phase 1.1 (handler state schemas) — required for
  H(σ, τ) correspondence (documented in handler.cljs ns docstring)
- [x] **10.2** Complete Phase 2.1 (`project`) — required for GFI completeness
- [x] **10.3** Complete Phase 1.4 (ChoiceMap algebra tests) — required for
  trace type monoid ⊕

### Formalization work

- [ ] **10.4** Define λ_MLX type grammar formally (extending λ_GEN Figure 10)
  - Types, terms, handler states, edit requests, diff types
  - Written in LAMBDA_MLX.md §3

- [ ] **10.5** Define denotational semantics in QBS (extending λ_GEN Figure 11)
  - Handler transition semantics as state monad
  - Written in LAMBDA_MLX.md §4

- [ ] **10.6** Define generate{−}, update{−}, regenerate{−}, edit{−} as program
  transformations (extending λ_GEN Figure 12's simulate{−} and assess{−})
  - Written in LAMBDA_MLX.md §6

- [ ] **10.7** Prove Proposition: correctness of generate and update
  - Analogous to λ_GEN Proposition 3.1
  - generate weight = log p(observations | args)
  - update weight = log p(new_choices | args) - log p(old_choices | args)

- [ ] **10.8** Prove broadcasting correctness theorem
  - Analogous to λ_GEN Theorem 3.3 but for broadcasting instead of vmap
  - Key lemma: handler shape-agnosticism preserves logical relations
  - Written in LAMBDA_MLX.md §5.3

- [ ] **10.9** Prove broadcasting commutativity corollary
  - Analogous to λ_GEN Corollary 3.4
  - Broadcasting-based vectorized inference = N independent sequential runs

- [ ] **10.10** Prove handler soundness
  - Each handler mode correctly implements its GFI operation
  - By induction on trace effect operations

- [ ] **10.11** Prove combinator compositionality
  - GFI contracts preserved by Map, Unfold, Switch, Scan, Mask, Mix

- [ ] **10.12** Prove edit/backward duality
  - Backward request correctly inverts forward transformation
  - MH acceptance ratio is valid

- [ ] **10.13** Prove diff-aware update correctness
  - MapCombinator vector-diff: only changed elements re-executed
  - Total weight = sum of per-element weight changes

### Trace type annotations (optional, aids formalization)

- [ ] **10.14** Add optional `:trace-schema` metadata to `gen` forms
  - Documents the grading γ in G_γ η
  - Not enforced at runtime, used for documentation and verification

- [ ] **10.15** Add optional `:retval-type` to `gen` forms
  - Documents the η in G_γ η

### Paper

- [ ] **10.16** Write λ_MLX paper (see LAMBDA_MLX.md §9 for structure)

---

## Phase 11: Input Validation & Error Messages

- [ ] **11.1** Malli schemas for all public API functions
  - Distribution constructors, GFI operations, inference algorithms
  - Instrument in development mode, strip in production
  - ~200 lines across multiple files

- [ ] **11.2** Helpful error messages for common mistakes
  - Calling `mx/eval!` inside a `gen` body during batched execution
  - Mismatched choice map structure in `generate`/`update`
  - Missing addresses in selections
  - Type mismatches in distribution parameters

---

## Phase 12: Ecosystem

*Build out the ecosystem with neural network integration, visualization,
and interop. Each sub-project is independently valuable.*

- [x] **12.1** Neural network integration — `nn->gen-fn`
  - NeuralNetGF wraps nn.Module as deterministic GF (simulate, generate, assess, propose)
  - Layer constructors (linear, sequential, relu, gelu, etc.)
  - Training utilities using MLX's native nn.valueAndGrad + optimizers
  - *Files*: `mlx.cljs`, `nn.cljs`, `test/genmlx/nn_test.cljs`

- [x] **12.2** Amortized inference via trained neural proposals
  - Reparameterized ELBO training (VAE-style) via nn.valueAndGrad
  - Neural importance sampling using trained guide as proposal
  - *Files*: `inference/amortized.cljs`, `test/genmlx/amortized_test.cljs`

- [ ] **12.3** Visualization (GenStudio-inspired)
  - `plot-trace`, `plot-posterior`, `animate-smc`
  - ClojureScript + Observable Plot or Vega-Lite
  - *File*: new `viz.cljs`

- [ ] **12.4** LLM-as-generative-function (following GenGPT3.jl)
  - Wrap LLM API calls as generative functions
  - Each token position is an addressed random choice
  - *File*: new `llm.cljs`

- [x] **12.5** Trace kernel DSL
  - Higher-level sugar for writing MCMC kernels
  - ~100 lines

- [ ] **12.6** Auto-genify
  - Automatically convert plain ClojureScript functions into generative functions
  - Macro that inserts `dyn/trace` at every stochastic call site

---

## Phase 13: Documentation & Packaging

- [ ] **13.1** Comprehensive API documentation
  - Every public function with docstring, signature, example
  - Generate from source via `codox` or similar

- [ ] **13.2** Tutorial notebooks
  - Getting started
  - Bayesian linear regression
  - Mixture models with stochastic branching
  - Custom inference algorithms
  - Vectorized inference

- [ ] **13.3** npm package for distribution
  - `package.json` with proper dependencies
  - `npx genmlx` CLI entry point

---

## Priority Order

For someone working through this linearly:

```
Immediate (solidify what exists):
  1.1  Handler state schemas + docs       ✅
  1.4  ChoiceMap algebra tests            ✅
  2.1  Implement project                  ✅
  3.7  Neg-binomial tests                 ✅
  6.1  ProposalEdit tests                ✅
  6.2  SMCP3 end-to-end test             ✅

Near-term (Gen.jl feature parity):
  3.1–3.5  Missing distributions          ✅
  3.6      map->dist bridge               ✅
  3.8      Batch sampling candidates      ✅
  4.1      Elliptical slice sampling      ✅
  4.2      MAP optimization               ✅
  5.1–5.3  Combinators (full Phase 5)     ✅
  6.3–6.4  Testing gaps                   ✅

Medium-term (GenJAX speed parity):
  7.1  Batch sampling coverage     ✅
  7.3  Vectorized MCMC                ✅
  7.4  Compiled MH chain              ✅
  7.5  Vectorized SMC sweep           ✅
  7.7  Benchmark suite                ✅

Medium-term (formal foundation):
  10.1–10.3  Prerequisites
  10.4–10.6  Calculus definition
  10.7–10.9  Core theorems

Long-term (ecosystem):
  4.3–4.4  VIMCO, ADEV                ✅
  8.1–8.2  Custom gradients            ✅
  9.1      Unfold/Scan per-step optimization
  11.1–11.2  Validation
  12.1–12.6  Ecosystem
  10.16  λ_MLX paper
```

---

## Tracking

| Phase | Items | Done | Remaining |
|-------|-------|------|-----------|
| 1. Functional Purity | 5 | 5 | 0 |
| 2. Missing GFI Ops | 2 | 2 | 0 |
| 3. Distributions | 8 | 8 | 0 |
| 4. Inference Algorithms | 4 | 4 | 0 |
| 5. Combinators | 3 | 3 | 0 |
| 6. Testing Gaps | 5 | 5 | 0 |
| 7. Vectorization & Perf | 8 | 6 | 2 |
| 8. Gradient Programming | 2 | 2 | 0 |
| 9. Incremental Computation | 2 | 1 | 1 |
| 10. Formal Foundation | 16 | 3 | 13 |
| 11. Validation | 2 | 0 | 2 |
| 12. Ecosystem | 6 | 2 | 4 |
| 13. Documentation | 3 | 0 | 3 |
| **Total** | **66** | **41** | **25** |
