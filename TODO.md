# GenMLX Master TODO

> Single source of truth for all remaining work. Consolidated from GAPS.md,
> AUDIT.md, LAMBDA_MLX.md, and a comprehensive code review (Feb 2026).
>
> **Goals**: Feature parity with Gen.jl (minimum). Speed parity with GenJAX.
> 100% idiomatic, purely functional ClojureScript. Formally grounded in lambda_MLX.
>
> Current state: ~10,800 lines across 29 source files, ~750-800 test assertions,
> 25 unique distribution types, 10 combinators, 60+ inference algorithms/functions.

---

## Phase 1: Functional Purity & Code Quality

*Eliminate all unnecessary mutation. Make every boundary between pure and
impure code explicit. Align the implementation with lambda_MLX's handler state
types so the formal correspondence is obvious.*

### Handler architecture

- [x] **1.1** Document handler state schemas and lambda_MLX correspondence
- [x] **1.2** Audit all uses of `volatile!` -- confirm confined to `run-handler`
- [x] **1.3** Audit dynamic vars (`*handler*`, `*state*`, `*param-store*`)

### ChoiceMap algebra

- [x] **1.4** Add property tests for the ChoiceMap algebra

### Effect boundary

- [x] **1.5** Document the three effect operations (`trace`, `splice`, `param`)

---

## Phase 2: Missing GFI Operations

- [x] **2.1** Implement `project` -- `(project trace selection) -> log-weight`
- [x] **2.2** Add `IProject` to Distribution record

---

## Phase 3: Distributions

### Already done (25 unique type keywords, 27 named constructors)

gaussian/normal, uniform, bernoulli/flip, beta-dist, gamma-dist, exponential,
categorical, poisson, laplace, student-t, log-normal, multivariate-normal,
dirichlet, delta, cauchy, inv-gamma, geometric, neg-binomial, binomial,
discrete-uniform, truncated-normal, mixture, piecewise-uniform,
beta-uniform-mixture, wishart, inv-wishart, broadcasted-normal

- [x] **3.1** Piecewise uniform
- [x] **3.2** Beta-uniform mixture
- [x] **3.3** Wishart
- [x] **3.4** Inverse Wishart
- [x] **3.5** Broadcasted normal
- [x] **3.6** External distribution compatibility bridge -- `map->dist`
- [x] **3.7** Add tests for neg-binomial
- [x] **3.8** Add native `dist-sample-n` for batchable distributions

---

## Phase 4: Inference Algorithms (Gen.jl Parity)

- [x] **4.1** Elliptical slice sampling
- [x] **4.2** MAP optimization
- [x] **4.3** VIMCO
- [x] **4.4** ADEV gradient estimation

---

## Phase 5: Combinators

- [x] **5.1** Recurse combinator
- [x] **5.2** Full GFI on Mask combinator
- [x] **5.3** Full GFI on Contramap / Dimap

---

## Phase 6: Testing Gaps (from AUDIT.md)

- [x] **6.1** ProposalEdit in the edit interface
- [x] **6.2** SMCP3 with actual forward/backward kernels
- [x] **6.3** Kernel combinators (chain, cycle, mix, seed)
- [x] **6.4** Batched update path (`vupdate`)
- [x] **6.5** Neg-binomial distribution tests

---

## Phase 7: Vectorization & Performance

- [x] **7.1** Native `dist-sample-n` for more distributions
- [x] **7.2** Vectorized splice support in batched mode
- [x] **7.3** Vectorized MCMC (batched MH chains)
- [x] **7.4** Compile entire MH chain into a single Metal program
- [x] **7.5** Vectorized SMC sweep
- [x] **7.6** Compile VI optimization loop
- [x] **7.7** Benchmark suite comparing GenMLX vs handcoded MLX
- [ ] **7.8** Benchmark suite comparing GenMLX vs GenJAX
- [x] **7.9** Fix `mx/realize`-on-value in discrete log-prob methods

---

## Phase 8: Gradient & Differentiable Programming

- [x] **8.1** Custom gradient generative functions -- `CustomGradientGF`
- [x] **8.2** Argument gradient annotations -- `has-argument-grads`

---

## Phase 9: Incremental Computation

- [x] **9.1** Per-step optimization for Unfold/Scan
- [ ] **9.2** Handler-level diff awareness for DynamicGF *(low priority)*

---

## Phase 10: Formal Foundation (lambda_MLX)

- [x] **10.1** -- **10.13** Formal specification and proofs (all complete)
- [ ] **10.14** Add optional `:trace-schema` metadata to `gen` forms
- [ ] **10.15** Add optional `:retval-type` to `gen` forms
- [ ] **10.16** Write lambda_MLX paper

---

## Phase 11: Input Validation & Error Messages

- [ ] **11.1** Malli schemas for all public API functions
- [x] **11.2** Helpful error messages for common mistakes

---

## Phase 12: Ecosystem

- [x] **12.1** Neural network integration -- `nn->gen-fn`
- [x] **12.2** Amortized inference via trained neural proposals
- [ ] **12.3** Visualization (GenStudio-inspired)
- [ ] **12.4** LLM-as-generative-function
- [x] **12.5** Trace kernel DSL
- [ ] **12.6** Auto-genify

---

## Phase 13: Documentation & Packaging

- [ ] **13.1** Comprehensive API documentation
- [ ] **13.2** Tutorial notebooks
- [ ] **13.3** npm package for distribution

---

## Phase 14: Gen.jl Parity (from survey)

- [x] **14.1** HMC mass matrix support
- [ ] **14.2** Enumerative / grid-based inference
- [x] **14.3** Residual resampling for SMC
- [x] **14.4** Stratified resampling for SMC
- [ ] **14.5** Product distribution
- [ ] **14.6** Kernel reversal declarations
- [ ] **14.7** Trace serialization
- [ ] **14.8** Directional statistics distributions

---

## Phase 15: Confirmed Bugs (from code review, Feb 2026)

*These are real bugs that should be fixed before any new feature work.*

- [x] **15.1** `mx/random-normal` instead of `mx/random-uniform` in vectorized compiled MH
  - **File**: `inference/mcmc.cljs:210`
  - **Severity**: High -- produces NaN from `mx/log` of negative values
  - **Impact**: `vectorized-compiled-mh` accept/reject is broken
  - **Fix**: Change `mx/random-normal` to `mx/random-uniform`

- [x] **15.2** REINFORCE estimator uses `log-q` as signal instead of objective values
  - **File**: `inference/vi.cljs:308-314`
  - **Severity**: High -- mathematically incorrect gradient estimator
  - **Impact**: `reinforce-estimator` and `programmable-vi` with `:reinforce` produce wrong gradients
  - **Fix**: Replace `(mx/stop-gradient (mx/subtract log-q baseline))` with
    `(mx/stop-gradient (mx/subtract obj-val baseline))` where `obj-val` is the
    objective function value (e.g., ELBO), not `log-q`

- [x] **15.3** Wake phase ELBO missing `log q(z|x)` entropy term
  - **File**: `learning.cljs:201-204`
  - **Severity**: High -- the guide weight variable is shadowed by the model weight
  - **Impact**: `wake-phase-loss` computes `-log p(x,z)` instead of `-(log p(x,z) - log q(z|x))`
  - **Fix**: Rename one of the destructured `weight` bindings; subtract guide log-prob from model weight

- [x] **15.4** ~~Possible 3-arg `mx/add` call in amortized inference~~ (not a bug: `mx/add` supports variadic args)
  - **File**: `inference/amortized.cljs:62`
  - **Severity**: Medium -- may cause runtime error if `mx/add` is strictly binary
  - **Impact**: `make-elbo-loss` log-q computation
  - **Fix**: Chain binary `mx/add` calls or verify `mx/add` supports 3+ args

- [x] **15.5** Neg-binomial sample reuses same PRNG key for gamma and Poisson steps
  - **File**: `dist.cljs:671-682`
  - **Severity**: Low -- correlated randomness, statistically incorrect samples
  - **Fix**: Split key before Poisson loop: use `k1` for gamma, `k2` for Poisson

---

## Phase 16: Correctness Concerns (from code review, Feb 2026)

*These are not necessarily bugs, but design issues that could produce subtly
wrong results in certain use cases.*

- [x] **16.1** Combinator sub-trace reconstruction uses `score = 0.0` and `retval = nil`
  - **Files**: `combinators.cljs` (Map update/regenerate/project, Unfold update/regenerate,
    Scan update/regenerate — 8 locations)
  - **Fix**: Now uses real element/step scores from trace metadata (`::element-scores`,
    `::step-scores`) with fallback to `(mx/scalar 0.0)` for backward compat

- [x] **16.2** `execute-sub` constructs fake traces with `score = 0.0` for spliced sub-GFs
  - **Files**: `handler.cljs`, `dynamic.cljs`
  - **Fix**: `merge-sub-result` now tracks per-splice scores, stored in trace metadata
    as `::splice-scores`, passed to handler state as `:old-splice-scores` for
    update/regenerate

- [x] **16.3** `assess` does not validate that all choices are constrained
  - **Files**: `handler.cljs`, `dynamic.cljs`
  - **Fix**: Added `assess-transition` and `assess-handler` that throw on unconstrained
    addresses, plus `execute-sub-assess` for nested GFs. DynamicGF.assess now uses the
    dedicated handler instead of generate-handler

- [x] **16.4** Switch combinator `update` does not support branch switching
  - **File**: `combinators.cljs`
  - **Fix**: Added `::switch-idx` metadata to simulate/generate/regenerate traces.
    Update reads old idx from metadata, compares to new idx from args. Same branch:
    updates in place. Different branch: generates new branch from scratch,
    weight = new_score - old_score, discard = old choices.

- [x] **16.5** Mix combinator `generate` passes full constraints (including `:component-idx`)
  to component GF
  - **File**: `combinators.cljs:949`
  - **Impact**: The component GF receives a `:component-idx` key in constraints it doesn't
    expect. May be silently ignored or may cause issues depending on the component.
  - **Fix**: Strip `:component-idx` from constraints before passing to component
    (as `update` already does at line 988)

- [ ] **16.6** PRNG keys frequently ignored -- non-reproducible inference
  - **Files**: Multiple
    - `importance-sampling` accepts `key` in opts but never uses it (`importance.cljs:22`)
    - `smcp3-step` generates per-particle keys but never passes them (`smcp3.cljs:116-118`)
    - `dist-simulate` always passes `nil` key (`dist/core.cljs:40`)
    - `systematic-resample-indices` ignores the passed key, uses global RNG (`vectorized.cljs:27-29`)
    - `choice-gradients` and `score-gradient` have no key parameter (`gradients.cljs`)
    - `wake-phase-loss` and `sleep-phase-loss` accept `key` but ignore it (`learning.cljs:192,221`)
    - `amortized.cljs:46` uses `mx/random-normal` with global state
  - **Impact**: Inference results are not reproducible even when users provide PRNG keys
  - **Fix**: Thread keys through all functions that involve randomness; audit every call
    to `mx/random-*` and `dist-sample` to ensure key propagation

---

## Phase 17: Missing Protocol Implementations (from code review, Feb 2026)

*Generative function types that are missing GFI protocol implementations,
causing runtime failures when used in certain inference contexts.*

- [x] **17.1** Add `IAssess` to all combinators
  - **Files**: `combinators.cljs`
  - **Impact**: `p/assess` cannot be called on Map, Unfold, Switch, Scan, Mask, Mix,
    Recurse, Contramap, or MapRetval. Any inference code that calls `assess` on a
    combinator-constructed GF will fail at runtime.
  - **Scope**: ~5-10 lines per combinator (delegate to `generate` with full constraints,
    similar to how DynamicGF implements it)

- [x] **17.2** Add `IPropose` to all combinators
  - **Files**: `combinators.cljs`
  - **Impact**: `p/propose` cannot be called on combinators. Affects custom proposal MH
    when the proposal is a combinator-constructed GF.
  - **Scope**: ~5-10 lines per combinator (simulate + return choices and score)

- [x] **17.3** Add `IUpdate` and `IRegenerate` to `CustomGradientGF`
  - **File**: `custom_gradient.cljs`
  - **Fix**: Added IUpdate, IRegenerate, IProject — deterministic, zero weight

- [x] **17.4** Add `IUpdate` and `IRegenerate` to `NeuralNetGF`
  - **File**: `nn.cljs`
  - **Fix**: Added IUpdate, IRegenerate, IProject — deterministic, zero weight

- [x] **17.5** Add `IProject` to `CustomGradientGF` and `NeuralNetGF`
  - **Files**: `custom_gradient.cljs`, `nn.cljs`
  - **Fix**: Included in 17.3/17.4 above — always returns 0 (no choices)

- [x] **17.6** Add `IUpdateWithDiffs` to `MixCombinator`
  - **File**: `combinators.cljs`
  - **Fix**: Added no-change fast path + fallback to regular update

---

## Phase 18: Distribution Quality (from code review, Feb 2026)

*Distribution-level issues that affect correctness, differentiability, or
performance.*

- [x] **18.1** Replace JS `log-gamma` with MLX-native `mlx-log-gamma` in beta, gamma,
  inv-gamma, student-t log-prob methods
  - **Files**: `dist.cljs` (beta ~line 225, gamma ~line 268, inv-gamma ~line 622,
    student-t ~line 477)
  - **Impact**: These distributions call `mx/realize` to extract parameters as JS numbers,
    compute `log-gamma` in JS, then mix back into MLX. This breaks the computation graph
    and prevents gradient flow through distribution parameters. The MLX-native
    `mlx-log-gamma` already exists (line 42-59) and is used by poisson, neg-binomial,
    and binomial -- it just needs to be adopted by the other distributions.
  - **Scope**: ~20 lines of changes per distribution

- [x] **18.2** Make categorical `dist-sample-n` native/vectorized
  - **File**: `dist.cljs:409-412`
  - **Impact**: Categorical is one of the most common distributions in PPL models.
    Its `sample-n` currently uses the sequential fallback (`split-n` + `mapv` + `stack`).
    This is a significant performance bottleneck for vectorized inference with categorical
    choices.
  - **Fix**: Use `mx/random-categorical` or implement batch Gumbel-max trick

- [x] **18.3** Extend parameter validation to all distributions
  - **File**: `dist.cljs`
  - **Impact**: Only 5 of 25 distributions validate parameters (gaussian sigma>0,
    uniform lo<hi, beta alpha>0/beta>0, gamma shape>0/rate>0, exponential rate>0).
    Invalid parameters to other distributions produce silent NaN or incorrect results.
  - **Missing validation**: truncated-normal (sigma>0, lo<hi), cauchy (scale>0),
    laplace (scale>0), student-t (df>0), poisson (rate>0), inv-gamma (shape>0, scale>0),
    wishart (df>0), multivariate-normal (positive-definite cov)
  - **Scope**: ~2-3 lines per distribution using existing `check-positive`/`check-less-than`

- [x] **18.4** Fix geometric `support` to not use hardcoded `(range 100)`
  - **File**: `dist.cljs:653`
  - **Impact**: For geometric distributions with very small `p`, significant probability
    mass lies beyond k=100. Enumeration-based inference (Gibbs) will miss this mass.
  - **Fix**: Compute support dynamically based on `p`, e.g., up to the 0.999 quantile

- [ ] **18.5** Wire `CustomGradientGF`'s `gradient-fn` into the autograd system
  - **File**: `custom_gradient.cljs`
  - **Impact**: The `gradient-fn` field is stored on the record and accessible via tests,
    but no code in the GFI pipeline dispatches to it. The custom gradient is effectively
    dead code from the framework's perspective.
  - **Fix**: Register the custom gradient with MLX's autograd via `mx/custom-vjp` or
    similar, or have `choice-gradients`/`score-gradient` check for and use it

---

## Phase 19: Code Quality & Cleanup (from code review, Feb 2026)

*Technical debt, dead code, and code duplication identified during review.*

- [ ] **19.1** Remove or complete the diff infrastructure
  - **File**: `diff.cljs`
  - **Status**: Only `no-change`, `no-change?`, and `changed?` are used in production.
    Everything else (`vector-diff`, `map-diff`, `value-change`, `compute-vector-diff`,
    `compute-map-diff`, `should-recompute?`, `unknown-change`) is only used in tests.
  - **Decision needed**: Either build the incremental computation system that uses these
    (Phase 9.2), or remove the unused infrastructure to reduce dead code

- [x] **19.2** Deduplicate resampling code
  - **Files**: `inference/util.cljs`, `inference/smc.cljs`, `inference/smcp3.cljs`
  - **Fix**: Moved `systematic-resample` and `compute-ess` to `inference/util.cljs`.
    Deleted private copies from smc.cljs and smcp3.cljs, replaced calls with `u/` prefix.
    (`vectorized.cljs` version intentionally kept separate — uses MLX array I/O.)

- [x] **19.3** Deduplicate Adam optimizer
  - **Files**: `learning.cljs` (adam-init, adam-step), `inference/vi.cljs` (adam-state, adam-step)
  - **Impact**: Two independent Adam implementations. The vi.cljs version may have subtle
    differences from the learning.cljs version.
  - **Fix**: Have `vi.cljs` import from `learning.cljs`

- [ ] **19.4** Remove or test `defdist-transform` macro
  - **File**: `dist/macros.cljc`
  - **Status**: The macro is defined and tested in `untested_features_test.cljs`, but is
    never used in production code. The most obvious candidate (log-normal) is defined
    directly with `defdist` instead.
  - **Decision needed**: Either use it for log-normal (validating the macro in practice)
    or remove it

- [x] **19.5** Merge `SelectAddrs` and `SelectSet` record types
  - **File**: `selection.cljs:23-43`
  - **Impact**: Both records have identical implementations (wrap a set, check `contains?`).
    The only difference is the constructor: `select` takes varargs, `from-set` takes a set.
  - **Fix**: Deleted `SelectSet`, `from-set` now constructs `SelectAddrs`

- [x] **19.6** Add missing `clojure.set` require in `diff.cljs`
  - **File**: `diff.cljs` ns form
  - **Impact**: `compute-map-diff` uses `clojure.set/difference` and
    `clojure.set/intersection` without requiring `clojure.set`. Works in nbb (SCI
    auto-loads it) but would break in standard ClojureScript.
  - **Fix**: Add `[clojure.set :as set]` to the ns `:require`

- [x] **19.7** Fix `requiring-resolve` usage in handler.cljs
  - **File**: `handler.cljs:591`
  - **Impact**: `requiring-resolve` is a Clojure JVM feature, fragile in nbb/ClojureScript.
    Used as a workaround to avoid circular dependency with `genmlx.protocols`.
  - **Fix**: Replaced with direct `p/simulate` call (already required as `genmlx.protocols :as p`)

- [x] **19.8** Add `inference/adev.cljs` and `inference/amortized.cljs` to the
  `inference.cljs` re-export facade
  - **File**: `inference.cljs`
  - **Impact**: These two modules are orphaned from the public API. Users must require
    them directly by full namespace.
  - **Fix**: Add re-exports for key public functions

- [x] **19.9** Deduplicate `run-kernel` (kernel.cljs) and `collect-samples` (mcmc.cljs)
  - **Files**: `inference/kernel.cljs:100-124`, `inference/mcmc.cljs:19-41`
  - **Impact**: Essentially identical MCMC loops with slightly different signatures.
  - **Fix**: Moved `collect-samples` to `kernel.cljs` as public fn, `run-kernel` delegates
    to it. Deleted private copy from `mcmc.cljs`, all 10 callers use `kern/collect-samples`.

---

## Phase 20: Amortized Inference Improvements (from code review, Feb 2026)

*The amortized inference module works but has significant limitations.*

- [ ] **20.1** Vectorize `neural-importance-sampling`
  - **File**: `inference/amortized.cljs:113-127`
  - **Impact**: Currently sequential (`mapv` over samples). Misses the 60-120x speedups
    from vectorized IS.
  - **Fix**: Use `dyn/vgenerate` with neural proposal as constraint source

- [ ] **20.2** Support non-Gaussian posteriors in amortized inference
  - **File**: `inference/amortized.cljs`
  - **Impact**: The encoder must output `[mu, log-sigma]` pairs -- only Gaussian posteriors
    are supported. No mixture of Gaussians, normalizing flows, or discrete latents.
  - **Scope**: Significant -- requires abstracting the reparameterization and log-q computation

- [ ] **20.3** Add minibatch training support
  - **File**: `inference/amortized.cljs:71-92`
  - **Impact**: `train-proposal!` cycles through individual data points one at a time.
    No minibatch training.
  - **Fix**: Accept batch size parameter, group dataset into minibatches

- [ ] **20.4** Add ADEV variance reduction (baseline subtraction)
  - **File**: `inference/adev.cljs`
  - **Impact**: The REINFORCE term has no baseline, leading to high variance gradients.
  - **Fix**: Implement moving-average baseline for REINFORCE surrogate term. ~15 lines.
    Store exponential moving average of cost values, subtract from `stop_gradient(cost)`
    in surrogate: `cost + stop_grad(cost - baseline) * reinforce_lp`

- [x] **20.5** Vectorized ADEV — batched gradient estimation on GPU
  - **File**: `inference/adev.cljs`
  - **Impact**: Currently `adev-gradient` loops over N samples in JS (`mapv`), running the
    model body N times sequentially. This is the single biggest ADEV performance bottleneck.
    GenJAX achieves GPU-parallel ADEV via `modular_vmap` over Jaxpr; GenMLX should use its
    existing shape-based batching (`[N]`-shaped arrays, same approach as `vsimulate`/`vgenerate`).
  - **Scope**: ~40 lines. Add `vadev-transition` (batched ADEV handler that samples `[N]`-shaped
    arrays at each trace site, accumulates `[N]`-shaped `reinforce-lp`), `vadev-execute`
    (runs model once with N particles), and `vadev-surrogate` (computes per-particle surrogate
    losses, returns mean). The handler is structurally identical to the existing batched
    generate handler, plus `reinforce-lp` accumulation for non-reparam sites.
  - **Expected speedup**: ~50-60x for N=100 (matching existing `vgenerate` benchmarks).
    MLX broadcasting handles all arithmetic naturally — no code changes to model bodies.
  - **Inspired by**: GenJAX ADEV (`src/genjax/adev/__init__.py`, ~1,719 lines using Jaxpr
    interpretation + `modular_vmap`). GenMLX's shape-based approach achieves the same
    parallelism in ~40 lines because MLX broadcasting replaces explicit vmap.

- [x] **20.6** Compiled ADEV optimization loop
  - **File**: `inference/adev.cljs`
  - **Impact**: `adev-optimize` rebuilds the computation graph each iteration. The existing
    `compiled-vi` and `compiled-programmable-vi` show 2-5x additional speedup from Metal JIT
    via `mx/compile-fn`.
  - **Scope**: ~30 lines. Add `compiled-adev-optimize` following the `compiled-programmable-vi`
    pattern: wrap `mx/value-and-grad(loss-fn)` in `mx/compile-fn`, separate sampling (key
    splitting) from the compiled gradient computation.
  - **Depends on**: 20.5 (vectorized ADEV) — compile the vectorized version, not the sequential one

- [x] **20.7** ADEV benchmark suite — measure vectorized + compiled speedups
  - **File**: new `test/genmlx/adev_benchmark.cljs` (~60 lines)
  - **Impact**: Required to validate that GPU ADEV delivers real speedups before merging.
    Must demonstrate measurable improvement or the feature should not ship.
  - **Benchmarks** (3 models × 3 configurations):
    - **Models**: (1) pure reparam (5 gaussians), (2) mixed reparam+REINFORCE (3 gaussians
      + 2 bernoullis), (3) 20-site model (stress test for dispatch amortization)
    - **Configurations**: sequential `adev-gradient` (baseline), vectorized `vadev-surrogate`,
      compiled+vectorized `compiled-adev-optimize`
    - **Particle counts**: N = 10, 100, 1000
    - Measure wall-clock time per gradient step, report speedup ratios
  - **Expected results**: 50-100x for vectorized, 100-300x for compiled+vectorized vs baseline
  - **Gate**: Only merge 20.5/20.6 if benchmarks show ≥10x speedup at N=100

---

## Phase 22: Practical Inference Improvements (from CODEBASE_ASSESSMENT.md)

- [x] **22.1** Adaptive step-size for HMC/NUTS via dual averaging
  - **File**: `inference/mcmc.cljs`
  - **Done**: `find-reasonable-epsilon`, `dual-averaging-warmup` (Algorithm 4+5, Hoffman &
    Gelman 2014), Welford diagonal mass matrix estimation. HMC defaults to target-accept 0.65,
    NUTS to 0.8. Options: `:adapt-step-size`, `:adapt-metric`. ~100 lines.
    Tests: `adaptive_hmc_test.cljs` (5 tests), `adaptive_nuts_test.cljs` (6 tests).

- [ ] **22.2** Remove deprecated stateful PRNG functions from `mlx.cljs`
  - **File**: `mlx.cljs:302-315`
  - **Impact**: `random-seed!`, `random-normal`, `random-uniform` use global MLX state,
    directly contradicting the functional PRNG design.
  - **Depends on 16.6**: `mx/random-normal` has ~15 callers and `mx/random-uniform` has
    ~10 callers in inference modules (mcmc.cljs, vi.cljs, amortized.cljs, smc.cljs,
    vectorized.cljs). Must thread PRNG keys through all call sites first (16.6).

- [x] **22.3** Add `mx/eval!` after resampling to prevent unbounded lazy graph growth
  - **File**: `vectorized.cljs:72-73`
  - **Impact**: `resample-vtrace` reindexes all leaf arrays via `mx/take` but never
    materializes the result. In a long SMC run with many resample steps, the computation
    graph chains `take(take(take(...)))` without bound.
  - **Fix**: Add `(mx/eval! (:choices new-vtrace))` after reindexing

---

## Phase 24: Verified Probabilistic Programming (from VERIFIED_PPL.md)

*Runtime verification of GFI contracts as executable theorems. Each contract
is a measure-theoretic theorem expressed as an executable predicate. The
contract registry serves as both a runtime verifier and a formal specification
that maps 1:1 to lambda_MLX theorems and (future) Lean 4 propositions.*

*See VERIFIED_PPL.md for the full design document.*

### Static validator (Level 1: structural correctness)

- [ ] **24.1** `validate-gen-fn` — static analysis of generative functions
  - **File**: new `src/genmlx/verify.cljs` (~100-150 lines)
  - **Checks**: address uniqueness (run once, collect addresses, check for dupes),
    score finiteness, parameter type validity, no side effects detected (heuristic),
    all code paths return a value
  - **Returns**: `{:valid? bool :violations [...]}`
  - **Use case**: Claude Code generates a model, run `validate-gen-fn` before
    executing any inference. Catches structural bugs immediately.

### GFI contract registry (Level 2: measure-theoretic soundness)

- [ ] **24.2** Data-driven GFI contract registry
  - **File**: new `src/genmlx/contracts.cljs` (~150 lines)
  - **Design**: A plain Clojure map of `{keyword -> {:theorem string, :check fn}}`.
    Each `:check` function takes `{:model :args :trace}` and returns boolean.
    Contracts are data — inspectable, composable, translatable.
  - **Contracts to implement** (11 core contracts):
    1. `generate` weight = score when fully constrained
    2. `update` with empty constraints = identity (weight 0)
    3. `update` weight = new_score - old_score
    4. `update` round-trip via discard recovers original trace
    5. `regenerate` with empty selection = identity (weight 0)
    6. `project(all)` = score
    7. `project(none)` = 0
    8. `assess` weight = `generate` score for same choices
    9. `propose` -> `generate` round-trip = weight 0
    10. Score decomposition: sum of `project` over address partition = score
    11. Broadcasting equivalence: `vsimulate(N)` statistically matches N `simulate` calls
  - **Corresponds to**: lambda_MLX Propositions 3.1, 3.2, Corollary 3.4, and
    handler soundness theorem

- [ ] **24.3** `verify-gfi-contracts` — automated contract verification for any model
  - **File**: same `src/genmlx/contracts.cljs` (~50 lines)
  - **API**: `(verify-gfi-contracts model args & {:keys [n-trials levels]})`
  - **Behavior**: For each contract in the registry, run N trials (default 50).
    Each trial: `simulate` to get a random trace, then check the contract.
    Returns detailed report with pass/fail counts per contract.
  - **Expected output**: 50 trials x 11 contracts = 550 checks per model.
    A model that passes all 550 checks has very high probability of being
    a valid generative function.

- [ ] **24.4** Canonical model suite for contract verification
  - **File**: new `test/genmlx/contract_verification_test.cljs` (~80 lines)
  - **Models**: 10-15 canonical models covering all features:
    single-site, multi-site, dependent addresses, discrete, mixed
    discrete/continuous, splice, Map, Unfold, Switch, Scan, Mask, Mix,
    Recurse, deep nesting
  - **Test**: Run `verify-gfi-contracts` on each model, assert zero failures
  - **Expected assertions**: 10 models x 550 checks = 5,500 contract checks

### Future: Lean 4 formalization (Level 4: mathematical certainty)

- [ ] **24.5** Formalize lambda_MLX core types in Lean 4
  - **Scope**: Significant (research-level effort)
  - Define: `ChoiceMap`, `Trace`, `Selection`, `Distribution`, `GenFn` as Lean types
  - Define: handler state transitions as pure Lean functions
  - Builds on Mathlib's existing probability theory and measure theory formalizations

- [ ] **24.6** Prove GFI axioms in Lean 4
  - **Scope**: Significant (research-level effort)
  - Prove: generate correctness, update weight, handler soundness,
    broadcasting commutativity
  - Proof structure: structural induction on trace operations (mirrors the
    handler's pure transition design)

- [ ] **24.7** Spec-to-Lean translator
  - **Scope**: ~200 lines of Clojure
  - Walk the contract registry, emit Lean 4 theorem statements
  - The `:check` predicates become theorem bodies; the `:theorem` strings
    become docstrings
  - Proofs still require human/tactic effort, but theorem STATEMENTS come
    from the contract registry automatically

---

## Phase 21: Testing Strategies (from TESTING.md, Feb 2026)

*Systematic testing gaps identified by comparing TESTING.md strategies against
existing test coverage. All tests use the existing `assert-true`/`assert-close`/
`assert-equal` + `println` pattern in standalone `.cljs` files (no test.check --
it doesn't work with nbb).*

### Distribution statistical verification

- [x] **21.1** E[X] and Var[X] for all distributions (10,000 samples each)
  - **Done**: `test/genmlx/dist_statistics_test.cljs` — 16 distributions tested
    (gaussian, uniform, bernoulli, beta, gamma, exponential, poisson, laplace,
    log-normal, geometric, neg-binomial, binomial, discrete-uniform, truncated-normal,
    student-t, inv-gamma) + MVN and dirichlet component-wise means. ~40 assertions.

- [x] **21.2** Discrete PMF sums to 1
  - **Done**: `test/genmlx/dist_statistics_test.cljs` — 8 discrete distributions tested
    (bernoulli×2, categorical, binomial, discrete-uniform, geometric, poisson, neg-binomial).
    ~8 assertions.

### GFI round-trip and cross-validation

- [x] **21.3** Update round-trip: `update(trace, c)` then `update(trace', discard)` recovers
  original trace
  - **Done**: `test/genmlx/gfi_contract_test.cljs` — tested on 5 canonical models
    (single-site, multi-site, linreg, splice, mixed). Verifies recovered values match.

- [x] **21.4** Edit round-trip: apply edit, then apply backward request, verify recovery
  - **Done**: `test/genmlx/combinator_contract_test.cljs` — ConstraintEdit round-trip,
    SelectionEdit backward type check, ProposalEdit weight finiteness. ~6 assertions.

- [x] **21.5** Cross-validate `assess` weight against `generate` score
  - **Done**: `test/genmlx/gfi_contract_test.cljs` — tested on all 5 canonical models.
    Verifies `assess(model, args, choices).weight ≈ generate(model, args, choices).trace.score`.

- [x] **21.6** Cross-validate `propose` -> `generate` round-trip
  - **Done**: `test/genmlx/gfi_contract_test.cljs` — tested on all 5 canonical models.
    Verifies propose produces choices that generate accepts with finite weight.

### Combinator compositionality

- [x] **21.7** Combinator degenerate case tests
  - **Done**: `test/genmlx/combinator_contract_test.cljs` — Map(single), Unfold(1),
    Switch(idx=0), Mask(true/false), Scan(1). ~8 assertions.

- [x] **21.8** Nested combinator tests
  - **Done**: `test/genmlx/combinator_contract_test.cljs` — Map(Switch), Unfold(Mask),
    Switch(Map, Map). ~5 assertions.

- [x] **21.9** Score additivity per combinator
  - **Done**: `test/genmlx/combinator_contract_test.cljs` — Map element-scores, Unfold
    step-scores, Switch branch score, Scan step-scores. ~4 assertions.

### Inference convergence

- [x] **21.10** Gamma-Poisson conjugate model convergence
  - **Done**: `test/genmlx/inference_convergence_test.cljs` — IS (100 particles) and
    MH (500 samples) both verify posterior mean ≈ 3.0 for Gamma(3,1)/Poisson data.
    ~4 assertions.

- [x] **21.11** HMC and NUTS acceptance rate checks
  - **Done**: `test/genmlx/inference_convergence_test.cljs` — HMC and NUTS on
    Normal-Normal model. Checks no NaN, acceptance rate > 0.3, posterior mean ≈ 3.
    ~6 assertions.

### GFI contract harness (from TESTING.md Phase A-C)

- [x] **21.12** Reusable `verify-gfi-contract` harness function
  - **Done**: `test/genmlx/gfi_contract_test.cljs` — `run-contracts` harness tests 10
    GFI contracts (simulate validity, generate full/empty, assess=generate, update no-op,
    update round-trip, regenerate none, propose round-trip, project all/none) on 5
    canonical models (single-site, multi-site, linreg, splice, mixed). ~65 assertions.

---

## Phase 23: Differential Testing Against Gen.jl (reference oracle)

*Use Gen.jl as a ground-truth oracle. A Julia script pre-computes deterministic
GFI outputs and log-prob values, saves to JSON. GenMLX loads the JSON and
asserts agreement. The Julia script runs manually (not in CI); the JSON file
is checked in. This eliminates all randomness — fixed choices make every GFI
operation deterministic.*

### Reference value generation (Julia side)

- [ ] **23.1** Julia script to generate `gen_jl_reference.json`
  - **File**: new `test/reference/gen_jl_reference.jl` (~200 lines)
  - **Output**: `test/reference/gen_jl_reference.json` (checked into repo)
  - **Contents**:
    - **Distribution log-probs** (~125 values): For each of the 25 distributions that
      have a Gen.jl equivalent, compute `logpdf` at 5+ test points (including boundary
      values, zero, negative, large). Cover all distributions in Gen.jl's standard
      library: normal, uniform, beta, gamma, exponential, categorical, poisson, bernoulli,
      laplace, geometric, binomial, neg-binomial, discrete-uniform, cauchy, inv-gamma,
      multivariate-normal, dirichlet
    - **GFI operation outputs** (~50 values): For 10 canonical models with fixed
      choicemaps, compute: `generate` weight, `generate` score, `update` weight (with
      specific new constraints), `update` discard values, `assess` weight, `project`
      with specific selections. Models:
      1. Single Gaussian (1 address)
      2. Two independent Gaussians
      3. Linear regression (dependent addresses: slope, intercept, observations)
      4. Single Bernoulli
      5. Mixed discrete/continuous (Bernoulli + Gaussian)
      6. Nested splice (sub-GF with `@trace`)
      7. Map combinator (3 elements)
      8. Unfold combinator (3 steps)
      9. Switch combinator (both branches)
      10. Model with 10+ addresses (stress test for weight accumulation)
    - **Conjugate posterior analytics** (~10 values): For Normal-Normal, Beta-Bernoulli,
      and Gamma-Poisson conjugate models, compute exact posterior parameters
  - **Requires**: Julia 1.9+ with Gen.jl installed on the developer's machine
  - **Run manually**: `julia test/reference/gen_jl_reference.jl`

### GenMLX test loader (ClojureScript side)

- [ ] **23.2** GenMLX test that loads and compares against `gen_jl_reference.json`
  - **File**: new `test/genmlx/gen_jl_differential_test.cljs` (~80 lines)
  - **Approach**:
    - Load JSON via `(js/JSON.parse (js/require "fs").readFileSync ...)`
    - Define the same 10 canonical models in GenMLX
    - For each distribution log-prob entry: compute GenMLX value, assert-close
      within float32 tolerance (1e-5)
    - For each GFI operation entry: run the same operation with the same fixed
      choicemap, assert-close on weight/score/discard values
    - For conjugate posteriors: verify analytical values match
  - **Expected assertions**: ~185 (125 log-prob + 50 GFI + 10 conjugate)
  - **Tolerance**: 1e-5 for most values (float32 vs float64), 1e-3 for values
    involving log-gamma approximations

### Extended comparison (optional, future)

- [ ] **23.3** Cross-process Gen.jl oracle for dynamic queries
  - **Scope**: ~100 lines Julia server + ~50 lines ClojureScript client
  - A persistent Julia process that accepts JSON commands over stdin/stdout:
    `{"op": "generate", "model": "linear_regression", "choices": {...}}` -> result
  - Enables property-style testing: generate random choicemaps in GenMLX, send to
    Gen.jl oracle, compare results
  - **Requires**: Julia installed on test machine, slower (Julia startup ~2-5s,
    or persistent server)
  - **When**: Only needed if the static reference file (23.1-23.2) proves insufficient

---

## Priority Order

```
CRITICAL (fix before any new feature work):
  (all fixed)

HIGH (correctness concerns affecting real use cases):
  22.2  Remove deprecated stateful PRNG            depends on 16.6
  22.3  mx/eval! after resampling                  ~2 lines  (done)
  16.1  Combinator sub-trace score=0              (done)
  16.2  execute-sub fake trace score=0            (done)
  16.3  assess not validating full constraints    (done)
  17.3  CustomGradientGF IUpdate/IRegenerate      (done)
  17.4  NeuralNetGF IUpdate/IRegenerate           (done)
  17.5  IProject on CustomGradientGF/NeuralNetGF  (done)
  17.6  MixCombinator IUpdateWithDiffs            (done)
  22.1  Adaptive HMC/NUTS step-size               (done)

MEDIUM (quality and completeness):
  18.1  MLX-native log-gamma in 4 distributions   ~80 lines
  18.2  Categorical native sample-n               ~15 lines
  18.3  Parameter validation for all dists        ~30 lines
  17.1  IAssess on combinators                    (done)
  17.2  IPropose on combinators                   (done)
  17.5  IProject on CustomGradientGF/NeuralNetGF  (done)
  19.2  Deduplicate resampling                    (done)
  19.3  Deduplicate Adam optimizer                ~5 lines refactor
  19.8  inference.cljs re-exports                 ~10 lines

LOW (cleanup and polish):
  16.4  Switch combinator branch switching        (done)
  16.6  PRNG key threading audit                  ~100+ lines
  18.4  Geometric support range                   ~5 lines
  18.5  CustomGradientGF gradient-fn wiring       ~20 lines
  19.1  Diff infrastructure cleanup               Decision needed
  19.4  defdist-transform usage                   Decision needed
  19.5  SelectAddrs/SelectSet merge               (done)
  19.6  clojure.set require                       ~1 line
  19.7  requiring-resolve workaround              (done)
  19.9  run-kernel/collect-samples dedup          (done)

MEDIUM-HIGH (verification -- catches bugs, path to formal proofs):
  24.1  Static validator (validate-gen-fn)          ~100-150 lines
  24.2  GFI contract registry (11 contracts)        ~150 lines
  24.3  verify-gfi-contracts function               ~50 lines
  24.4  Canonical model suite + contract test       ~80 lines, ~5500 checks
  23.1  Gen.jl reference value generation (Julia)   ~200 lines Julia
  23.2  GenMLX differential test loader             ~80 lines, ~185 assertions

MEDIUM-HIGH (testing -- catches bugs in existing code):
  ~~21.12 GFI contract harness + canonical models~~ ✅ DONE
  ~~21.1  Distribution E[X]/Var[X] for all dists~~ ✅ DONE
  ~~21.3  Update round-trip via discard~~ ✅ DONE
  ~~21.5  assess vs generate cross-validation~~ ✅ DONE
  ~~21.9  Score additivity per combinator~~ ✅ DONE

MEDIUM (testing -- deeper coverage):
  ~~21.2  Discrete PMF sums to 1~~ ✅ DONE
  ~~21.4  Edit round-trip via backward request~~ ✅ DONE
  ~~21.6  Propose -> generate round-trip~~ ✅ DONE
  ~~21.7  Combinator degenerate cases~~ ✅ DONE
  ~~21.10 Gamma-Poisson conjugate convergence~~ ✅ DONE
  ~~21.11 HMC/NUTS acceptance rate checks~~ ✅ DONE

LOW-MEDIUM (testing -- nice to have):
  ~~21.8  Nested combinator tests~~ ✅ DONE

HIGH (GPU ADEV — inspired by GenJAX, gate on benchmarks):
  ~~20.5  Vectorized ADEV (batched GPU gradient est.)~~ ✅ DONE (75-86x speedup)
  ~~20.6  Compiled ADEV optimization loop~~ ✅ DONE
  ~~20.7  ADEV benchmark suite (gate: ≥10x at N=100)~~ ✅ DONE (all PASS)
  20.4  ADEV variance reduction (baseline)           ~15 lines

FUTURE (new features, lower priority):
  7.8   GenJAX benchmark comparison
  9.2   Handler-level diff awareness
  10.14-10.16  Formal foundation completion
  11.1  Malli schemas
  12.3-12.4, 12.6  Ecosystem features
  13.1-13.3  Documentation & packaging
  14.2, 14.5-14.8  Gen.jl parity features
  20.1-20.3  Amortized inference improvements

RESEARCH (Lean 4 formalization):
  24.5  Formalize lambda_MLX types in Lean 4       Research-level
  24.6  Prove GFI axioms in Lean 4                 Research-level
  24.7  Spec-to-Lean translator                    ~200 lines
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
| 7. Vectorization & Perf | 9 | 7 | 2 |
| 8. Gradient Programming | 2 | 2 | 0 |
| 9. Incremental Computation | 2 | 1 | 1 |
| 10. Formal Foundation | 16 | 13 | 3 |
| 11. Validation | 2 | 1 | 1 |
| 12. Ecosystem | 6 | 3 | 3 |
| 13. Documentation | 3 | 0 | 3 |
| 14. Gen.jl Parity | 8 | 4 | 4 |
| 15. Confirmed Bugs | 5 | 5 | 0 |
| 16. Correctness Concerns | 6 | 5 | **1** |
| 17. Missing Protocols | 6 | 6 | 0 |
| 18. Distribution Quality | 5 | 4 | **1** |
| 19. Code Quality | 9 | 7 | **2** |
| 20. Amortized + GPU ADEV | 7 | 0 | **7** |
| 21. Testing Strategies | 12 | 12 | **0** |
| 22. Practical Inference | 3 | 2 | **1** |
| 23. Gen.jl Differential Testing | 3 | 0 | **3** |
| 24. Verified PPL | 7 | 0 | **7** |
| **Total** | **138** | **99** | **39** |
