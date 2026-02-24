# GenMLX Long-Term Roadmap: Closing the Gap with Gen.jl and GenJAX

> A phased plan to bring GenMLX to full feature parity with Gen.jl and GenJAX,
> while remaining 100% idiomatic ClojureScript -- purely functional, data-driven,
> and GPU-accelerated on Apple Silicon via MLX.
>
> *Updated 2026-02-24. Reflects elliptical slice, MAP, Recurse, project,
> loop compilation, adaptive HMC, batch dist-sample-n for rejection dists.*

---

## Current State of GenMLX

GenMLX is a ~14,000-line probabilistic programming system implementing the
**Generative Function Interface (GFI)** in ClojureScript on Node.js (nbb) with
MLX for GPU acceleration. It provides:

| Category | What exists today |
|---|---|
| **GFI protocols (8)** | `simulate`, `generate`, `assess`, `update`, `regenerate`, `propose`, `project`, `update-with-diffs` |
| **DSL** | `gen` macro, `dyn/trace`, `dyn/splice`, `dyn/param` (Dynamic DSL only) |
| **Distributions (27)** | gaussian, uniform, bernoulli, beta, gamma, exponential, categorical, poisson, laplace, student-t, log-normal, multivariate-normal, dirichlet, delta, cauchy, inv-gamma, geometric, neg-binomial, binomial, discrete-uniform, truncated-normal, mixture, half-normal, wishart, inv-wishart, von-mises, piecewise-uniform |
| **Combinators (9)** | Map, Unfold, Switch, Scan, Mask, Mix, Recurse, Contramap/MapRetval (dimap) |
| **Inference (20+)** | IS, Importance Resampling, MH, Custom Proposal MH, MALA, HMC (with adaptive step-size), NUTS, Enumerative Gibbs, Elliptical Slice Sampling, Involutive MCMC (with Jacobian), MAP, SMC, Conditional SMC, SMCP3, ADVI, Programmable VI (ELBO/IWELBO/PWake/QWake), Wake-Sleep |
| **Inference composition** | chain, repeat-kernel, seed, cycle-kernels, mix-kernels, run-kernel |
| **Loop compilation** | Entire MCMC chains compiled into single Metal dispatches (MH 5.6x, MALA 2.3x, HMC 3.9x) |
| **Edit interface** | IEdit protocol, ConstraintEdit, SelectionEdit, ProposalEdit with backward requests |
| **Incremental computation** | Argdiffs/retdiffs, diff-aware update on Map/Unfold/Scan/Switch combinators |
| **Trainable parameters** | `dyn/param`, functional parameter store, SGD/Adam optimizers, `train`, `make-param-loss-fn` |
| **Choice gradients** | Per-choice and score gradients via MLX autograd |
| **Vectorized inference** | `vsimulate`, `vgenerate`, `vupdate`, `vregenerate`, vectorized IS/SMC, vectorized switch (29-60x speedup via dispatch amortization) |
| **Diagnostics** | ESS, R-hat, summary statistics |
| **MLX integration** | Full autograd (`grad`, `value-and-grad`, `jvp`, `vjp`), `compile-fn` (JIT to Metal), `vmap`, `tidy`, functional PRNG |
| **Data structures** | Hierarchical choice maps, immutable traces, `VectorizedTrace`, composable selections |
| **Distribution constructors** | `defdist` macro, `defdist-transform` (derived dists via transforms), `mixture` constructor |

---

## Gap Analysis

The tables below identify every significant capability present in Gen.jl or
GenJAX that GenMLX does not yet have, organized by subsystem.

### 1. Distributions

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| Normal / Gaussian | `normal` | `normal` | `gaussian` | -- |
| Uniform continuous | `uniform` | `uniform` | `uniform` | -- |
| Bernoulli | `bernoulli` | `bernoulli`/`flip` | `bernoulli` | -- |
| Beta | `beta` | `beta` | `beta-dist` | -- |
| Gamma | `gamma` | -- | `gamma-dist` | -- |
| Exponential | `exponential` | -- | `exponential` | -- |
| Categorical | `categorical` | `categorical` | `categorical` | -- |
| Poisson | `poisson` | -- | `poisson` | -- |
| Laplace | `laplace` | -- | `laplace` | -- |
| Student-t | -- | -- | `student-t` | -- |
| Log-Normal | -- | -- | `log-normal` | -- |
| Multivariate Normal | `mvnormal` | `mv_normal` | `multivariate-normal` | -- |
| Dirichlet | `dirichlet` | -- | `dirichlet` | -- |
| Delta | -- | -- | `delta` | -- |
| Cauchy | `cauchy` | -- | `cauchy` | -- |
| Inverse Gamma | `inv_gamma` | -- | `inv-gamma` | -- |
| Geometric | `geometric` | -- | `geometric` | -- |
| Negative Binomial | `neg_binom` | -- | `neg-binomial` | -- |
| Binomial | `binom` | -- | `binomial` | -- |
| Discrete Uniform | `uniform_discrete` | -- | `discrete-uniform` | -- |
| Truncated Normal | -- | `truncated_normal` | `truncated-normal` | -- |
| Mixture | `HomogeneousMixture` | -- | `mixture` constructor | -- |
| `@dist` constructor | `@dist` transform DSL | -- | `defdist-transform` macro | -- |
| Piecewise Uniform | `piecewise_uniform` | -- | `piecewise-uniform` | -- |
| **Beta-Uniform** | `beta_uniform` | -- | -- | **GAP** |
| **Broadcasted Normal** | `broadcasted_normal` | -- | -- | **GAP** |
| Wishart / Inv-Wishart | via Distributions.jl | -- | `wishart`, `inv-wishart` | -- |
| **External dist compat** | GenDistributions.jl | ExactDensity / Distribution base | -- | **GAP** |

### 2. Inference Algorithms

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| Importance Sampling | `importance_sampling` | ImportanceK | `importance-sampling` | -- |
| Importance Resampling | `importance_resampling` | -- | `importance-resampling` | -- |
| MH (selection) | `mh(trace, sel)` | MH | `mh` | -- |
| MH (custom proposal GF) | `mh(trace, proposal, args)` | Custom proposals | `mh-custom` | -- |
| Involutive MCMC | `mh(trace, proposal, involution)` | Involutive MCMC | `involutive-mh` (with Jacobian) | -- |
| Enumerative Gibbs | via enumeration | Enumerative Gibbs | `gibbs` | -- |
| MALA | `mala` | -- | `mala` | -- |
| HMC | `hmc` | HMC | `hmc` | -- |
| NUTS | -- | -- | `nuts` | -- (GenMLX ahead) |
| SMC / Particle Filter | `initialize_particle_filter` | Bootstrap PF | `smc` | -- |
| Conditional SMC | -- | cSMC | `csmc` | -- |
| SMCP3 | GenSMCP3.jl | First-class SMCP3 | `smcp3` | -- |
| ADVI | -- | -- | `vi` | -- (GenMLX ahead) |
| Programmable VI | -- | PLDI 2024 module | `programmable-vi` (ELBO/IWELBO/PWake/QWake) | -- |
| Wake-Sleep | `lecture!` | -- | `wake-sleep` (with auto-discovery) | -- |
| Inference composition | -- | Kernel combinators | `chain`, `repeat-kernel`, `seed`, `cycle-kernels`, `mix-kernels` | -- |
| Elliptical Slice Sampling | `elliptic_slice` | -- | `elliptical-slice` | -- |
| MAP Optimization | `map_optimize` | -- | `map-optimize` + `vectorized-map-optimize` | -- (GenMLX ahead) |
| **VIMCO** | `black_box_vimco!` | -- | -- | **GAP** |
| **ADEV gradient estimation** | -- | ADEV integration | -- | **GAP** |

### 3. Combinators & Model Composition

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| Map | `Map` | `Map` | `map-combinator` (full GFI + diffs) | -- |
| Unfold | `Unfold` | `Unfold` | `unfold-combinator` (full GFI) | -- |
| Switch / Cond | `Switch` | `Cond` | `switch-combinator` (full GFI) | -- |
| Scan | -- | `Scan` (lax.scan) | `scan-combinator` (full GFI) | -- |
| Mask | -- | Masking combinator | `mask-combinator` | -- |
| Mix | -- | -- | `mix-combinator` (full GFI) | -- (GenMLX ahead) |
| Contramap / Dimap | -- | -- | `contramap-gf`, `map-retval`, `dimap` (full GFI) | -- (GenMLX ahead) |
| Vectorized Cond | -- | Vectorized stochastic branching | `vectorized-switch` (mx/where masking) | -- |
| Recurse | `Recurse` | -- | `recurse-combinator` (full GFI) | -- |

### 4. Gradient & Differentiable Programming

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| `grad(f)` | -- | `jax.grad` | `mx/grad` | -- |
| `value-and-grad` | -- | `jax.value_and_grad` | `mx/value-and-grad` | -- |
| `jvp` / `vjp` | -- | native JAX | `mx/jvp`, `mx/vjp` | -- |
| `stop-gradient` | -- | `jax.lax.stop_gradient` | `mx/stop-gradient` | -- |
| `choice_gradients` | Per-choice gradients from trace | -- | `choice-gradients`, `score-gradient` | -- |
| Param gradient accumulation | `accumulate_param_gradients!` | -- | `make-param-loss-fn` (via MLX autograd) | -- |
| **Custom gradient GFs** | `CustomGradientGF` | -- | -- | **GAP** |
| **Argument gradient annotations** | `has_argument_grads` | -- | -- | **GAP** |

### 5. Trainable Parameters & Learning

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| `@param` declarations | Trainable params in GFs | -- | `dyn/param` + `*param-store*` | -- |
| Parameter store | `init_param!`, `get_params`, `set_param!` | -- | `make-param-store`, `get-param`, `set-param` | -- |
| Parameter update configs | `ParamUpdate`, `FixedStepGradientDescent`, Flux.ADAM | -- | `sgd-step`, `adam-step` | -- |
| `train!` | Supervised training loop | -- | `train` (generic loop with pluggable optimizer) | -- |
| Wake-Sleep learning | `lecture!`, `lecture_batched!` | -- | `wake-sleep` (with auto-discovery) | -- |
| **Amortized inference** | Via trained proposals | Neural proposals via Equinox | -- | **GAP** |

### 6. Trace Operations & Incremental Computation

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| `simulate` | yes | yes | yes | -- |
| `generate` | yes | yes | yes | -- |
| `assess` | yes | yes | yes | -- |
| `update` | yes | `edit` | yes | -- |
| `regenerate` | yes | -- | yes | -- |
| `propose` | `propose(gf, args)` | -- | `propose` on DynamicGF + Distribution | -- |
| Argdiffs / Retdiffs | `NoChange`, `UnknownChange`, `VectorDiff` | `Retdiff` change propagation | `no-change`, `unknown-change`, `value-change`, `vector-diff`, `map-diff` | -- |
| Diff-aware update | Static DSL skips unchanged | Retdiff-based | `update-with-diffs` on Map (per-element), Unfold, Scan, Switch | -- |
| EditRequest types | -- | Parametric edit interface | `ConstraintEdit`, `SelectionEdit`, `ProposalEdit` + backward requests | -- |
| `project` | `project(trace, sel)` | -- | `project` via project-handler | -- |
| **Static DSL incremental** | Static DSL skips unchanged subgraphs | XLA-compiled updates | -- | **GAP** |

### 7. Static DSL / Compilation

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| **Static computation graph** | `@gen (static)` compiles DAG | All GFs are JAX-traceable | -- | **GAP** |
| **Incremental trace update** | Skips unchanged subgraphs | XLA-compiled updates | -- | **GAP** |
| **Static choice maps** | `StaticChoiceMap` | JAX Pytree CMs | -- | **GAP** |
| JIT-compiled inference loops | -- | `jax.jit` over full loops | `mx/compile-fn` loop compilation (MH 5.6x, MALA 2.3x, HMC 3.9x) | -- |

### 8. Vectorization & Hardware Acceleration

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| GPU acceleration | CPU only | GPU/TPU via XLA | GPU via MLX (Apple Silicon) | -- |
| `compile-fn` / JIT | -- | `jax.jit` | `mx/compile-fn` | -- |
| Vectorized traces | -- | First-class trace vmap | `VectorizedTrace` (shape-based) | -- |
| Vectorized particle inference | -- | Vectorized particle inference | `vsimulate`/`vgenerate`/`vupdate`/`vregenerate` (29-60x speedup via dispatch amortization) | -- |
| Vectorized stochastic branching | -- | Cond via masking | `vectorized-switch` + `mask-combinator` | -- |
| **Formal vectorization correctness** | -- | lambda_GEN proof (POPL '26) | -- | **GAP** |

### 9. Ecosystem & Tooling

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| **Neural network integration** | GenFlux.jl (Flux), GenTF.jl | Equinox, Flax | -- | **GAP** |
| **LLM-as-GF** | GenGPT3.jl | -- | -- | **GAP** |
| **Visualization** | -- | GenStudio | -- | **GAP** |
| **Auto-genify** | Genify.jl (auto-convert Julia fns) | -- | -- | **GAP** |
| **Stochastic probabilities** | GenSP.jl | -- | -- | **GAP** |
| **Trace kernel DSL** | GenTraceKernelDSL.jl | -- | -- | **GAP** |
| **External dist compat** | GenDistributions.jl | ExactDensity API | -- | **GAP** |
| **Input validation** | -- | Static tracing-time typecheck | -- | **GAP** |
| **Error messages** | -- | -- | -- | **GAP** |

---

## Roadmap

The roadmap is organized into **7 phases**, roughly ordered by
(a) foundational dependencies, (b) impact on user modeling power, and
(c) feasibility. Each phase is independently valuable and shippable.

### Design Principles (apply to every phase)

1. **Purely functional first.** State lives in persistent data structures passed
   through function arguments. No atoms, no mutation, no side effects in the
   modeling layer. The only mutable boundary is the handler's `volatile!`
   (already isolated).

2. **Data-driven, open for extension.** Prefer multimethods and records over
   closed hierarchies. New distributions, combinators, and inference algorithms
   should be addable without modifying core code.

3. **MLX arrays end-to-end.** Values stay as MLX arrays from sampling through
   scoring through gradient computation. Only extract to JS numbers at
   inference boundaries.

4. **Lazy graph + explicit eval.** Leverage MLX's lazy evaluation. Mark
   materialization points explicitly with `mx/eval!` to bound memory.

5. **ClojureScript idioms.** Protocols, multimethods, `defrecord`, transducers,
   persistent data structures, structural sharing, spec/Malli for validation.

---

### Phase 1: Distribution Library Completeness ✅ COMPLETE

**Goal:** Match Gen.jl's full distribution library and add distribution
composition primitives.

**Status:** 27 distributions total. All have native `dist-sample-n` batch
sampling (including rejection-sampling dists: beta, gamma, dirichlet, student-t).
`defdist-transform` macro and `mixture` constructor implemented.

#### 1a. Primitive Distributions — DONE + Remaining

All 7 planned distributions implemented via `defdist`. Remaining niche distributions:

| Distribution | Constructor | Status |
|---|---|---|
| Cauchy | `(cauchy loc scale)` | ✅ Done |
| Inverse Gamma | `(inv-gamma shape scale)` | ✅ Done |
| Geometric | `(geometric p)` | ✅ Done |
| Negative Binomial | `(neg-binomial r p)` | ✅ Done |
| Binomial | `(binomial n p)` | ✅ Done |
| Discrete Uniform | `(discrete-uniform lo hi)` | ✅ Done |
| Truncated Normal | `(truncated-normal mu sigma lo hi)` | ✅ Done |
| Piecewise Uniform | `(piecewise-uniform bounds probs)` | ✅ Done |
| Wishart | `(wishart df scale)` | ✅ Done |
| Inverse Wishart | `(inv-wishart df scale)` | ✅ Done |
| Half-Normal | `(half-normal sigma)` | ✅ Done |
| Von Mises | `(von-mises mu kappa)` | ✅ Done |

#### 1b. Distribution Constructors — DONE

**`defdist-transform` macro** ✅ — Implemented. Declares derived distributions
via deterministic transforms of base distributions. Tested with log-normal.

**`mixture` constructor** ✅ — Implemented as first-class Distribution record.
`sample` draws component index from categorical, delegates to selected component.
`log-prob` computes `logsumexp` over weighted component log-probs.

#### 1c. External Distribution Compatibility

A protocol bridge so that user-defined distributions need only supply
`{:sample (fn [key] ...) :log-prob (fn [v] ...)}` as a plain map:

```clojure
(defn map->dist
  "Lift a {:sample fn, :log-prob fn} map into a Distribution record."
  [{:keys [sample log-prob reparam support] :as spec}]
  ...)
```

**Remaining scope:** External dist bridge (~50 lines). Niche: beta-uniform.

---

### Phase 2: Custom Proposals & Advanced MCMC ✅ COMPLETE

**Goal:** Enable user-written proposal generative functions for MH, add
Gibbs sampling, and advanced MCMC.

**Status:** All sub-items implemented and tested.

#### 2a. Custom Proposal MH — DONE ✅

`IPropose` protocol implemented on DynamicGF and Distribution. `mh-custom`
accepts a proposal GF and optional backward-gf. Uses update weight for
acceptance ratio: `log-alpha = update-weight + backward-score - forward-score`.

#### 2b. Elliptical Slice Sampling — DONE ✅

`elliptical-slice-step` and `elliptical-slice` in `inference/mcmc.cljs`.
Specialized MCMC kernel for models with multivariate Gaussian priors.

#### 2c. MAP Optimization — DONE ✅

`map-optimize` and `vectorized-map-optimize` in `inference/mcmc.cljs`.
Gradient ascent on log-probability via `mx/grad`. Vectorized variant
runs N random restarts in parallel.

#### 2d. Gibbs Sampling with Enumeration — DONE ✅

`gibbs-step-with-support` and `gibbs` implemented. Enumerates all values
in support, computes scores, samples from softmax. Tested with posterior
concentration.

---

### Phase 3: Involutive MCMC & Reversible Jump ✅ CORE COMPLETE

**Goal:** Implement Gen.jl's involutive MCMC framework.

**Status:** `involutive-mh-step` and `involutive-mh` implemented with full
Jacobian support. Involution can return 2-tuple (volume-preserving) or
3-tuple with `log|det J|`. Uses update weight for acceptance ratio.

#### 3a. Involutive MCMC Kernel — DONE ✅

Three-phase kernel: propose auxiliary choices, apply involution, accept/reject
with `log-alpha = update-weight + bwd-score - fwd-score + log|det J|`.
Tested with both volume-preserving and non-volume-preserving transforms.

#### 3b. Reversible Jump MCMC Helpers — REMAINING

Higher-level sugar for split/merge and birth/death moves.
~60 lines over involutive-mh.

---

### Phase 4: Trainable Parameters & Learning ✅ COMPLETE

**Goal:** Add trainable parameters, gradient infrastructure, and learning
algorithms.

**Status:** All sub-items implemented in `genmlx.learning` and `genmlx.gradients`.

#### 4a. Parameter Store — DONE ✅

Purely functional parameter store: `make-param-store`, `get-param`, `set-param`,
`params->array`, `array->params`. Inside gen bodies, `dyn/param` reads from a
`*param-store*` dynamic var bound by `simulate-with-params`/`generate-with-params`.

#### 4b. Choice Gradients — DONE ✅

`choice-gradients` and `score-gradient` in `genmlx.gradients`. Uses `mx/grad`
to compute per-choice gradients of log p(trace).

#### 4c. Parameter Gradient Accumulation — DONE ✅

`make-param-loss-fn` creates a differentiable loss-gradient function from a
model + observations + param names. Gradients flow through MLX arrays correctly.

#### 4d. Optimizers & Training — DONE ✅

`sgd-step`, `adam-step`, `adam-init` optimizers. Generic `train` loop with
pluggable optimizer, learning rate, callback.

#### 4e. Wake-Sleep Learning — DONE ✅

`wake-phase-loss`, `sleep-phase-loss`, and `wake-sleep` with auto-discovery
of guide addresses.

---

### Phase 5: Incremental Computation & Performance ✅ CORE COMPLETE

**Goal:** Add change-propagation hints (argdiffs/retdiffs) so `update` can
skip unchanged computation.

**Status:** Diff types, `IUpdateWithDiffs` protocol, and combinator
implementations all complete. Static analysis pass remains aspirational.

#### 5a. Diff Types — DONE ✅

`no-change`, `unknown-change`, `value-change`, `vector-diff`, `map-diff` in
`genmlx.diff`. Utility functions: `compute-diff`, `compute-vector-diff`,
`compute-map-diff`, `should-recompute?`.

#### 5b. Diff-Aware Update — DONE ✅

`IUpdateWithDiffs` protocol implemented on DynamicGF (no-change fast path),
MapCombinator (full per-element optimization with stored element scores),
and Unfold/Scan/Switch (no-change fast path).

#### 5c. Static Analysis Pass — REMAINING

Compile-time dependency graph extraction for `gen` bodies.
Aspirational — requires significant macro engineering.

---

### Phase 6: SMCP3 & Programmable Sequential Inference ✅ COMPLETE

**Goal:** Implement SMCP3 and the edit interface.

**Status:** All sub-items implemented.

#### 6a. Edit Requests — DONE ✅

`IEdit` protocol with `edit-dispatch` handling `ConstraintEdit` (→ update),
`SelectionEdit` (→ regenerate), `ProposalEdit` (→ propose + update + assess).
Backward requests generated automatically. Implemented on all 9 GF record types.

#### 6b. Inference Kernel Composition — DONE ✅

`chain`, `repeat-kernel`, `seed`, `cycle-kernels`, `mix-kernels`, `mh-kernel`,
`run-kernel` in `genmlx.inference.kernel`.

#### 6c. SMCP3 Inference — DONE ✅

`smcp3-init`, `smcp3-step`, `smcp3` in `genmlx.inference.smcp3`.
Supports custom forward/backward kernels, ESS-based resampling, and
optional MCMC rejuvenation.

---

### Phase 7: Vectorized Inference & Advanced Compilation

**Goal:** Bring GenJAX's key innovation -- compositional vectorization of
traces and inference -- to GenMLX via shape-based batching.

**Priority:** Medium -- significant performance gains for particle methods.

**Status:** MOSTLY COMPLETE. Achieved 29-122x vectorized speedup via shape-based
batching, plus 2.3-5.6x loop compilation speedup for MCMC chains.
No `splice` in batched mode. No compiled SMC sweeps.

#### 7a. Vectorized Traces ✅ DONE

Struct-of-arrays trace where each leaf is a batched `[N]`-shaped MLX array:

```clojure
(defrecord VectorizedTrace [gen-fn args choices score weight n-particles retval])
;; choices: ChoiceMap where leaves hold [N]-shaped arrays
;; score/weight: [N]-shaped MLX arrays
```

Includes `resample-vtrace`, `vtrace-log-ml-estimate`, `vtrace-ess`,
`systematic-resample-indices`. Implemented in `genmlx.vectorized`.

#### 7b. Vectorized GFI Operations ✅ DONE

Shape-based batching: sample `[N]` values at each site instead of running
N sequential particles. Broadcasting handles all score/weight arithmetic.

```clojure
(dyn/vsimulate model args n key)     ;; → VectorizedTrace
(dyn/vgenerate model args obs n key) ;; → VectorizedTrace with weights
```

Also: `vectorized-importance-sampling`, `vsmc-init`.

**Key insight:** No `vmap` needed. `dist-sample-n` produces `[N]`-shaped
samples, and `dist-log-prob` broadcasts naturally. The handler state
threading is already shape-agnostic. All distributions have native batch
sampling via `dist-sample-n`.

**Benchmarks (N=100, 5-site model, dispatch amortization):**
- `dist-sample-n`: 68x (N=1000, sub-ms at N=100)
- `vgenerate`: 57x
- Vectorized IS: 53x
- Vectorized SMC init: 62x

**Limitations (remaining):**
- No `splice` (sub-GF calls) in batched mode

#### 7c. Vectorized Stochastic Branching ✅ DONE

`vectorized-switch` executes all branches with N independent samples each,
combines results using `mx/where` based on [N]-shaped index arrays.
`mask-combinator` gates execution on a boolean condition.

#### 7d. Full Inference Loop Compilation ✅ DONE

Entire MCMC chains compiled into single Metal dispatches via `mx/compile-fn`.
Pre-generated noise arrays passed as inputs (compile-fn freezes random ops).

- **compiled-mh**: 5.6x speedup (K-step chains as one dispatch)
- **MALA**: 2.3x speedup (score+grad caching, 3→1 val-grad/step)
- **HMC**: 3.9x speedup (K outer × L inner leapfrog unrolling)
- **HMC adaptive step-size**: Dual averaging (Hoffman & Gelman 2014) during burn-in

---

### Phase 8: Ecosystem & Interop (Long-Term)

**Goal:** Build out the ecosystem with neural network integration,
visualization, and interop.

#### 8a. Neural Network Integration

A protocol for wrapping MLX neural networks as generative functions:

```clojure
(defn nn->gen-fn
  "Wrap an MLX neural network as a generative function.
   The network's forward pass becomes the body; parameters become @param values."
  [network param-map]
  ...)
```

Enables neural amortized proposals and neural likelihood models.

#### 8b. Visualization (GenStudio-inspired)

A ClojureScript visualization layer using Observable Plot or Vega-Lite:

```clojure
(defn plot-trace [trace]
  "Render a trace as an interactive visualization.")

(defn plot-posterior [samples addresses]
  "Plot marginal posteriors from MCMC samples.")

(defn animate-smc [smc-result]
  "Animate particle filter evolution over time.")
```

Could leverage ClojureScript's excellent JavaScript interop and
libraries like `oz` or `vega-lite-clj`.

#### 8c. LLM-as-Generative-Function

Following GenGPT3.jl, wrap LLM API calls as generative functions:

```clojure
(defn llm-gen-fn
  "Create a generative function that calls an LLM API.
   Each token position is an addressed random choice."
  [model-name prompt-template]
  ...)
```

#### 8d. Input Validation (Malli)

Malli schemas for all public API functions:

```clojure
(def DistributionSchema
  [:map
   [:type keyword?]
   [:params [:vector :any]]])

(def InferenceOptsSchema
  [:map
   [:samples pos-int?]
   [:burn {:optional true} nat-int?]
   [:thin {:optional true} pos-int?]])
```

Instrument in development mode, strip in production for zero overhead.

**Estimated scope:** Open-ended. Each sub-project is independently valuable.

---

## Summary: Phase Status

```
Phase 1: Distributions          ✅ COMPLETE (27 distributions)
Phase 2: Custom Proposals       ✅ COMPLETE (ESS, MAP, Gibbs all done)
Phase 3: Involutive MCMC        ✅ CORE COMPLETE (RJ helpers remaining)
Phase 4: Parameters & Learning  ✅ COMPLETE
Phase 5: Incremental Compute    ✅ CORE COMPLETE (static analysis remaining)
Phase 6: SMCP3                  ✅ COMPLETE
Phase 7: Vectorized Inference   ✅ MOSTLY COMPLETE (no splice in batched mode, no compiled SMC)
Phase 8: Ecosystem              ❌ NOT STARTED
```

**Remaining gaps (by priority):**
1. Static DSL / computation graph (significant effort)
2. VIMCO, ADEV gradient estimation (~200 lines)
3. Ecosystem: NN integration, visualization, LLM-as-GF (open-ended)

---

## Design Philosophy: GenMLX vs Gen.jl vs GenJAX

| Aspect | Gen.jl | GenJAX | GenMLX |
|---|---|---|---|
| **Language** | Julia (imperative + functional) | Python + JAX (functional transforms) | ClojureScript (purely functional) |
| **Mutation** | Mutable traces, parameter store | JAX Pytrees (immutable-ish) | Persistent data structures, no mutation |
| **Dispatch** | Julia multiple dispatch | Python class hierarchy | Clojure multimethods + protocols |
| **Extension** | Subtype `GenerativeFunction` | Subclass / Pytree | Implement protocols, add multimethods |
| **Compilation** | Julia JIT | XLA (GPU/TPU) | MLX Metal (Apple Silicon GPU) |
| **Vectorization** | Manual | `jax.vmap` (first-class) | Shape-based batching (29-60x speedup via dispatch amortization) |
| **PRNG** | Global RNG | JAX splittable keys | Splittable keys (functional) |
| **Memory** | Julia GC | XLA managed | MLX unified memory (zero-copy CPU/GPU) |

GenMLX's unique advantage: **unified memory architecture** on Apple Silicon
means CPU control flow and GPU numerics share the same memory with zero
transfer cost. This is ideal for the "thin CPU control + fat GPU compute"
pattern of MCMC inference.

---

## What GenMLX Already Does Better

- **NUTS sampler** -- neither Gen.jl nor GenJAX ship a built-in NUTS
- **ADVI** -- Gen.jl's `black_box_vi!` is more general but less turnkey
- **Adaptive HMC** -- dual averaging step-size tuning (Hoffman & Gelman 2014)
- **Vectorized MAP** -- N random restarts optimized simultaneously
- **Loop-compiled MCMC** -- entire chains as single Metal dispatches (MH 5.6x, MALA 2.3x, HMC 3.9x)
- **Purely functional handler design** -- cleaner than Gen.jl's mutable traces
- **`defdist` macro** -- more ergonomic than Gen.jl's manual `Distribution` subtyping
- **MLX unified memory** -- zero-copy CPU/GPU is architecturally superior to PCIe transfer
- **Vectorized inference without `vmap`** -- shape-based batching amortizes Metal
  dispatch across N particles (53-68x at N=100), simpler than GenJAX's JAX vmap
- **~14,000 lines** -- dramatically smaller than Gen.jl (~20k+) or GenJAX (~10k+),
  making the system auditable and hackable

---

*This roadmap is a living document. Phases should be re-prioritized based on
user needs and upstream MLX capabilities.*
