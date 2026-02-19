# GenMLX Long-Term Roadmap: Closing the Gap with Gen.jl and GenJAX

> A phased plan to bring GenMLX to full feature parity with Gen.jl and GenJAX,
> while remaining 100% idiomatic ClojureScript -- purely functional, data-driven,
> and GPU-accelerated on Apple Silicon via MLX.

---

## Current State of GenMLX

GenMLX is a ~2000-line probabilistic programming system implementing the
**Generative Function Interface (GFI)** in ClojureScript on Node.js (nbb) with
MLX for GPU acceleration. It already provides:

| Category | What exists today |
|---|---|
| **GFI protocols** | `simulate`, `generate`, `assess`, `update`, `regenerate` |
| **DSL** | `gen` macro, `dyn/trace`, `dyn/splice` (Dynamic DSL only) |
| **Distributions (14)** | gaussian, uniform, bernoulli, beta, gamma, exponential, categorical, poisson, laplace, student-t, log-normal, multivariate-normal, dirichlet, delta |
| **Combinators (3)** | Map, Unfold, Switch |
| **Inference (7)** | Importance Sampling, Importance Resampling, MH, MALA, HMC, NUTS, SMC (with rejuvenation), ADVI |
| **Vectorized inference** | `vsimulate`, `vgenerate`, vectorized IS, vectorized SMC init (29-122x speedup via shape-based batching) |
| **Diagnostics** | ESS, R-hat, summary statistics |
| **MLX integration** | Full autograd (`grad`, `value-and-grad`, `jvp`, `vjp`), `compile-fn` (JIT to Metal), `vmap`, `tidy`, functional PRNG |
| **Data structures** | Hierarchical choice maps, immutable traces, `VectorizedTrace`, composable selections |

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
| **Cauchy** | `cauchy` | -- | -- | **GAP** |
| **Inverse Gamma** | `inv_gamma` | -- | -- | **GAP** |
| **Geometric** | `geometric` | -- | -- | **GAP** |
| **Negative Binomial** | `neg_binom` | -- | -- | **GAP** |
| **Binomial** | `binom` | -- | -- | **GAP** |
| **Discrete Uniform** | `uniform_discrete` | -- | -- | **GAP** |
| **Piecewise Uniform** | `piecewise_uniform` | -- | -- | **GAP** |
| **Beta-Uniform** | `beta_uniform` | -- | -- | **GAP** |
| **Truncated Normal** | -- | `truncated_normal` | -- | **GAP** |
| **Broadcasted Normal** | `broadcasted_normal` | -- | -- | **GAP** |
| **Wishart / Inv-Wishart** | via Distributions.jl | -- | -- | **GAP** |
| **`@dist` constructor** | `@dist` transform DSL | -- | -- | **GAP** |
| **Mixture constructors** | `HomogeneousMixture`, `HeterogeneousMixture` | -- | -- | **GAP** |
| **External dist compat** | GenDistributions.jl | ExactDensity / Distribution base | -- | **GAP** |

### 2. Inference Algorithms

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| Importance Sampling | `importance_sampling` | ImportanceK | `importance-sampling` | -- |
| Importance Resampling | `importance_resampling` | -- | `importance-resampling` | -- |
| MH (selection) | `mh(trace, sel)` | MH | `mh` | -- |
| MALA | `mala` | -- | `mala` | -- |
| HMC | `hmc` | HMC | `hmc` | -- |
| NUTS | -- | -- | `nuts` | -- (GenMLX ahead) |
| SMC / Particle Filter | `initialize_particle_filter` | Bootstrap PF | `smc` | -- |
| ADVI | -- | -- | `vi` | -- (GenMLX ahead) |
| **MH (custom proposal GF)** | `mh(trace, proposal, args)` | Custom proposals | -- | **GAP** |
| **Involutive MCMC** | `mh(trace, proposal, args, involution)` | Involutive MCMC | -- | **GAP** |
| **Elliptical Slice Sampling** | `elliptic_slice` | -- | -- | **GAP** |
| **MAP Optimization** | `map_optimize` | -- | -- | **GAP** |
| **Gibbs Sampling** | via enumeration | Enumerative Gibbs | -- | **GAP** |
| **Black-Box VI** | `black_box_vi!` | genjax.vi | -- | **GAP** |
| **VIMCO** | `black_box_vimco!` | -- | -- | **GAP** |
| **SMCP3** | GenSMCP3.jl | First-class SMCP3 | -- | **GAP** |
| **Programmable VI** | -- | PLDI 2024 module | -- | **GAP** |
| **ADEV gradient estimation** | -- | ADEV integration | -- | **GAP** |

### 3. Combinators & Model Composition

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| Map | `Map` | `Map` | `map-combinator` | -- |
| Unfold | `Unfold` | `Unfold` | `unfold-combinator` | -- |
| Switch / Cond | `Switch` | `Cond` | `switch-combinator` | -- |
| **Recurse** | `Recurse` | -- | -- | **GAP** |
| **Scan** | -- | `Scan` (lax.scan) | -- | **GAP** |
| **Vectorized Cond (masking)** | -- | Vectorized stochastic branching | -- | **GAP** |

### 4. Gradient & Differentiable Programming

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| `grad(f)` | -- | `jax.grad` | `mx/grad` | -- |
| `value-and-grad` | -- | `jax.value_and_grad` | `mx/value-and-grad` | -- |
| `jvp` / `vjp` | -- | native JAX | `mx/jvp`, `mx/vjp` | -- |
| `stop-gradient` | -- | `jax.lax.stop_gradient` | `mx/stop-gradient` | -- |
| **`choice_gradients`** | Per-choice gradients from trace | -- | -- | **GAP** |
| **`accumulate_param_gradients!`** | Gradient accum for `@param` | -- | -- | **GAP** |
| **Custom gradient GFs** | `CustomGradientGF` | -- | -- | **GAP** |
| **Argument gradient annotations** | `has_argument_grads` | -- | -- | **GAP** |

### 5. Trainable Parameters & Learning

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| **`@param` declarations** | Trainable params in GFs | -- | -- | **GAP** |
| **Parameter store** | `init_param!`, `get_params`, `set_param!` | -- | -- | **GAP** |
| **Parameter update configs** | `ParamUpdate`, `FixedStepGradientDescent`, Flux.ADAM | -- | -- | **GAP** |
| **`train!`** | Supervised training loop | -- | -- | **GAP** |
| **Wake-Sleep learning** | `lecture!`, `lecture_batched!` | -- | -- | **GAP** |
| **Amortized inference** | Via trained proposals | Neural proposals via Equinox | -- | **GAP** |

### 6. Trace Operations & Incremental Computation

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| `simulate` | yes | yes | yes | -- |
| `generate` | yes | yes | yes | -- |
| `assess` | yes | yes | yes | -- |
| `update` | yes | `edit` | yes | -- |
| `regenerate` | yes | -- | yes | -- |
| **`propose`** | `propose(gf, args)` | -- | -- | **GAP** |
| **`project`** | `project(trace, sel)` | -- | -- | **GAP** |
| **Argdiffs / Retdiffs** | `NoChange`, `UnknownChange`, `VectorDiff` | `Retdiff` change propagation | -- | **GAP** |
| **Incremental update** | Static DSL skips unchanged computation | Retdiff-based | -- | **GAP** |
| **EditRequest types** | -- | Parametric edit interface | -- | **GAP** |

### 7. Static DSL / Compilation

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| **Static computation graph** | `@gen (static)` compiles DAG | All GFs are JAX-traceable | -- | **GAP** |
| **Incremental trace update** | Skips unchanged subgraphs | XLA-compiled updates | -- | **GAP** |
| **Static choice maps** | `StaticChoiceMap` | JAX Pytree CMs | -- | **GAP** |
| **JIT-compiled inference loops** | -- | `jax.jit` over full loops | `mx/compile-fn` (partial) | **PARTIAL** |

### 8. Vectorization & Hardware Acceleration

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| GPU acceleration | CPU only | GPU/TPU via XLA | GPU via MLX (Apple Silicon) | -- |
| `compile-fn` / JIT | -- | `jax.jit` | `mx/compile-fn` | -- |
| Vectorized traces | -- | First-class trace vmap | `VectorizedTrace` (shape-based) | -- |
| Vectorized particle inference | -- | Vectorized particle inference | `vsimulate`/`vgenerate` (29-122x speedup) | -- |
| **Vectorized stochastic branching** | -- | Cond via masking | -- | **GAP** |
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

### Phase 1: Distribution Library Completeness

**Goal:** Match Gen.jl's full distribution library and add distribution
composition primitives.

**Priority:** High -- distributions are the atoms of every model.

#### 1a. Missing Primitive Distributions

All implementable with the existing `defdist` macro:

| Distribution | Constructor | Notes |
|---|---|---|
| Cauchy | `(cauchy loc scale)` | Standard Cauchy via ratio of normals |
| Inverse Gamma | `(inv-gamma shape scale)` | 1/Gamma transform |
| Geometric | `(geometric p)` | Discrete, has `support` |
| Negative Binomial | `(neg-binomial r p)` | Polya / waiting time |
| Binomial | `(binomial n p)` | Discrete, has `support` for small n |
| Discrete Uniform | `(discrete-uniform lo hi)` | Integer-valued, has `support` |
| Truncated Normal | `(truncated-normal mu sigma lo hi)` | Rejection or inverse-CDF |
| Piecewise Uniform | `(piecewise-uniform bounds probs)` | Continuous, piecewise |
| Beta-Uniform | `(beta-uniform theta alpha beta)` | Mixture of Beta and Uniform |
| Wishart | `(wishart df scale)` | Matrix-variate, via Bartlett decomposition |
| Inverse Wishart | `(inv-wishart df scale)` | Matrix-variate |

Add `reparam` methods where possible (Cauchy, Truncated Normal, Inv-Gamma via
transform).

#### 1b. Distribution Constructors

**`defdist-transform` macro** -- ClojureScript equivalent of Gen.jl's `@dist`:

```clojure
;; Declare a derived distribution via a deterministic transform of a base
(defdist-transform log-normal-alt [mu sigma]
  :base gaussian
  :forward exp
  :inverse log
  :log-det-jac (fn [v] v))  ;; log |d/dv exp(v)| = v
```

The macro generates `sample`, `log-prob`, and optionally `reparam` methods
by composing the base distribution with the transform + change-of-variables
Jacobian correction.

**`mixture` constructor** -- data-driven mixture distributions:

```clojure
;; Homogeneous mixture
(def mix (mixture gaussian [[0 1] [5 2]] [0.3 0.7]))

;; Heterogeneous mixture
(def hetmix (mixture [gaussian exponential] [[[0 1]] [[2]]] [0.6 0.4]))
```

Returns a `Distribution` record whose `sample` draws a component index from
`categorical`, then delegates to the selected component. `log-prob` computes
`logsumexp` over weighted component log-probs.

#### 1c. External Distribution Compatibility

A protocol bridge so that user-defined distributions need only supply
`{:sample (fn [key] ...) :log-prob (fn [v] ...)}` as a plain map:

```clojure
(defn map->dist
  "Lift a {:sample fn, :log-prob fn} map into a Distribution record."
  [{:keys [sample log-prob reparam support] :as spec}]
  ...)
```

**Estimated scope:** ~300 lines. No architectural changes needed.

---

### Phase 2: Custom Proposals & Advanced MCMC

**Goal:** Enable user-written proposal generative functions for MH, add
elliptical slice sampling, MAP optimization, and Gibbs sampling.

**Priority:** High -- custom proposals are the core of "programmable inference."

#### 2a. Custom Proposal MH

Gen.jl's `mh(trace, proposal, proposal_args)` variant, where `proposal` is
itself a generative function that proposes new values:

```clojure
(defn mh-custom
  "MH with a user-supplied proposal generative function.
   proposal: a gen fn (fn [trace & args] ...) that proposes new choices
   Returns updated trace or original (accept/reject)."
  [trace proposal proposal-args]
  ...)
```

The proposal GF is `simulate`d to get proposed choices, then the acceptance
ratio is computed from the model score difference and the forward/backward
proposal log-densities. This is the standard MH formula:

```
log-alpha = (model-score-new - model-score-old)
          + (backward-proposal-score - forward-proposal-score)
```

Requires implementing `propose` on `DynamicGF`:

```clojure
(defprotocol IPropose
  (propose [gf args]
    "Sample choices and return {:choices cm :weight log-q :retval v}."))
```

#### 2b. Elliptical Slice Sampling

A specialized MCMC kernel for models with multivariate Gaussian priors:

```clojure
(defn elliptical-slice
  "Elliptical slice sampling update for address `addr` with prior N(mu, cov).
   Requires no tuning parameters."
  [trace addr mu cov]
  ...)
```

Implementation: sample a random ellipse in the joint space of the current
value and a fresh prior sample, then shrink a bracket until a point with
higher likelihood is found. ~40 lines, purely functional.

#### 2c. MAP Optimization

Gradient ascent on the log-probability of selected continuous choices:

```clojure
(defn map-optimize
  "Backtracking gradient ascent on log p(trace) over selected choices.
   Returns optimized trace."
  [{:keys [selection learning-rate max-iters tolerance]} trace]
  ...)
```

Uses `mx/grad` of the score function, with optional line search. Returns
an updated trace with optimized values.

#### 2d. Gibbs Sampling with Enumeration

For discrete random variables with finite support, exact conditional
distributions can be computed by enumeration:

```clojure
(defn gibbs-step
  "One Gibbs step: enumerate all values of `addr`, compute conditional,
   and sample from it."
  [trace addr]
  (let [support (dist/dist-support (get-dist trace addr))
        scores  (mapv #(score-with-value trace addr %) support)
        probs   (softmax scores)]
    (sample-from-categorical trace addr support probs)))
```

Requires that distributions declare their support via the existing
`dist-support` multimethod (already implemented for `bernoulli`, `categorical`,
`delta`). Add `support` to `geometric`, `binomial`, `discrete-uniform`,
`neg-binomial`.

**Estimated scope:** ~400 lines across 4 files.

---

### Phase 3: Involutive MCMC & Reversible Jump

**Goal:** Implement Gen.jl's most powerful MCMC primitive -- the involutive
MCMC framework that enables trans-dimensional inference.

**Priority:** Medium-High -- unlocks structure learning and model selection.

#### 3a. Involutive MCMC Kernel

The three-phase kernel:

1. **Propose:** Simulate an auxiliary generative function to get forward
   choices `u`.
2. **Apply involution:** A user-supplied pure function
   `h : (trace-choices, u) -> (trace-choices', u')` where `h(h(x)) = x`.
3. **Accept/reject:** Compute weight from the Jacobian of the involution
   and the score difference.

```clojure
(defn involutive-mh
  "Involutive MCMC step.
   - proposal:   generative function producing auxiliary choices
   - involution: pure fn (trace-cm, aux-cm) -> (new-trace-cm, new-aux-cm)
   - jacobian?:  if true, compute log |det J| (for continuous transforms)"
  [trace proposal proposal-args involution & {:keys [jacobian?]}]
  ...)
```

The involution must be its own inverse -- GenMLX can verify this property
in debug mode by checking `h(h(input)) == input`.

#### 3b. Reversible Jump MCMC Helpers

Higher-level helpers for common trans-dimensional moves:

```clojure
;; Split/merge move
(defn split-merge-kernel [trace split-proposal merge-fn]
  ...)

;; Birth/death move
(defn birth-death-kernel [trace birth-proposal death-selection]
  ...)
```

These are sugar over `involutive-mh` with appropriate involutions.

**Estimated scope:** ~300 lines. Requires `propose` from Phase 2a.

---

### Phase 4: Trainable Parameters & Learning

**Goal:** Add trainable parameters to generative functions, enabling
gradient-based learning of model parameters (supervised, unsupervised,
amortized inference).

**Priority:** Medium-High -- essential for neural-network-infused models.

#### 4a. Parameter Store (Functional)

Unlike Gen.jl's mutable parameter store, GenMLX uses a purely functional
approach -- parameters are an explicit map threaded through training:

```clojure
(defrecord ParamStore [params grads])

(defn init-params
  "Initialize a parameter store from a generative function's @param declarations."
  [gen-fn initial-values]
  (->ParamStore initial-values (zipmap (keys initial-values) (repeat nil))))
```

Inside `gen` bodies, a new `dyn/param` form reads from the store:

```clojure
(gen [x]
  (let [theta (dyn/param :theta)]  ;; reads from implicit param store
    (dyn/trace :y (gaussian theta 1))))
```

The parameter store is threaded through handlers like the PRNG key --
no global mutable state.

#### 4b. Choice Gradients from Traces

Extract per-choice gradients from an executed trace:

```clojure
(defn choice-gradients
  "Compute gradients of log p(trace) w.r.t. selected continuous choices.
   Returns {:arg-grads [...] :choice-values cm :choice-grads cm}."
  [trace selection ret-grad]
  ...)
```

This uses `mx/grad` applied to a function that reconstructs the score from
choices, decomposing the gradient per-address.

#### 4c. Parameter Gradient Accumulation

```clojure
(defn accumulate-param-gradients
  "Compute gradient of log p(trace) w.r.t. trainable parameters.
   Returns updated ParamStore with accumulated gradients."
  [trace param-store ret-grad]
  ...)
```

#### 4d. Optimizers & Training Loop

Data-driven optimizer configs (purely functional -- no mutation):

```clojure
(defn sgd [lr] {:type :sgd :lr lr})
(defn adam [lr & {:keys [beta1 beta2 eps] :or {beta1 0.9 beta2 0.999 eps 1e-8}}]
  {:type :adam :lr lr :beta1 beta1 :beta2 beta2 :eps eps})

(defn apply-update
  "Apply one optimizer step. Returns [new-params new-opt-state]."
  [optimizer-config opt-state params grads]
  ...)

(defn train-step
  "One training step: run model, accumulate gradients, apply update."
  [model args observations param-store optimizer opt-state]
  ...)
```

#### 4e. Wake-Sleep Learning

```clojure
(defn lecture
  "Sleep phase: simulate from model p, train recognition model q on the trace."
  [model model-args recog-model recog-args param-store]
  ...)

(defn wake-sleep-step
  "One wake-sleep iteration: wake phase (train p) + sleep phase (train q)."
  [model recog-model data-batch param-store-p param-store-q optimizer]
  ...)
```

**Estimated scope:** ~500 lines. New namespace `genmlx.learning`.

---

### Phase 5: Incremental Computation & Performance

**Goal:** Add change-propagation hints (argdiffs/retdiffs) and a static
analysis pass so that `update` and `regenerate` can skip unchanged
computation -- critical for MCMC performance.

**Priority:** Medium -- performance optimization for iterative inference.

#### 5a. Diff Types

Purely functional change descriptors:

```clojure
(def no-change  {:type :no-change})
(def unknown-change {:type :unknown-change})

(defn vector-diff [changed-indices]
  {:type :vector-diff :changed changed-indices})

(defn map-diff [added removed changed]
  {:type :map-diff :added added :removed removed :changed changed})
```

#### 5b. Diff-Aware Update

Modify the `update` handler to accept and propagate diffs:

```clojure
(defprotocol IUpdateWithDiffs
  (update-with-diffs [gf trace new-args argdiffs constraints]
    "Like update, but with change hints for efficient incremental computation.
     Returns {:trace Trace :weight MLX-scalar :retdiff diff :discard ChoiceMap}."))
```

Combinators benefit enormously: `Map` can skip unchanged indices, `Unfold`
can start from the first changed timestep.

#### 5c. Static Analysis Pass (Optional)

A compile-time analysis of `gen` bodies to extract a dependency graph.
When a `gen` function's body is sufficiently simple (no recursion, bounded
loops), the macro can emit a static plan:

```clojure
(gen ^:static [x]
  (let [a (dyn/trace :a (gaussian 0 1))
        b (dyn/trace :b (gaussian a 1))]
    (dyn/trace :obs (gaussian b x))))
```

The `^:static` hint instructs the macro to analyze data flow and produce a
dependency map `{:b #{:a}, :obs #{:b}}`. During `update`, only addresses
downstream of changed values are re-executed.

**Estimated scope:** ~400 lines. `genmlx.diff` namespace + handler changes.

---

### Phase 6: SMCP3 & Programmable Sequential Inference

**Goal:** Implement Sequential Monte Carlo with Probabilistic Program Proposals
(SMCP3), bringing GenJAX's most distinctive inference capability to GenMLX.

**Priority:** Medium -- advanced but very powerful for sequential models.

#### 6a. Edit Requests

Generalize `update` into a parametric `edit` operation:

```clojure
(defprotocol IEdit
  (edit [gf trace edit-request]
    "Apply an edit request to a trace.
     Returns {:trace Trace :weight MLX-scalar :retdiff diff
              :discard ChoiceMap :backward-request EditRequest}."))
```

Different `EditRequest` types correspond to different SMCP3 moves:
- `ConstraintEdit` -- equivalent to current `update`
- `RegenerateEdit` -- equivalent to current `regenerate`
- `ProposalEdit` -- forward proposal with backward kernels

#### 6b. Trace Kernels

A DSL for defining stochastic maps between traces (inspired by
GenTraceKernelDSL.jl):

```clojure
(defn trace-kernel
  "Define a stochastic map from one trace to another.
   The body has access to the old trace and produces new choices."
  [kernel-fn]
  ...)
```

#### 6c. SMCP3 Inference

```clojure
(defn smcp3
  "Sequential Monte Carlo with Probabilistic Program Proposals.
   - forward-kernel: gen fn producing new choices given old trace + new obs
   - backward-kernel: gen fn producing auxiliary choices for weight computation
   Returns {:traces [...] :log-weights [...] :log-ml-estimate scalar}."
  [{:keys [particles forward-kernel backward-kernel ess-threshold]}
   model args observations-seq]
  ...)
```

**Estimated scope:** ~600 lines. New namespaces `genmlx.edit`, `genmlx.smcp3`.

---

### Phase 7: Vectorized Inference & Advanced Compilation

**Goal:** Bring GenJAX's key innovation -- compositional vectorization of
traces and inference -- to GenMLX via shape-based batching.

**Priority:** Medium -- significant performance gains for particle methods.

**Status:** Phase 7a-7b COMPLETE. Achieved 29-122x speedup via shape-based
batching (no `vmap` needed -- MLX broadcasting handles everything).

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
threading is already shape-agnostic. 7 distributions have native batch
sampling; the rest fall back to sequential `dist-sample-n :default`.

**Benchmarks (N=100, 5-site model):**
- `dist-sample-n`: 29x (N=100), 66x (N=1000)
- `vgenerate`: 61-122x
- Vectorized IS: 81x
- Vectorized SMC init: 65x

**Limitations (7.0):**
- No `splice` (sub-GF calls) in batched mode
- No vectorized `update`/`regenerate` (needed for multi-step SMC)
- Rejection-sampling distributions (beta, gamma, poisson, student-t,
  dirichlet) fall back to sequential

#### 7c. Vectorized Stochastic Branching

For the `Switch`/`Cond` combinator, replace Python-level branching with
masking so that all branches execute and results are selected:

```clojure
(defn vcond
  "Vectorized conditional: execute all branches, mask-select results.
   Compatible with vmap and compile-fn."
  [selector & branches]
  ...)
```

This is critical for JIT-compiling models with discrete latent structure.

#### 7d. Full Inference Loop Compilation

Compile entire MCMC chains or SMC sweeps into a single Metal program:

```clojure
(defn compiled-mh-chain
  "Compile an entire MH chain into a single Metal kernel."
  [model args observations selection n-samples]
  (mx/compile-fn
    (fn [key]
      ;; entire chain as a pure function of the PRNG key
      ...)))
```

**Remaining scope:** ~400 lines for 7c-7d. Vectorized branching and
compiled inference loops.

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

## Summary: Phase Dependencies

```
Phase 1: Distributions          (no deps -- start immediately)
    |
Phase 2: Custom Proposals       (uses Phase 1 distributions)
    |
Phase 3: Involutive MCMC        (requires Phase 2 propose)
    |
Phase 4: Parameters & Learning  (requires Phase 2 for proposal training)
    |
Phase 5: Incremental Compute    (independent, but benefits Phase 6)
    |
Phase 6: SMCP3                  (requires Phase 2, 3, 5)
    |
Phase 7: Vectorized Inference   (independent, but benefits all phases)
    |
Phase 8: Ecosystem              (independent, ongoing)
```

Phases 1, 5, 7, and 8 can proceed in parallel with the main sequence.

---

## Design Philosophy: GenMLX vs Gen.jl vs GenJAX

| Aspect | Gen.jl | GenJAX | GenMLX |
|---|---|---|---|
| **Language** | Julia (imperative + functional) | Python + JAX (functional transforms) | ClojureScript (purely functional) |
| **Mutation** | Mutable traces, parameter store | JAX Pytrees (immutable-ish) | Persistent data structures, no mutation |
| **Dispatch** | Julia multiple dispatch | Python class hierarchy | Clojure multimethods + protocols |
| **Extension** | Subtype `GenerativeFunction` | Subclass / Pytree | Implement protocols, add multimethods |
| **Compilation** | Julia JIT | XLA (GPU/TPU) | MLX Metal (Apple Silicon GPU) |
| **Vectorization** | Manual | `jax.vmap` (first-class) | Shape-based batching (29-122x speedup) |
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
- **Purely functional handler design** -- cleaner than Gen.jl's mutable traces
- **`defdist` macro** -- more ergonomic than Gen.jl's manual `Distribution` subtyping
- **MLX unified memory** -- zero-copy CPU/GPU is architecturally superior to PCIe transfer
- **Vectorized inference without `vmap`** -- shape-based batching achieves 29-122x
  speedup by broadcasting, simpler than GenJAX's JAX vmap approach
- **~2000 lines** -- dramatically smaller than Gen.jl (~20k+) or GenJAX (~10k+),
  making the system auditable and hackable

---

*This roadmap is a living document. Phases should be re-prioritized based on
user needs and upstream MLX capabilities.*
