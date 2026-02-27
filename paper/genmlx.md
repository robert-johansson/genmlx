# GenMLX: A Probabilistic Programming Language on Apple Silicon

## Abstract

We present GenMLX, a probabilistic programming language implemented in approximately 10,400 lines of ClojureScript, targeting Apple's MLX framework for GPU-accelerated inference on Apple Silicon. GenMLX implements the Generative Function Interface (GFI), the same compositional abstraction used by Gen.jl and GenJAX, but with a purely functional design built on persistent data structures, open multimethods, and shape-based vectorization. The system provides 27 probability distributions (with a `defdist` macro for zero-boilerplate definitions and a first-class mixture constructor), ten model combinators (Map, Unfold, Switch, Scan, Mask, Mix, Recurse, Contramap, Dimap, and a vmap combinator), and a comprehensive inference toolkit spanning importance sampling, nine MCMC algorithms (MH, custom-proposal MH, enumerative Gibbs, involutive MCMC, MALA, HMC, NUTS, elliptical slice sampling, MAP), four SMC variants (standard, conditional, vectorized, SMCP3), variational inference (ADVI, programmable VI with five objectives including VIMCO), ADEV gradient estimation, amortized inference via trained neural proposals, and wake-sleep learning. Adaptive tuning (dual averaging for step-size, Welford metric estimation) and loop compilation (fusing multi-step MCMC chains into single Metal dispatches) further improve gradient-based samplers. A parametric edit interface generalizes trace mutations into typed requests with automatic backward computation for reversible kernels. Diff-aware updates enable incremental computation in combinators. Trainable parameters (`dyn/param`) integrate the learning infrastructure with the generative function interface. Vectorized execution achieves significant speedup over sequential particle methods by exploiting MLX's broadcasting semantics rather than explicit vectorization transforms. All computation stays on GPU as lazy MLX arrays throughout the inference pipeline, with materialization deferred to inference boundaries. The system passes 1,300+ test assertions across 77 test files, including compatibility suites adapted from Gen.jl (165 tests) and GenJAX (73 tests).

---

## 1. Introduction

Probabilistic programming languages (PPLs) allow practitioners to express probabilistic models as programs and perform inference automatically. The field has produced several mature systems: Stan for Bayesian statistics, Pyro and NumPyro for deep probabilistic models, and the Gen family --- Gen.jl in Julia, GenJAX on JAX --- which introduced the Generative Function Interface (GFI) as a compositional abstraction for programmable inference.

Despite the maturity of these systems, none targets Apple Silicon's unified memory architecture. Apple's MLX framework provides a distinctive computational model: CPU and GPU share the same memory with zero transfer cost, operations are lazy by default (building computation graphs that fuse before execution), and the framework provides automatic differentiation, JIT compilation to Metal shaders, and vectorization primitives. This architecture is particularly well-suited to probabilistic programming, where MCMC and SMC algorithms require tight interleaving of CPU control flow (accept/reject decisions, resampling, tree traversal) with GPU numerics (score evaluation, gradient computation, leapfrog integration).

GenMLX explores what happens when a PPL is designed from the ground up for this architecture. The result is a system that:

- Implements the full GFI in ~10,400 lines of ClojureScript, running on Node.js via nbb with native MLX bindings
- Keeps all probabilistic values as lazy MLX arrays throughout the inference pipeline
- Achieves significant speedup for particle methods through shape-based vectorization that requires no code transformation
- Provides gradient-based inference (MALA, HMC, NUTS, ADVI) with JIT-compiled score functions, fused leapfrog integration, and adaptive step-size and metric tuning
- Matches and extends GenJAX's inference algorithm repertoire: custom-proposal MH, enumerative Gibbs, involutive MCMC, elliptical slice sampling, SMCP3, conditional SMC, programmable variational inference with five objectives (ELBO, IWELBO, PWake, QWake, VIMCO), ADEV gradient estimation, amortized inference, and composable inference kernels
- Provides a parametric edit interface, diff-aware incremental updates, trainable parameters inside model bodies, a neural network integration layer, and a vmap combinator for batched generative functions
- Maintains semantic equivalence with Gen.jl and GenJAX, verified by 1,300+ test assertions

### 1.1 The Gen Ecosystem

The Generative Function Interface, introduced by Cusumano-Towner et al. in Gen.jl, decomposes probabilistic programming into a small set of operations on generative functions:

- **simulate**: forward-sample all random choices, producing a trace
- **generate**: execute under constraints (observations), returning a trace and importance weight
- **update**: modify a trace with new constraints, returning weight and discarded choices
- **regenerate**: resample selected addresses, returning the acceptance weight
- **propose**: forward-sample all choices and return choices + their joint log-probability
- **assess**: score fully-specified choices under the model

This interface is compositional: distributions, user-defined models, and combinators all implement the same protocols, enabling generic inference algorithms that work across model structures. GenJAX brought this interface to JAX, adding hardware acceleration, vectorization via `jax.vmap`, and a parametric edit interface that generalizes update and regenerate into a single operation.

GenMLX is the third implementation of the GFI, bringing it to ClojureScript and Apple Silicon, and the first to match GenJAX's full inference algorithm repertoire outside the JAX ecosystem.

---

## 2. System Architecture

GenMLX is organized into seven layers, each building on the layers below. This section describes each layer and the design decisions that shape it.

### 2.1 MLX Foundation (Layer 0)

The foundation is a thin ClojureScript wrapper over `@frost-beta/mlx`, the Node.js native addon for Apple's MLX framework. The wrapper (`genmlx.mlx`, 440 lines) exposes approximately 130 MLX operations as idiomatic ClojureScript functions:

```clojure
(mx/add a b)              ;; element-wise addition (lazy)
(mx/multiply a b)         ;; element-wise multiplication (lazy)
(mx/grad f)               ;; automatic differentiation
(mx/compile-fn f)         ;; JIT compile to Metal shader
(mx/eval! a b c)          ;; materialize lazy computation graph
(mx/vmap f)               ;; vectorized map
```

All MLX operations are lazy by default. Calling `(mx/add a b)` does not compute the sum; it records the operation in a computation graph. The graph is materialized only when `mx/eval!` is called, at which point MLX fuses operations into optimized Metal kernels. This laziness is critical: it allows GenMLX to build up entire score computations --- spanning multiple distribution log-probabilities and accumulations --- as a single fused GPU dispatch.

The wrapper covers arithmetic, math functions (including special functions like `lgamma`, `digamma`, `erf`, `erfinv`), reductions with axis support, comparison and selection, shape manipulation, indexing, matrix operations, linear algebra (Cholesky, solve, QR, SVD, eigendecomposition), autograd (`grad`, `value-and-grad`, `jvp`, `vjp`), transforms (`compile-fn`, `vmap`), Metal GPU device discovery and monitoring, and explicit memory management (`tidy`, `dispose!`, cache control).

A companion namespace (`genmlx.mlx.random`, 166 lines) provides functional PRNG key management. Every random operation consumes a key and produces sub-keys via deterministic splitting, following the splittable PRNG design pioneered by JAX:

```clojure
(rng/fresh-key)           ;; create initial key
(rng/split key)           ;; -> [k1 k2]
(rng/normal key [n])      ;; sample n standard normals
```

This eliminates hidden PRNG state --- every sample is a pure function of its key. A dynamic variable `*prng-key*` enables reproducible inference by threading a PRNG key through all GFI operations within a `with-key` scope. The module provides 10 key-based sampling functions: `normal`, `uniform`, `bernoulli`, `categorical`, `randint`, `gumbel`, `laplace`, `truncated-normal`, `multivariate-normal`, and `permutation`.

### 2.2 Core Data Types (Layer 1)

Three persistent data structures form the vocabulary of GenMLX:

**Choice Maps** (`genmlx.choicemap`, 228 lines) are hierarchical trees mapping addresses to values. They use ClojureScript's persistent maps for structural sharing, with a thin protocol (`IChoiceMap`) distinguishing leaf values from interior nodes:

```clojure
(cm/choicemap :slope (mx/scalar 2.0) :intercept (mx/scalar 1.0))
(cm/get-choice cm [:params :slope])
(cm/set-choice cm [:obs :y0] (mx/scalar 3.0))
```

Choice maps are the universal data interchange format: observations are choice maps, trace contents are choice maps, and update constraints are choice maps. The module also provides batching utilities (`stack-choicemaps`, `unstack-choicemap`) for converting between N scalar choice maps and one choice map with `[N]`-shaped leaf arrays.

**Traces** (`genmlx.trace`, 13 lines) are immutable records holding the result of a generative function execution:

```clojure
(defrecord Trace [gen-fn args choices retval score])
```

The `score` field holds `log p(choices | args)` as a lazy MLX array, staying on GPU until explicitly materialized.

**Selections** (`genmlx.selection`, 60 lines) are a composable algebra for identifying subsets of addresses. They support flat selection (`(sel/select :x :y)`), hierarchical selection, complement, and the universal selections `all` and `none`. Selections drive the `regenerate` operation, which resamples only the selected addresses while holding others fixed.

### 2.3 GFI Protocols and Execution Engine (Layer 2)

The GFI is expressed as nine ClojureScript protocols:

```clojure
(defprotocol IGenerativeFunction (simulate [gf args]))
(defprotocol IGenerate (generate [gf args constraints]))
(defprotocol IUpdate (update [gf trace constraints]))
(defprotocol IRegenerate (regenerate [gf trace selection]))
(defprotocol IAssess (assess [gf args choices]))
(defprotocol IPropose (propose [gf args]))
(defprotocol IProject (project [gf trace selection]))
(defprotocol IUpdateWithDiffs (update-with-diffs [gf trace constraints argdiffs]))
(defprotocol IHasArgumentGrads (has-argument-grads [gf]))
```

The first four are the core GFI from Gen.jl. `IAssess` and `IPropose` enable custom-proposal MH and involutive MCMC. `IProject` scores a subset of trace addresses. `IUpdateWithDiffs` enables diff-aware incremental computation in combinators (Section 2.9). `IHasArgumentGrads` reports which arguments support gradient computation.

The execution engine (`genmlx.handler`, 608 lines) implements these operations through a handler-based architecture. When a model body calls `(dyn/trace :addr dist)`, the call dispatches to whichever handler is currently active. Each handler is decomposed into a **pure state transition function** and a **thin mutable wrapper**:

```clojure
;; Pure transition: (state, addr, dist) -> (value, state')
(defn- simulate-transition [state addr dist]
  (let [[k1 k2] (rng/split (:key state))
        value (dc/dist-sample dist k2)
        lp    (dc/dist-log-prob dist value)]
    [value (-> state
             (assoc :key k1)
             (update :choices #(cm/set-choice % [addr] value))
             (update :score #(mx/add % lp)))]))

;; Mutable wrapper: reads/writes a volatile!
(defn simulate-handler [addr dist]
  (let [[value state'] (simulate-transition @*state* addr dist)]
    (vreset! *state* state')
    value))
```

The state is an immutable map containing `:choices`, `:score`, `:weight`, `:key`, `:constraints`, and optionally `:param-store` for trainable parameters. The volatile (`*state*`) never escapes the `run-handler` boundary --- it exists solely to allow the model body to be written in direct style rather than continuation-passing style.

This separation has a crucial consequence: the pure transition functions are entirely agnostic to the shapes of the values flowing through them. A score that starts as a scalar `(mx/scalar 0.0)` and accumulates scalar log-probabilities works identically to a score that starts as `(mx/scalar 0.0)` and accumulates `[N]`-shaped log-probabilities via broadcasting. This shape-agnosticism is what enables vectorized execution without modifying the handler (Section 5).

The handler provides six scalar transitions (simulate, generate, assess, update, regenerate, project) and four batched transitions (batched-simulate, batched-generate, batched-update, batched-regenerate) that sample `[N]`-shaped arrays at each trace site, enabling all particles to be processed in a single model execution. Sub-generative-function calls via `dyn/splice` are handled transparently, with native batched execution for DynamicGF sub-models and a per-particle fallback for combinator sub-models.

### 2.4 Dynamic DSL (Layer 3)

The user-facing modeling language is built on a `gen` macro (`genmlx.gen`, 19 lines) that wraps a ClojureScript function body into a `DynamicGF` record:

```clojure
(def model
  (gen [xs]
    (let [slope     (dyn/trace :slope (dist/gaussian 0 10))
          intercept (dyn/trace :intercept (dist/gaussian 0 10))]
      (mx/eval! slope intercept)
      (let [s (mx/item slope) i (mx/item intercept)]
        (doseq [[j x] (map-indexed vector xs)]
          (dyn/trace (keyword (str "y" j))
                     (dist/gaussian (+ (* s x) i) 1)))
        [s i]))))
```

Inside `gen` bodies, `dyn/trace` addresses a random choice, `dyn/splice` calls a sub-generative function at a namespaced address, and `dyn/param` reads a trainable parameter from the active parameter store:

```clojure
(let [mu (dyn/param :mu 0.0)]        ;; trainable parameter (default 0.0)
  (dyn/trace :obs (dist/gaussian mu 1)))
```

The `DynamicGF` record (`genmlx.dynamic`, 380 lines) implements all nine GFI protocols by selecting the appropriate handler and running the body function. It also provides four vectorized entry points (`vsimulate`, `vgenerate`, `vupdate`, `vregenerate`) that run the model body once with batched handlers for N-particle inference.

Unlike Gen.jl's approach of compiling the model body into a static computation graph, GenMLX's dynamic DSL supports full ClojureScript control flow --- loops, recursion, higher-order functions, conditionals on random values. This flexibility comes at the cost of incremental computation (a static graph can skip unchanged subexpressions during `update`), though the diff-aware update system (Section 2.9) recovers some of this performance at the combinator level.

### 2.5 Distributions (Layer 4)

GenMLX provides 27 probability distributions, all built on a single `Distribution` record with open multimethods:

```clojure
(defrecord Distribution [type params]
  IGenerativeFunction
  (simulate [this _] (dist-simulate this))
  IGenerate
  (generate [this _ constraints] (dist-generate this constraints)))
```

The `defdist` macro (`genmlx.dist.macros`, 153 lines) reduces distribution definitions to their mathematical essence:

```clojure
(defdist gaussian
  "Gaussian (normal) distribution with mean mu and std sigma."
  [mu sigma]
  (sample [key]
    (mx/add mu (mx/multiply sigma (rng/normal key []))))
  (log-prob [v]
    (let [z (mx/divide (mx/subtract v mu) sigma)]
      (mx/negative
        (mx/add (mx/scalar (* 0.5 LOG-2PI))
                (mx/log sigma)
                (mx/multiply (mx/scalar 0.5) (mx/square z))))))
  (reparam [key]
    (mx/add mu (mx/multiply sigma (rng/normal key [])))))
```

The macro generates the constructor function, `defmethod` implementations for `dist-sample`, `dist-log-prob`, `dist-reparam`, and optionally `dist-support` (for enumeration in Gibbs sampling), and wraps all constructor parameters with `mx/ensure-array` so that plain numbers are automatically promoted to MLX scalars.

The 27 distributions span:
- **Continuous**: gaussian, uniform, exponential, laplace, log-normal, beta, gamma, student-t, cauchy, inverse-gamma, truncated-normal, broadcasted-normal, piecewise-uniform
- **Discrete**: bernoulli, categorical, poisson, geometric, negative-binomial, binomial, discrete-uniform
- **Multivariate**: multivariate-normal, dirichlet
- **Matrix**: wishart, inverse-wishart
- **Structural**: delta (point mass)
- **Composite**: mixture (first-class mixture constructor)

Twenty distributions provide native batch sampling via `dist-sample-n` for vectorized inference; the remainder fall back to sequential sampling with automatic stacking.

The `mixture` constructor in `dist/core.cljs` creates mixture distributions as first-class `Distribution` records with proper log-prob computation via log-sum-exp. A `map->dist` function enables creating distributions dynamically from maps of functions, enabling runtime distribution construction without macros.

Because distributions are GFI participants --- they implement `IGenerativeFunction` and `IGenerate` --- they can be used directly as arguments to inference algorithms, as components in combinators, and as building blocks in hierarchical models.

### 2.6 Combinators (Layer 5)

Ten combinators compose generative functions into higher-level models (`genmlx.combinators`, 1,571 lines):

**Map** applies a kernel generative function independently to each element of input sequences, analogous to Gen.jl's `Map`. Each invocation is addressed by its integer index, enabling per-element constraints and updates. Map stores per-element scores as trace metadata, enabling diff-aware updates that skip unchanged elements (Section 2.9).

**Unfold** applies a kernel sequentially, threading state from one step to the next. This is the natural combinator for time-series and sequential models: the kernel receives `[t state & extra-args]` and returns the next state. Supports prefix-skip optimization: only recomputes from the first changed step.

**Switch** selects between multiple branch generative functions based on a runtime index, enabling mixture models and discrete structural choices.

**Scan** is a state-threading sequential combinator equivalent to GenJAX's `scan` (and `jax.lax.scan`). The kernel takes `[carry input]` and returns `[new-carry output]`, applied over a sequence while accumulating both carry-state and outputs. More general than Unfold for models that need to accumulate per-step outputs.

**Mask** gates execution of a generative function on a boolean condition. When masked (condition = false), the GF is not executed and contributes zero score. Used internally by vectorized switch for all-branch execution with mask-selection.

**Mix** is a first-class mixture model combinator that combines multiple component generative functions with mixing weights. It samples a component index from a categorical distribution, then executes the selected component. Both the index and the component's choices appear in the trace. Log-weights can be a function of arguments for adaptive mixtures.

**Recurse** is a fixed-point combinator for recursive models. A maker function receives a reference to the combinator itself and returns a generative function, enabling recursive calls. This supports random trees, linked lists, context-free grammars, and other recursive stochastic structures.

**Contramap** transforms arguments before passing to an inner generative function, enabling argument preprocessing without modifying the inner model.

**Dimap** transforms both arguments and return values, composing a Contramap with a return-value map (MapRetvalGF).

**Vmap** (`genmlx.vmap`, 428 lines) applies a kernel GF independently to N elements, storing choices with `[N]`-shaped leaf arrays instead of integer-indexed sub-choicemaps. This is GenMLX's analog of `jax.vmap` applied to generative functions.

All combinators implement the complete GFI --- `simulate`, `generate`, `update`, `regenerate`, `assess`, `propose`, `project`, `edit`, and `update-with-diffs` --- so they compose seamlessly with all inference algorithms. A vectorized switch mechanism executes all branches with N independent samples each, then combines results using `mx/where` based on `[N]`-shaped index arrays, enabling vectorized mixture models.

### 2.7 Inference Algorithms (Layer 6)

GenMLX provides a comprehensive inference toolkit spanning the major paradigms.

**Importance Sampling and Resampling** (`importance.cljs`, 81 lines). The simplest particle method: run `generate` with observations to get weighted samples, then optionally resample proportional to weights. A vectorized variant (`vectorized-importance-sampling`) runs all particles in a single model execution (Section 5).

**Metropolis-Hastings** (`mcmc.cljs`, 1,730 lines). Nine MCMC algorithms:

- *Standard MH*: Uses the GFI `regenerate` operation as the proposal mechanism, resampling selected addresses from the prior.
- *Custom-proposal MH*: Accepts a user-written proposal generative function. Runs `propose` on the proposal GF, applies proposed choices via `update`, and computes the Metropolis-Hastings acceptance ratio from the update weight, forward score, and backward score. This is the foundation of programmable inference.
- *Enumerative Gibbs*: Exact conditional sampling for discrete variables with finite support. Enumerates all values in `dist-support`, computes the conditional score for each via `update`, and samples from the resulting categorical distribution.
- *Involutive MCMC*: The general MCMC framework from Gen.jl for trans-dimensional inference. Three-phase kernel: (1) propose auxiliary choices from a proposal GF, (2) apply a user-supplied involution `h(trace-cm, aux-cm) -> (new-trace-cm, new-aux-cm)`, (3) accept/reject with score difference.
- *MALA* (Metropolis-Adjusted Langevin Algorithm): Gradient-informed proposals shifted in the direction of the score gradient, with a Metropolis correction for the asymmetric proposal.
- *HMC* (Hamiltonian Monte Carlo): Simulates Hamiltonian dynamics via fused leapfrog integration that merges adjacent half-kicks, reducing gradient evaluations from 2L to L+1 for L steps. Supports diagonal and dense mass matrices.
- *NUTS* (No-U-Turn Sampler): Adaptively selects trajectory length by recursively doubling the tree of leapfrog states until a U-turn is detected, following the Hoffman and Gelman algorithm with generalized U-turn detection.
- *Elliptical Slice Sampling*: A tuning-free sampler for models with Gaussian priors that proposes along elliptical slices through the current state.
- *MAP* (Maximum A Posteriori): Gradient ascent on the log-posterior, returning the mode rather than samples.

**Adaptive Tuning.** HMC and NUTS support automatic tuning during warmup via dual averaging (Algorithm 4, Hoffman & Gelman 2014) for step-size adaptation and Welford online variance estimation for diagonal mass matrix learning. HMC defaults to a target acceptance rate of 0.65, NUTS to 0.8.

**Loop Compilation.** MH, MALA, and HMC support loop-compiled chains: multiple MCMC steps are fused into a single Metal dispatch via `mx/compile-fn`, with pre-generated noise arrays passed as inputs. This amortizes the N-API call overhead across K steps, yielding significant speedup over per-step eager execution.

**Vectorized MCMC.** MH, MALA, HMC, and MAP provide vectorized variants that run N independent chains simultaneously via `[N,D]`-shaped parameter arrays, with per-particle accept/reject decisions.

Score functions and their gradients are JIT-compiled via `mx/compile-fn` for amortized Metal dispatch.

**Sequential Monte Carlo** (`smc.cljs`, 406 lines). Four SMC variants:

- *Standard SMC*: Particle filtering with adaptive resampling and optional MCMC rejuvenation. Each timestep: extend particles with new observations via `update`, reweight, resample if ESS falls below threshold, rejuvenate. Supports three resampling strategies: systematic, residual, and stratified.
- *Conditional SMC* (cSMC): SMC with a retained reference particle for particle MCMC / PMCMC. The reference particle is never resampled; its trajectory is preserved through conditional resampling.
- *Vectorized SMC*: Uses shape-based batching for initialization and rejuvenation, running all particles in a single model execution with vectorized MH rejuvenation.
- *Vectorized SMC init*: Uses shape-based batching for the initialization step, running all particles in a single model execution.

**SMCP3** (`smcp3.cljs`, 182 lines). Sequential Monte Carlo with Probabilistic Program Proposals --- the most powerful inference algorithm in GenJAX. Uses custom generative functions as proposals in SMC, with automatic incremental weight computation via the edit interface (Section 2.8). Supports per-particle forward and backward proposal kernels.

**Variational Inference** (`vi.cljs`, 459 lines):

- *ADVI*: Fits a mean-field Gaussian guide by maximizing the ELBO via Adam optimization. Uses reparameterized sampling and `mx/vmap` to batch ELBO sample evaluation.
- *Programmable VI*: Supports five variational objectives and two gradient estimators. Pluggable objectives enable different trade-offs between bound tightness and gradient variance.

The five variational objectives are:
- **ELBO**: Standard evidence lower bound --- `E_q[log p(x,z) - log q(z)]`
- **IWELBO**: Importance-weighted ELBO (IWAE bound) --- tighter bound that approaches `log p(x)` as K increases
- **PWake**: P-Wake objective --- trains model parameters by minimizing `KL(q || p)`
- **QWake**: Q-Wake objective --- trains guide parameters using self-normalized importance weights
- **VIMCO**: Variational Inference with Multi-sample Objectives --- leave-one-out control variates for lower variance than REINFORCE with non-reparameterizable distributions

Two gradient estimators:
- **Reparameterization**: Gradient flows through the sampling process (for continuous, reparameterizable distributions)
- **REINFORCE**: Score function estimator with variance-reducing mean baseline (for discrete or non-reparameterizable distributions)

**ADEV Gradient Estimation** (`inference/adev.cljs`, 301 lines). Automatic Differentiation of Expected Values computes unbiased gradients of expected costs by automatically choosing reparameterization or score-function estimation at each trace site, following Lew et al. (POPL 2023).

**Amortized Inference** (`inference/amortized.cljs`, 136 lines). Trains neural encoder networks `q(z|x; θ)` to approximate posteriors via reparameterized ELBO (VAE-style training), then uses trained encoders as proposals in importance sampling.

**Wake-Sleep Learning** (`learning.cljs`, 324 lines). Amortized inference via alternating wake and sleep phases: wake phase minimizes KL(q||p) by optimizing guide parameters, sleep phase minimizes KL(p||q) by training the guide on model prior samples.

**Inference Composition** (`kernel.cljs`, 223 lines). Composable inference kernels with algebraic operators:

```clojure
(chain k1 k2 k3)           ;; sequential composition
(repeat-kernel 10 k)        ;; apply k 10 times
(seed k fixed-key)           ;; fix PRNG key
(cycle-kernels 30 [k1 k2])  ;; round-robin cycling
(mix-kernels [[k1 0.7] [k2 0.3]])  ;; random mixture
(run-kernel {:samples 1000 :burn 200 :thin 2} k init-trace)
```

An inference kernel is simply a function `(fn [trace key] -> trace)`. The `run-kernel` executor handles burn-in, thinning, callbacks, and acceptance rate tracking.

**Choice Gradients** (`gradients.cljs`, 68 lines). Per-choice gradients of log p(trace) w.r.t. individual continuous choices, and score gradients w.r.t. flat parameter arrays. Foundation for gradient-based learning and parameter training.

**Custom Gradient GFs** (`custom_gradient.cljs`, 69 lines). Generative functions with user-defined gradient behavior, enabling gradient overrides for distributions or model components where automatic differentiation is insufficient.

**Diagnostics** (`diagnostics.cljs`, 125 lines). Effective sample size (ESS) via the autocorrelation-based estimator with Geyer's initial positive sequence truncation, R-hat (Gelman-Rubin) convergence diagnostic for multiple chains, and summary statistics (mean, standard deviation, quantiles).

### 2.8 Edit Interface

The edit interface (`edit.cljs`, 108 lines) generalizes `update` and `regenerate` into a single parametric operation with typed EditRequests and automatic backward request generation:

```clojure
(defprotocol IEdit
  (edit [gf trace edit-request]))
```

Three EditRequest types cover the space of trace mutations:

- **ConstraintEdit**: equivalent to `update` --- changes constrained values. The backward request contains the discarded values.
- **SelectionEdit**: equivalent to `regenerate` --- resamples selected addresses. The backward request is the same selection (regenerate is its own inverse as a proposal).
- **ProposalEdit**: forward + backward proposal GFs for SMCP3-style reversible kernels. Runs `propose` on the forward GF, applies choices via `update`, scores the reverse move via `assess` on the backward GF, and combines weights. The backward request swaps forward and backward GFs.

`IEdit` is implemented on all ten generative function types (DynamicGF, MapCombinator, UnfoldCombinator, SwitchCombinator, ScanCombinator, MaskCombinator, MixCombinator, RecurseCombinator, ContramapGF, MapRetvalGF). Each delegates to a default `edit-dispatch` that handles all three request types, but individual types can override with specialized implementations. SMCP3 uses the polymorphic `edit/edit` method, so custom GF types automatically work with SMCP3 inference.

### 2.9 Diff-Aware Incremental Updates

The diff system (`diff.cljs`, 121 lines) provides change-tagged values for incremental computation:

```clojure
diff/no-change              ;; value has not changed
diff/unknown-change         ;; assume everything changed (conservative)
(diff/value-change old new) ;; specific old/new values
(diff/vector-diff #{2 5})   ;; indices 2 and 5 changed
(diff/map-diff changed added removed)  ;; map key changes
```

The `IUpdateWithDiffs` protocol enables combinators to skip unchanged sub-computations:

```clojure
(p/update-with-diffs model trace constraints (diff/vector-diff #{0}))
```

**MapCombinator** provides the key optimization: it stores per-element scores as trace metadata. When called with a `vector-diff`, only the changed elements are updated --- unchanged elements reuse their stored choices and scores. Elements with new constraints are also detected and updated even when the argdiff says no-change. This makes updating one element of a 1000-element mapped model O(1) instead of O(1000).

**UnfoldCombinator** and **ScanCombinator** provide prefix-skip optimization: they store per-step scores and carry states, and only recompute from the first changed step forward.

Other combinators (Switch, Mask, Mix, Recurse) provide `no-change` fast paths that return the trace unchanged with zero weight when no arguments or constraints have changed. DynamicGF delegates to regular `update` (since the body must be re-executed), with a `no-change` shortcut when there are no constraints.

### 2.10 Trainable Parameters

The `dyn/param` function enables declaring trainable parameters inside `gen` bodies:

```clojure
(def model
  (gen [obs-val]
    (let [mu (dyn/param :mu 0.0)]    ;; reads from param store, or default 0.0
      (dyn/trace :obs (dist/gaussian mu 1))
      mu)))
```

The implementation uses a dynamic variable `*param-store*` in the handler. When no param store is bound, `dyn/param` returns the default value. When a param store is bound (via `learning/simulate-with-params` or `learning/generate-with-params`), it reads the stored MLX array, which participates in the computation graph for gradient flow.

The learning module provides integration utilities:

```clojure
;; Run model with parameter store
(learn/simulate-with-params model args param-store)
(learn/generate-with-params model args observations param-store)

;; Create differentiable loss-gradient function for training
(learn/make-param-loss-fn model args observations [:mu :sigma])
;; Returns (fn [params-array key] -> {:loss :grad})
```

The `make-param-loss-fn` builds a param store from a flat parameter array, binds it, runs `generate`, and returns the negative log-weight as the loss. Gradients flow through the MLX arrays correctly, enabling standard optimization (SGD, Adam) over model parameters.

### 2.11 Neural Network Integration

The neural network module (`nn.cljs`, 119 lines) bridges MLX's native `nn.Module` infrastructure with the GFI:

- **Layer constructors**: `linear`, `sequential`, `relu`, `gelu`, `tanh-act`, `sigmoid-act`, `dropout`, `layer-norm`, `embedding` wrap MLX's native nn.Module classes
- **GFI bridge**: `nn->gen-fn` wraps any nn.Module as a deterministic generative function (delta distribution over the forward pass output), enabling neural networks to be composed with probabilistic models via `dyn/splice`
- **Training**: `value-and-grad` and `step!` provide gradient computation and parameter updates for nn.Module parameters via MLX's native `nn.valueAndGrad`

This enables amortized inference architectures where neural networks serve as recognition models or proposal distributions within the GFI framework.

---

## 3. The MLX Advantage: Unified Memory and Lazy Evaluation

### 3.1 Unified Memory Architecture

On Apple Silicon, CPU and GPU share the same physical memory. There is no PCIe bus, no explicit data transfer, no `cuda.memcpy`. An MLX array created on CPU is immediately accessible to GPU kernels, and vice versa.

This eliminates the dominant bottleneck in GPU-accelerated MCMC: the synchronization cost of transferring data between host and device for each accept/reject decision. In a typical CUDA-based implementation, each MH step requires:

1. GPU: evaluate score at proposed state
2. Transfer: copy score to CPU
3. CPU: compute acceptance probability, draw uniform, decide
4. Transfer: copy decision back to GPU (or copy new state to GPU)

On unified memory, steps 2 and 4 are zero-cost. The score evaluation produces a lazy MLX array; `mx/item` materializes it in-place without transfer. The acceptance decision is pure CPU control flow. The next iteration's score evaluation picks up right where the previous one left off.

### 3.2 Lazy Evaluation as a Compilation Strategy

MLX's lazy evaluation model means that a sequence of operations like:

```clojure
(-> (mx/subtract v mu)
    (mx/divide sigma)
    mx/square
    (mx/multiply (mx/scalar 0.5))
    mx/negative)
```

builds a five-node computation graph without executing anything. When `mx/eval!` is called, MLX fuses these operations into a single Metal kernel. This is effectively JIT compilation at the operation level.

GenMLX exploits this by deferring `mx/eval!` as late as possible. A typical score computation spans multiple `dyn/trace` calls, each accumulating a log-probability into the running score. The entire score --- across all trace sites --- remains a lazy graph until the inference algorithm needs its numerical value. At that point, a single `mx/eval!` materializes the entire computation in one fused GPU dispatch.

For gradient-based inference, `mx/compile-fn` takes this further: it traces the score function once, caches the resulting Metal program, and reuses it across all subsequent evaluations. Combined with `mx/grad` (which is also lazy), this means the entire gradient computation --- forward pass, backward pass, and parameter update --- can be a single cached Metal dispatch.

### 3.3 Loop Compilation

Loop compilation extends the compilation strategy to multi-step MCMC chains. Instead of dispatching one Metal kernel per MCMC step, an entire K-step chain is compiled into a single Metal dispatch:

```clojure
(let [chain-fn (fn [params noise uniforms]
                 (loop [p params, i 0]
                   (if (>= i k-steps)
                     p
                     (let [proposal (mx/add p (mx/multiply std (mx/index noise i)))
                           ...]
                       (recur (mx/where accept? proposal p) (inc i))))))
      compiled (mx/compile-fn chain-fn)]
  ;; Each call dispatches ONE Metal program for K steps
  (compiled init-params (rng/normal key [k n]) (rng/uniform key [k])))
```

The noise arrays are pre-generated outside the compiled function, ensuring correct randomness while keeping the entire chain within a single dispatch. This technique applies to MH, MALA, and HMC, and is particularly effective for burn-in where per-step output is not needed.

### 3.4 Memory Management

MLX's lazy evaluation can accumulate large computation graphs if left unchecked. GenMLX manages this through:

- **Explicit `mx/eval!` at step boundaries**: each MCMC step, leapfrog step, or VI iteration materializes its results, bounding graph size
- **`mx/tidy` blocks**: wrap computations that produce intermediate arrays, automatically disposing them after the block returns
- **Periodic cache clearing**: long-running inference loops (SMC, compiled VI, wake-sleep) periodically call `mx/clear-cache!` to prevent Metal cache exhaustion
- **Score scalars, not tensors**: the typical GenMLX trace score is a scalar MLX array, not a large tensor, so the memory pressure from score accumulation is minimal

---

## 4. Distribution Design: Data-Driven and Extensible

### 4.1 Single Record, Open Multimethods

A common pattern in PPLs is to define each distribution as a separate class or type. Gen.jl uses Julia's type system (`struct Normal <: Distribution{Float64}`), NumPyro uses Python classes, and Stan uses a fixed set of built-in distributions.

GenMLX takes a different approach: all distributions share a single `Distribution` record with a `:type` keyword, and behavior is defined through open multimethods:

```clojure
(defrecord Distribution [type params])

(defmulti dist-sample   (fn [d _key] (:type d)))
(defmulti dist-log-prob (fn [d _value] (:type d)))
(defmulti dist-sample-n (fn [d _key _n] (:type d)))
```

This design means:

1. **No class hierarchy.** There is no abstract base class to subclass, no interface to implement beyond the multimethods.
2. **Extension from any namespace.** A user can define a new distribution in their own code without modifying GenMLX:

```clojure
(defdist my-custom-dist [param1 param2]
  (sample [key] ...)
  (log-prob [v] ...))
```

3. **Data serialization for free.** Distribution instances are plain Clojure maps: `{:type :gaussian :params {:mu <mlx-array> :sigma <mlx-array>}}`. They can be serialized, compared, and inspected with standard Clojure tools.
4. **GFI participation for free.** The `Distribution` record implements `IGenerativeFunction` and `IGenerate`, so every distribution is automatically a valid generative function that can be used in combinators, passed to inference algorithms, and composed in model hierarchies.
5. **Runtime construction.** The `map->dist` function creates distributions dynamically from maps of functions, enabling distribution construction at runtime without macros.

### 4.2 The `defdist` Macro

The `defdist` macro eliminates boilerplate by generating the constructor function and all multimethod implementations from a declarative specification:

```clojure
(defdist exponential
  "Exponential distribution with the given rate."
  [rate]
  (sample [key]
    (let [u (rng/uniform key [])]
      (mx/divide (mx/negative (mx/log (mx/subtract (mx/scalar 1.0) u))) rate)))
  (log-prob [v]
    (let [log-density (mx/subtract (mx/log rate) (mx/multiply rate v))
          non-neg (mx/greater-equal v (mx/scalar 0.0))]
      (mx/where non-neg log-density (mx/scalar ##-Inf))))
  (reparam [key]
    (let [u (rng/uniform key [])]
      (mx/divide (mx/negative (mx/log (mx/subtract (mx/scalar 1.0) u))) rate))))
```

The macro generates:
- A constructor function `(exponential rate)` that wraps `rate` with `mx/ensure-array`
- `defmethod dist-sample :exponential` with PRNG key management
- `defmethod dist-log-prob :exponential` with value wrapping
- `defmethod dist-reparam :exponential` (optional, for gradient flow)
- `defmethod dist-support :exponential` (optional, for enumeration)

### 4.3 Log-Probability Broadcasting

A critical design property: all `dist-log-prob` implementations use element-wise MLX operations. This means they broadcast naturally over batched inputs. If `dist-sample` returns a scalar and `dist-log-prob` is called on that scalar, the result is a scalar log-probability. If `dist-sample-n` returns an `[N]`-shaped array and `dist-log-prob` is called on that array, the result is an `[N]`-shaped array of log-probabilities --- with no code changes.

This broadcasting property is not an accident; it is the foundation of GenMLX's vectorized execution (Section 5).

---

## 5. Vectorized Inference via Shape-Based Batching

### 5.1 The Performance Problem

Sequential particle methods --- importance sampling, SMC --- run the model independently for each particle. With N particles and S trace sites, this means:

- N x S distribution samples (N x S N-API calls from JavaScript to MLX)
- N x S log-probability evaluations
- N `mx/eval!` calls to materialize weights (N CPU-GPU synchronization points)

For N=100 and S=5, this is 1,000 N-API calls and 100 sync points. The GPU compute per call is trivial (scalar operations), so overhead dominates. The GPU is idle most of the time, waiting for the next tiny kernel to be dispatched.

### 5.2 The Key Insight: Broadcasting Instead of `vmap`

GenJAX solves this problem with `jax.vmap`: transform a scalar model function into a batched version that processes all particles simultaneously. This is powerful but imposes constraints --- the model must be JAX-traceable, which excludes Python control flow over random values.

GenMLX takes a different approach: **change the shape of what flows through the model, not the model itself.** Instead of sampling a scalar at each trace site, sample an `[N]`-shaped array:

```clojure
;; Sequential: sample 1 value, run N times
(rng/normal key [])        ;; -> scalar

;; Batched: sample N values, run once
(rng/normal key [N])       ;; -> [N]-shaped array
```

Because all subsequent operations (log-prob, score accumulation, weight computation) use element-wise MLX operations, they broadcast automatically:

```clojure
;; scalar + [N] -> [N]
(mx/add (mx/scalar 0.0) (dist-log-prob gaussian-dist batch-values))
```

The handler's state threading is already shape-agnostic (Section 2.3), so a batched handler is structurally identical to the scalar one --- it just calls `dist-sample-n` instead of `dist-sample`.

### 5.3 Implementation

The implementation adds several components:

**`dist-sample-n` multimethod** (`dist/core.cljs`). A new multimethod with a sequential fallback:

```clojure
(defmulti dist-sample-n (fn [d _key _n] (:type d)))

;; Default: sequential fallback for distributions that can't batch
(defmethod dist-sample-n :default [d key n]
  (let [keys (rng/split-n (rng/ensure-key key) n)]
    (mx/stack (mapv #(dist-sample d %) keys))))
```

Twenty distributions provide native batch sampling (one or two lines each: call `rng/normal` or `rng/uniform` with shape `[n]` instead of `[]`). Distributions with complex structure (neg-binomial, wishart, inv-wishart) or those defined via `map->dist` use the sequential fallback.

**Batched handler transitions** (`handler.cljs`). Four batched transitions (simulate, generate, update, regenerate) that are structurally identical to the scalar transitions but calling `dist-sample-n`:

```clojure
(defn- batched-simulate-transition [state addr dist]
  (let [n (:batch-size state)
        [k1 k2] (rng/split (:key state))
        value (dc/dist-sample-n dist k2 n)     ;; [N]-shaped
        lp    (dc/dist-log-prob dist value)]    ;; [N] via broadcasting
    [value (-> state
             (assoc :key k1)
             (update :choices #(cm/set-choice % [addr] value))
             (update :score #(mx/add % lp)))]))
```

For constrained sites, the observation is a scalar. Its log-probability is also a scalar. When this scalar is added to the `[N]`-shaped running score, MLX broadcasts it across all particles --- exactly the correct semantics (all particles see the same observation).

**`VectorizedTrace` record** (`vectorized.cljs`, 153 lines). A trace where choice map leaves hold `[N]`-shaped arrays, and score/weight are `[N]`-shaped. With utilities for systematic resampling (both CPU loop-based and GPU-native implementations), ESS computation, log marginal likelihood estimation, and per-particle trace merging via boolean masks.

**`vsimulate` / `vgenerate` / `vupdate` / `vregenerate`** (`dynamic.cljs`). Public API functions that run the model body once with the appropriate batched handler, producing or updating a `VectorizedTrace`.

**Vectorized switch** (`combinators.cljs`). For models with discrete latent structure, executes all branches with N independent samples each, then combines results using `mx/where` based on `[N]`-shaped index arrays. This enables vectorized mixture models and clustering models.

### 5.4 Performance Characteristics

The speedup comes entirely from eliminating per-particle overhead. The GPU compute is the same --- evaluating N scalar Gaussians sequentially takes the same number of FLOPs as evaluating one `[N]`-shaped Gaussian. But the sequential version makes Nx more N-API calls and Nx more synchronization points.

The speedup scales with particle count: at higher N, the overhead per particle in the batched case is effectively zero (bounded by a constant number of API calls regardless of N), while the sequential version scales linearly in N.

### 5.5 Limitations

The current vectorized execution has several limitations:

- **No `splice`**: sub-generative-function calls are not supported in batched mode, because the sub-GF may have control flow that depends on sampled values (which are now `[N]`-shaped arrays rather than scalars)
- **Models must not materialize**: calling `mx/eval!` or `mx/item` on traced values inside the model body would force scalar semantics on `[N]`-shaped arrays

Vectorized `update` and `regenerate` have been implemented, enabling multi-step SMC where particles can be updated and resampled incrementally. Vectorized stochastic branching via the vectorized switch mechanism is also available for models with discrete latent structure.

---

## 6. Comparison with Related Systems

### 6.1 Gen.jl

Gen.jl is the original GFI implementation, written in Julia (~20,000 lines). Key differences:

- **Language**: Julia vs ClojureScript. Julia provides multiple dispatch natively; GenMLX uses multimethods and protocols for equivalent extensibility.
- **Static DSL**: Gen.jl provides a `@gen (static)` variant that compiles model bodies into dependency graphs for incremental computation. GenMLX has only the dynamic DSL, but provides diff-aware combinators for partial incremental updates.
- **Mutation**: Gen.jl uses mutable traces and a mutable parameter store. GenMLX uses persistent data structures throughout.
- **Hardware**: Gen.jl runs on CPU only (GPU support via GenFlux.jl for neural network components). GenMLX runs all numerics on Apple Silicon GPU.
- **Gradient-based MCMC**: GenMLX provides built-in MALA, HMC, NUTS, and elliptical slice sampling with adaptive tuning; Gen.jl does not include these built-in.

### 6.2 GenJAX

GenJAX brings the GFI to JAX (~10,000 lines). Key differences:

- **Vectorization**: GenJAX uses `jax.vmap` to transform scalar trace operations into batched ones, requiring all code to be JAX-traceable. GenMLX uses shape-based batching, which works with arbitrary ClojureScript control flow but cannot vectorize data-dependent branching.
- **Compilation**: GenJAX benefits from XLA's whole-program compilation. GenMLX uses per-function `mx/compile-fn`, lazy graph fusion, and loop compilation for multi-step chains.
- **Hardware**: GenJAX targets NVIDIA GPUs and Google TPUs. GenMLX targets Apple Silicon.
- **Edit interface**: Both systems provide a parametric edit interface with typed requests and backward computation. GenMLX implements `IEdit` on all GF types via a default `edit-dispatch` with per-type overridability.
- **Inference parity**: GenMLX matches GenJAX's inference algorithm repertoire (custom-proposal MH, enumerative Gibbs, involutive MCMC, SMCP3, conditional SMC, programmable VI, composable inference kernels, trainable parameters) and additionally provides MALA, HMC, NUTS, elliptical slice sampling, ADVI, ADEV, VIMCO, amortized inference, and adaptive step-size/metric tuning.

### 6.3 Size Comparison

| System | Language | Lines of Code | Distributions | Combinators | Inference Algorithms | Hardware |
|---|---|---|---|---|---|---|
| Gen.jl | Julia | ~20,000+ | 12 | 3 | 4 (MH, IS, SMC, MAP) | CPU |
| GenJAX | Python/JAX | ~10,000+ | 8 | 5 | 6+ (MH, Gibbs, SMCP3, VI) | GPU/TPU (NVIDIA, Google) |
| GenMLX | ClojureScript | ~10,400 | 27 | 10 | 20+ (MH, Gibbs, MALA, HMC, NUTS, ESS, MAP, IS, SMC, cSMC, vSMC, SMCP3, ADVI, PVI, VIMCO, ADEV, amortized, wake-sleep) | GPU (Apple Silicon) |

The size difference between GenMLX and its predecessors is smaller than before, reflecting the system's growth from its initial ~5,700-line core to include adaptive MCMC, loop compilation, amortized inference, ADEV, neural network integration, vmap, and comprehensive combinator coverage. GenMLX achieves its relative compactness by relying on ClojureScript's built-in persistent data structures, open multimethods, and MLX's lazy evaluation model, which together replace significant amounts of infrastructure code.

---

## 7. Verification and Testing

GenMLX is verified by 1,300+ test assertions across 77 test files:

**Gen.jl compatibility** (165 tests): distribution log-probability spot checks verified against scipy.stats and Gen.jl (within float32 tolerance), mathematical properties (symmetry, normalization, shift invariance), GFI semantics (simulate, generate, update with discard/weight, regenerate), dynamic DSL semantics (gen macro, trace, splice, nested tracing, score computation), and end-to-end inference (line model with constrained observations, importance sampling with branching models).

**GenJAX compatibility** (73 tests): GFI invariants (generate weight equals score under full constraints, update weight equals score difference), MCMC convergence (normal-normal and beta-Bernoulli conjugate posteriors), importance sampling and SMC diagnostics, variational inference convergence, combinator GFI contracts, gradient flow (per-choice gradients, value-and-grad, reparameterized sampling, compiled gradients), diagnostics (ESS, R-hat), numerical stability (extreme parameters, boundary values, large models), and score consistency (hierarchical round-trips, no-op update invariance).

**Vectorized inference** (60+ tests): shape correctness for all batchable distributions, log-probability broadcasting, `vsimulate`/`vgenerate`/`vupdate`/`vregenerate` shape and statistical equivalence, resampling, vectorized importance sampling, vectorized SMC initialization, splice guard in batched mode, and sequential fallback for non-batchable distributions.

**Combinators** (62+ tests): Map, Unfold, Switch, Scan, Mask, Mix, Recurse, Contramap, Dimap --- full GFI coverage including simulate, generate, update, regenerate, edit, update-with-diffs. Combinator-specific tests for element-level score tracking, prefix-skip optimization, branch switching, and recursive structure.

**Vmap** (103 tests): comprehensive coverage of the vmap combinator across simulate, generate, update, regenerate, project, edit, and update-with-diffs for single-site, multi-site, and nested models.

**MCMC algorithms**: dedicated test suites for adaptive HMC (step-size and metric adaptation), adaptive NUTS, loop-compiled chains (MH, MALA, HMC), elliptical slice sampling, vectorized MCMC, and compiled benchmarks.

**Additional test suites** cover distributions (including batch sampling, gamma batching, statistical moments), ADEV, amortized inference, neural network integration, PRNG key threading (16 reproducibility tests), custom gradient GFs, SMCP3 kernels, kernel DSL composition, assess/propose protocols, the edit interface (constraint, selection, proposal edits), diff-aware updates, trainable parameters, error messages, memory management, and resource stress testing.

---

## 8. Usage Example: Bayesian Linear Regression

The following example demonstrates the complete workflow --- model definition, observation, inference, and posterior analysis:

```clojure
(ns example
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Define model
(def linear-regression
  (gen [xs]
    (let [slope     (dyn/trace :slope (dist/gaussian 0 10))
          intercept (dyn/trace :intercept (dist/gaussian 0 10))]
      (mx/eval! slope intercept)
      (let [s (mx/item slope) i (mx/item intercept)]
        (doseq [[j x] (map-indexed vector xs)]
          (dyn/trace (keyword (str "y" j))
                     (dist/gaussian (+ (* s x) i) 1)))
        [s i]))))

;; Observed data
(def xs [1.0 2.0 3.0 4.0 5.0])
(def observations
  (reduce (fn [cm [j y]]
            (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector [2.1 3.9 6.2 7.8 10.1])))

;; Run HMC inference with adaptive tuning
(def samples
  (mcmc/hmc {:samples 1000 :burn 200 :step-size 0.01
             :leapfrog-steps 20 :addresses [:slope :intercept]
             :adapt-step-size true :adapt-metric true}
            linear-regression [xs] observations))

;; Posterior analysis
(println "Posterior slope:"
         (/ (reduce + (map first samples)) (count samples)))
;; => ~2.0 (true slope)
```

---

## 9. Conclusion

GenMLX demonstrates that a full-featured probabilistic programming system --- exceeding the inference algorithm repertoire of GenJAX while adding NUTS, MALA, elliptical slice sampling, ADVI, ADEV, VIMCO, amortized inference, and adaptive MCMC tuning --- can be built in approximately 10,400 lines of ClojureScript by leveraging three properties of the target platform:

1. **MLX's unified memory** eliminates the CPU-GPU transfer overhead that dominates MCMC performance on discrete GPUs
2. **MLX's lazy evaluation** enables entire score computations to be fused into single GPU dispatches, `mx/compile-fn` caches these as reusable Metal programs, and loop compilation extends this to multi-step MCMC chains
3. **MLX's broadcasting semantics** enable vectorized particle inference through shape manipulation rather than code transformation

The purely functional design --- persistent data structures, open multimethods, explicit state threading --- results in a system that is extensible without modifying core code. New distributions, inference algorithms, and combinators can be added from any namespace.

The system now provides the complete GenJAX feature set and more: a parametric edit interface with typed requests and automatic backward computation, diff-aware incremental updates for combinators, trainable parameters integrated with the GFI, composable inference kernels, enumerative Gibbs sampling, involutive MCMC, SMCP3 with probabilistic program proposals, conditional SMC, programmable variational inference with five objectives and two gradient estimators, ADEV gradient estimation, amortized inference via trained neural proposals, neural network integration, and a vmap combinator. All ten model combinators implement the complete GFI.

The system's remaining limitations --- no static DSL, no handler-level incremental computation, volume-preserving-only involutions --- represent clear directions for future work. The functional handler architecture, which proved to be shape-agnostic by construction, suggests that these extensions can be made without architectural changes.

GenMLX is open source under the MIT license at https://github.com/robert-johansson/genmlx.

---

## References

1. Cusumano-Towner, M. F., Saad, F. A., Lew, A. K., & Mansinghka, V. K. (2019). Gen: A General-Purpose Probabilistic Programming System with Programmable Inference. PLDI 2019.

2. Lew, A. K., Matheos, G., Zhi-Xuan, T., Jameel, M. N., & Mansinghka, V. K. (2023). ADEV: Sound Automatic Differentiation of Expected Values of Probabilistic Programs. POPL 2023.

3. Lew, A. K., Huot, M., Staton, S., & Mansinghka, V. K. (2024). Probabilistic Programming with Programmable Variational Inference. PLDI 2024.

4. Awad, H., Lew, A., Huot, M., Staton, S., & Mansinghka, V. K. (2025). Compositional Vectorization for Probabilistic Programming. POPL 2026.

5. Hoffman, M. D. & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. JMLR 2014.

6. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. Handbook of Markov Chain Monte Carlo.

7. Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). Automatic Differentiation Variational Inference. JMLR 2017.

8. Apple Inc. (2023). MLX: An array framework for Apple silicon. https://github.com/ml-explore/mlx.

9. Mnih, A. & Gregor, K. (2014). Neural Variational Inference and Learning in Belief Networks. ICML 2014.

10. Murray, I., Adams, R. P., & MacKay, D. J. C. (2010). Elliptical Slice Sampling. AISTATS 2010.
