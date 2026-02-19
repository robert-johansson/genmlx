# GenMLX: A Probabilistic Programming Language on Apple Silicon

## Abstract

We present GenMLX, a probabilistic programming language implemented in approximately 3,100 lines of ClojureScript, targeting Apple's MLX framework for GPU-accelerated inference on Apple Silicon. GenMLX implements the Generative Function Interface (GFI), the same compositional abstraction used by Gen.jl and GenJAX, but with a purely functional design built on persistent data structures, open multimethods, and shape-based vectorization. The system provides 14 probability distributions, seven inference algorithms (importance sampling, MH, MALA, HMC, NUTS, SMC, ADVI), three model combinators, and a vectorized execution mode that achieves 29--122x speedup over sequential particle methods by exploiting MLX's broadcasting semantics rather than explicit vectorization transforms. All computation stays on GPU as lazy MLX arrays throughout the inference pipeline, with materialization deferred to inference boundaries. The system passes 238 compatibility tests adapted from Gen.jl and GenJAX, demonstrating semantic equivalence with the established implementations.

---

## 1. Introduction

Probabilistic programming languages (PPLs) allow practitioners to express probabilistic models as programs and perform inference automatically. The field has produced several mature systems: Stan for Bayesian statistics, Pyro and NumPyro for deep probabilistic models, and the Gen family --- Gen.jl in Julia, GenJAX on JAX --- which introduced the Generative Function Interface (GFI) as a compositional abstraction for programmable inference.

Despite the maturity of these systems, none targets Apple Silicon's unified memory architecture. Apple's MLX framework provides a distinctive computational model: CPU and GPU share the same memory with zero transfer cost, operations are lazy by default (building computation graphs that fuse before execution), and the framework provides automatic differentiation, JIT compilation to Metal shaders, and vectorization primitives. This architecture is particularly well-suited to probabilistic programming, where MCMC and SMC algorithms require tight interleaving of CPU control flow (accept/reject decisions, resampling, tree traversal) with GPU numerics (score evaluation, gradient computation, leapfrog integration).

GenMLX explores what happens when a PPL is designed from the ground up for this architecture. The result is a system that:

- Implements the full GFI in ~3,100 lines of ClojureScript, running on Node.js via nbb with native MLX bindings
- Keeps all probabilistic values as lazy MLX arrays throughout the inference pipeline
- Achieves 29--122x speedup for particle methods through shape-based vectorization that requires no code transformation
- Provides gradient-based inference (MALA, HMC, NUTS, ADVI) with JIT-compiled score functions and fused leapfrog integration
- Maintains semantic equivalence with Gen.jl and GenJAX, verified by 238 compatibility tests

### 1.1 The Gen Ecosystem

The Generative Function Interface, introduced by Cusumano-Towner et al. in Gen.jl, decomposes probabilistic programming into a small set of operations on generative functions:

- **simulate**: forward-sample all random choices, producing a trace
- **generate**: execute under constraints (observations), returning a trace and importance weight
- **update**: modify a trace with new constraints, returning weight and discarded choices
- **regenerate**: resample selected addresses, returning the acceptance weight

This interface is compositional: distributions, user-defined models, and combinators all implement the same protocols, enabling generic inference algorithms that work across model structures. GenJAX brought this interface to JAX, adding hardware acceleration and vectorization via `jax.vmap`.

GenMLX is the third implementation of the GFI, bringing it to ClojureScript and Apple Silicon.

---

## 2. System Architecture

GenMLX is organized into seven layers, each building on the layers below. This section describes each layer and the design decisions that shape it.

### 2.1 MLX Foundation (Layer 0)

The foundation is a thin ClojureScript wrapper over `@frost-beta/mlx`, the Node.js native addon for Apple's MLX framework. The wrapper (`genmlx.mlx`, 406 lines) exposes MLX operations as idiomatic ClojureScript functions:

```clojure
(mx/add a b)              ;; element-wise addition (lazy)
(mx/multiply a b)         ;; element-wise multiplication (lazy)
(mx/grad f)               ;; automatic differentiation
(mx/compile-fn f)         ;; JIT compile to Metal shader
(mx/eval! a b c)          ;; materialize lazy computation graph
(mx/vmap f)               ;; vectorized map
```

All MLX operations are lazy by default. Calling `(mx/add a b)` does not compute the sum; it records the operation in a computation graph. The graph is materialized only when `mx/eval!` is called, at which point MLX fuses operations into optimized Metal kernels. This laziness is critical: it allows GenMLX to build up entire score computations --- spanning multiple distribution log-probabilities and accumulations --- as a single fused GPU dispatch.

A companion namespace (`genmlx.mlx.random`, 115 lines) provides functional PRNG key management. Every random operation consumes a key and produces sub-keys via deterministic splitting, following the splittable PRNG design pioneered by JAX:

```clojure
(rng/fresh-key)           ;; create initial key
(rng/split key)           ;; -> [k1 k2]
(rng/normal key [n])      ;; sample n standard normals
```

This eliminates hidden PRNG state --- every sample is a pure function of its key.

### 2.2 Core Data Types (Layer 1)

Three persistent data structures form the vocabulary of GenMLX:

**Choice Maps** (`genmlx.choicemap`, 155 lines) are hierarchical trees mapping addresses to values. They use ClojureScript's persistent maps for structural sharing, with a thin protocol (`IChoiceMap`) distinguishing leaf values from interior nodes:

```clojure
(cm/choicemap :slope (mx/scalar 2.0) :intercept (mx/scalar 1.0))
(cm/get-choice cm [:params :slope])
(cm/set-choice cm [:obs :y0] (mx/scalar 3.0))
```

Choice maps are the universal data interchange format: observations are choice maps, trace contents are choice maps, and update constraints are choice maps.

**Traces** (`genmlx.trace`, 13 lines) are immutable records holding the result of a generative function execution:

```clojure
(defrecord Trace [gen-fn args choices retval score])
```

The `score` field holds `log p(choices | args)` as a lazy MLX array, staying on GPU until explicitly materialized.

**Selections** (`genmlx.selection`, 66 lines) are a composable algebra for identifying subsets of addresses. They support flat selection (`(sel/select :x :y)`), hierarchical selection, complement, and the universal selections `all` and `none`. Selections drive the `regenerate` operation, which resamples only the selected addresses while holding others fixed.

### 2.3 GFI Protocols and Execution Engine (Layer 2)

The GFI is expressed as five ClojureScript protocols:

```clojure
(defprotocol IGenerativeFunction
  (simulate [gf args]))

(defprotocol IGenerate
  (generate [gf args constraints]))

(defprotocol IUpdate
  (update [gf trace constraints]))

(defprotocol IRegenerate
  (regenerate [gf trace selection]))

(defprotocol IAssess
  (assess [gf args choices]))
```

The execution engine (`genmlx.handler`, 241 lines) implements these operations through a handler-based architecture. When a model body calls `(dyn/trace :addr dist)`, the call dispatches to whichever handler is currently active. Each handler is decomposed into a **pure state transition function** and a **thin mutable wrapper**:

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

The state is an immutable map containing `:choices`, `:score`, `:weight`, `:key`, and `:constraints`. The volatile (`*state*`) never escapes the `run-handler` boundary --- it exists solely to allow the model body to be written in direct style rather than continuation-passing style.

This separation has a crucial consequence: the pure transition functions are entirely agnostic to the shapes of the values flowing through them. A score that starts as a scalar `(mx/scalar 0.0)` and accumulates scalar log-probabilities works identically to a score that starts as `(mx/scalar 0.0)` and accumulates `[N]`-shaped log-probabilities via broadcasting. This shape-agnosticism is what enables vectorized execution without modifying the handler (Section 5).

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

Inside `gen` bodies, `dyn/trace` addresses a random choice and `dyn/splice` calls a sub-generative function at a namespaced address. The `DynamicGF` record (`genmlx.dynamic`, 175 lines) implements all five GFI protocols by selecting the appropriate handler and running the body function.

Unlike Gen.jl's approach of compiling the model body into a static computation graph, GenMLX's dynamic DSL supports full ClojureScript control flow --- loops, recursion, higher-order functions, conditionals on random values. This flexibility comes at the cost of incremental computation (a static graph can skip unchanged subexpressions during `update`), but the vectorized execution approach described in Section 5 recovers much of the performance.

### 2.5 Distributions (Layer 4)

GenMLX provides 14 probability distributions, all defined through a single `Distribution` record with open multimethods:

```clojure
(defrecord Distribution [type params]
  IGenerativeFunction
  (simulate [this _] (dist-simulate this))
  IGenerate
  (generate [this _ constraints] (dist-generate this constraints)))
```

The `defdist` macro (`genmlx.dist.macros`, 90 lines) reduces distribution definitions to their mathematical essence:

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

The macro generates the constructor function, `defmethod` implementations for `dist-sample`, `dist-log-prob`, `dist-reparam`, and optionally `dist-support`, and wraps all constructor parameters with `mx/ensure-array` so that plain numbers are automatically promoted to MLX scalars.

The 14 distributions span continuous (gaussian, uniform, exponential, laplace, log-normal, beta, gamma, student-t), discrete (bernoulli, categorical, poisson), multivariate (multivariate-normal, dirichlet), and point mass (delta) families. Six distributions (gaussian, uniform, exponential, laplace, log-normal, multivariate-normal) provide reparameterized sampling for gradient-based inference.

Because distributions are GFI participants --- they implement `IGenerativeFunction` and `IGenerate` --- they can be used directly as arguments to inference algorithms, as components in combinators, and as building blocks in hierarchical models.

### 2.6 Combinators (Layer 5)

Three combinators compose generative functions into higher-level models:

**Map** applies a kernel generative function independently to each element of input sequences, analogous to Gen.jl's `Map`. Each invocation is addressed by its integer index, enabling per-element constraints and updates.

**Unfold** applies a kernel sequentially, threading state from one step to the next. This is the natural combinator for time-series and sequential models: the kernel receives `[t state & extra-args]` and returns the next state.

**Switch** selects between multiple branch generative functions based on a runtime index, enabling mixture models and discrete structural choices.

All three combinators implement the full GFI --- `simulate`, `generate`, `update`, and `regenerate` --- so they compose seamlessly with all inference algorithms.

### 2.7 Inference Algorithms (Layer 6)

GenMLX provides seven inference algorithms spanning the major paradigms:

**Importance Sampling and Resampling** (`importance.cljs`, 77 lines). The simplest particle method: run `generate` with observations to get weighted samples, then optionally resample proportional to weights. A vectorized variant (`vectorized-importance-sampling`) runs all particles in a single model execution (Section 5).

**Metropolis-Hastings** (`mcmc.cljs`). Uses the GFI `regenerate` operation as the proposal mechanism: resample selected addresses from the prior, compute the acceptance weight from the score difference, and accept or reject. This is Gen.jl's standard MH, where the proposal is implicitly defined by the prior over selected addresses.

**MALA** (Metropolis-Adjusted Langevin Algorithm). Gradient-informed proposals: the proposal mean is shifted in the direction of the score gradient, with a Metropolis correction for the asymmetric proposal. Score functions and their gradients are JIT-compiled via `mx/compile-fn` for amortized Metal dispatch.

**HMC** (Hamiltonian Monte Carlo). Simulates Hamiltonian dynamics via leapfrog integration. GenMLX provides a fused leapfrog implementation that merges adjacent half-kicks between steps, reducing the number of gradient evaluations from 2L to L+1 for L leapfrog steps. The entire trajectory is built as a single lazy computation graph, materialized once at the end:

```clojure
(defn- leapfrog-trajectory-fused [grad-U q p eps half-eps L]
  (let [g (grad-U q)
        p (mx/subtract p (mx/multiply half-eps g))
        q (mx/add q (mx/multiply eps p))]
    (loop [i 1, q q, p p]
      (if (>= i L)
        (let [g (grad-U q)
              p (mx/subtract p (mx/multiply half-eps g))]
          [q p])
        (let [g (grad-U q)
              p (mx/subtract p (mx/multiply eps g))
              q (mx/add q (mx/multiply eps p))]
          (recur (inc i) q p))))))
```

**NUTS** (No-U-Turn Sampler). Adaptively selects trajectory length by recursively doubling the tree of leapfrog states until a U-turn is detected. GenMLX's implementation follows the original Hoffman and Gelman algorithm with multinomial sampling from the balanced tree.

**SMC** (Sequential Monte Carlo). Particle filtering with systematic resampling and optional MCMC rejuvenation. At each timestep: extend particles with new observations via `update`, reweight, resample if ESS falls below threshold, and optionally apply MH rejuvenation steps. A vectorized initialization step (`vsmc-init`) uses shape-based batching for the first timestep.

**ADVI** (Automatic Differentiation Variational Inference). Fits a mean-field Gaussian guide by maximizing the ELBO via Adam optimization. Uses reparameterized sampling for low-variance gradient estimates and `mx/vmap` to batch ELBO sample evaluation.

All MCMC algorithms share a generic `collect-samples` loop that handles burn-in, thinning, callbacks, and functional PRNG threading. Gradient-based methods (MALA, HMC, NUTS) construct a score function from the model and observations, then JIT-compile both the score and its gradient for amortized Metal dispatch.

**Diagnostics** (`diagnostics.cljs`, 125 lines) provide effective sample size (ESS) via the autocorrelation-based estimator with Geyer's initial positive sequence truncation, R-hat (Gelman-Rubin) convergence diagnostic for multiple chains, and summary statistics (mean, standard deviation, quantiles).

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

### 3.3 Memory Management

MLX's lazy evaluation can accumulate large computation graphs if left unchecked. GenMLX manages this through:

- **Explicit `mx/eval!` at step boundaries**: each MCMC step, leapfrog step, or VI iteration materializes its results, bounding graph size
- **`mx/tidy` blocks**: wrap computations that produce intermediate arrays, automatically disposing them after the block returns
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

The implementation adds four components:

**`dist-sample-n` multimethod** (`dist/core.cljs`). A new multimethod with a sequential fallback:

```clojure
(defmulti dist-sample-n (fn [d _key _n] (:type d)))

;; Default: sequential fallback for distributions that can't batch
(defmethod dist-sample-n :default [d key n]
  (let [keys (rng/split-n (rng/ensure-key key) n)]
    (mx/stack (mapv #(dist-sample d %) keys))))
```

Seven distributions provide native batch sampling (one line each: call `rng/normal` or `rng/uniform` with shape `[n]` instead of `[]`). Distributions with rejection loops (beta, gamma, poisson, student-t) or non-trivial structure (dirichlet, categorical, multivariate-normal) use the sequential fallback.

**Batched handler transitions** (`handler.cljs`). Structurally identical to the scalar transitions but calling `dist-sample-n`:

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

For constrained sites in `batched-generate-transition`, the observation is a scalar. Its log-probability is also a scalar. When this scalar is added to the `[N]`-shaped running score, MLX broadcasts it across all particles --- exactly the correct semantics (all particles see the same observation).

**`VectorizedTrace` record** (`vectorized.cljs`, 94 lines). A trace where choice map leaves hold `[N]`-shaped arrays, and score/weight are `[N]`-shaped:

```clojure
(defrecord VectorizedTrace
  [gen-fn args choices score weight n-particles retval])
```

With utilities for systematic resampling (reindex all choice map leaves by ancestor indices), ESS computation, and log marginal likelihood estimation.

**`vsimulate` / `vgenerate`** (`dynamic.cljs`). Public API functions that run the model body once with the batched handler, producing a `VectorizedTrace`.

### 5.4 Performance Results

Benchmarks on a 5-site Gaussian model with N=100 particles:

| Operation | Sequential | Batched | Speedup |
|---|---|---|---|
| `dist-sample` (N=100) | 29 ms | 1 ms | **29x** |
| `dist-sample` (N=1000) | 265 ms | 4 ms | **66x** |
| `generate` (5 sites, N=100) | 122 ms | 2 ms | **61x** |
| Importance sampling (N=100) | 81 ms | 1 ms | **81x** |
| SMC init (N=100) | 65 ms | 1 ms | **65x** |

The speedup comes entirely from eliminating per-particle overhead. The GPU compute is the same --- evaluating N scalar Gaussians sequentially takes the same number of FLOPs as evaluating one `[N]`-shaped Gaussian. But the sequential version makes 100x more N-API calls and 100x more synchronization points.

The speedup scales linearly with particle count: at N=1000, the `dist-sample-n` speedup reaches 66x, and the overhead per particle in the batched case is effectively zero (bounded by a constant number of API calls regardless of N).

### 5.5 Limitations and Future Work

The current vectorized execution has several limitations:

- **No `splice`**: sub-generative-function calls are not supported in batched mode, because the sub-GF may have control flow that depends on sampled values (which are now `[N]`-shaped arrays rather than scalars)
- **No vectorized `update`/`regenerate`**: needed for multi-step SMC where particles must be updated incrementally
- **Rejection-sampling distributions**: beta, gamma, poisson, student-t, and dirichlet fall back to sequential because their sampling algorithms contain data-dependent loops
- **Models must not materialize**: calling `mx/eval!` or `mx/item` on traced values inside the model body would force scalar semantics on `[N]`-shaped arrays

The first two limitations are addressable in future work. Vectorized stochastic branching (executing all branches and mask-selecting results) would enable splice support, and vectorized update/regenerate would require extending the batched handler with the corresponding transition functions.

---

## 6. Comparison with Related Systems

### 6.1 Gen.jl

Gen.jl is the original GFI implementation, written in Julia (~20,000 lines). Key differences:

- **Language**: Julia vs ClojureScript. Julia provides multiple dispatch natively; GenMLX uses multimethods and protocols for equivalent extensibility.
- **Static DSL**: Gen.jl provides a `@gen (static)` variant that compiles model bodies into dependency graphs for incremental computation. GenMLX has only the dynamic DSL.
- **Mutation**: Gen.jl uses mutable traces and a mutable parameter store. GenMLX uses persistent data structures throughout.
- **Hardware**: Gen.jl runs on CPU only (GPU support via GenFlux.jl for neural network components). GenMLX runs all numerics on Apple Silicon GPU.
- **NUTS**: GenMLX provides a built-in NUTS sampler; Gen.jl does not.

### 6.2 GenJAX

GenJAX brings the GFI to JAX (~10,000 lines). Key differences:

- **Vectorization**: GenJAX uses `jax.vmap` to transform scalar trace operations into batched ones, requiring all code to be JAX-traceable. GenMLX uses shape-based batching, which works with arbitrary ClojureScript control flow but cannot vectorize data-dependent branching.
- **Compilation**: GenJAX benefits from XLA's whole-program compilation. GenMLX uses per-function `mx/compile-fn` and lazy graph fusion.
- **Hardware**: GenJAX targets NVIDIA GPUs and Google TPUs. GenMLX targets Apple Silicon.
- **Advanced inference**: GenJAX provides SMCP3 (Sequential Monte Carlo with Probabilistic Program Proposals) and programmable variational inference. GenMLX provides NUTS and standard ADVI.

### 6.3 Size Comparison

| System | Language | Lines of Code | Hardware |
|---|---|---|---|
| Gen.jl | Julia | ~20,000+ | CPU |
| GenJAX | Python/JAX | ~10,000+ | GPU/TPU (NVIDIA, Google) |
| GenMLX | ClojureScript | ~3,100 | GPU (Apple Silicon) |

The order-of-magnitude size difference reflects different design trade-offs: Gen.jl includes a static DSL compiler, incremental computation, and a parameter learning framework; GenJAX includes JAX-compatible pytree structures and XLA compilation infrastructure. GenMLX achieves its compact size by relying on ClojureScript's built-in persistent data structures, open multimethods, and MLX's lazy evaluation model, which together replace significant amounts of infrastructure code.

---

## 7. Verification and Testing

GenMLX is verified by 238 compatibility tests adapted from Gen.jl and GenJAX:

**Gen.jl compatibility** (165 tests): distribution log-probability spot checks verified against scipy.stats and Gen.jl (within float32 tolerance), mathematical properties (symmetry, normalization, shift invariance), GFI semantics (simulate, generate, update with discard/weight, regenerate), dynamic DSL semantics (gen macro, trace, splice, nested tracing, score computation), and end-to-end inference (line model with constrained observations, importance sampling with branching models).

**GenJAX compatibility** (73 tests): GFI invariants (generate weight equals score under full constraints, update weight equals score difference), MCMC convergence (normal-normal and beta-Bernoulli conjugate posteriors), importance sampling and SMC diagnostics, variational inference convergence, combinator GFI contracts, gradient flow (per-choice gradients, value-and-grad, reparameterized sampling, compiled gradients), diagnostics (ESS, R-hat), numerical stability (extreme parameters, boundary values, large models), and score consistency (hierarchical round-trips, no-op update invariance).

**Vectorized inference** (34 tests): shape correctness for all seven batchable distributions, log-probability broadcasting, `vsimulate`/`vgenerate` shape and statistical equivalence, resampling, vectorized importance sampling, vectorized SMC initialization, splice guard in batched mode, and sequential fallback for non-batchable distributions.

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

;; Run HMC inference
(def samples
  (mcmc/hmc {:samples 1000 :burn 200 :step-size 0.01
             :leapfrog-steps 20 :addresses [:slope :intercept]}
            linear-regression [xs] observations))

;; Posterior analysis
(println "Posterior slope:"
         (/ (reduce + (map first samples)) (count samples)))
;; => ~2.0 (true slope)
```

---

## 9. Conclusion

GenMLX demonstrates that a full-featured probabilistic programming system can be built in approximately 3,100 lines of ClojureScript by leveraging three properties of the target platform:

1. **MLX's unified memory** eliminates the CPU-GPU transfer overhead that dominates MCMC performance on discrete GPUs
2. **MLX's lazy evaluation** enables entire score computations to be fused into single GPU dispatches, and `mx/compile-fn` caches these as reusable Metal programs
3. **MLX's broadcasting semantics** enable vectorized particle inference through shape manipulation rather than code transformation, achieving 29--122x speedup with minimal implementation complexity

The purely functional design --- persistent data structures, open multimethods, explicit state threading --- results in a system that is both smaller and more extensible than its predecessors. New distributions, inference algorithms, and combinators can be added without modifying core code.

The system's current limitations --- no static DSL, no incremental computation, no vectorized update/regenerate --- represent clear directions for future work. The shape-based vectorization approach could be extended to support vectorized stochastic branching (via masking) and multi-step SMC (via vectorized update transitions). The functional handler architecture, which proved to be shape-agnostic by construction, suggests that these extensions can be made without architectural changes.

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
