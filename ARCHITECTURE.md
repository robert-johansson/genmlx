# GenMLX Architecture: The Generative Function Interface as Universal Integration Boundary

*This document describes GenMLX's architecture as built: the Generative Function Interface as an external contract, the pure-handler mechanism that implements it, the compilation ladder layered on top, and the data-driven dispatch that keeps these separate. It defines the abstractions precisely enough to guide implementation and extension.*

*Speculative material — the domain-integration pattern (LLMs, vision, audio, databases) and forward-looking research directions — has been moved out of the repo to keep this document an accurate record of the implemented system.*

---

# Part I -- The Protocol

> **Status: IMPLEMENTED** -- Every part of this document describes the current GenMLX architecture as built.

A probabilistic programming system needs a contract. Not a set of library functions, not an API surface area, but a *mathematical contract* that specifies exactly what operations are available on probabilistic computations, what guarantees they provide, and what callers may assume. The Generative Function Interface (GFI) is that contract.

The GFI was formalized by Cusumano-Towner in the Gen probabilistic programming framework. GenMLX implements it faithfully in ClojureScript, with one additional constraint: all numerical values are MLX arrays that remain on the GPU throughout the protocol boundary. The GPU backend is per-host — Metal on Apple Silicon, CUDA on Linux (validated on Jetson AGX Thor sm_110 and RTX PRO 6000 sm_120) — behind one membrane, with an explicit device knob (`mx/set-default-device!`; the `:device :cpu|:gpu` option on the inference entry points that accept it) for work that belongs on the CPU. Numeric tolerances are measured per-architecture, not per-backend. The protocol is the same; the substrate is different.

This section defines the GFI as GenMLX's external contract, the data algebra it operates on, and why this particular composition boundary is the right one for a system that must support inference algorithms from importance sampling through variational inference to sequential Monte Carlo with programmable proposals.


## 1.1 The GFI Contract

A **generative function** is a four-tuple *P* = (*X*, *Y*, *p*, *f*) in an address universe (*A*, *V*, *M*) where:

- *X* is the argument type (model inputs: data, hyperparameters, configuration)
- *Y* is the return type (the deterministic output of the computation)
- *p*(·; *x*) is a family of structured probability densities over choice dictionaries, parameterized by *x* ∈ *X*
- *f*(*x*, *τ*) is the return value function, deterministic given arguments and choices

The address universe gives structure to randomness. Each random choice in the computation is identified by an **address** *a* ∈ *A*, takes a **value** *v* ∈ *V*, and has a **measure** *μ* ∈ *M*. A **choice dictionary** *τ* maps a finite set of addresses to values. The density *p*(*τ*; *x*) factorizes over addresses according to the structure of the computation.

The GFI defines seven operations on generative functions. Each has precise mathematical semantics, and together they are sufficient for all known inference algorithms.

**1. simulate**(*x*) → Trace *t*

Forward-sample all random choices from the prior. Returns a trace *t* = (*P*, *x*, *τ*) where *τ* ~ *p*(·; *x*). This is the simplest operation: run the generative function, sample every `trace` site from its distribution, record everything. The trace carries the generative function reference, the arguments, the complete choice dictionary, the return value *f*(*x*, *τ*), and the joint log-density log *p*(*τ*; *x*).

**2. generate**(*x*, *σ*) → (Trace *t*, log *w*)

Constrained execution. The caller provides a partial choice dictionary *σ* (the **constraints** -- typically observed data). The generative function must produce a trace *t* consistent with *σ*: at every address where *σ* specifies a value, *t* must agree. At unconstrained addresses, *t* samples from the prior. The log-weight is:

> log *w* = log *p̄*(*σ*; *x*)

where *p̄*(*σ*; *x*) is the marginal likelihood of the constraints -- the density of the observed values integrated over all unobserved choices.

This single operation is the foundation of importance sampling. Call `generate` *N* times with the same observations; the resulting log-weights are importance weights targeting the posterior. No special machinery required.

**3. update**(*t*, *x'*, *v*) → (Trace *t'*, log *w*, *v'*)

Modify an existing trace. Given a previous trace *t* with arguments *x* and choices *τ*, new arguments *x'*, and new constraints *v* (a partial choice dictionary of proposed changes), produce a new trace *t'* with choices *τ'* that incorporates the changes. The log-weight is the density ratio:

> log *w* = log *p*(*τ'*; *x'*) - log *p*(*τ*; *x*)

The **discard** *v'* contains the old values at addresses that were overwritten -- exactly the information needed to reverse the update. This reversibility is what makes update the basis for Metropolis-Hastings with custom proposals: propose new values via a proposal distribution, use `update` to install them and compute the acceptance ratio, and use the discard to construct the reverse proposal density.

**4. regenerate**(*t*, *S*) → (Trace *t'*, log *w*)

Resample from the prior at selected addresses. Given a trace *t* and a **selection** *S* (a set of addresses), produce a new trace *t'* where addresses in *S* are freshly sampled from their prior distributions, and addresses not in *S* retain their previous values. The log-weight accounts for the density change:

> log *w* = Σ_{a ∈ S} [log *p_a*(*τ'_a*) - log *p_a*(*τ_a*)]

This is the primitive that Metropolis-Hastings uses directly. One MH step: call `regenerate` on the current trace with the selection identifying which latent variables to resample, then accept or reject based on the returned log-weight. The weight *is* the log acceptance ratio (when the proposal is the prior).

**5. assess**(*x*, *τ*) → log *p*(*τ*; *x*)

Score a fully-specified choice dictionary. Every address that the generative function would visit must have a value in *τ*. Returns the joint log-density.

**6. project**(*t*, *S*) → scalar

Compute the log-density of choices at the selected addresses:

> Σ_{a ∈ S} log *p_a*(*τ_a*)

This is the "partial scoring" operation. It enables computing the density contribution of any subset of random choices without re-executing the model.

**7. propose**(*x*) → (*τ*, log *p*(*τ*; *x*), *f*(*x*, *τ*))

Forward-sample and return the choices, their joint log-density, and the return value. Semantically equivalent to `simulate` followed by extracting the choices, score, and return value, but expressed as a single operation to allow optimized implementations.

### Why Seven Operations Suffice

These seven operations are not an arbitrary API. They are the minimal set that closes over all standard inference algorithms:

- **Importance sampling** uses `generate` alone. Each call produces a weighted sample from the posterior.
- **Metropolis-Hastings** uses `regenerate` (prior proposals) or `update` (custom proposals). The log-weight is the acceptance ratio.
- **Sequential Monte Carlo** uses `generate` to initialize, then `update` to incorporate new observations. The incremental weights from `update` are the SMC weights.
- **Variational inference** uses `generate` to compute the ELBO. Gradients flow through the MLX computation graph.
- **Analytical elimination** uses `generate` with middleware that intercepts conjugate pairs and computes their contribution analytically. The caller sees the same interface but gets better estimates.
- **SMCP3** uses `update` with the full discard-and-reverse mechanism for reversible kernels.

In the implementation this surface is factored into **eleven** single-operation protocols (`protocols.cljs`): the seven operations above, the diff-aware update variants (`update-with-args` — the thesis's *x′* move — and `update-with-diffs`), gradient introspection (`has-argument-grads`), and vectorized splice support (`batched-splice`).

GenMLX extends the base GFI with the **edit interface** (`edit.cljs`), which unifies `update` and `regenerate` into a single parametric operation with typed edit requests: `ConstraintEdit` (equivalent to `update`), `SelectionEdit` (equivalent to `regenerate`), `ProposalEdit` (a forward/backward generative function pair for reversible-jump MCMC), `ArgsUpdateEdit` (equivalent to `update-with-args`), and `CompositeEdit` (left-to-right composition of edit requests). Every edit returns a **backward request** -- the edit that would reverse the transformation. For `ConstraintEdit`, the backward carries the discarded values. For `ProposalEdit`, it swaps the forward and backward generative functions. This automatic backward computation is the foundation for SMCP3 and reversible-jump proposals, and is essential for structure-changing moves on text (insert/delete clauses, rewrite sections) in the LLM integration.

The GFI is closed under composition. A generative function that internally calls other generative functions (via `splice`) is itself a generative function with a valid GFI. Combinators -- `Map`, `Unfold`, `Switch`, `Scan` -- produce generative functions from generative functions. The protocol composes.


## 1.2 The Data Algebra

The GFI operates on three data structures. Each exists for a precise reason, and together they provide exactly the structure that compositional inference requires.

### Traces

A **trace** is an immutable record:

```
Trace = {gen-fn, args, choices, retval, score, omega?}
```

A trace is one complete execution. It records not just *what* happened (the choices) but *in what context* (the generative function and arguments) and *at what probability* (the score). Inference algorithms need to re-enter the computation at any point, and the trace carries everything needed to do so. `update` takes a trace and produces a new trace. `regenerate` takes a trace and produces a new trace. The generative function reference inside the trace tells these operations which model to re-execute.

The optional `omega` field carries **encapsulated randomness** (thesis §4.5; nil for ordinary exact-density traces). When a generative function's score is an unbiased density *estimator* rather than an exact density, `omega` records the internal randomness that realized that estimate, making the score reproducible — see Part IV §4.1.

Traces are immutable. An `update` does not mutate the old trace; it produces a new one. Clojure's persistent data structures make this efficient through structural sharing.

### Choice Maps

A **choice map** is a hierarchical mapping from addresses to values, with two node types:

- **Value** (leaf): wraps a single random choice value (an MLX array)
- **Node** (interior): a persistent map from addresses to child choice maps

The key invariant: **choices are the sufficient statistic for the execution**. Given a generative function, its arguments, and a complete choice map, the entire execution is deterministic. The return value, the score, every intermediate computation -- all are determined.

Choice maps support `stack` (combine *N* scalar maps into one map with [*N*]-shaped leaves) and `unstack` (the reverse). These operations bridge per-particle and batched representations for vectorized inference.

### Selections

A **selection** is a composable specification of which addresses to operate on. The algebra has five forms:

- `all`: every address is selected
- `none`: no address is selected
- `(select :a :b :c)`: specific top-level addresses
- `(hierarchical :sub sel)`: recursive structure matching the choice map hierarchy
- `(complement-sel s)`: the set complement

Selections form a Boolean algebra. They enable composable address specification for `regenerate` and `project` without coupling inference algorithms to model structure.

### Why These Three

Traces, choice maps, and selections are the minimal algebra for compositional inference:

- **Traces** are execution records. They carry enough context to continue, modify, or reverse a computation.
- **Choice maps** are the random state. They separate *what was sampled* from *how it was sampled*. This separation is what enables different execution strategies (sampling, enumeration, analytical computation) to produce the same choice map types.
- **Selections** identify subsets for partial operations without enumerating addresses or coupling to model structure.


## 1.3 The Composition Boundary

The GFI is a **composition boundary**: the surface across which inference algorithms interact with probabilistic models. Its design determines what can compose with what, and how much each side must know about the other.

The boundary has one defining property: **any computational object that can sample and score participates as a first-class citizen**. A generative function might be:

- A ClojureScript function written with the `gen` macro, executing trace-by-trace through the handler system
- A compiled tensor computation that fuses all trace sites into a single MLX graph
- A neural network wrapped as a generative function via `nn->gen-fn`
- A combinator like `Map` or `Unfold` that builds structured models from simpler components
- A domain-specific system -- an LLM, a renderer, a physics simulator -- that implements the relevant GFI subset

The inference algorithm does not know which of these it is talking to. It calls `generate`, gets back a trace and a weight. It calls `regenerate`, gets back a trace and a weight. The types are the same. The weight accounting is correct. The algorithm proceeds.

This is the meaning of the phrase: **"The handler is ground truth; compilation is optimization."**

GenMLX's handler system -- the pure state transitions in `handler.cljs` driven by the `volatile!`-based runtime in `runtime.cljs` -- is the reference implementation of the GFI. Every generative function can execute through the handler. The handler is simple, correct, and general. Compilation levels (L1 through L4) produce *faster* implementations of the same operations. But these compiled paths must produce identical traces, identical scores, identical weights as the handler path.

This separation has a deep consequence for inference algorithms:

1. **Importance sampling** calls `generate` *N* times. It does not know whether each call executes through the handler, through a compiled path, or through vectorized execution. The weights are correct in all cases.
2. **MH** calls `regenerate` on a trace. It does not know whether the trace was produced by sampling, by enumeration, or by analytical computation. The weight is correct regardless.
3. **SMC** calls `update` to incorporate new data. Whether the generative function computes the density ratio by re-executing the model or by looking up a cached analytical result is invisible to SMC.
4. **Analytical elimination** wraps the handler transitions with middleware that replaces sampling with analytical marginalization. The outer GFI operations are unchanged. Inference algorithms get better estimates without modification.

The contract does not leak. Callers never access handler state, never inspect the volatile cell, never depend on execution order of trace sites. They see traces, choice maps, selections, and log-weights. These are the protocol's currency. Everything else is private.

---

# Part II -- The Handler

The previous section introduced the Generative Function Interface as an external contract. This section describes the internal mechanism that implements that contract. In GenMLX, every GFI operation reduces to a single abstraction -- the **handler transition** -- a pure function that receives an immutable state, an address, and a distribution, and returns a value paired with a new state.

This design has three consequences. First, the entire GFI becomes testable as pure data transformations. Second, alternative execution strategies -- exact enumeration, analytical elimination, grammar-constrained decoding -- are expressed as alternative transition functions, not as new protocol implementations. Third, middleware composition follows an algebraic structure: transitions compose under function composition, forming a monoid that enables layered execution modes without modification to existing code.


## 2.1 Transitions as Values

A handler transition is a pure function of three arguments:

```
transition : (State, Address, Distribution) → (Value, State')
```

State is an immutable Clojure map. The exact keys depend on the GFI operation being implemented:

| Mode       | State keys                                                        |
|------------|-------------------------------------------------------------------|
| simulate   | `:key` `:choices` `:score` `:param-store` `:executor`             |
| generate   | `:key` `:choices` `:score` `:weight` `:constraints` `:param-store` `:executor` |
| assess     | `:key` `:choices` `:score` `:weight` `:constraints` `:param-store` `:executor` |
| update     | `:key` `:choices` `:score` `:weight` `:constraints` `:old-choices` `:discard` `:old-splice-scores` `:old-nested-splice-scores` `:param-store` `:executor` |
| regenerate | `:key` `:choices` `:score` `:weight` `:old-choices` `:selection` `:old-splice-scores` `:old-nested-splice-scores` `:param-store` `:executor` |
| project    | `:key` `:choices` `:score` `:weight` `:old-choices` `:selection` `:constraints` `:param-store` `:executor` |
| propose    | as simulate                                                       |

`:executor` holds the splice-execution function and is per-operation (`execute-sub`, with `execute-sub-assess` for assess and `execute-sub-project` for project); the two `:old-*-splice-scores` keys let the splice closure reuse recorded sub-scores on update/regenerate. Batched variants add `:batch-size`, `:batched?`, and `:batched-sub-regen`.

`handler.cljs` defines thirteen transitions: seven scalar and six batched (alternative execution strategies — enumerate, ADEV, Kalman/EKF, compiled replay — define further same-signature transitions in their own namespaces). The seventh scalar transition is `regenerate-transition-general` (the retained-only general regenerate path, genmlx-hmch/yep2), used when a selection is not fast-path-eligible (i.e., the model has branches, or selected sites are not mutually independent). The scalar transitions implement the core GFI semantics. Here is `simulate-transition`, the simplest:

```clojure
(defn simulate-transition [state addr dist]
  (let [[k1 k2] (rng/split (:key state))
        value    (dc/dist-sample dist k2)
        lp       (dc/dist-log-prob dist value)]
    [value (-> state
               (assoc :key k1)
               (update :choices cm/set-value addr value)
               (update :score #(mx/add % lp)))]))
```

Split the PRNG key. Sample a value. Compute the log-probability. Thread all three results into the new state. No mutation. No side effects.

`generate-transition` adds constraint checking: if the address appears in the constraint map, use the constrained value and add its log-probability to both `:score` and `:weight`. Otherwise, delegate to `simulate-transition`:

```clojure
(defn generate-transition [state addr dist]
  (let [constraint (cm/get-submap (:constraints state) addr)]
    (assert-leaf-constraint! constraint addr :generate) ;; a node choicemap at a primitive site throws
    (if (cm/has-value? constraint)
      (let [value (cm/get-value constraint)
            lp    (dc/dist-log-prob dist value)]
        [value (-> state
                   (update :choices cm/set-value addr value)
                   (update :score #(mx/add % lp))
                   (update :weight #(mx/add % lp)))])
      (simulate-transition state addr dist))))
```

The six batched transitions (`batched-simulate-transition`, `batched-generate-transition`, `batched-update-transition`, `batched-regenerate-transition`, `batched-project-transition`, `batched-regenerate-transition-general`) are structurally identical to their scalar counterparts. The main difference: they call `dist-sample-n` to draw `[N]`-shaped tensors instead of scalars. They also add a per-site `check-batched-lp!` shape-invariant guard (no scalar counterpart) — a no-eval, O(1) check that fails loudly if a site's log-prob is not `[]`, `[1]`, or `[N]`. Because MLX arithmetic broadcasts, an `[N]`-shaped sample paired with a scalar distribution parameter produces `[N]`-shaped log-probabilities. The state threading logic is unchanged. The handler never inspects value shapes -- this is precisely what makes shape-based vectorization transparent.

The mutable boundary of handler execution sits in `runtime.cljs`. The function `run-handler` wraps a transition in a single `volatile!` cell. (System-wide, a small audited set of other mutable points exists outside the execution path — resource-management counters in `mlx.cljs`, dev-mode extension atoms, memoization caches, caller-owned training state; CLAUDE.md "Key design principles" carries the full inventory. None of them affect computation results.)

```clojure
;; Abridged — the real run-handler also derefs a dev-mode validation hook,
;; calls mx/auto-cleanup! inside trace-fn, and defines ~110 lines of
;; splice-fn/param-fn (including a recursive run-handler for sub-gen-fns).
(defn run-handler [transition init-state body-fn]
  (let [vol (volatile! init-state)
        trace-fn (fn [addr dist]
                   (let [[value state'] (transition @vol addr dist)]
                     (vreset! vol state')
                     value))
        rt #js {:trace trace-fn :splice splice-fn :param param-fn}]
    (let [retval (body-fn rt)]
      (assoc @vol :retval retval))))
```

The volatile is created, used, and consumed within the dynamic extent of `run-handler`. It never escapes. The design is analogous to re-frame's `app-db`: one encapsulated mutable cell provides the illusion of imperative sequencing, while everything below (the transitions) and above (the model body) remains purely functional.


## 2.2 Three Fundamental Interpretations

Inside a `gen` body, a `trace` call declares an algebraic effect: "I need a value from distribution D at address A." The call says nothing about how that value is obtained. The handler transition determines the interpretation.

There are three fundamental interpretations:

**Sample.** The transition draws a value from the distribution. `simulate-transition` always samples. `generate-transition` samples at unconstrained addresses. The model receives a random draw, and the log-probability is accumulated into `:score`.

**Constrain.** The transition fixes the value to an observation. `generate-transition` constrains at addresses present in the constraint map. `assess-transition` constrains at all addresses. The model receives the observed value, and the log-probability is accumulated into both `:score` and `:weight`.

**Enumerate.** The transition expands the entire support of the distribution as a new tensor axis. Instead of returning a single scalar, `enumerate-transition` returns a tensor of shape `[K, 1, 1, ...]` where K is the support size and the trailing 1s broadcast against all previously enumerated axes.

The snippet below is a **simplified illustration** of the structure (not literal source — see `inference/exact.cljs:51-104` for the full implementation, which includes a k=1 deterministic short-circuit, correct lp broadcast-dim insertion between K and param dims, and `:support` in the `:axes` entry):

```clojure
;; Simplified illustration — see exact.cljs for the full implementation
(defn enumerate-transition [state addr dist]
  (let [constraint (cm/get-submap (:constraints state) addr)]
    (if (cm/has-value? constraint)
      ;; Constrained: scalar value, no new axis
      (let [value (cm/get-value constraint)
            lp    (dc/dist-log-prob dist value)]
        [value (-> state
                   (update :choices cm/set-value addr value)
                   (update :score #(mx/add % lp)))])
      ;; Free: expand all support values as new leftmost axis
      (let [support   (dc/dist-support dist)
            k         (count support)
            ndim      (:ndim state)
            val-shape (into [k] (repeat ndim 1))
            values-nd (mx/reshape (mx/stack support) val-shape)
            lp-all    (dc/dist-log-prob-support dist)
            lp-nd     (mx/reshape lp-all (into [k] (repeat ndim 1)))]
        [values-nd (-> state
                       (update :choices cm/set-value addr values-nd)
                       (update :score #(mx/add % lp-nd))
                       (update :axes conj {:addr addr :size k :dim ndim :support support})
                       (update :ndim inc))]))))
```

The model code is identical in all three cases. The same `gen` body runs under `simulate-transition`, `generate-transition`, or `enumerate-transition` without modification. The handler determines the semantics. This is algebraic effects in the Plotkin-Pretnar sense: the effectful operation (`trace`) is syntax; the handler provides the denotation.

A critical architectural point: **enumerate is not a new GFI protocol operation**. It is an alternative *implementation* of the existing operations. A generative function using `enumerate-transition` internally still exposes the standard GFI interface externally. Callers see `simulate`, `generate`, `update`, `regenerate`, `assess`, `project`, `propose` -- the same seven operations defined in Part I. The enumeration is invisible to callers.


## 2.3 Middleware Composition

Handler transitions compose via the Ring middleware pattern. A middleware is a function from transition to transition:

```
middleware : Transition → Transition
```

This is an endomorphism on the type `(State, Address, Distribution) → (Value, State')`. The identity middleware returns its argument unchanged. Composition is associative. These two properties make middlewares a monoid under function composition.

The canonical example is `wrap-analytical`, which intercepts trace sites with conjugate distributions and computes marginal likelihoods exactly:

```clojure
(defn wrap-analytical [base-transition dispatch-map]
  (fn [state addr dist]
    (if-let [handler-fn (get dispatch-map (:type dist))]
      (or (handler-fn state addr dist)
          (base-transition state addr dist))
      (base-transition state addr dist))))
```

Multiple middleware layers compose by function composition:

```clojure
(-> base-transition
    (wrap-analytical conjugacy-dispatch)
    (wrap-analytical kalman-dispatch)
    (wrap-analytical hmm-dispatch))
```

Each layer gets first refusal. If it cannot handle a site, the next layer tries. The base transition at the bottom always succeeds. This is the same pattern as Ring middleware in Clojure web development: `(Handler → Handler)` composes associatively, and the identity handler is the unit element.

This middleware approach extends to any domain-specific execution strategy. A grammar-constrained language model wraps the base transition with validity checking. An adaptive weighted rejection sampler wraps the base transition with a proposal-and-accept loop. Each wrapper is a `Transition → Transition` function. They compose with each other and with the existing infrastructure without modification.


## 2.4 Handler Substitution

GenMLX provides two mechanisms for attaching custom execution to a generative function, both via metadata annotation.

**Transition substitution** replaces the handler transition while keeping the standard handler machinery (init-state construction, result packaging, `run-handler` orchestration):

```clojure
(defn with-handler [gf transition]
  (vary-meta gf assoc ::custom-transition transition))
```

The transition has the standard handler signature `(fn [state addr dist] -> [value state'])`. The dispatcher wraps it into `run-handler` with the correct init-state per GFI operation. This is the common case for domain-specific execution strategies:

```clojure
;; Grammar-constrained decoding
(def constrained-llm
  (with-handler llm (wrap-grammar generate-transition json-schema)))

;; Analytical elimination
(def marginalized-model
  (with-handler model (wrap-analytical generate-transition conjugacy-map)))
```

**Full dispatch override** replaces the entire execution path per GFI operation, for cases that need custom init-state or post-processing:

```clojure
(defn with-dispatch [gf dispatch-fn]
  (vary-meta gf assoc ::custom-dispatch dispatch-fn))
```

The dispatch function has signature `(fn [op gf args key opts] -> gfi-result)`. Exact enumeration uses this because it needs custom state (`:axes`, `:ndim`) and custom post-processing (logsumexp normalization):

```clojure
;; Exact enumeration — internally uses with-dispatch
(def exact-model (enumerate model))
```

Both mechanisms produce generative functions that satisfy the full GFI. No new record types. No manual protocol reimplementation. The existing `DynamicGF` machinery handles everything. Prefer `with-handler` when a transition substitution suffices; use `with-dispatch` only when the standard handler state shape is insufficient.


## 2.5 The Dispatcher Stack

The DynamicGF protocol methods delegate to a dispatcher stack: a sequence of dispatch functions, each implementing:

```clojure
(defprotocol IDispatcher
  (resolve-transition [this op schema opts]))
  ;; Returns a transition-spec map or nil
```

The GFI method walks the stack and uses the first non-nil result:

```clojure
(def ^:private default-dispatcher-stack
  [custom-dispatcher              ;; with-handler / with-dispatch metadata
   analytical-dispatcher           ;; L3 conjugacy
   compiled-dispatcher             ;; L1 compiled paths
   handler-dispatcher])            ;; L0 fallback (always succeeds)
```

`handler-dispatcher` at the bottom always returns the appropriate base transition. Every dispatcher above it is optional. Adding a new execution mode means adding a dispatcher to the stack. No changes to `DynamicGF`. No changes to existing dispatchers. The stack is data: a vector of functions that can be extended per-model, per-inference-run, or globally.

---

**Note on vectorized execution.** The batched functions (`vsimulate`, `vgenerate`, `vupdate`, `vupdate-args`, `vregenerate` — plus `vgenerate-compiled`, the persistent-compiled sweep for static models) bypass the dispatcher stack and directly use batched handler transitions. This is by design: vectorized execution is shape-based (`[N]`-shaped arrays via MLX broadcasting) and always runs through the handler path. Batched splices take fused fast paths through `IBatchedSplice` combinators (Map/Mask), and batched standard-SMC (`vsmcp3`) rides the same machinery. The dispatcher stack applies to scalar GFI operations, not to batched particle execution.

The handler system reduces GenMLX's execution model to a single primitive: a pure function from `(State, Address, Distribution)` to `(Value, State')`. GFI operations are choices of initial state and transition function. Vectorization is a transition that samples tensors instead of scalars. Exact enumeration is a transition that expands support sets as tensor axes. Analytical elimination is middleware that intercepts conjugate sites. Grammar-constrained decoding is middleware that rejects invalid extensions. All of these compose under function composition, and all of them produce generative functions that satisfy the same external GFI contract.

The mutable boundary of handler execution is a single `volatile!` in `run-handler`, created and consumed within one dynamic extent. Below it, transitions are pure. Above it, model bodies are pure. The volatile bridges the gap between the sequential imperative style that model code expects and the functional state-threading that the transitions implement. It is the thinnest possible shim between two pure layers.

---

# Part III -- The Compilation Ladder

The compilation ladder is five levels of progressive optimization. Each level moves more computation from the ClojureScript host into fused MLX computation graphs. The critical property: every level preserves the GFI semantic contract. A model written against Level 0 runs unchanged at Level 4. The handler is ground truth; compilation is optimization.

This is not a pipeline where each level feeds the next. It is a set of independently applicable strategies. A model might use L3 analytical elimination for conjugate substructure and L0 handler execution for everything else. The dispatcher stack selects the right strategy per trace site, per splice, per inference iteration.


## 3.1 Levels as Dispatchers

**Level 0: The handler-dispatcher.** The base transition functions in `handler.cljs` are pure functions of type `(fn [state addr dist] -> [value state'])`. They operate on scalar arrays. Their batched variants operate on `[N]`-shaped arrays. MLX broadcasting handles all arithmetic. Level 0 is the identity dispatcher -- it always works, for any model, with any distribution, under any GFI operation.

**Level 1: The compiled-dispatcher.** The `gen` macro captures source forms. At construction time, `schema.cljs` walks the quoted form to extract trace sites, classify the model, and compute a topological sort of trace addresses. For static models, `compiled.cljs` uses noise transforms to bypass multimethod dispatch: distribution-specific transforms (Gaussian: `mean + std * noise`) compile into a pure function. The result is a single lazy MLX graph that dispatches to the GPU in one `eval!` (note: `mx/compile-fn` is *permanently* an identity pass-through by verdict, genmlx-819v — GenMLX's fusion comes from lazy-graph construction plus the noise-transform expression compiler; persistent MLX compilation exists only behind the explicit handles of §3.5). Partial compilation (L1-M3) handles mixed models: the static prefix compiles, the dynamic suffix falls back to the handler. Branch rewriting (L1-M4) converts same-address conditionals into branch-free selection — `mx/where` on the density side, a precomputed values-map select on the simulate path. Single-site MH moves are cone-restricted: regenerate re-executes only the changed site's dependency cone (O(|direct children|) graph work), with batched N-chain (`vmh`) and fused whole-chain variants.

**Level 2: The compiled-sweep-dispatcher.** At L0 and L1, the inference loop is host-driven. Level 2 eliminates this. Pre-generated randomness (all noise tensors allocated upfront as `[T, N, K]`) makes the entire sweep deterministic given its inputs. The compiled particle filter unrolls the loop into one lazy graph. Differentiable resampling enables gradient flow through the sweep via Gumbel-softmax.

**Level 3: The analytical-dispatcher.** `wrap-analytical` is Ring-style middleware. Conjugacy detection statically analyzes the schema to find conjugate prior-likelihood pairs. The dependency graph provides d-separation testing. When the analytical dispatcher intercepts a conjugate site, it replaces sampling with exact posterior computation. The Kalman filter middleware handles linear-Gaussian SSMs; the HMM forward algorithm handles discrete latent chains. These compose: a model with both Kalman chains and Beta-Bernoulli pairs gets both middleware layers. Beyond scalar conjugacy: joint linear-Gaussian block elimination (`linear_gaussian.cljs`) handles coupled multi-latent regression, where per-prior scalar conjugacy would be silently wrong for affine means; the extended analytical tier (CLAUDE.md's Level 3.5) adds EKF and N-dimensional EKF middleware (`ekf.cljs`, `ekf_nd.cljs` — full cross-covariance by scalar decomposition) and conjugacy through combinators; `rb_mixture.cljs` Rao-Blackwellizes Gaussian-mixture marginal likelihoods.

**Level 4: The fused-graph-dispatcher.** The compiled optimizer fuses model + inference + gradient + Adam into one GPU dispatch. Method selection is a pure decision tree that reads the schema. The `fit` API is the one-call entry point.


## 3.2 Each Level as Middleware or Substitution

The five levels decompose into two mechanisms:

**Substitution** replaces the execution path entirely. L1 substitutes the handler path with a compiled pure function. L2 substitutes the host-driven inference loop with a compiled sweep. L4 substitutes the entire training loop with a fused graph.

**Middleware** wraps the base transition, intercepting specific sites while passing others through. L3 is middleware: `wrap-analytical` composes on top of any base transition.

L0 is neither -- it is the identity. It is the base transition that everything else composes on or substitutes for.

The middleware pattern means L3 composes with L0 or L1. A partially-compiled model can have its static prefix compiled (L1) while its dynamic suffix runs under an analytical-wrapped handler (L3 on L0). The dispatcher stack resolves per-site, not per-model.


## 3.3 Score Encoding as Enforced Metadata

Different transitions assign different meanings to the score field:

- **Standard transitions**: score = log *p*(*τ*; *x*) (joint log-probability)
- **Analytical middleware**: score = marginal log-likelihood (conjugate sites collapsed)
- **Enumerate transition**: score = exact marginal likelihood (all latents collapsed)
- **Tensor-trace placeholder**: score = a 0.0 stand-in inside compiled sweeps (`:placeholder`)
- **Beam transition**: score = beam-approximated marginal likelihood (reserved)

The score encoding is explicit, checked metadata, not an implicit convention.
Every trace carries `:genmlx.trace/score-type` metadata (`:joint`,
`:marginal`, `:collapsed`, `:placeholder` — the zero-score tensor-trace
placeholder used by compiled SMC — plus the reserved `:beam-marginal`), set
by every producing path: handler and compiled paths tag `:joint`, the
analytical path tags `:marginal` when a handler actually marginalized,
enumerate tags `:collapsed`. Composite producers propagate:
`merge-sub-result` lubs a spliced sub-result's score-type into the parent
state, and combinators tag their result traces with the lub over element
traces — a marginal sub-score cannot launder into a joint-looking total.
(The lub is a max over a severity rank from which `:collapsed` is
deliberately absent: an enumerate splice records no internal choices and
composes like a deterministic factor, so a collapsed *part* folds to
`:joint`; top-level `:collapsed` is only ever assigned directly by
enumerate.)

Consumers enforce the contract at joint-scoring boundaries
(update/project/regenerate, via the dispatcher guard for `DynamicGF` and
`ensure-joint-self` for combinator records): `:marginal` traces convert by
re-generating fully constrained from their own choices (one handler
generate; alternate paths stay semantically invisible), while `:collapsed`
traces throw — their choicemaps are empty, so no joint conversion exists —
and `:placeholder` traces throw likewise: there is no density behind the
score.
Trace-MH entry points additionally assert their regenerate results are
joint-scored, so a forgotten analytical strip throws instead of silently
anchoring chains at the posterior mean (the genmlx-540f failure class).
The law `:score-type-soundness` in `gfi.cljs` and `score_type_test.cljs`
guard the convention.

## 3.4 Path-Equivalence Fuzzing: Enforcing "the Handler is Ground Truth"

The compilation ladder's contract — every alternate path (compiled, prefix,
branch-rewrite, analytical, batched) must be observationally equivalent to
the handler interpretation — was historically enforced by hand-written
tests. The 2026-07 audit showed the failure mode of that regime: every
critical it found was a *divergence between two implementations of the same
math*, discovered by hand-comparing paths on a hand-built model. PEF
(`src/genmlx/pef.cljs`, bean genmlx-0bgi) institutionalizes that audit as a
generator:

- **Model generator** — a togglable feature grammar (dist families,
  site dependencies, affine/nonlinear parameter expressions, let-rebinding,
  nested trace calls in dist-args, splices in both dependency directions,
  M4-eligible and structure-changing branches, model args) drives a
  test.check genome generator. Genomes materialize through the same
  `make-gen-fn` + quoted-source mechanism as hand-written models, so schema
  extraction, L1 compilation, and L3 conjugacy analysis all run for real.
  Everything derives from one seed; every failure prints as a
  paste-runnable `pef/reproduce` one-liner.
- **Path registry** — pair specs (data, extensible) compare the model
  against `strip-compiled` / `strip-analytical` / forced-general-regenerate
  / batched execution per GFI op, composing on the `gfi.cljs` laws where
  they already encode the property (compiled-equivalence, fast-vs-general
  regenerate, vectorized shapes). Comparison policies: `:eps` for exact
  math under float32 reduction jitter, `:statistical` with
  variance-derived bands for estimator pairs (SMC vs IS), bit-exact where
  no resampling is involved (update round-trips).
- **Invariants** — alternate-path *soundness* (selected ⇒ equivalent),
  compiled-path *liveness* (on the independent-sites profile the L1 path
  must fire above a pinned threshold — an eligibility gate that silently
  routes everything to the handler fails the suite, not just perf),
  crash-freedom (any un-graceful throw from any op is a failure artifact),
  discard round-trips, and score-type conservation.
- **Tiers** — a fast-core smoke slice (`pef_test.cljs`, ~100 models,
  seconds), a deep mode (`pef_deep_test.cljs`, `PEF_MODELS` env, default
  2000), and a frozen regression corpus (`pef_corpus.cljs`) holding the
  minimal model for every divergence ever found, starting with the audit's
  own bug classes. New PEF findings get shrunk, filed as beans, and
  appended; the fuzzer found its first real bug (the M3 prefix walker
  accepting nested-trace dist-args) on its first smoke run.

PEF's counterpart at the battery level is the **test-suite honesty contract**
(genmlx-n061): `test/run.sh` classifies a file by exit code plus one anchored
grep for its summary line, `run.sh check` enforces that every asserting file
can actually FAIL (four accepted gate shapes, `@gate` markers for
delegated/exception files), and deliberate SKIPs are scored apart from FAILs.
There is no CI; every published number is a manual attestation with a
hand-run battery behind it.

## 3.5 Persistent Compilation: Captured-Graph Replay

`mx/compile-fn` stays an identity by verdict, but persistent MLX compilation
exists as an opt-in, explicit-handle API on the membrane (epic genmlx-z2gt):
`compile-create` builds a native compiled callable for a fixed-structure
function; `compiled-call` / `compiled-call-captured` invoke it (the captured
variant replays a recorded graph against updated captured inputs); and
`with-compiled` / `compiled-free!` scope the handle's lifetime — the same
blessed-scope pattern as every other native resource. On top of the handles,
fixed-structure inference paths get whole-call compiled factories: a
persistent-compiled batched generate sweep for static models
(`vgenerate-compiled`) and fused whole-call MALA/HMC/MH chain factories in
`inference/mcmc.cljs`. Equivalence with the interpreted path is enforced like
every other alternate path: a dedicated persistent-compile equivalence suite
plus the PEF pair registry.

---

# Part IV -- Extended Thesis Mechanisms

Two mechanisms from later chapters of the Cusumano-Towner thesis extend the
engine beyond the seven core operations. Both compose on the existing GFI; the
handler remains ground truth.

## 4.1 Encapsulated Randomness (§4.5)

A generative function may own internal randomness *ω* that is not part of its
choice dictionary *τ*. Its realized score is then a value of an unbiased density
*estimator* *ξ*(*x*, *τ*, *ω*) rather than the exact density *p*(*τ*; *x*):

> E_*ω*[ *ξ*(*x*, *τ*, *ω*) ] = *p*(*τ*; *x*)     (Eq 4.3)

The `Trace` record carries the optional `omega` field so the realized score is
reproducible: re-evaluating the estimator at the recorded *ω* yields the same
*ξ*. This makes identity `update`/`project` cost weight 0 exactly, while a genuine
move (changed value or arguments) resamples *ω* and pays the pseudo-marginal
ratio log *ξ'* − log *ξ*_old.

`genmlx.encapsulated` provides `EncapsulatedGF` (the full GFI over one observed
address), two estimator families with closed-form oracles — `marginalized-gaussian`
(black-box stochastic code, importance sampling over a nuisance) and
`mixture-density` (a finite mixture whose component index is integrated by Monte
Carlo) — and `pseudo-marginal-mh` (Andrieu-Roberts), which infers a parameter
whose likelihood is available only as an unbiased estimate and still targets the
exact posterior. The laws `:encapsulated-estimator-unbiased`,
`:encapsulated-identity-update-zero`, and `:pseudo-marginal-stationarity` in
`gfi.cljs` pin the contract.

## 4.2 Trace Translators (§3.6-3.7)

A **trace translator** maps a trace of model *P₁* to a trace of model *P₂* whose
choice dictionaries live in different spaces. It is built from forward/backward
auxiliary proposals *Q₁*, *Q₂* and a bijection *h* between (*τ₁*, *ρ₁*) and
(*τ₂*, *ρ₂*) (Def 3.6.1), with importance weight (Eq 3.12):

> log *w* = (log *p₂* − log *p₁*) + (log *q₂* − log *q₁*) + log|det *J_h*|

The forward auxiliary density is the denominator, the backward the numerator, and
the Jacobian is taken over the continuous coordinates only (discrete coordinates
contribute no column). The involutive-MCMC kernel in `inference/mcmc.cljs` is the
symmetric special case (*P₁*=*P₂*, *Q₁*=*Q₂*, *h* an involution, §3.7).

`genmlx.inference.translator` provides `trace-translator` (the constructor),
`apply-translator`/`translator-weight` (Eq 3.12), an AD Jacobian
(`jacobian-logdet`, via `mx/grad`; the log|det| uses `mx/logabsdet` — a QR-based, lazy on-device determinant correct for the non-symmetric Jacobians bijections produce, where the SPD-only Cholesky path would be silently wrong, genmlx-rqp9) with a
sparsity-aware variant (`sparse-jacobian-logdet`, §3.6.2: coordinates *h* copies
unchanged are identity columns excluded from the determinant block), a
`read-choice`/`write-choice`/`copy-choice` introspection API for writing bijections,
`reversible-jump-mh` (structure-changing split/merge moves, §3.7.4), and
`coarse-to-fine-smc` (a model sequence bridged by translators, §3.6.4). Because
translators drive sampling-based methods, `apply-translator` strips the L3
analytical path before scoring so a conjugate target yields joint scores, not
posterior-mean-pinned marginals (the genmlx-540f failure class). Five laws
(`:translator-weight-formula`, `:translator-jacobian-ad`,
`:translator-sparsity-equiv`, `:translator-bijection-roundtrip`,
`:reversible-jump-detailed-balance`) pin the contract.

---

# Part V -- Separation of Concerns

The previous sections argued that GenMLX's architecture embodies the right abstractions: pure transitions, immutable traces, composable protocols. The remaining concern is where the dispatch logic lives. The naive design puts a `cond` ladder in every GFI method of `dynamic.cljs` — each checking schema flags to select between handler, compiled, analytical, and prefix paths. Those ladders are structurally identical: same priority order, operation-specific wiring. Adding an execution strategy would mean updating every ladder in lockstep.

This part describes how GenMLX avoids that: a data-driven dispatcher stack (`dispatch.cljs` + the dispatchers in `dynamic.cljs`) that replaces the ladders without changing any external behavior. It is implemented, not aspirational.


## 5.1 The Design

Three pieces: a dispatcher protocol, a stack of dispatchers, and thin GFI methods that delegate to the stack.

**The dispatcher protocol** (`dispatch.cljs`):

```clojure
(defprotocol IDispatcher
  (resolve-transition [this op schema opts]
    "Return a dispatch-spec or nil.
     op:     :simulate | :generate | :update | :regenerate | :assess | :project | :propose
     schema: the model's schema map
     opts:   {:gf gen-fn, :op op, :constraints cm, :trace trace, :selection sel}
     Returns: {:run (fn [gf args key opts] -> gfi-result)
               :score-type :joint|:marginal|:collapsed|:beam-marginal}
     or nil."))
```

Four dispatcher implementations (defined in `dynamic.cljs` where they access the private run-* helpers):

```clojure
(def ^:private default-dispatcher-stack
  [custom-dispatcher               ;; with-handler / with-dispatch metadata
   analytical-dispatcher           ;; L3 conjugacy
   compiled-dispatcher             ;; L1 compiled paths
   handler-dispatcher])            ;; L0 fallback (always succeeds)
```

Resolution walks the stack, returns the first non-nil dispatch-spec. The `:run` function in the spec encapsulates the full execution path for that level. One recovery path wraps the walk: if an analytical handler aborts mid-run (a `:genmlx.analytical/bail` throw), `run-dispatched*` re-runs the whole operation with the analytical path stripped — a conservative bail degrades to the compiled/handler path instead of an error.

**`with-handler`** (in `dispatch.cljs`): Attaches a custom transition under `::dispatch/custom-transition` metadata via `vary-meta` (a single `(fn [state addr dist])` or a per-op map). The custom dispatcher checks for it, alongside `::dispatch/custom-dispatch` for full dispatch override via `with-dispatch`.

**Score encoding**: Each dispatch-spec carries `:score-type` (`:joint`, `:marginal`, `:collapsed`, `:beam-marginal`).

**DynamicGF protocol methods**: There are no `cond` ladders. Each method delegates to the stack:

```clojure
(simulate [this args]
  (let [key (ensure-key this)
        result (@dispatch-fn this :simulate args key {})]
    (mx/gfi-cleanup!)
    result))
```

(`dispatch-fn` defaults to `run-dispatched*`; `genmlx.dev/start!` swaps in a
validating wrapper, which is where Malli return-schema validation lives.)


## 5.2 What the Dispatch Layer Leaves Untouched

The dispatch layer is internal. Every external surface below is independent of it.

- **`handler.cljs`**: All 13 transitions (11 core + 2 regenerate-general variants) remain as-is. They are already pure functions with the right signature.
- **`runtime.cljs`**: The `volatile!` boundary stays. `run-handler` stays.
- **`protocols.cljs`**: The 11 GFI protocols are unchanged. This IS the composition boundary.
- **`dist/core.cljs`**: The distribution-as-GF pattern stays. It's the template for domain integrations.
- **`choicemap.cljs`, `trace.cljs`, `selection.cljs`**: The data algebra stays unchanged.
- **All inference algorithms**: They're written against the GFI and don't need changes.
- **All combinators**: They implement the GFI and compose through it.


## 5.3 File-Level Map

Layer numbering follows CLAUDE.md's project-structure section (the two
documents previously disagreed on the 6/7/8/9 assignments).

```
src/genmlx/
  ;; Layer 0: MLX + Runtime — the compute membrane
  ;; (native substrate: the vendored mlx-node fork, a git submodule carrying
  ;;  its own patched mlx; sync architecture in docs/fork/)
  mlx.cljs                    ;; THE membrane: lazy graph ops, eval!/tidy boundary, buffer-count
                              ;;   discipline, CPU/GPU device knob, persistent-compile handles (§3.5);
                              ;;   surface pinned by the coverage matrix (docs/membrane-coverage.md)
  mlx/random.cljs             ;; Functional PRNG: split, fresh-key, ensure-key
  mlx/constants.cljs          ;; MLX dtype/device constants
  runtime.cljs                ;; run-handler, volatile! boundary
  dispatch.cljs               ;; IDispatcher protocol, stack walk, with-handler / with-dispatch

  ;; Layer 1: Data Algebra (unchanged)
  choicemap.cljs              ;; Value/Node, hierarchical address->value maps
  trace.cljs                  ;; Immutable Trace record (+ score-type metadata, §3.3)
  selection.cljs              ;; Composable address selection algebra
  diff.cljs                   ;; Argdiff types for update-with-args

  ;; Layer 2: GFI & Execution
  protocols.cljs              ;; The 11 GFI protocols
  handler.cljs                ;; 13 pure transitions (7 scalar + 6 batched, including 2 regenerate-general variants)
  edit.cljs                   ;; Edit interface (ConstraintEdit, SelectionEdit, ProposalEdit,
                              ;;   ArgsUpdateEdit, CompositeEdit)
  tensor_trace.cljs           ;; TensorTrace/TensorChoiceMap for Level-2 compiled inference
                              ;;   (VectorizedTrace lives in vectorized.cljs)

  ;; Layer 3: DSL + Schema
  gen.cljc                    ;; gen macro
  schema.cljs                 ;; Schema extraction
  schemas.cljs                ;; Malli schemas for GFI types
  inspect.cljs                ;; inspect API (compilation level, dispatch map)
  dynamic.cljs                ;; DynamicGF + the four dispatchers (dispatch/resolve)

  ;; Layer 4: Distributions (unchanged)
  dist/core.cljs, dist/macros.cljc, dist.cljs

  ;; Layer 5: Combinators (unchanged)
  combinators.cljs, vmap.cljs

  ;; Layer 6: Inference
  inference.cljs              ;; public aggregator namespace
  inference/
    importance.cljs, mcmc.cljs, smc.cljs, vi.cljs, adev.cljs
    kernel.cljs, smcp3.cljs, pmcmc.cljs
    exact.cljs                 ;; enumerate-transition
    enumerate.cljs             ;; enumerate support helpers
    analytical.cljs            ;; wrap-analytical middleware
    auto_analytical.cljs       ;; Address-dispatch analytical handlers
    conjugate.cljs, kalman.cljs, ekf.cljs, ekf_nd.cljs, hmm_forward.cljs
    rb_mixture.cljs            ;; Rao-Blackwellized Gaussian-mixture marginal likelihood
    compiled_smc.cljs, compiled_optimizer.cljs, compiled_gradient.cljs
    differentiable.cljs, differentiable_resample.cljs, amortized.cljs
    translator.cljs            ;; General trace translators (§3.6-3.7): Eq 3.12, RJMCMC, coarse-to-fine
    util.cljs                  ;; inference utilities (materialize-weights, etc.)
    diagnostics.cljs           ;; ESS, convergence diagnostics
    fisher.cljs                ;; Fisher information / Laplace approximation
    cost.cljs                  ;; CostMeter: monotonic, additive cost counters (measurement, not estimation)
    steppable.cljs             ;; steppable/budgeted inference (SMC-first)

  ;; Layer 7: Compiled Paths + Static Analysis
  compiled.cljs, compiled_ops.cljs, compiled_gen.cljs
  affine.cljs, conjugacy.cljs, dep_graph.cljs
  rewrite.cljs                ;; algebraic rewrite engine: Kalman collapse, conjugacy
                              ;;   elimination, Rao-Blackwellization (build-analytical-plan)
  linear_gaussian.cljs        ;; joint linear-Gaussian block elimination (L3 linreg)
  method_selection.cljs       ;; pure decision tree over schemas (L4 method selection)

  ;; Layer 8: Supporting Systems
  vectorized.cljs             ;; VectorizedTrace + shape-based batching support
  gradients.cljs, learning.cljs, custom_gradient.cljs, nn.cljs, serialize.cljs
  fit.cljs                    ;; one-call fit entry point (L4)
  encapsulated.cljs           ;; Encapsulated randomness (§4.5): EncapsulatedGF, estimators, pseudo-marginal-mh
  memory.cljs                 ;; PERSISTENCE face of the world membrane: synchronous bun:sqlite
                              ;;   durable read/write (a world/ face that predates the directory)
  program.cljs                ;; two-level GFI: structural inference over probabilistic programs
                              ;;   (SCI-compiled gen fns, scored via p/generate)
  sensorimotor.cljs           ;; Hammer-style sensorimotor learning: Beta-Bernoulli concept
                              ;;   memory with exponential temporal projection
  tabular.cljs                ;; vectorized deterministic primitives over enumerated state spaces
  sandbox.cljs                ;; budgeted, interruptible CLJS eval in a SIGKILL-able subprocess
  sandbox_child.cljs          ;;   (its child-process evaluator script)
  dev.cljs                    ;; dev mode start!/stop! (swaps dispatch-fn/validate-fn atoms)

  ;; Verification
  gfi.cljs                     ;; the GFI algebraic law catalog from the Cusumano-Towner thesis
                               ;;   (85 laws as of 2026-08; count the vector)
  verify.cljs                  ;; static validator (validate-gen-fn)
  pef.cljs                     ;; path-equivalence fuzzing: generator + pair registry + invariants (§3.4)

  ;; Layer 9: LLM Integration
  llm/
    core.cljs                  ;; make-llm-gf: wrap LLM as DynamicGF (token = trace site)
    backend.cljs               ;; mlx-node loader, forward pass, KV cache
    forward.cljs               ;; generic LLM forward pass
    qwen3_forward.cljs         ;; owned Qwen3 text forward pass
    qwen35_forward.cljs        ;; owned Qwen3.5 text forward: dense + MoE (gather-qmm),
                               ;;   fused GDN prefill/decode, [K]-particle batch dim, router taps
    qwen35_vision_forward.cljs ;; owned Qwen3.5-VL vision tower + M-RoPE VLM decode
    sampling.cljs              ;; pure logit-transform sampler for the owned decode loop
    grammar.cljs               ;; DFA-constrained generation (regex → token mask)
    bytes.cljs                 ;; byte-level marginalization via TokenByteTrie
    codegen.cljs               ;; reader-as-grammar for valid ClojureScript
    schema_grammar.cljs        ;; Malli schema → grammar constraint
    structured.cljs            ;; structured generation API
    toolcall.cljs              ;; grammar-constrained tool calls (per-argument + reader-level :cljs)
    msa.cljs                   ;; Model Synthesis Architecture (LLM proposes programs)
    msa_score.cljs             ;; native-free model eval/scoring spine (split out of msa)
    vision.cljs                ;; VLM input adaptation
    branched.cljs              ;; branch-using GFI on the native branchable KV cache
    smc.cljs                   ;; token-SMC: particle filtering where each particle IS a
                               ;;   cache branch (genmlx-5qk7) — twisted SMC/SMC-steering
                               ;;   semantics (target ∝ p_LM · Π φ_s; grammar-masked
                               ;;   proposal weights = the mask log-normalizer), prefill
                               ;;   once per prompt, fork-on-resample with immediate loser
                               ;;   disposal (live handles ≤ N+1; all disposed on return —
                               ;;   with-token-smc* scopes live branches like
                               ;;   with-llm-branches*). The decoder is an abstraction
                               ;;   (native branches / dense replay / synthetic table), so
                               ;;   the exactness suite runs model-free against enumerable
                               ;;   posteriors. This is the second path made concrete:
                               ;;   resource-rational synthesis = many cheap partial
                               ;;   hypotheses + principled reallocation on the resident
                               ;;   model. A batched-lane mode instead runs K particles in
                               ;;   ONE [K]-shaped branch with lane-axis resampling.
    pi_provider.cljs           ;; pi coding-agent turn engine over the owned forward
    pi_session.cljs            ;; pi session JSONL → chat messages
    pi_assess.cljs             ;; per-turn log-prob assessment of administered sessions
    pi_edit.cljs               ;; trajectory editing / resample-from-step
  codegen/
    eval.cljs                  ;; native-free reader/eval spine (SCI)

  ;; Agents
  agents/
    agent.cljs                 ;; agent record + act/step API
    belief.cljs                ;; tensor belief-update kernel (pure-MLX observation-Bayes filter)
    biased_planners.cljs       ;; biased/MC planner baselines
    differentiable.cljs        ;; differentiable MDP utility/alpha learning (gradient through the planner)
    gridworld.cljs             ;; gridworld environment
    pacman.cljs                ;; Pac-Man maze substrate (the shared example environment)
    helpers.cljs               ;; agent utility functions
    inverse.cljs               ;; inverse planning / IRL
    pomdp.cljs                 ;; POMDP agent
    pomdp_env.cljs             ;; POMDP environment protocol
    presentation.cljs          ;; result presentation / formatting
    remote.cljs                ;; PROVISIONAL gym-transport RL seam above world.net
    rollout.cljs               ;; trajectory rollout
    worlds.cljs                ;; agentmodels.org environments-as-data (line-mdp, hike-mdp)

  ;; Control (layer contract in control/CONTRACTS.md)
  control/
    decision_value.cljs        ;; decision-theoretic value functions
    meta_mdp.cljs              ;; meta-MDP / rational metareasoning
    cybernetic.cljs            ;; comparator/TOTE controlled-steppable — the stop decision is a traced site
    synth_steppable.cljs       ;; synthesis steppable {:propose :deepen :stop}

  ;; World (effect membranes — one blessed scope around one side effect per face)
  world/
    net.cljs                   ;; NETWORK face: Bun.serve / HTTP, scoped by with-server
    proc.cljs                  ;; PROCESS face: control's eval!-equivalent — advancing compute
                               ;;   under a wall-clock deadline (synchronous; never a GF)
    train.cljs                 ;; TRAINING face: boundary over the native GRPO engine; in-place
                               ;;   weight/AdamW updates are training's eval!-equivalent (with-trainer)
    train_reward.cljs          ;; GRPO where the reward is a GFI quantity
    session_grpo.cljs          ;; pi sessions → GRPO prompts
    session_reward.cljs        ;; reward builders for GRPO-on-sessions
    probe.cljs                 ;; pure statistics for GRPO probe-set eval
    synth.cljs                 ;; REPL-driven program-synthesis kernel (phase 1)
    search.cljs                ;; particle search over construction steps (phase 2)
    llm_proposer.cljs          ;; real-LLM proposer for the synthesis loop
    curriculum.cljs            ;; graded, oracle-grounded curriculum generator
    harvest.cljs               ;; REPL-trace harvest (run side)
    repl_corpus.cljs           ;; harvested traces → SFT corpus
    sft.cljs                   ;; pure core of the cljs-coder SFT step
    distill.cljs               ;; offline teacher→student oracle-filter distillation
    distill_sandbox.cljs       ;; process isolation for the distillation filter
    distill_tasks.cljs         ;; canonical distillation task set
    t1_battery.cljs            ;; Tier-1 bake-off task battery
```


## 5.4 How Execution Strategies Map Onto the Stack

Each execution strategy maps onto the dispatch design cleanly:

- **ExactGF record** → `(enumerate model)` via `with-dispatch`. The record's manual protocol reimplementations are deleted. The `enumerate-transition` function, the post-processing algebra (`marginal`, `condition-on`, `joint-marginal`, `extract-table`, `expectation`, `entropy`, `variance`, `mutual-info`), the high-level API (`exact-posterior`, `exact-joint`, `pr`, `observes`), and the utilities (`categorical-argmax`, `with-cache`) all stay in `inference/exact.cljs` -- they operate on the raw handler output, not on a separate record.
- **`exact/Exact` annotation** → `(enumerate model)`. Same metadata mechanism, cleaner name.
- **`analytical-applicable?` predicate** → moves into `AnalyticalDispatcher/resolve-transition`.
- **Compiled path selection** → moves into `CompiledDispatcher/resolve-transition`.
- **The `cond` ladder in each GFI method** → single call to `(dispatch/resolve stack op schema opts)`.

The GFI protocol surface is identical regardless of which dispatcher resolves an operation. The dispatch mechanism is internal -- callers never see it.

