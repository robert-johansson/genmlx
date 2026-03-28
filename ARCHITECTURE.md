# GenMLX Architecture: The Generative Function Interface as Universal Integration Boundary

*This document is a blueprint for GenMLX's architecture, its refactoring toward clean separation of concerns, and its integration with domain-specific systems including language models, vision, audio, and databases. It defines the abstractions precisely enough to guide both implementation and future research.*

---

# Part I -- The Protocol

A probabilistic programming system needs a contract. Not a set of library functions, not an API surface area, but a *mathematical contract* that specifies exactly what operations are available on probabilistic computations, what guarantees they provide, and what callers may assume. The Generative Function Interface (GFI) is that contract.

The GFI was formalized by Cusumano-Towner in the Gen probabilistic programming framework. GenMLX implements it faithfully in ClojureScript, with one additional constraint: all numerical values are MLX arrays that remain on the GPU throughout the protocol boundary. The protocol is the same; the substrate is different.

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

GenMLX extends the base GFI with the **edit interface** (`edit.cljs`), which unifies `update` and `regenerate` into a single parametric operation with typed edit requests: `ConstraintEdit` (equivalent to `update`), `SelectionEdit` (equivalent to `regenerate`), and `ProposalEdit` (a forward/backward generative function pair for reversible-jump MCMC). Every edit returns a **backward request** -- the edit that would reverse the transformation. For `ConstraintEdit`, the backward carries the discarded values. For `ProposalEdit`, it swaps the forward and backward generative functions. This automatic backward computation is the foundation for SMCP3 and reversible-jump proposals, and is essential for structure-changing moves on text (insert/delete clauses, rewrite sections) in the LLM integration.

The GFI is closed under composition. A generative function that internally calls other generative functions (via `splice`) is itself a generative function with a valid GFI. Combinators -- `Map`, `Unfold`, `Switch`, `Scan` -- produce generative functions from generative functions. The protocol composes.


## 1.2 The Data Algebra

The GFI operates on three data structures. Each exists for a precise reason, and together they provide exactly the structure that compositional inference requires.

### Traces

A **trace** is an immutable record:

```
Trace = {gen-fn, args, choices, retval, score}
```

A trace is one complete execution. It records not just *what* happened (the choices) but *in what context* (the generative function and arguments) and *at what probability* (the score). Inference algorithms need to re-enter the computation at any point, and the trace carries everything needed to do so. `update` takes a trace and produces a new trace. `regenerate` takes a trace and produces a new trace. The generative function reference inside the trace tells these operations which model to re-execute.

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
| simulate   | `:key` `:choices` `:score` `:executor`                            |
| generate   | `:key` `:choices` `:score` `:weight` `:constraints` `:executor`   |
| assess     | `:key` `:choices` `:score` `:weight` `:constraints` `:executor`   |
| update     | `:key` `:choices` `:score` `:weight` `:constraints` `:old-choices` `:discard` `:executor` |
| regenerate | `:key` `:choices` `:score` `:weight` `:old-choices` `:selection` `:executor` |
| project    | `:key` `:choices` `:score` `:weight` `:old-choices` `:selection` `:constraints` `:executor` |

GenMLX defines ten transitions: six scalar and four batched. The scalar transitions implement the core GFI semantics. Here is `simulate-transition`, the simplest:

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
    (if (cm/has-value? constraint)
      (let [value (cm/get-value constraint)
            lp    (dc/dist-log-prob dist value)]
        [value (-> state
                   (update :choices cm/set-value addr value)
                   (update :score #(mx/add % lp))
                   (update :weight #(mx/add % lp)))])
      (simulate-transition state addr dist))))
```

The four batched transitions (`batched-simulate-transition`, `batched-generate-transition`, `batched-update-transition`, `batched-regenerate-transition`) are structurally identical to their scalar counterparts. The sole difference: they call `dist-sample-n` to draw `[N]`-shaped tensors instead of scalars. Because MLX arithmetic broadcasts, an `[N]`-shaped sample paired with a scalar distribution parameter produces `[N]`-shaped log-probabilities. The state threading logic is unchanged. The handler never inspects value shapes -- this is precisely what makes shape-based vectorization transparent.

The only mutable boundary in the entire system sits in `runtime.cljs`. The function `run-handler` wraps a transition in a single `volatile!` cell:

```clojure
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

**Enumerate.** The transition expands the entire support of the distribution as a new tensor axis. Instead of returning a single scalar, `enumerate-transition` returns a tensor of shape `[K, 1, 1, ...]` where K is the support size and the trailing 1s broadcast against all previously enumerated axes:

```clojure
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
                       (update :axes conj {:addr addr :size k :dim ndim})
                       (update :ndim inc))]))))
```

The model code is identical in all three cases. The same `gen` body runs under `simulate-transition`, `generate-transition`, or `enumerate-transition` without modification. The handler determines the semantics. This is algebraic effects in the Plotkin-Pretnar sense: the effectful operation (`trace`) is syntax; the handler provides the denotation.

A critical architectural point: **enumerate is not a new GFI protocol operation**. It is an alternative *implementation* of the existing operations. A generative function using `enumerate-transition` internally still exposes the standard GFI interface externally. Callers see `simulate`, `generate`, `update`, `regenerate`, `assess`, `project` -- the same seven operations defined in Part I. The enumeration is invisible to callers.


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

GenMLX provides a mechanism for attaching a custom transition to any generative function: metadata annotation.

```clojure
(defn with-handler [gf transition]
  (vary-meta gf assoc ::custom-transition transition))
```

`vary-meta` attaches metadata to a value without changing its identity or behavior. The generative function remains a `DynamicGF` record. Its `body-fn`, `source`, and `schema` are untouched. But when the GFI dispatch encounters `::custom-transition` in the metadata, it uses the custom transition instead of the default.

The same mechanism enables all domain-specific execution modes:

```clojure
;; Exact enumeration
(def exact-model (with-handler model enumerate-transition))

;; Grammar-constrained decoding
(def constrained-llm
  (with-handler llm (wrap-grammar generate-transition json-schema)))

;; Analytical elimination
(def marginalized-model
  (with-handler model (wrap-analytical generate-transition conjugacy-map)))
```

Each produces a generative function that satisfies the full GFI. No new record types. No manual protocol reimplementation. The existing `DynamicGF` machinery handles everything.


## 2.5 The Dispatcher Stack

The current DynamicGF protocol methods use a `cond` ladder to select the execution path. The natural generalization is a dispatcher stack: a sequence of dispatch functions, each implementing:

```clojure
(defprotocol IDispatcher
  (resolve-transition [this op schema opts]))
  ;; Returns a transition-spec map or nil
```

The GFI method walks the stack and uses the first non-nil result:

```clojure
(def default-dispatchers
  [custom-transition-dispatcher    ;; with-handler metadata
   analytical-dispatcher           ;; L3 conjugacy
   compiled-dispatcher             ;; L1 compiled paths
   handler-dispatcher])            ;; L0 fallback (always succeeds)
```

`handler-dispatcher` at the bottom always returns the appropriate base transition. Every dispatcher above it is optional. Adding a new execution mode means adding a dispatcher to the stack. No changes to `DynamicGF`. No changes to existing dispatchers. The stack is data: a vector of functions that can be extended per-model, per-inference-run, or globally.

---

The handler system reduces GenMLX's execution model to a single primitive: a pure function from `(State, Address, Distribution)` to `(Value, State')`. GFI operations are choices of initial state and transition function. Vectorization is a transition that samples tensors instead of scalars. Exact enumeration is a transition that expands support sets as tensor axes. Analytical elimination is middleware that intercepts conjugate sites. Grammar-constrained decoding is middleware that rejects invalid extensions. All of these compose under function composition, and all of them produce generative functions that satisfy the same external GFI contract.

The mutable boundary is a single `volatile!` in `run-handler`, created and consumed within one dynamic extent. Below it, transitions are pure. Above it, model bodies are pure. The volatile bridges the gap between the sequential imperative style that model code expects and the functional state-threading that the transitions implement. It is the thinnest possible shim between two pure layers.

---

# Part III -- The Compilation Ladder

The compilation ladder is five levels of progressive optimization. Each level moves more computation from the ClojureScript host into fused MLX computation graphs. The critical property: every level preserves the GFI semantic contract. A model written against Level 0 runs unchanged at Level 4. The handler is ground truth; compilation is optimization.

This is not a pipeline where each level feeds the next. It is a set of independently applicable strategies. A model might use L3 analytical elimination for conjugate substructure and L0 handler execution for everything else. The dispatcher stack selects the right strategy per trace site, per splice, per inference iteration.


## 3.1 Levels as Dispatchers

**Level 0: The handler-dispatcher.** The base transition functions in `handler.cljs` are pure functions of type `(fn [state addr dist] -> [value state'])`. They operate on scalar arrays. Their batched variants operate on `[N]`-shaped arrays. MLX broadcasting handles all arithmetic. Level 0 is the identity dispatcher -- it always works, for any model, with any distribution, under any GFI operation.

**Level 1: The compiled-dispatcher.** The `gen` macro captures source forms. At construction time, `schema.cljs` walks the quoted form to extract trace sites, classify the model, and compute a topological sort of trace addresses. For static models, `compiled.cljs` uses noise transforms to bypass multimethod dispatch: distribution-specific transforms (Gaussian: `mean + std * noise`) compile into a pure function. `mx/compile-fn` fuses this into a single Metal kernel. Partial compilation (L1-M3) handles mixed models: the static prefix compiles, the dynamic suffix falls back to the handler. Branch rewriting (L1-M4) converts conditionals to `mx/where` operations.

**Level 2: The compiled-sweep-dispatcher.** At L0 and L1, the inference loop is host-driven. Level 2 eliminates this. Pre-generated randomness (all noise tensors allocated upfront as `[T, N, K]`) makes the entire sweep deterministic given its inputs. The compiled particle filter unrolls the loop into one lazy graph. Differentiable resampling enables gradient flow through the sweep via Gumbel-softmax.

**Level 3: The analytical-dispatcher.** `wrap-analytical` is Ring-style middleware. Conjugacy detection statically analyzes the schema to find conjugate prior-likelihood pairs. The dependency graph provides d-separation testing. When the analytical dispatcher intercepts a conjugate site, it replaces sampling with exact posterior computation. The Kalman filter middleware handles linear-Gaussian SSMs; the HMM forward algorithm handles discrete latent chains. These compose: a model with both Kalman chains and Beta-Bernoulli pairs gets both middleware layers.

**Level 4: The fused-graph-dispatcher.** The compiled optimizer fuses model + inference + gradient + Adam into one Metal dispatch. Method selection is a pure decision tree that reads the schema. The `fit` API is the one-call entry point.


## 3.2 Each Level as Middleware or Substitution

The five levels decompose into two mechanisms:

**Substitution** replaces the execution path entirely. L1 substitutes the handler path with a compiled pure function. L2 substitutes the host-driven inference loop with a compiled sweep. L4 substitutes the entire training loop with a fused graph.

**Middleware** wraps the base transition, intercepting specific sites while passing others through. L3 is middleware: `wrap-analytical` composes on top of any base transition.

L0 is neither -- it is the identity. It is the base transition that everything else composes on or substitutes for.

The middleware pattern means L3 composes with L0 or L1. A partially-compiled model can have its static prefix compiled (L1) while its dynamic suffix runs under an analytical-wrapped handler (L3 on L0). The dispatcher stack resolves per-site, not per-model.


## 3.3 Score Encoding as Transition Metadata

Different transitions assign different meanings to the score field:

- **Standard transitions**: score = log *p*(*τ*; *x*) (joint log-probability)
- **Analytical middleware**: score = marginal log-likelihood (conjugate sites collapsed)
- **Enumerate transition**: score = exact marginal likelihood (all latents collapsed)
- **Beam transition**: score = beam-approximated marginal likelihood

The score encoding should be explicit metadata on the transition result, not an implicit convention. When `merge-sub-result` accumulates a sub-result's score into the parent state, it must know whether that score is a joint density, a marginal likelihood, or an approximation. Making it metadata makes the convention checkable.

---

# Part IV -- Domain Integration Pattern

## 4.1 The Universal Recipe

The GFI boundary is domain-agnostic. Any domain integrates through the same four-step recipe:

1. **Define what the generative function represents.** What is the sample space? What does `trace` name? What does `score` measure?
2. **Implement a transition function.** Either a standard `(fn [state addr dist] -> [value state'])` for domains that fit the distribution-per-site pattern, or a full `IGenerativeFunction` record for domains with specialized structure.
3. **Optionally provide middleware.** Domain-specific constraints, analytical solvers, or custom proposals compose as middleware layers on the base transition.
4. **All inference algorithms work automatically.** IS, MCMC, SMC, VI, ADEV, MAP -- they operate through the GFI boundary. A new domain gets the entire inference library for free.


## 4.2 Language Models

The most detailed integration. GenMLX subsumes the entire GenLM ecosystem (genlm-backend, genlm-bytes, genlm-grammar, genlm-control) by making LLMs first-class generative functions in the GFI.

**Token-level LLM generative function.** Each trace site is a token position. The transition samples a token from the next-token logit distribution, scores it as log *p*(token | context), and advances the KV cache in the handler state:

```clojure
(defn token-transition [state addr dist]
  (let [logits (forward-pass (:kv-cache state) (:context state))
        [k1 k2] (rng/split (:key state))
        token (categorical-sample logits k2)
        lp (log-softmax-at logits token)]
    [token (-> state
               (assoc :key k1)
               (update :choices cm/set-value addr token)
               (update :score #(mx/add % lp))
               (update :kv-cache advance-cache token)
               (update :context conj token))]))
```

This is a transition function. It plugs into `run-handler`. All GFI operations work: `generate` constrains token positions, `update` changes constraints, `regenerate` resamples selected positions.

**Byte-level combinator.** Handler substitution marginalizes over tokenizations:

```clojure
(def byte-llm (with-handler token-llm beam-transition))
```

The `beam-transition` maintains K tokenization hypotheses via a TokenByteTrie. At each byte position, it extends all K hypotheses, scores them under the LLM's token distribution, and prunes to the top K. The score is the log of the sum of probabilities across surviving hypotheses -- a beam-approximated marginal likelihood over tokenizations.

**Grammar constraints.** Middleware on the transition:

```clojure
(def grammar-llm
  (with-handler byte-llm
    (wrap-grammar byte-transition (compile-wfsa #"[A-Z][a-z]+"))))
```

WFSAs from regular expressions and WCFGs from Lark grammars implement the same interface: given a prefix, return the set of valid next bytes and their weights. Earley parsing provides incremental prefix weight computation for context-free grammars.

**AWRS sampler.** An inference algorithm in `inference/awrs.cljs`, alongside `mcmc.cljs` and `smc.cljs`. It proposes from the LLM, accepts/rejects based on a boolean constraint, and produces properly weighted samples for SMC.

**What GenMLX adds beyond GenLM:**

- **Full GFI interface.** `update` and `regenerate` enable MCMC on text. GenLM has only SMC.
- **The edit interface.** `ProposalEdit` with forward/backward generative functions enables reversible-jump MCMC on text structure -- insert a paragraph, delete a clause, restructure an argument -- with correct acceptance probabilities. GenLM has no equivalent.
- **Heterogeneous composition via splice.** LLM + continuous + discrete in one model.
- **ExactGF composition.** Small discrete latent spaces controlling LLM generation, enumerated exactly.
- **Vectorized inference.** `[N]`-shaped batched particles through shape polymorphism.
- **The compilation ladder.** Eventually fusing LLM + constraint + inference into compiled graphs.


## 4.3 Vision

The renderer is a deterministic computation inside the `gen` body, not a generative function:

```clojure
(gen [observed-image]
  (let [n-objects (trace :n-objects (dist/poisson 3))
        poses (for [i (range n-objects)]
                (trace (keyword (str "pose-" i))
                       (dist/mvn-gaussian prior-mean prior-cov)))
        shapes (for [i (range n-objects)]
                 (trace (keyword (str "shape-" i))
                        (dist/categorical shape-logits)))
        rendered (render-scene poses shapes)]  ;; deterministic
    (trace :image (dist/gaussian rendered noise-std))))
```

Latent variables: object count, poses, shapes. The renderer maps latents to a predicted image. The likelihood scores rendered vs. observed. Inference: SMC with coarse-to-fine proposals, HMC on continuous pose parameters. The renderer's Jacobian flows through `mx/grad`.


## 4.4 Audio

The synthesizer is a deterministic function inside the `gen` body. Latent variables: pitch, timbre, duration, onset, envelope. The likelihood compares synthesized and observed spectrograms. Inference: HMC on continuous parameters, Gibbs on discrete parameters.

The real-time case uses `update`. As new audio frames arrive, `update` incorporates new observations without re-running inference from scratch. The `Unfold` combinator models temporal structure; `update` adds each new step's constraints. This is online inference -- each `update` is a state transition on the immutable trace.


## 4.5 Databases

A table model is a generative function over rows. Each column is a trace site. Row clustering is a latent discrete variable. GFI operations map directly to SQL semantics:

- `SELECT` = `simulate` (generate a synthetic row)
- `WHERE` = `generate` with constraints (condition on column values)
- `PROBABILITY OF` = `assess` (score a specific row)
- SQL `UPDATE` = GFI `update` (incorporate new rows)

Small categorical columns use exact enumeration. Conjugate continuous columns use L3 analytical elimination. The combination yields exact inference for a significant class of tabular models. GenSQL, written in Clojure, is the most natural integration target.


## 4.6 Cross-Domain Composition

The payoff: splice heterogeneous generative functions into one model.

```clojure
(gen [image audio transcript]
  (let [scene   (splice :scene scene-model)
        sounds  (splice :audio audio-model)
        text    (splice :language llm-model)]
    (trace :coherent (dist/bernoulli (coherence scene sounds text)))))
```

Each domain has its own transition. The dispatcher stack selects the right one per splice. Inference composes domain-specific proposals through the kernel algebra:

```clojure
(def cross-domain-kernel
  (kernel/cycle
    [(kernel/repeat (mcmc/hmc :scene 10) 5)   ;; HMC on scene poses
     (kernel/repeat (mcmc/gibbs :audio) 3)     ;; Gibbs on audio params
     (mcmc/mh :language)                       ;; MH on language tokens
     (mcmc/mh :coherent)]))                    ;; MH on coherence
```

The compilation ladder applies per-domain. The vision sub-model might run at L0. The audio sub-model at L3. The LLM at L2. There is no requirement that all sub-models operate at the same level -- the GFI boundary abstracts over the implementation.

---

# Part V -- Separation of Concerns

The previous sections argued that GenMLX's architecture already embodies the right abstractions: pure transitions, immutable traces, composable protocols. The problem is not the architecture -- it is where the dispatch logic lives. Six GFI methods in `dynamic.cljs` each contain a `cond` ladder that checks schema flags to select between handler, compiled, analytical, and prefix paths. These ladders are structurally identical. They encode the same priority order with operation-specific wiring. When a new execution strategy is added, every ladder must be updated in lockstep.

This part describes the concrete refactoring that eliminates the ladders without changing any external behavior.


## 5.1 What Changes

Three additions, one modification. Total new code: approximately 60 lines.

**The dispatcher protocol** (`dispatch.cljs`, new file, ~30 lines):

```clojure
(defprotocol IDispatcher
  (resolve-transition [this op schema opts]
    "Return a transition-spec or nil.
     op:     keyword (:simulate :generate :update :regenerate :assess :project)
     schema: the model's schema map
     opts:   operation context {:constraints :selection :trace :in-grad?}
     Returns: {:transition fn :init-state-fn fn :post-fn fn :score-type kw} or nil."))
```

Four dispatcher implementations:

```clojure
(def default-dispatcher-stack
  [(->CustomTransitionDispatcher)    ;; with-handler metadata
   (->AnalyticalDispatcher)          ;; L3 conjugacy
   (->CompiledDispatcher)            ;; L1 compiled paths
   (->HandlerDispatcher)])           ;; L0 fallback (always succeeds)
```

Resolution walks the stack, returns the first non-nil result.

**`with-handler`** (~10 lines): Attaches a custom transition to a GF's metadata. The custom-transition-dispatcher checks for it.

**Score encoding** (~20 lines): Transitions carry `:score-type` metadata (`:joint`, `:marginal`, `:collapsed`, `:beam-marginal`). `merge-sub-result` respects this.

**DynamicGF protocol methods** (modified): The six `cond` ladders collapse. Each method becomes:

```clojure
(simulate [this args]
  (let [key (ensure-key this)
        spec (dispatch/resolve dispatcher-stack :simulate schema
               {:gf this})
        result (execute-spec spec this args key {})]
    (mx/gfi-cleanup!)
    result))
```


## 5.2 What Stays

The refactoring is internal. Every external surface is unchanged.

- **`handler.cljs`**: All 10 transitions remain as-is. They are already pure functions with the right signature.
- **`runtime.cljs`**: The `volatile!` boundary stays. `run-handler` stays.
- **`protocols.cljs`**: The 7 GFI protocols are unchanged. This IS the composition boundary.
- **`dist/core.cljs`**: The distribution-as-GF pattern stays. It's the template for domain integrations.
- **`choicemap.cljs`, `trace.cljs`, `selection.cljs`**: The data algebra stays unchanged.
- **All inference algorithms**: They're written against the GFI and don't need changes.
- **All combinators**: They implement the GFI and compose through it.


## 5.3 File-Level Map

```
src/genmlx/
  ;; Layer 0: MLX + Runtime (unchanged)
  mlx.cljs                    ;; MLX bindings, lazy graph, eval, tidy, auto-cleanup
  mlx/random.cljs             ;; Functional PRNG: split, seed!, fresh-key
  runtime.cljs                ;; run-handler, volatile! boundary

  ;; Layer 1: Data Algebra (unchanged)
  choicemap.cljs              ;; Value/Node, hierarchical address->value maps
  trace.cljs                  ;; Immutable Trace record
  selection.cljs              ;; Composable address selection algebra

  ;; Layer 2: GFI Protocols + Dispatch
  protocols.cljs              ;; 7 GFI protocols (unchanged)
  handler.cljs                ;; 10 pure transitions (unchanged)
  dispatch.cljs               ;; NEW: IDispatcher protocol, stack, with-handler

  ;; Layer 3: DSL + Schema (simplified)
  gen.cljc                    ;; gen macro (unchanged)
  schema.cljs                 ;; Schema extraction (unchanged)
  dynamic.cljs                ;; DynamicGF: cond ladders → dispatch/resolve

  ;; Layer 4: Distributions (unchanged)
  dist/core.cljs, dist/macros.cljc, dist.cljs

  ;; Layer 5: Combinators (unchanged)
  combinators.cljs, vmap.cljs

  ;; Layer 6: Compiled Paths (unchanged)
  compiled.cljs, compiled_ops.cljs, compiled_gen.cljs
  tensor_trace.cljs, rewrite.cljs

  ;; Layer 7: Inference (unchanged + new algorithms)
  inference/
    importance.cljs, mcmc.cljs, smc.cljs, vi.cljs, adev.cljs
    kernel.cljs, smcp3.cljs, pmcmc.cljs
    exact.cljs                 ;; enumerate-transition stays here
    analytical.cljs            ;; wrap-analytical middleware
    auto_analytical.cljs       ;; Address-dispatch analytical handlers
    conjugate.cljs, kalman.cljs, ekf.cljs, hmm_forward.cljs
    compiled_smc.cljs, compiled_optimizer.cljs, compiled_gradient.cljs
    awrs.cljs                  ;; NEW: AWRS sampler for constrained LLMs

  ;; Layer 8: Domain Integrations (new namespace tree)
  llm/
    backend.cljs               ;; MLX model loading, forward pass, KV cache
    tokenizer.cljs             ;; byte_vocab extraction
    trie.cljs                  ;; TokenByteTrie, sparse weight propagation
    bytes.cljs                 ;; ByteBeamState, beam-transition
  grammar/
    semiring.cljs              ;; Boolean, Float, Log, MaxPlus, Entropy
    wfsa.cljs                  ;; Weighted FSA
    wcfg.cljs                  ;; Weighted CFG + Earley parser
    fst.cljs                   ;; Weighted FST + Mohri composition

  ;; Layer 9: Analysis (unchanged)
  affine.cljs, conjugacy.cljs, dep_graph.cljs
  method_selection.cljs, fit.cljs

  ;; Layer 10: Verification (unchanged)
  contracts.cljs, verify.cljs

  ;; Support (unchanged)
  edit.cljs, diff.cljs, gradients.cljs, learning.cljs
  custom_gradient.cljs, nn.cljs, vectorized.cljs, serialize.cljs
```


## 5.4 Migration Path

Each existing pattern maps cleanly:

- **ExactGF record** → `(with-handler model enumerate-transition)`. The record's manual protocol reimplementations are deleted. The `enumerate-transition` function, the post-processing algebra (`marginal`, `condition-on`, `joint-marginal`, `extract-table`, `expectation`, `entropy`, `variance`, `mutual-info`), the high-level API (`exact-posterior`, `exact-joint`, `pr`, `observes`), and the utilities (`categorical-argmax`, `with-cache`) all stay in `inference/exact.cljs` -- they operate on the raw handler output, not on the ExactGF record.
- **`exact/Exact` annotation** → `(enumerate model)`. Same metadata mechanism, cleaner name.
- **`analytical-applicable?` predicate** → moves into `AnalyticalDispatcher/resolve-transition`.
- **Compiled path selection** → moves into `CompiledDispatcher/resolve-transition`.
- **The `cond` ladder in each GFI method** → single call to `(dispatch/resolve stack op schema opts)`.

Existing tests pass unchanged because the GFI protocol surface is identical. The refactoring is internal -- callers never see the dispatch mechanism.

---

# Part VI -- Research Directions

## 6.1 L5 -- LLMs in the Fused Graph

An LLM forward pass is a sequence of matrix multiplications and attention operations -- already an MLX computation graph. The target: constrained text generation where the host never orchestrates individual tokens. `mx/compile-fn` receives a function that takes `(params, kv-cache, grammar-state, particle-weights)` and returns `(new-kv-cache, new-grammar-state, new-particle-weights, selected-tokens)`. One Metal dispatch per token.

The dispatcher protocol makes this possible without modifying the model. The model still says `(trace :next-token (constrained-lm grammar))`, and the compiled dispatcher sees a pattern it can fuse.


## 6.2 Theory Search

Meta-generative models over model structure. The outer model samples which mechanisms to include (discrete choices). The inner model implements each mechanism. The data provides the likelihood signal. The posterior over mechanism configurations is the posterior over theories.

`Switch` selects between sub-models. Exact enumeration marginalizes discrete structure variables. SMC handles sequential data. The compilation ladder makes the inner loop tractable by fusing each candidate theory's evaluation into Metal.

This connects to the cfunc cognitive architecture: habituation to theory-based RL as probabilistic program induction.


## 6.3 Cross-Modal Inference

Vision, language, and physics composed in one model, with joint inference across modalities:

```clojure
(def scene-model
  (gen [image utterance]
    (let [objects (splice :scene  (physics-prior))
          pixels  (splice :render (renderer objects))
          caption (splice :lang   (language-model objects))]
      (trace :image (dist/gaussian pixels noise))
      (trace :text  (constrained-lm grammar caption utterance)))))
```

Each splice site brings its own transition. SMC with domain-specific proposals handles the heterogeneous latent space. The kernel algebra composes HMC (continuous), MH (discrete), and AWRS (constrained text).


## 6.4 Formal Verification

GenMLX's 11 contracts verify GFI correctness via property-based testing. A stronger guarantee: mechanized proofs (Lean) that handler transitions satisfy the GFI axioms. The pure-handler architecture makes this tractable -- transitions are pure functions with algebraic specifications.

The dispatcher stack adds a proof obligation: composition of dispatchers must preserve the GFI contract. Score-type declarations make this obligation explicit and checkable.


## 6.5 Real-Time Online Inference

GenMLX's natural mode is online inference. `update` modifies existing traces incrementally as new data arrives. For robotics, audio, and interactive systems: real-time posterior updates without re-running inference from scratch. The `IUpdateWithDiffs` protocol tells combinators which sub-computations to skip. Combined with compiled update transitions, this gives microsecond-level latency for incremental updates.


## 6.6 The Functional-Probabilistic Correspondence, Extended

The GenMLX paper identifies 14 correspondences between functional and probabilistic programming. The domain integration pattern adds more:

| Functional concept | Probabilistic concept | Domain instance |
|---|---|---|
| Pure function in gen body | Deterministic subroutine | Renderer, synthesizer, simulator |
| Middleware on transition | Domain constraint | Grammar, physics, schema validation |
| Address/value space | Domain vocabulary | Tokens, pixels, frequencies, rows |
| Likelihood connecting splices | Cross-domain coherence | Image matches described scene |
| Dispatcher stack | Execution strategy selection | Handler, AWRS, enumerate, beam |

The thesis extends: probabilistic programming, functional programming, and domain-specific computation are aspects of a single framework for compositional computation under uncertainty. A generative function is a function. A trace is a value. A transition is a state machine. A constraint is middleware. A score is a number. The mathematical structure is the same at every level. The compilation ladder, the dispatcher, the domain integrations, and the inference algorithms all compose because they all respect the same algebraic interface.
