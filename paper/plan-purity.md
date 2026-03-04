# Plan: Pushing GenMLX Toward Pure Functional Architecture

**Inspired by:** re-frame's effects-as-data, coeffect injection, interceptor chains,
and flows

**Status:** Research complete, ready for implementation planning

---

## 1. Motivation

GenMLX is already remarkably pure for a GPU-accelerated probabilistic programming
system — 33 source files, only 8 contain any mutation, zero global mutable state.
But mutation is still scattered across inference loops (memory management), training
code (`nn/step!`), and the handler dispatch layer (10 `vreset!` sites).

[re-frame](https://day8.github.io/re-frame/re-frame/) solved the same structural
problem for web UIs. The browser DOM is a mutable external resource, but re-frame
keeps all application logic pure by treating the DOM as an effect target. Event
handlers return **data descriptions** of side effects; the framework executes them.
This separation is the key insight: **handlers describe effects, they don't perform
them**.

GenMLX's GPU (MLX/Metal) is directly analogous to re-frame's DOM. Both are mutable
external resources that the framework interacts with. Both require a clear boundary
between pure application logic and impure world-touching code. The question is: can
we apply re-frame's architecture to push that boundary outward, making more of
GenMLX's code demonstrably pure?

---

## 2. The GPU-as-DOM Analogy

### 2.1 Structural parallel

| re-frame concept | GenMLX equivalent | Analogy |
|------------------|-------------------|---------|
| Browser DOM | Metal GPU | Mutable external resource |
| Hiccup `[:div {:class "x"} "hello"]` | MLX lazy computation graph | Data description of desired output |
| React reconciliation + DOM commit | `mx/eval!` materializing on Metal | Executing the description |
| `app-db` (Reagent atom) | `*state*` (volatile! in handler) | Single mutable container for app state |
| Event handler (pure fn) | Handler transition (pure fn) | Pure function that computes what should change |
| Effects map `{:db ... :dispatch ...}` | Currently implicit — GPU ops inline | Data description of side effects |
| Coeffects `{:db ... :now ... :local-store ...}` | Dynamic vars (`*param-store*`, `*prng-key*`) | External world state injected as data |
| Interceptor chain | Not present | Composable middleware around event handling |
| Subscriptions / Signal graph | Not present | Reactive derived values from app state |
| Flows | Not present | Declarative derived state stored back in app-db |

### 2.2 Why the analogy is precise

MLX operations are lazy by design. When you write `(mx/add a b)`, no computation
happens — MLX records the operation in a computation graph. Only when you call
`mx/eval!` does the graph execute on the GPU. This is exactly how React/Reagent
works: calling `[:div "hello"]` doesn't touch the DOM — it returns a data
description. Only when React commits does the DOM mutate.

This means GenMLX already has the "effects-as-data" pattern at the lowest level:

```
re-frame:  view fn -> hiccup data -> React commit -> DOM mutations
GenMLX:    model fn -> MLX graph data -> eval! commit -> GPU mutations
```

The MLX computation graph **is** GenMLX's hiccup. `eval!` **is** GenMLX's React
commit. The architecture is already there at Layer 0. The opportunity is to extend
this pattern upward through the inference and training layers.

### 2.3 Where the analogy breaks

re-frame event handlers fire-and-forget: they receive an event, return an effects
map, and never need intermediate results. GenMLX's `dyn/trace` is different — it's
called mid-execution inside a model body and must return a value immediately:

```clojure
;; re-frame: fire-and-forget
(reg-event-fx :click
  (fn [{:keys [db]} _]
    {:db (update db :count inc)}))   ;; returns effects, done

;; GenMLX: need value to continue
(gen [x]
  (let [slope (dyn/trace :slope (dist/gaussian 0 10))]  ;; NEED slope NOW
    (dyn/trace :y (dist/gaussian (* slope x) 1))))      ;; slope used here
```

This means the handler dispatch mechanism (`*handler*`, `*state*`) must remain as
dynamic scope — it's the irreducible core of imperative probabilistic programming
in a language without algebraic effects. But everything **around** the handler —
inference loops, training, memory management, diagnostics — can adopt re-frame
patterns.

### 2.4 Comparison with other PPL systems

| System | Purity approach | Statefulness |
|--------|----------------|--------------|
| **Gen.jl** (Julia) | Imperative, mutable traces. `update!` modifies in place. No purity discipline. | High — mutable by default |
| **GenJAX** (JAX) | Functional via JAX's pure-function requirement. `vmap`/`jit` enforce purity. But JAX's functional transforms are the mechanism, not a design choice. | Medium — enforced by JAX, not by design |
| **Turing.jl** (Julia) | Imperative sampling with `@model` macro. Mutable accumulator for log-prob. | High — no purity goals |
| **Pyro** (Python) | Effect handlers (Poutine) + mutable message passing. Side-effect-heavy. | Very high |
| **GenMLX** (current) | Pure handler transitions, volatile dispatch wrapper, dynamic var injection. | Low — but scattered `eval!`/`tidy`/`step!` |
| **GenMLX** (proposed) | Effects-as-data, coeffect injection, interceptor chains, single mutation point. | Minimal — only `*handler*`/`*state*` + GPU boundary |

GenMLX is already more disciplined about purity than any other PPL system. The
proposed changes would make it the first PPL with a re-frame-like effects-as-data
architecture, where inference algorithms are composable pure transformations and
all mutation is concentrated at explicit boundaries.

---

## 3. Current State: Complete Audit of Mutation

### 3.1 Inventory by category

**Category A — Already effectively pure (dispatch wrapper for pure transitions):**

| File | Site | Pattern | Detail |
|------|------|---------|--------|
| `handler.cljs:605` | `(volatile! init-state)` | Created per `run-handler` call | Single volatile, scoped to binding block |
| `handler.cljs:290-354` | 10× `vreset!` | One per handler type | simulate, generate, assess, update, regenerate, project + 4 batched |
| `handler.cljs:566,572,591` | Splice `vreset!`/`vswap!` | Sub-GF result merging | Recursive execution merge |
| `adev.cljs:51,200` | 2× `vreset!` | ADEV handler | Mirrors handler pattern |
| `verify.cljs:56` | `vreset!` | Contract checking | Same pattern |
| `util.cljs:172,267` | 2× `volatile!` | Temporary containers | Tree walk accumulation, tidy boundary crossing |

All transitions are pure `(state, addr, dist) -> [value, state']`. The volatile is
never exposed to user code and never escapes the binding block.

**Category B — Dynamic variable injection (could become explicit coeffects):**

| Var | File | Default | Bound by | Could be coeffect? |
|-----|------|---------|----------|-------------------|
| `*handler*` | handler.cljs | `nil` | `run-handler` | **No** — structural necessity |
| `*state*` | handler.cljs | `nil` | `run-handler` | **No** — structural necessity |
| `*param-store*` | handler.cljs | `nil` | `learning.cljs`, `adev.cljs` | **Yes** — read-only from handler |
| `*prng-key*` | mlx/random.cljs | `nil` | `dynamic.cljs`, `learning.cljs` | **Yes** — deterministic key |
| `*batched-exec?*` | mlx.cljs | `false` | `run-handler` | **Yes** — boolean flag |

12 `binding` forms across handler.cljs, dynamic.cljs, learning.cljs, adev.cljs.

**Category C — GPU boundary (fundamentally requires mutation):**

| Function | File | Call count | What it mutates |
|----------|------|-----------|----------------|
| `mx/eval!` | mlx.cljs | ~40 across codebase | Materializes computation graph on GPU |
| `mx/tidy` | mlx.cljs | ~30 across codebase | Frees intermediate Metal buffers |
| `mx/dispose!` | mlx.cljs | Via `dispose-trace!` | Explicit buffer deallocation |
| `mx/clear-cache!` | mlx.cljs | Inference utilities | Flushes Metal buffer cache |
| `mx/set-cache-limit!` | mlx.cljs | Module init | Configures GPU resources |
| `mx/set-memory-limit!` | mlx.cljs | User-facing | GPU memory cap |
| `mx/set-wired-limit!` | mlx.cljs | User-facing | Wired memory cap |
| `mx/reset-peak-memory!` | mlx.cljs | Benchmarks | Reset peak counter |
| `nn/step!` | nn.cljs | Training loops | Mutates nn module weights |
| `train-proposal!` | amortized.cljs | Training | Wraps nn/step! |
| `seed-from-key!` | mlx/random.cljs | PRNG threading | Seeds MLX global PRNG |
| `force-gc!` | util.cljs | Long inference | Triggers JS GC |
| `dispose-trace!` | util.cljs | Long inference | Bulk array disposal |

**Category D — Other:**

| Item | File | Detail |
|------|------|--------|
| 1 atom | `dist.cljs:1074` | Counter in Wishart sampling. Function-local, trivially replaceable. |
| 2 `js/console.warn` | mlx.cljs, dynamic.cljs | Development diagnostics only |
| 8 `defonce` | mlx.cljs, random.cljs | Immutable module references, not mutable state |

### 3.2 What's already pure

The following are completely free of mutation:

- All 27 distributions (sample, log-prob, sample-n)
- All 11 combinators (Map, Unfold, Switch, Scan, Mask, Mix, Recurse, Vmap, etc.)
- All choicemap operations (construction, access, merge, enumeration)
- All trace/selection/diff operations
- The `gen` macro and model body execution
- All handler state transitions (the pure functions inside the handlers)
- Score and gradient computation (builds lazy MLX graphs, no eval)
- All GFI protocol dispatch (simulate, generate, update, regenerate, assess)
- The `defdist` and `defdist-transform` macros
- All kernel constructors and combinators (chain, cycle, mix, repeat)

This is ~85% of the codebase by line count. The remaining ~15% is where the
proposed changes apply.

### 3.3 Detailed `mx/eval!` placement audit

Every `mx/eval!` call is a "commit point" — the boundary between the pure world
(building computation graphs) and the impure world (executing them on GPU). Here
is where they currently live:

**learning.cljs (4 calls):**
- Line 92: Adam optimizer state evaluation — `(mx/eval! new-params m v)`
- Line 125: Loss/gradient materialization — `(mx/eval! loss grad)`
- Line 305: Wake phase loss — `(mx/eval! loss grad)`
- Line 315: Sleep phase loss — `(mx/eval! loss grad)`

**gradients.cljs (2 calls):**
- Line 38: Choice gradient — `(mx/eval! grad-arr)`
- Line 67: Vectorized gradient — `(mx/eval! score grad)`

**inference/util.cljs (2 calls):**
- Line 191: `eval-state!` helper
- Lines 275-276: Bulk eval in `tidy-step`

**inference/mcmc.cljs (~10 calls):**
- Lines 184, 238: MH chain steps
- Line 614: Compiled MH
- Lines 849, 909: HMC leapfrog trajectory
- Line 988: NUTS
- Line 1741: MAP optimization
- Several more in vectorized variants

**inference/vi.cljs (~8 calls):**
- Lines 70, 97, 101: ADVI initialization and loop
- Lines 152, 180, 184: Compiled VI
- Lines 439-440: Programmable VI

**inference/adev.cljs (1 call):**
- Line 290: Compiled ADEV loop

**mlx.cljs (1 call):**
- Line 89: Inside `->clj` for array conversion

**Total: ~40 `mx/eval!` calls.** All correctly placed outside model bodies, but
scattered across 7 files in 6 different inference modules.

### 3.4 Detailed `mx/tidy` placement audit

`mx/tidy` wraps a function call and disposes all intermediate arrays created
during execution, preserving only the returned arrays. Essential for preventing
Metal buffer exhaustion during long inference runs.

**inference/mcmc.cljs (~12 calls):**
- MH sample loop wrapping
- HMC leapfrog step wrapping
- NUTS tree building
- MAP gradient steps
- Compiled variants

**inference/vi.cljs (~8 calls):**
- ADVI gradient computation
- ELBO estimation
- Compiled VI variants
- Programmable VI

**inference/util.cljs (1 call):**
- `tidy-step` — generic wrapper for inference steps

**inference/adev.cljs (1 call):**
- Compiled ADEV gradient

**Total: ~30 `mx/tidy` calls.** These are the primary source of memory management
code pollution in inference modules.

---

## 4. Re-frame Patterns: Deep Analysis

### 4.1 Effects-as-data

**How re-frame does it:**

```clojure
;; IMPURE handler (the wrong way):
(reg-event-db :save-item
  (fn [db [_ item]]
    (.setItem js/localStorage "item" (pr-str item))  ;; side effect!
    (dispatch [:notify "Saved!"])                     ;; side effect!
    (assoc db :saved true)))

;; PURE handler (the re-frame way):
(reg-event-fx :save-item
  (fn [{:keys [db]} [_ item]]
    {:db           (assoc db :saved true)           ;; effect: update state
     :local-store  {:key "item" :value item}        ;; effect: write storage
     :dispatch     [:notify "Saved!"]}))            ;; effect: trigger event
```

The handler returns a plain Clojure map. Each key-value pair describes one effect.
The framework's `do-fx` interceptor iterates over this map and calls the registered
effect handler for each key:

```clojure
;; Effect handlers are registered separately:
(reg-fx :local-store
  (fn [{:keys [key value]}]
    (.setItem js/localStorage key (pr-str value))))

(reg-fx :dispatch
  (fn [event-vector]
    (router/dispatch event-vector)))
```

**Why this works:** The handler function is a pure function of `(coeffects, event)
-> effects-map`. No side effects, no mutation, no I/O. Testing requires no mocks —
just call the function and assert on the returned map:

```clojure
(deftest save-item-test
  (let [result (save-handler {:db {}} [:save-item {:name "x"}])]
    (is (= true (get-in result [:db :saved])))
    (is (= {:key "item" :value {:name "x"}} (:local-store result)))
    (is (= [:notify "Saved!"] (:dispatch result)))))
```

**How GenMLX could adopt this:**

The pattern maps directly to GenMLX's training and inference loops. Instead of
performing GPU operations inline, steps would return effect descriptions:

```clojure
;; Current GenMLX: imperative training step
(defn step! [module optim vg-fn & inputs]
  (let [[loss grads] (apply vg-fn inputs)]
    (.update optim module grads)          ;; side effect: weight update
    (mx/eval! loss)                       ;; side effect: GPU sync
    loss))

;; Re-frame-inspired: pure training step
(defn training-step [weights grads optimizer-state]
  {:weights         (apply-gradients weights grads optimizer-state)
   :optimizer-state (update-momentum optimizer-state grads)
   :effects         {:apply-weights true    ;; effect: update nn module
                     :eval          true    ;; effect: GPU sync
                     :gc            false}}) ;; effect: memory management
```

The effect executor is the only impure code:

```clojure
(defn execute-training-effects! [module optimizer result]
  (when (:apply-weights (:effects result))
    (.update optimizer module (:weights result)))
  (when (:eval (:effects result))
    (mx/eval! (:weights result)))
  (when (:gc (:effects result))
    (mx/clear-cache!)))
```

### 4.2 Coeffect injection

**How re-frame does it:**

Coeffects are the dual of effects: they inject the current state of the external
world into a handler as pure data. The handler never reaches out to impure sources.

```clojure
;; Register a coeffect handler (impure — reads external world):
(reg-cofx :now
  (fn [coeffects _]
    (assoc coeffects :now (js/Date.))))

(reg-cofx :local-store
  (fn [coeffects key]
    (assoc coeffects :local-store
           (js->clj (.getItem js/localStorage key)))))

;; Inject coeffects during handler registration:
(reg-event-fx :load-defaults
  [(inject-cofx :local-store "defaults")
   (inject-cofx :now)]
  (fn [{:keys [db local-store now]} _]
    {:db (assoc db :defaults local-store :loaded-at now)}))
```

`inject-cofx` returns an interceptor whose `:before` function calls the registered
coeffect handler and `assoc`s the result into the context's `:coeffects` map. By the
time the event handler runs, everything it needs is already in its first argument.

A critical hidden detail: `reg-event-db` and `reg-event-fx` both automatically
prepend `(inject-cofx :db)` to every interceptor chain. The `:db` key in coeffects
is not magic — it's injected through the same mechanism as any other coeffect.

**Why this matters for GenMLX:**

GenMLX currently injects three things via dynamic scope:

1. **Parameter store** — `(binding [handler/*param-store* store] ...)`
2. **PRNG key** — `(binding [rng/*prng-key* (volatile! key)] ...)`
3. **Batched mode flag** — `(binding [mx/*batched-exec?* true] ...)`

With coeffect injection, these become explicit data:

```clojure
;; Current: implicit injection via dynamic scope
(binding [handler/*param-store* store
          rng/*prng-key* (volatile! key)]
  (p/simulate model args))

;; Proposed: explicit coeffect data
(run-handler simulate-transition body-fn
  {:param-store store
   :prng-key    key
   :batched?    false})
```

**Testing benefit:** Instead of complex `binding` setup, tests pass a simple map:

```clojure
;; Current test:
(binding [handler/*param-store* (make-param-store {:w (mx/scalar 1.0)})
          rng/*prng-key* (volatile! (rng/fresh-key 42))]
  (let [trace (p/simulate model [x])]
    (assert-close ...)))

;; Proposed test:
(let [trace (simulate-with-cofx model [x]
              {:param-store {:w (mx/scalar 1.0)}
               :prng-key    (rng/fresh-key 42)})]
  (assert-close ...))
```

### 4.3 Interceptor chain

**How re-frame does it:**

Every event is processed by walking through a chain of interceptors in two phases.
Each interceptor is a data map:

```clojure
{:id      :my-interceptor
 :before  (fn [context] ... return modified context ...)
 :after   (fn [context] ... return modified context ...)}
```

The context map flows through the chain:

```clojure
{:coeffects {:event [:some-id :param]
             :db    {... current app-db ...}}
 :effects   {:db    {... new app-db ...}
             :dispatch [:event-id :param]}
 :queue     [... interceptors not yet processed ...]
 :stack     [... interceptors already processed ...]}
```

Execution flow:

```
Forward sweep (:before functions, left to right):
  inject-db → inject-cofx:now → trim-v → handler-wrapper
                                              |
                                         (handler runs here,
                                          result goes into :effects)
                                              |
Backward sweep (:after functions, right to left):
  handler-wrapper → trim-v → inject-cofx:now → inject-db → do-fx
                                                              |
                                                        (effects executed here)
```

The event handler itself is wrapped as the innermost interceptor. `db-handler->
interceptor` extracts `db` and `event` from `:coeffects`, calls the handler, and
stores the result into `:effects`:

```clojure
(defn db-handler->interceptor [handler-fn]
  (->interceptor
    :id :db-handler
    :before (fn [context]
              (let [{:keys [db event]} (:coeffects context)]
                (assoc-in context [:effects :db]
                          (handler-fn db event))))))
```

**Why this matters for GenMLX:**

Inference algorithms are currently monolithic functions that interleave proposal
logic, scoring, acceptance, GPU sync, memory management, and diagnostics. An
interceptor chain would decompose them into orthogonal, composable pieces:

```clojure
;; Current MH (monolithic):
(defn mh [opts model args observations]
  (let [init-trace ...]
    (loop [trace init-trace, samples [], i 0]
      ;; mixed concerns in one loop body:
      (let [new-trace (mx/tidy                        ;; memory management
                        #(let [{:keys [trace weight]}
                               (p/regenerate model     ;; proposal
                                 trace selection)]
                           (mx/eval! (:score trace))  ;; GPU sync
                           (if (accept? weight)        ;; acceptance
                             trace
                             trace)))]
        (when callback (callback i new-trace))         ;; diagnostics
        (recur new-trace ...)))))

;; Proposed (interceptor chain):
(def mh-step-chain
  [inject-trace-interceptor      ;; coeffect: current trace
   inject-prng-interceptor       ;; coeffect: PRNG key
   propose-interceptor           ;; :before — regenerate selection
   accept-reject-interceptor     ;; :before — MH decision
   update-trace-interceptor      ;; :before — finalize trace
   diagnostics-interceptor       ;; :after — acceptance rate, ESS
   gpu-sync-interceptor          ;; :after — mx/eval!
   memory-interceptor])          ;; :after — mx/tidy, dispose
```

Each interceptor is independently testable:

```clojure
(deftest propose-test
  (let [ctx {:coeffects {:trace mock-trace :selection (sel/select :x)}}
        result ((:before propose-interceptor) ctx)]
    (is (contains? (:effects result) :proposed-trace))
    (is (contains? (:effects result) :log-alpha))))
```

Users compose custom inference by assembling interceptors:

```clojure
;; Standard MH
(def my-mh [propose accept-reject gpu-sync])

;; MH with gradient diagnostics
(def my-mh-debug [propose accept-reject gradient-check gpu-sync diagnostics])

;; MH with adaptive step size
(def my-adaptive-mh [propose accept-reject adapt-step-size gpu-sync])

;; HMC: swap the proposal interceptor
(def my-hmc [leapfrog-propose accept-reject gpu-sync])

;; MALA: leapfrog with 1 step + gradient
(def my-mala [mala-propose accept-reject gpu-sync])

;; SMC step: different structure entirely
(def smc-step [inject-particles propose-all reweight resample gpu-sync])

;; Add convergence monitoring to any algorithm
(def monitored-mh (conj my-mh convergence-check-interceptor))
```

### 4.4 Flows

**How re-frame does it:**

Flows are declarative derived values. When inputs at certain `app-db` paths change,
an output is automatically recomputed and stored at another path:

```clojure
{:id     :room-area
 :inputs {:w [:room :width] :h [:room :length]}
 :output (fn [{:keys [w h]}] (* w h))
 :path   [:room :area]}
```

Flows run via an interceptor that executes after every event handler. The system:
1. Resolves current values at input paths
2. Compares with previous values to detect changes
3. If inputs changed, runs the `:output` function
4. Stores the result back into `app-db` at `:path`

Flows can depend on other flows via `flow<-`, forming a dependency graph that
re-frame topologically sorts:

```clojure
{:id     :main-room-ratio
 :inputs {:kitchen    (rf/flow<- :kitchen-area)
          :living     (rf/flow<- :living-room-area)}
 :output (fn [{:keys [kitchen living]}] (/ kitchen living))
 :path   [:ratios :main-rooms]}
```

Flows also support lifecycle: `:live?` controls activation, `:cleanup` runs on
deactivation.

**How this maps to GenMLX:**

Derived inference quantities are currently computed manually at arbitrary points
in inference loops. Flows would automate this:

```clojure
;; Inference flows
(def smc-flows
  [{:id     :ess
    :inputs {:weights [:particles :log-weights]}
    :output (fn [{:keys [weights]}]
              (let [w (mx/softmax weights)]
                (mx/reciprocal (mx/sum (mx/square w)))))
    :path   [:diagnostics :ess]}

   {:id     :should-resample?
    :inputs {:ess (flow<- :ess) :n [:particles :count]}
    :output (fn [{:keys [ess n]}] (< (mx/item ess) (* 0.5 n)))
    :path   [:diagnostics :should-resample?]}

   {:id     :log-ml-estimate
    :inputs {:weights [:particles :log-weights]}
    :output (fn [{:keys [weights]}]
              (mx/subtract (mx/logsumexp weights)
                           (mx/log (mx/scalar (count weights)))))
    :path   [:diagnostics :log-ml]}

   {:id     :convergence
    :inputs {:samples [:chain :samples]}
    :output (fn [{:keys [samples]}]
              {:r-hat (diag/r-hat samples)
               :ess   (diag/ess samples)})
    :path   [:diagnostics :convergence]
    :live?  (fn [state] (:compute-diagnostics? state))}])
```

When particle weights change, ESS auto-recomputes. When ESS changes, the resample
decision updates. The `:live?` predicate activates expensive diagnostics (R-hat,
trace autocorrelation) only when requested.

### 4.5 Async effects

**How re-frame does it:**

re-frame handles async through effect descriptions. Handlers never perform async
operations directly:

```clojure
;; Pure handler: describes HTTP request
(reg-event-fx :fetch-data
  (fn [{:keys [db]} _]
    {:db   (assoc db :loading? true)
     :http {:url "/api/data"
            :on-success [:data-received]
            :on-fail    [:fetch-failed]}}))

;; Pure handler: processes HTTP response
(reg-event-fx :data-received
  (fn [{:keys [db]} [_ response]]
    {:db (assoc db :loading? false :data response)}))
```

Async is modeled as a sequence of pure synchronous events connected by
effect-described async operations.

**How this maps to GenMLX:**

GPU computation is fundamentally asynchronous — Metal command buffers are submitted
and completed asynchronously, and MLX uses lazy evaluation. An async effects pattern
could model inference as event sequences:

```clojure
;; Inference step as event
(reg-inference-fx :mcmc-step
  (fn [{:keys [trace key]} _]
    (let [proposed (propose trace key)]
      {:trace    (if (accept? proposed) (:trace proposed) trace)
       :gpu-eval {:arrays [(:score proposed)]
                  :on-complete [:mcmc-step-complete]}})))

;; Next step triggered by GPU completion
(reg-inference-fx :mcmc-step-complete
  (fn [{:keys [trace step]} _]
    (if (< step max-steps)
      {:dispatch [:mcmc-step]}
      {:result trace})))
```

This enables non-blocking inference, interruptible computation, and replay-based
debugging.

---

## 5. Proposed Changes

### 5.1 Change 1: Single mutation point for handler dispatch

**Effort:** Small (half a day)
**Purity gain:** Modest
**Priority:** High (enables everything else)

**Problem:** 10 handler wrapper functions each do `vreset!` directly in
`handler.cljs`. Mutation is scattered across 10+ sites.

**Current code (`handler.cljs:287-354`):**

```clojure
(defn simulate-handler [addr dist]
  (let [[value state'] (simulate-transition @*state* addr dist)]
    (vreset! *state* state')
    value))

(defn generate-handler [addr dist]
  (let [[value state'] (generate-transition @*state* addr dist)]
    (vreset! *state* state')
    value))

(defn assess-handler [addr dist]
  (let [[value state'] (assess-transition @*state* addr dist)]
    (vreset! *state* state')
    value))

;; ... 7 more identical patterns
```

**Proposed code:**

```clojure
(defn execute-transition! [addr dist]
  (let [transition-fn @*handler*
        [value state'] (transition-fn @*state* addr dist)]
    (vreset! *state* state')   ;; THE ONLY vreset! in core handler dispatch
    value))
```

The `*handler*` dynamic var already holds the current transition function. Each GFI
operation (simulate, generate, update, etc.) binds `*handler*` to the appropriate
transition when calling `run-handler`. So we don't need 10 separate wrapper functions
— we just call the bound transition directly.

**`trace-choice!` simplifies from:**

```clojure
(defn trace-choice! [addr dist]
  (if *handler*
    (*handler* addr dist)           ;; calls e.g. simulate-handler
    (dc/dist-sample dist (rng/next-key))))
```

**To:**

```clojure
(defn trace-choice! [addr dist]
  (if *handler*
    (execute-transition! addr dist) ;; single dispatch point
    (dc/dist-sample dist (rng/next-key))))
```

**Cross-cutting concerns become trivial to add:**

```clojure
(defn execute-transition! [addr dist]
  (let [transition-fn @*handler*
        [value state'] (transition-fn @*state* addr dist)]
    ;; Contract validation (optional, dev mode)
    (when *validate-contracts?*
      (validate-transition-output addr dist value state'))
    ;; State transition logging (optional, debug mode)
    (when *log-transitions?*
      (log-transition addr dist @*state* state'))
    ;; The single mutation point
    (vreset! *state* state')
    value))
```

**Splice handling also simplifies.** Currently `handler.cljs:566-591` has separate
`vreset!` and `vswap!` calls for splice operations. These would route through the
same `execute-transition!` or a parallel `execute-splice!` function.

**Files changed:** `handler.cljs`
**Tests affected:** None — behavior is identical, only internal structure changes.
**Backward compatibility:** Full — no API changes.

---

### 5.2 Change 2: Explicit coeffect injection

**Effort:** Small-Medium (1-2 days)
**Purity gain:** Medium
**Priority:** Medium

**Problem:** Three dynamic vars inject external state implicitly:

```clojure
;; learning.cljs:144
(binding [handler/*param-store* store]
  (p/simulate model args))

;; learning.cljs:203
(binding [rng/*prng-key* (volatile! key)]
  (loss-fn params))

;; handler.cljs:604-606
(binding [*handler* handler-fn
          *state*   (volatile! init-state)
          mx/*batched-exec?* (boolean (:batched? init-state))]
  ...)
```

These make testing harder (requires `binding` setup) and hide dependencies.

**Proposed: Coeffects as data in `run-handler`:**

```clojure
;; run-handler accepts explicit coeffects
(defn run-handler [transition-fn body-fn
                   & {:keys [param-store prng-key batched?]
                      :or {batched? false}}]
  (let [key (or prng-key (rng/fresh-key))
        init-state {:choices    cm/EMPTY
                    :score      (mx/scalar 0)
                    :weight     (mx/scalar 0)
                    :key        key
                    :param-store param-store    ;; coeffect: in state map
                    :batched?    batched?}]     ;; coeffect: in state map
    (binding [*handler* transition-fn
              *state*   (volatile! init-state)]
      (let [retval (body-fn)]
        (assoc @*state* :retval retval)))))
```

**The param-store becomes a field in the state map** instead of a separate dynamic
var. The `trace-param!` function reads from the state:

```clojure
;; Current: reads from dynamic var
(defn trace-param! [name default]
  (if *param-store*
    (or (get-in *param-store* [:params name]) default)
    default))

;; Proposed: reads from state map
(defn trace-param! [name default]
  (let [store (:param-store @*state*)]
    (if store
      (or (get-in store [:params name]) default)
      default)))
```

**Dynamic vars reduced from 5 to 2:**

| Var | Status | Reason |
|-----|--------|--------|
| `*handler*` | Keep | Structural necessity for `dyn/trace` dispatch |
| `*state*` | Keep | Structural necessity for `dyn/trace` state |
| `*param-store*` | **Remove** | Becomes field in state map |
| `*prng-key*` | **Remove** | Becomes `:key` in state map (already is) |
| `*batched-exec?*` | **Remove** | Becomes `:batched?` in state map |

**GFI operations gain optional coeffect arguments:**

```clojure
;; Current API (preserved for backward compat):
(p/simulate model args)

;; New API with explicit coeffects:
(p/simulate model args {:prng-key (rng/fresh-key 42)
                        :param-store my-store})
```

**Testing becomes trivial:**

```clojure
;; No more binding gymnastics:
(let [trace (p/simulate model [x]
              {:prng-key (rng/fresh-key 42)
               :param-store {:w (mx/scalar 1.0)}})]
  (assert-close "slope" expected (get-choice trace [:slope]) 0.01))
```

**Files changed:** `handler.cljs`, `dynamic.cljs`, `learning.cljs`,
`inference/adev.cljs`
**Backward compatibility:** Full — existing `binding` API still works, coeffects
are optional.

---

### 5.3 Change 3: Effects-as-data for training

**Effort:** Medium (2-3 days)
**Purity gain:** High
**Priority:** High

**Problem:** Training code in `nn.cljs` and `learning.cljs` performs GPU mutation
inline:

```clojure
;; nn.cljs:111-119 — step! is impure
(defn step! [module optim vg-fn & inputs]
  (let [[loss grads] (apply vg-fn inputs)]
    (.update optim module grads)   ;; MUTATION: updates nn module weights
    (mx/eval! loss)                ;; MUTATION: GPU sync
    loss))

;; learning.cljs:118-130 — training loop is impure
(defn train [{:keys [iterations optimizer lr callback key]} loss-grad-fn init-params]
  (let [adam-state (adam-init init-params)]
    (loop [params init-params, state adam-state, i 0]
      (if (>= i iterations)
        params
        (let [[loss grad] (mx/tidy #(loss-grad-fn params))     ;; MUTATION: tidy
              [new-params new-state] (adam-step params grad state
                                               {:lr lr})]
          (mx/eval! loss grad)                                  ;; MUTATION: GPU sync
          (when callback (callback i new-params (mx/item loss)));; MUTATION: item
          (recur new-params new-state (inc i)))))))
```

**Proposed: Pure training step returning effects map:**

```clojure
;; Pure training step — zero side effects
(defn training-step [state loss-grad-fn]
  (let [{:keys [params adam-state step]} state
        [loss grad] (loss-grad-fn params)
        [new-params new-adam] (adam-step params grad adam-state
                                        {:lr (:lr state)})]
    {:params     new-params
     :adam-state new-adam
     :step       (inc step)
     :loss       loss
     :effects    {:eval   [loss new-params]
                  :tidy   true
                  :gc     (when (zero? (mod step 100)) :clear-cache)}}))

;; Effect executor — the only impure training code
(defn execute-training-effects! [effects]
  (when (:tidy effects)
    (mx/tidy
      #(when-let [arrays (:eval effects)]
         (apply mx/eval! arrays))))
  (when (= :clear-cache (:gc effects))
    (mx/clear-cache!)))

;; Pure training loop as reduce
(defn train-pure [opts loss-grad-fn init-params]
  (let [init-state {:params     init-params
                    :adam-state (adam-init init-params)
                    :step       0
                    :lr         (:lr opts)}]
    (reduce
      (fn [state _]
        (let [result (training-step state loss-grad-fn)]
          (execute-training-effects! (:effects result))
          (when (:callback opts)
            ((:callback opts) (:step result) (:params result)
                              (mx/item (:loss result))))
          (dissoc result :effects :loss)))
      init-state
      (range (:iterations opts)))))
```

**Same pattern for `nn/step!`:**

```clojure
;; Pure: returns effect description
(defn nn-training-step [module-weights grads optimizer-state]
  (let [new-weights (compute-new-weights module-weights grads optimizer-state)]
    {:weights         new-weights
     :optimizer-state (update-optimizer-state optimizer-state grads)
     :effects         {:apply-weights new-weights
                       :eval          [new-weights]}}))

;; Impure executor (called once per step)
(defn execute-nn-effects! [module optimizer effects]
  (when-let [w (:apply-weights effects)]
    (.update optimizer module w))
  (when-let [a (:eval effects)]
    (apply mx/eval! a)))
```

**Same pattern for `amortized.cljs:train-proposal!`:**

```clojure
;; Current: imperative
(defn train-proposal! [encoder loss-fn dataset & opts]
  (let [optim (nn/optimizer :adam lr)]
    (doseq [i (range iterations)]
      (doseq [data dataset]
        (nn/step! encoder optim vg-fn data)))))  ;; mutation each step

;; Proposed: pure step + executor
(defn amortized-training-step [state encoder-weights data loss-fn]
  (let [[loss grads] (loss-fn encoder-weights data)
        new-weights (apply-gradients encoder-weights grads (:optimizer state))]
    {:weights new-weights
     :step    (inc (:step state))
     :loss    loss
     :effects {:apply-weights new-weights :eval [loss new-weights]}}))
```

**Testing benefit — train without a GPU:**

```clojure
(deftest training-step-test
  (let [mock-loss-grad (fn [params] [(mx/scalar 0.5) (mx/scalar 0.1)])
        state {:params (mx/scalar 1.0) :adam-state (adam-init (mx/scalar 1.0))
               :step 0 :lr 0.01}
        result (training-step state mock-loss-grad)]
    ;; Pure assertions on the effects map:
    (is (contains? (:effects result) :eval))
    (is (mx/array? (:loss result)))
    (is (= 1 (:step result)))
    ;; Never touched the GPU — no mx/eval! needed for the test
    ))
```

**Files changed:** `nn.cljs`, `learning.cljs`, `inference/amortized.cljs`
**Backward compatibility:** Add new pure API alongside existing `step!`/`train`.
Deprecate old API gradually.

---

### 5.4 Change 4: Effects-as-data for inference memory management

**Effort:** Medium (3-4 days)
**Purity gain:** High
**Priority:** High

**Problem:** `mx/tidy`, `mx/eval!`, `dispose-trace!`, and `force-gc!` are called
imperatively throughout inference loops. These are scattered across ~70 call sites
in 7 files, interleaved with pure inference logic.

**Current pattern (repeated ~40 times across inference modules):**

```clojure
;; mcmc.cljs — typical MCMC loop
(loop [trace init-trace, samples [], i 0]
  (if (>= i (+ burn samples-count))
    samples
    (let [new-trace (mx/tidy                              ;; memory mgmt
                      #(let [{:keys [trace weight]}
                             (p/regenerate model trace selection)]
                         (mx/eval! (:score trace))        ;; GPU sync
                         (if (> (mx/item weight) threshold)
                           trace
                           trace)))]
      (when (> i burn)
        (dispose-trace! trace))                            ;; memory mgmt
      (recur new-trace
             (if (> i burn)
               (conj samples new-trace)
               samples)
             (inc i)))))
```

**Proposed: Pure inference step + generic effect executor:**

```clojure
;; Pure MH step — no side effects
(defn mh-step-pure [trace selection key]
  (let [[k1 k2] (rng/split key)
        {:keys [trace weight]} (p/regenerate (:gen-fn trace) trace selection)
        log-u (mx/log (rng/uniform k2 []))
        accept? (mx/greater weight log-u)]
    {:trace     (if (mx/item accept?) trace trace)  ;; new or old trace
     :accepted? accept?
     :key       k1
     :effects   {:eval    [(:score trace)]          ;; GPU sync needed
                 :dispose []                        ;; traces to free
                 :tidy    true}}))                  ;; memory management

;; Generic inference loop with effect execution at boundaries
(defn run-mcmc [step-fn init-state opts]
  (loop [state init-state, samples (transient []), i 0]
    (if (>= i (+ (:burn opts) (:samples opts)))
      (persistent! samples)
      (let [result (step-fn (:trace state) (:key state))
            old-trace (:trace state)
            effects (assoc-in (:effects result) [:dispose]
                              (when (and (> i (:burn opts))
                                         (not= old-trace (:trace result)))
                                [old-trace]))]
        ;; Effect execution at the boundary
        (execute-inference-effects! effects)
        (recur
          (select-keys result [:trace :key])
          (if (and (> i (:burn opts))
                   (zero? (mod (- i (:burn opts)) (:thin opts 1))))
            (conj! samples (:trace result))
            samples)
          (inc i))))))

;; Effect executor — handles all GPU/memory operations
(defn execute-inference-effects! [effects]
  (when (:tidy effects)
    (mx/tidy
      #(do
         (when-let [arrays (:eval effects)]
           (apply mx/eval! (remove nil? arrays)))
         nil)))
  (when-let [traces (:dispose effects)]
    (doseq [t (remove nil? traces)]
      (dispose-trace! t)))
  (when (:gc effects)
    (force-gc!)))
```

**Pluggable memory strategies:**

```clojure
;; Aggressive: dispose every old trace, GC every 100 steps
(def aggressive-memory
  {:dispose-old? true
   :gc-interval  100
   :tidy?        true})

;; Lazy: keep traces in memory, GC only when Metal buffers fill
(def lazy-memory
  {:dispose-old? false
   :gc-interval  nil
   :tidy?        false})

;; The inference loop uses the strategy to populate effects
(defn add-memory-effects [result step strategy]
  (assoc result :effects
    {:eval    (:eval (:effects result))
     :tidy    (:tidy? strategy)
     :dispose (when (:dispose-old? strategy) ...)
     :gc      (when-let [interval (:gc-interval strategy)]
                (zero? (mod step interval)))}))
```

**This pattern applies uniformly across all inference modules:**

| Module | Current `mx/tidy` calls | Current `mx/eval!` calls | After: effect sites |
|--------|------------------------|------------------------|---------------------|
| `mcmc.cljs` | ~12 | ~10 | 1 executor call per step |
| `vi.cljs` | ~8 | ~8 | 1 executor call per step |
| `smc.cljs` | ~3 | ~5 | 1 executor call per step |
| `adev.cljs` | ~1 | ~1 | 1 executor call per step |
| Total | ~24 | ~24 | ~4 executor functions |

**Files changed:** `inference/mcmc.cljs`, `inference/smc.cljs`, `inference/vi.cljs`,
`inference/adev.cljs`, `inference/util.cljs`
**Backward compatibility:** Internal refactor — public API unchanged.

---

### 5.5 Change 5: Interceptor chain for composable inference

**Effort:** Large (5-7 days)
**Purity gain:** Very high
**Priority:** Medium (ambitious but transformative)

**Problem:** Inference algorithms are monolithic functions that mix orthogonal
concerns. Adding diagnostics, gradient checks, or custom memory strategies requires
modifying the algorithm implementation.

**Proposed: New `interceptor.cljs` module:**

```clojure
(ns genmlx.interceptor)

;; An interceptor is a data map
(defn ->interceptor [& {:keys [id before after]}]
  {:id     id
   :before (or before identity)
   :after  (or after identity)})

;; The context map
;; {:coeffects {:trace ... :key ... :selection ...}
;;  :effects   {:trace ... :accepted? ... :eval [...]}
;;  :queue     [remaining interceptors]
;;  :stack     [processed interceptors]}

;; Execute a chain: forward sweep of :before, backward sweep of :after
(defn execute-chain [chain coeffects]
  (let [init-ctx {:coeffects coeffects
                  :effects   {}
                  :queue     chain
                  :stack     []}]
    ;; Forward sweep
    (let [after-forward
          (reduce (fn [ctx interceptor]
                    (-> ctx
                        (update :stack conj interceptor)
                        (update :queue rest)
                        ((:before interceptor) )))
                  init-ctx
                  chain)]
      ;; Backward sweep
      (reduce (fn [ctx interceptor]
                ((:after interceptor) ctx))
              after-forward
              (reverse (:stack after-forward))))))
```

**Standard interceptor library:**

```clojure
;; Coefficient injection interceptors
(def inject-trace
  (->interceptor
    :id :inject-trace
    :before (fn [ctx]
              (assoc-in ctx [:coeffects :trace]
                        (get-in ctx [:coeffects :init-trace])))))

(def inject-prng
  (->interceptor
    :id :inject-prng
    :before (fn [ctx]
              (let [[k1 k2] (rng/split (get-in ctx [:coeffects :key]))]
                (-> ctx
                    (assoc-in [:coeffects :proposal-key] k1)
                    (assoc-in [:coeffects :key] k2))))))

;; Core MCMC interceptors
(def propose-regenerate
  (->interceptor
    :id :propose-regenerate
    :before (fn [ctx]
              (let [{:keys [trace selection proposal-key]} (:coeffects ctx)
                    result (p/regenerate (:gen-fn trace) trace selection)]
                (-> ctx
                    (assoc-in [:effects :proposed-trace] (:trace result))
                    (assoc-in [:effects :log-alpha] (:weight result)))))))

(def propose-random-walk
  (->interceptor
    :id :propose-random-walk
    :before (fn [ctx]
              (let [{:keys [trace addresses proposal-std proposal-key]}
                    (:coeffects ctx)
                    params (u/extract-params trace addresses)
                    noise  (mx/multiply (mx/scalar proposal-std)
                                        (rng/normal proposal-key
                                                    (mx/shape params)))
                    proposed (mx/add params noise)
                    constraints (u/params->constraints addresses proposed)]
                (let [{:keys [trace weight]} (p/update (:gen-fn trace)
                                                        trace constraints)]
                  (-> ctx
                      (assoc-in [:effects :proposed-trace] trace)
                      (assoc-in [:effects :log-alpha] weight)))))))

(defn leapfrog-propose [leapfrog-steps step-size]
  (->interceptor
    :id :leapfrog-propose
    :before (fn [ctx]
              (let [{:keys [trace addresses proposal-key]} (:coeffects ctx)
                    params (u/extract-params trace addresses)
                    momentum (rng/normal proposal-key (mx/shape params))
                    {:keys [new-params new-momentum log-alpha]}
                    (leapfrog-trajectory params momentum trace
                                         leapfrog-steps step-size)]
                (-> ctx
                    (assoc-in [:effects :proposed-trace]
                              (reconstruct-trace trace addresses new-params))
                    (assoc-in [:effects :log-alpha] log-alpha)
                    (assoc-in [:effects :momentum] new-momentum))))))

(def accept-reject-mh
  (->interceptor
    :id :accept-reject
    :before (fn [ctx]
              (let [log-alpha (get-in ctx [:effects :log-alpha])
                    key       (get-in ctx [:coeffects :proposal-key])
                    accept?   (u/accept-mh? log-alpha key)
                    trace     (if accept?
                                (get-in ctx [:effects :proposed-trace])
                                (get-in ctx [:coeffects :trace]))]
                (-> ctx
                    (assoc-in [:effects :trace] trace)
                    (assoc-in [:effects :accepted?] accept?))))))

;; Cross-cutting interceptors
(def gpu-sync
  (->interceptor
    :id :gpu-sync
    :after (fn [ctx]
             (when-let [trace (get-in ctx [:effects :trace])]
               (mx/eval! (:score trace)))
             ctx)))

(def memory-management
  (->interceptor
    :id :memory
    :after (fn [ctx]
             (when-let [old (get-in ctx [:effects :old-trace])]
               (dispose-trace! old))
             ctx)))

(defn diagnostics [& {:keys [track-acceptance? track-ess?]}]
  (->interceptor
    :id :diagnostics
    :after (fn [ctx]
             (cond-> ctx
               track-acceptance?
               (update-in [:effects :diagnostics :accept-count]
                          (fnil + 0) (if (get-in ctx [:effects :accepted?]) 1 0))
               track-ess?
               (assoc-in [:effects :diagnostics :ess]
                         (compute-ess (get-in ctx [:effects :trace])))))))

(defn adapt-step-size [target-accept]
  (->interceptor
    :id :adapt-step-size
    :after (fn [ctx]
             (let [accepted? (get-in ctx [:effects :accepted?])
                   current   (get-in ctx [:coeffects :step-size] 0.01)
                   ;; Dual averaging adaptation
                   new-size  (if accepted?
                               (* current 1.02)
                               (* current 0.98))]
               (assoc-in ctx [:effects :adapted-step-size] new-size)))))
```

**Assembling algorithms from interceptors:**

```clojure
;; MH with prior proposal
(def mh-chain
  [inject-prng propose-regenerate accept-reject-mh gpu-sync memory-management])

;; MH with random walk
(def rw-mh-chain
  [inject-prng propose-random-walk accept-reject-mh gpu-sync memory-management])

;; HMC
(def hmc-chain
  [inject-prng (leapfrog-propose 10 0.01) accept-reject-mh
   gpu-sync memory-management])

;; Adaptive HMC with diagnostics
(def adaptive-hmc-chain
  [inject-prng (leapfrog-propose 10 0.01) accept-reject-mh
   (adapt-step-size 0.65)
   (diagnostics :track-acceptance? true)
   gpu-sync memory-management])

;; SMC step (entirely different structure, same interceptor pattern)
(def smc-step-chain
  [inject-particles inject-observations
   propose-extensions reweight
   check-ess-and-resample
   gpu-sync memory-management])
```

**Running inference with a chain:**

```clojure
(defn run-chain-mcmc [chain model args observations opts]
  (let [init-trace (first (:traces (is/importance-sampling
                                     {:samples 1} model args observations)))
        init-cofx {:trace     init-trace
                   :key       (or (:key opts) (rng/fresh-key))
                   :selection (or (:selection opts) sel/all)
                   :model     model}]
    (loop [cofx init-cofx, samples [], i 0]
      (if (>= i (+ (:burn opts 0) (:samples opts)))
        samples
        (let [result (execute-chain chain cofx)
              new-cofx (merge cofx
                              {:trace (get-in result [:effects :trace])
                               :key   (get-in result [:coeffects :key])})]
          (recur new-cofx
                 (if (>= i (:burn opts 0))
                   (conj samples (get-in result [:effects :trace]))
                   samples)
                 (inc i)))))))

;; Usage:
(run-chain-mcmc mh-chain model args obs {:samples 1000 :burn 200})
(run-chain-mcmc adaptive-hmc-chain model args obs {:samples 500 :burn 100})
```

**Files changed:** New `interceptor.cljs`, refactor `inference/mcmc.cljs`,
`inference/smc.cljs`, `inference/vi.cljs`
**Backward compatibility:** Existing API preserved. Interceptor chain is a new,
parallel API for advanced users.

---

### 5.6 Change 6: Flows for derived inference quantities

**Effort:** Medium-Large (3-5 days)
**Purity gain:** High
**Priority:** Low (nice-to-have, becomes natural after interceptors)

**Problem:** Derived quantities (ESS, log-ML, acceptance rate, R-hat) are computed
ad-hoc at specific points in inference loops. Adding new diagnostics requires
modifying algorithm implementations.

**Proposed: New `flows.cljs` module:**

```clojure
(ns genmlx.flows)

(defrecord Flow [id inputs output path live? cleanup])

(defn flow<- [flow-id]
  {:type :flow-ref :id flow-id})

(defn make-flow [& {:keys [id inputs output path live? cleanup]}]
  (->Flow id inputs output path (or live? (constantly true)) cleanup))

;; Resolve inputs from state map, checking for changes
(defn resolve-inputs [flow state prev-state]
  (reduce-kv
    (fn [m k input-spec]
      (let [v (cond
                (vector? input-spec) (get-in state input-spec)
                (and (map? input-spec) (= :flow-ref (:type input-spec)))
                (get-in state (:path (get flows-by-id (:id input-spec))))
                :else input-spec)]
        (assoc m k v)))
    {}
    (:inputs flow)))

;; Run all flows, topologically sorted, with change detection
(defn run-flows [flows state prev-state]
  (reduce
    (fn [state flow]
      (if (not ((:live? flow) state))
        ;; Flow is inactive — run cleanup if transitioning
        (if ((:live? flow) prev-state)
          (if-let [cleanup (:cleanup flow)]
            (cleanup state)
            state)
          state)
        ;; Flow is active — check if inputs changed
        (let [inputs (resolve-inputs flow state prev-state)
              prev-inputs (resolve-inputs flow prev-state prev-state)]
          (if (= inputs prev-inputs)
            state  ;; No change — skip computation (deduplication)
            (let [output ((:output flow) inputs)]
              (assoc-in state (:path flow) output))))))
    state
    (topo-sort flows)))

;; Register flows with an inference loop
(defn with-flows [flows step-fn]
  (fn [state & args]
    (let [prev-state state
          result (apply step-fn state args)]
      (run-flows flows result prev-state))))
```

**Standard inference flows:**

```clojure
(def smc-flows
  [(make-flow
     :id     :ess
     :inputs {:weights [:particles :log-weights]}
     :output (fn [{:keys [weights]}]
               (u/compute-ess weights))
     :path   [:diagnostics :ess])

   (make-flow
     :id     :normalized-weights
     :inputs {:log-weights [:particles :log-weights]}
     :output (fn [{:keys [log-weights]}]
               (mx/softmax log-weights))
     :path   [:particles :normalized-weights])

   (make-flow
     :id     :should-resample?
     :inputs {:ess (flow<- :ess) :n [:particles :count]}
     :output (fn [{:keys [ess n]}]
               (< (mx/item ess) (* 0.5 n)))
     :path   [:diagnostics :should-resample?])

   (make-flow
     :id     :log-ml-estimate
     :inputs {:weights [:particles :log-weights]}
     :output (fn [{:keys [weights]}]
               (vec/vtrace-log-ml-estimate weights))
     :path   [:diagnostics :log-ml])

   (make-flow
     :id     :convergence
     :inputs {:samples [:chain :samples]}
     :output (fn [{:keys [samples]}]
               {:r-hat      (diag/r-hat samples)
                :ess        (diag/ess samples)
                :mean       (diag/sample-mean samples)
                :std        (diag/sample-std samples)
                :quantiles  (diag/sample-quantiles samples)})
     :path   [:diagnostics :convergence]
     :live?  (fn [state] (:compute-convergence? state false)))])

(def mcmc-flows
  [(make-flow
     :id     :acceptance-rate
     :inputs {:accepted [:step :accepted?] :total [:step :count]}
     :output (fn [{:keys [accepted total]}]
               (/ accepted (max 1 total)))
     :path   [:diagnostics :acceptance-rate])

   (make-flow
     :id     :step-size-recommendation
     :inputs {:rate (flow<- :acceptance-rate)}
     :output (fn [{:keys [rate]}]
               (cond
                 (< rate 0.1) :decrease-step-size
                 (> rate 0.9) :increase-step-size
                 :else        :step-size-ok))
     :path   [:diagnostics :step-size-recommendation])])
```

**Integration with interceptor chain:**

```clojure
;; Flows run as an :after interceptor
(defn flows-interceptor [flows]
  (->interceptor
    :id :flows
    :after (fn [ctx]
             (let [state (:effects ctx)
                   prev  (:coeffects ctx)]
               (assoc ctx :effects (run-flows flows state prev))))))

;; Add flows to any inference chain
(def mh-with-diagnostics
  (conj mh-chain (flows-interceptor mcmc-flows)))
```

**Files changed:** New `flows.cljs`, integration with inference modules
**Backward compatibility:** Additive — no existing API changes.

---

### 5.7 Change 7: Remove the Wishart atom

**Effort:** Tiny (15 minutes)
**Purity gain:** Tiny
**Priority:** Low (completionism)

**Current (`dist.cljs:1074`):**

```clojure
(let [keys (rng/split-n key (* n (dec n)))
      ki (atom 0)
      next-key! (fn [] (let [i @ki] (swap! ki inc) (nth keys i)))]
  ;; Build Bartlett decomposition matrix using next-key! for each element
  ...)
```

**Proposed:**

```clojure
(let [keys (rng/split-n key (* n (dec n)))]
  ;; Use reduce with index, or loop with key-index threading
  (loop [i 0, matrix initial-matrix]
    (if (>= i (* n (dec n)))
      matrix
      (let [k (nth keys i)]
        (recur (inc i) (update-matrix matrix i k))))))
```

**Files changed:** `dist.cljs`

---

## 6. Proposed Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         PURE WORLD                              │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  User Code                                                │  │
│  │  ├── gen bodies: (gen [x] (trace :y (gaussian x 1)))     │  │
│  │  ├── distributions: (gaussian mu sigma)                   │  │
│  │  ├── combinators: (unfold-combinator kernel)              │  │
│  │  └── choicemaps/traces/selections                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────────┐  │
│  │  Handler Transitions (pure)                               │  │
│  │  (fn [state addr dist] -> [value state'])                 │  │
│  │  ├── simulate-transition                                  │  │
│  │  ├── generate-transition                                  │  │
│  │  ├── update-transition                                    │  │
│  │  ├── regenerate-transition                                │  │
│  │  └── batched-* variants                                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────────┐  │
│  │  Inference Steps (pure, return effects-as-data)           │  │
│  │  ├── mh-step-pure: trace -> {:trace :effects}             │  │
│  │  ├── hmc-step-pure: trace -> {:trace :effects}            │  │
│  │  ├── smc-step-pure: particles -> {:particles :effects}    │  │
│  │  └── training-step: params -> {:params :effects}          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────────┐  │
│  │  Interceptor Chain (pure context transformations)         │  │
│  │  ├── :before — coeffect injection, proposal, acceptance   │  │
│  │  └── :after  — diagnostics, adaptation, flows             │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────────┐  │
│  │  Flows (pure derived values, auto-maintained)             │  │
│  │  ├── ESS, log-ML, acceptance rate                         │  │
│  │  ├── convergence diagnostics (R-hat, autocorrelation)     │  │
│  │  └── dependency graph with change deduplication           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    COEFFECT BOUNDARY                            │
│                                                                 │
│  inject :prng-key ────── deterministic PRNG key                 │
│  inject :param-store ─── trainable parameters                   │
│  inject :trace ───────── current execution trace                │
│  inject :gpu-info ────── device capabilities, memory state      │
│  inject :memory-strategy  aggressive vs lazy disposal           │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                     EFFECT BOUNDARY                             │
│               (execute-effects! — single function)              │
│                                                                 │
│  :state-update ──── single vreset! point (handler dispatch)     │
│  :eval ──────────── mx/eval! (materialize computation graph)    │
│  :tidy ──────────── mx/tidy (free intermediate buffers)         │
│  :dispose ───────── mx/dispose! (free specific arrays)          │
│  :apply-weights ─── nn module weight update                     │
│  :gc ────────────── mx/clear-cache!, force-gc!                  │
│  :seed ──────────── seed-from-key! (MLX PRNG)                   │
│  :memory-config ─── set-memory-limit!, set-cache-limit!         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    GPU / METAL (the "DOM")                       │
│                                                                 │
│  MLX computation graph execution                                │
│  Metal command buffer submission & completion                   │
│  Unified memory buffer allocation & deallocation                │
│  PRNG state management                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Implementation Plan

### Phase 1: Foundation (1 day)

**Change 1: Single `vreset!` point**

- Refactor `handler.cljs` to use single `execute-transition!`
- Verify all 165 Gen.clj compat tests pass
- Verify all 73 GenJAX compat tests pass
- Verify all core tests pass

### Phase 2: Pure training (2-3 days)

**Change 3: Effects-as-data for training**

- Add pure `training-step` alongside existing `train`
- Add pure `nn-training-step` alongside existing `step!`
- Add `execute-training-effects!`
- Refactor `amortized.cljs:train-proposal!`
- Write tests for pure training steps (no GPU required)
- Keep existing `step!` and `train` for backward compat

### Phase 3: Pure inference memory (3-4 days)

**Change 4: Effects-as-data for inference memory management**

- Add `execute-inference-effects!` to `inference/util.cljs`
- Refactor `mcmc.cljs`: MH, compiled-MH, MALA, HMC, NUTS
- Refactor `smc.cljs`: SMC, cSMC, vSMC
- Refactor `vi.cljs`: ADVI, compiled-VI, programmable-VI
- Refactor `adev.cljs`: ADEV, compiled-ADEV
- Verify all inference tests pass
- Verify vectorized benchmarks unchanged

### Phase 4: Explicit coeffects (1-2 days)

**Change 2: Coeffect injection**

- Move `*param-store*` into state map
- Move `*batched-exec?*` into state map
- Fold PRNG key handling into state map (already partially there)
- Add optional coeffects argument to GFI operations
- Verify backward compatibility (old binding API still works)

### Phase 5: Interceptor chain (5-7 days)

**Change 5: Composable inference middleware**

- Implement `interceptor.cljs` (execute-chain, ->interceptor)
- Build standard interceptor library (propose, accept, gpu-sync, memory, diagnostics)
- Add `run-chain-mcmc` as parallel API to existing `mh`
- Build HMC, MALA, NUTS from interceptors
- Write comprehensive tests
- Document the interceptor API

### Phase 6: Flows (3-5 days)

**Change 6: Derived inference quantities**

- Implement `flows.cljs` (Flow record, dependency resolution, change detection)
- Build standard SMC and MCMC flow sets
- Integrate flows as interceptor `:after` phase
- Write tests for flow dependency graph
- Document the flows API

### Phase 7: Cleanup (15 minutes)

**Change 7: Remove Wishart atom**

- Replace atom with loop/reduce in `dist.cljs`
- Run distribution tests

### Summary

| Phase | Effort | Cumulative | What becomes pure |
|-------|--------|-----------|------------------|
| 1 | 1 day | 1 day | Handler dispatch (10 vreset! → 1) |
| 2 | 2-3 days | 3-4 days | Training loops (nn/step!, train, train-proposal!) |
| 3 | 3-4 days | 6-8 days | Inference memory (~70 eval!/tidy/dispose calls) |
| 4 | 1-2 days | 7-10 days | Context injection (5 dynamic vars → 2) |
| 5 | 5-7 days | 12-17 days | Inference algorithms (composable interceptors) |
| 6 | 3-5 days | 15-22 days | Derived quantities (auto-maintained flows) |
| 7 | 15 min | 15-22 days | Wishart distribution (1 atom → 0) |

**Phases 1-4 are the high-value core** — they deliver most of the purity gains
with moderate effort and no API changes.

**Phases 5-6 are the ambitious extension** — they require new API design but
enable composable, user-extensible inference.

---

## 8. Testing Strategy

### 8.1 Purity verification

After each phase, verify that the pure functions are genuinely pure:

```clojure
;; A pure function called twice with same inputs gives same outputs
(deftest training-step-purity
  (let [state {:params (mx/scalar 1.0) :adam-state ... :step 0 :lr 0.01}
        result-1 (training-step state mock-loss-grad)
        result-2 (training-step state mock-loss-grad)]
    (is (= (:step result-1) (:step result-2)))
    (is (= (mx/item (:loss result-1)) (mx/item (:loss result-2))))
    (is (= (keys (:effects result-1)) (keys (:effects result-2))))))
```

### 8.2 Effect executor testing

Test that effect executors correctly handle all effect types:

```clojure
(deftest execute-inference-effects-test
  ;; Verify eval! is called with correct arrays
  ;; Verify dispose! is called for discarded traces
  ;; Verify gc is called when requested
  ;; Verify tidy wrapping works correctly
  ...)
```

### 8.3 Interceptor testing

Each interceptor is testable in isolation:

```clojure
(deftest propose-interceptor-test
  (let [ctx {:coeffects {:trace mock-trace :selection (sel/select :x)}}
        result ((:before propose-regenerate) ctx)]
    (is (contains? (:effects result) :proposed-trace))
    (is (contains? (:effects result) :log-alpha))
    (is (number? (mx/item (get-in result [:effects :log-alpha]))))))
```

### 8.4 Regression testing

All existing tests must pass unchanged:
- 165/165 Gen.clj compatibility tests
- 73/73 GenJAX compatibility tests
- All core unit tests
- All vectorized tests
- All inference convergence tests

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Performance regression from effect indirection | Medium | Profile before/after. Effect maps are small; overhead should be negligible vs GPU cost. |
| API complexity for users | Medium | Keep existing API. Effects/interceptors are opt-in for advanced users. |
| Incomplete interceptor coverage | High | Start with MH only, verify pattern works before expanding to HMC/SMC/VI. |
| Memory management correctness | High | Extensive testing of tidy/dispose in effect executor. Compare memory profiles. |
| Breaking backward compat | High | All changes additive. Old API preserved. Deprecation warnings only. |

---

## 10. What Cannot Be Made Pure

### 10.1 Irreducible dynamic vars

Two dynamic vars are structurally required:

1. **`*handler*`** — `dyn/trace` is called mid-execution inside `gen` bodies. The
   handler must be available via dynamic scope because user code doesn't (and
   shouldn't) thread it explicitly. Without algebraic effects or monadic
   do-notation, there is no way to thread the handler through imperative code.

2. **`*state*`** — Same reason. The volatile holds the handler state that
   `dyn/trace` reads and writes during imperative model body execution.

These are GenMLX's equivalent of re-frame's `app-db` atom — the minimal mutable
shell that the pure functional core lives inside.

### 10.2 GPU operations

`mx/eval!`, `mx/dispose!`, `mx/tidy`, and `seed-from-key!` are fundamentally
stateful. They mutate Metal buffer state, GPU memory, and PRNG state. This is
GenMLX's DOM: the external world that the framework manages through a layer of
indirection.

The goal is not to eliminate these mutations but to **push them to the boundary**
so all code above the effect executor is pure.

### 10.3 The `dyn/trace` tension

The fundamental difference between re-frame and GenMLX is that re-frame handlers
fire-and-forget (return effects, done), while `dyn/trace` must return a value for
the model body to continue. This tension is irresolvable in ClojureScript without
language-level support for:

- **Algebraic effects** (like OCaml 5, Koka, or Eff) — would allow `dyn/trace`
  to suspend, hand control to the handler, and resume with the value.
- **Monadic do-notation** (like Haskell) — would allow threading state through
  a monad without explicit passing.
- **Delimited continuations** — would allow capturing the continuation at
  `dyn/trace` and resuming it later.

None of these are available in ClojureScript. Dynamic vars + volatile are the
idiomatic Clojure solution.

---

## 11. Relation to Other Papers

### System paper (`plan-system.md`)

The purity architecture is a key differentiator vs Gen.jl (Julia, imperative) and
GenJAX (JAX, functional but vmap-heavy). A section on "effects-as-data for
GPU-accelerated inference" would strengthen the systems contribution. The comparison
table in Section 2.4 directly supports the novelty claim.

### Formal paper (`plan-formal.md`)

The interceptor chain formalizes inference as composable pure transformations on a
context map. Each interceptor's `:before` and `:after` are pure functions
`context -> context'`. This connects to the measure-theoretic GFI contracts:
contract verification can run as an interceptor.

### Vectorization paper (`plan-vectorization.md`)

The effects boundary naturally supports batched effect execution. Multiple `eval!`
calls from vectorized inference can be fused into one at the executor level. The
`batched?` coeffect replaces the `*batched-exec?*` dynamic var.

### Compilation paper (`plan-compilation.md`)

Compiled inference (compiled-mh, compiled-vi, compiled-adev) currently inlines
`mx/eval!` and `mx/tidy` for performance. The effects-as-data pattern must be
compatible with compilation — the effect executor could itself be compiled.

---

## 12. References

### re-frame

- [re-frame documentation](https://day8.github.io/re-frame/re-frame/)
- [A Data Loop (6 dominoes)](https://github.com/day8/re-frame/blob/master/docs/a-loop.md)
- [Effectful Handlers](https://day8.github.io/re-frame/EffectfulHandlers/)
- [Coeffects](https://day8.github.io/re-frame/Coeffects/)
- [Interceptors](https://github.com/day8/re-frame/blob/master/docs/Interceptors.md)
- [Flows](https://github.com/day8/re-frame/blob/master/docs/Flows.md)
- [API Reference](https://day8.github.io/re-frame/api-re-frame.core/)

### Probabilistic programming systems (comparison targets)

- Gen.jl: Cusumano-Towner et al., "Gen: A General-Purpose Probabilistic
  Programming System with Programmable Inference" (PLDI 2019)
- GenJAX: Lew et al., "GenJAX: Scaling Programmable Inference in Gen with JAX"
- Pyro: Bingham et al., "Pyro: Deep Universal Probabilistic Programming" (JMLR 2019)
- Turing.jl: Ge et al., "Turing: A Language for Flexible Probabilistic Inference"

### Functional programming and effects

- Algebraic effects and handlers: Plotkin & Pretnar (2013)
- Monads for functional programming: Wadler (1995)
- The Expression Problem: Wadler (1998)
