# Plan: Pushing GenMLX Toward Pure Functional Architecture

**Inspired by:** re-frame's effects-as-data, coeffect injection, interceptor chains, and flows

**Status:** Research complete, ready for implementation planning

---

## Motivation

GenMLX is already remarkably pure for a GPU-accelerated probabilistic programming
system — 33 source files, only 8 contain any mutation, zero global mutable state.
But mutation is still scattered across inference loops (memory management), training
code (`nn/step!`), and the handler dispatch layer (10 `vreset!` sites).

[re-frame](https://day8.github.io/re-frame/re-frame/) solved the same problem for
web UIs: the browser DOM is a mutable external resource, but re-frame keeps all
application logic pure by treating the DOM as an effect target. Event handlers return
**data descriptions** of side effects; the framework executes them.

GenMLX's GPU (MLX/Metal) is directly analogous to re-frame's DOM:

| re-frame | GenMLX |
|----------|--------|
| Browser DOM | Metal GPU |
| Hiccup (data describing DOM) | MLX lazy computation graph |
| React reconciliation + commit | `mx/eval!` materializing on Metal |
| `app-db` atom | `*state*` volatile in handler |
| Event handlers (pure) | Handler transitions (already pure) |
| Effects map `{:db ... :dispatch ...}` | Currently implicit — GPU ops inline |
| Coeffects `{:db ... :now ...}` | Dynamic vars (`*param-store*`, `*prng-key*`) |
| Interceptor chain | Not present |
| Flows (derived values) | Not present |

The key re-frame insight: **handlers describe effects, they don't perform them**.
GenMLX's handler transitions are already pure functions. But training, memory
management, and inference loops still perform effects directly.

---

## Current State: Complete Audit

### Where mutation lives (8 of 33 source files)

**Handler dispatch (Category B — already effectively pure):**
- `handler.cljs:605` — single `volatile!` created per `run-handler` call
- `handler.cljs:290-354` — 10 handler wrappers each do `vreset!`
- `handler.cljs:566,572,591` — splice `vreset!`/`vswap!`
- `adev.cljs:51,200` — ADEV handler `vreset!` (mirrors handler pattern)
- `verify.cljs:56` — contract checking `vreset!`

All state transitions are pure `(state, addr, dist) -> [value, state']`.
The volatile! is a dispatch wrapper that never escapes the binding block.

**Context injection (Category B — could become explicit coeffects):**
- `*handler*` — active handler function (handler.cljs)
- `*state*` — volatile holding handler state map (handler.cljs)
- `*param-store*` — optional parameter store (handler.cljs)
- `*prng-key*` — optional volatile PRNG key (mlx/random.cljs)
- `*batched-exec?*` — vectorized mode flag (mlx.cljs)

All scoped via `binding`, all unwind automatically.

**GPU boundary (Category C — fundamentally requires mutation):**
- ~40 `mx/eval!` calls across learning, inference, gradients
- ~30 `mx/tidy` calls across inference modules
- `mx/dispose!`, `mx/clear-cache!`, memory limit functions
- `nn/step!` — mutates neural network weights in place
- `seed-from-key!` — seeds MLX's global PRNG
- `train-proposal!` — wraps `nn/step!`

**Other:**
- 1 atom in `dist.cljs:1074` (Wishart sampling counter — trivially replaceable)
- 2 `js/console.warn` calls (development diagnostics)

### What's already pure

- All handler state transitions
- All distributions (sample, log-prob)
- All combinators
- All choicemap/trace/selection operations
- The `gen` macro and model body execution
- Score/gradient computation (builds lazy graphs)

---

## Proposed Changes

### Change 1: Single mutation point for handler dispatch

**Effort:** Small — **Purity gain:** Modest — **Priority:** High (enables everything else)

**Current:** 10 handler wrapper functions each do `vreset!` directly.

```clojure
;; handler.cljs — scattered across 10 functions
(defn simulate-handler [addr dist]
  (let [[value state'] (simulate-transition @*state* addr dist)]
    (vreset! *state* state')
    value))

(defn generate-handler [addr dist]
  (let [[value state'] (generate-transition @*state* addr dist)]
    (vreset! *state* state')
    value))
;; ... 8 more
```

**Proposed:** Single dispatch function, analogous to re-frame's `do-fx`:

```clojure
(defn execute-transition! [addr dist]
  (let [transition-fn @*handler*
        [value state'] (transition-fn @*state* addr dist)]
    (vreset! *state* state')   ;; THE ONLY vreset! in the system
    value))
```

All 10 handler wrappers collapse into one. The `*handler*` dynamic var already
holds the transition function — we just need to use it directly instead of wrapping
each handler individually.

**Benefits:**
- Single place to add cross-cutting concerns (contract validation, logging, undo)
- Easier to reason about mutation (one site instead of 10+)
- `trace-choice!` simplifies to calling `execute-transition!` directly

**Files changed:** `handler.cljs`

---

### Change 2: Explicit coeffect injection

**Effort:** Small-Medium — **Purity gain:** Medium — **Priority:** Medium

**Current:** External resources are injected via dynamic variable `binding`:

```clojure
;; learning.cljs
(binding [handler/*param-store* store
          rng/*prng-key* (volatile! key)]
  (p/simulate model args))
```

**Proposed:** Coeffects as an explicit data map passed through the GFI:

```clojure
;; Option A: coeffects map as optional last argument
(p/simulate model args {:param-store store :prng-key key})

;; Option B: coeffects injected into handler init-state
(defn run-handler [transition-fn body-fn & {:keys [param-store prng-key batched?]}]
  (let [init-state {:choices cm/EMPTY
                    :score   (mx/scalar 0)
                    :weight  (mx/scalar 0)
                    :key     (or prng-key (rng/fresh-key))
                    :param-store param-store}]
    ...))
```

**Which dynamic vars can become coeffects:**

| Var | Can become coeffect? | Notes |
|-----|---------------------|-------|
| `*param-store*` | Yes | Read-only from handler's perspective |
| `*prng-key*` | Yes | Could be a key in init-state |
| `*batched-exec?*` | Yes | Boolean flag, easily passed as data |
| `*handler*` | No | Structural necessity — dispatch mid-execution |
| `*state*` | No | Structural necessity — volatile for `dyn/trace` |

This reduces dynamic vars from 5 to 2 (the two that are structurally required
by the imperative `dyn/trace` mid-execution dispatch pattern).

**Benefits:**
- Testing becomes trivial: pass coeffects map with fixed PRNG, mock params
- No `binding` gymnastics in test code
- Dependencies become explicit in function signatures
- Matches re-frame's `inject-cofx` pattern exactly

**Files changed:** `handler.cljs`, `dynamic.cljs`, `learning.cljs`,
`inference/adev.cljs`

---

### Change 3: Effects-as-data for training

**Effort:** Medium — **Purity gain:** High — **Priority:** High

**Current:** `nn/step!` mutates weights in place, `mx/eval!` and memory management
are imperative:

```clojure
;; nn.cljs — imperative training
(defn step! [module optim vg-fn & inputs]
  (let [[loss grads] (apply vg-fn inputs)]
    (.update optim module grads)          ;; mutation: weight update
    (mx/eval! loss)                       ;; mutation: GPU sync
    loss))

;; learning.cljs — imperative training loop
(loop [params init-params, state adam-state, i 0]
  (let [[loss grad] (mx/tidy #(loss-grad-fn params))
        [new-params new-state] (adam-step params grad state opts)]
    (mx/eval! loss grad)                  ;; mutation: GPU sync
    (when callback (callback i new-params (mx/item loss)))
    (recur new-params new-state (inc i))))
```

**Proposed:** Training step as pure function returning effects map:

```clojure
;; Pure training step — no mutation
(defn training-step [state batch]
  (let [{:keys [params adam-state step]} state
        [loss grad] (compute-loss-and-grad params batch)
        [new-params new-adam] (adam-step params grad adam-state)]
    ;; Return effects description as data
    {:params     new-params
     :adam-state new-adam
     :step       (inc step)
     :effects    {:eval   [loss new-params]                ;; GPU sync
                  :gc     (when (zero? (mod step 100))
                            :clear-cache)                  ;; memory mgmt
                  :report {:iteration step :loss loss}}})) ;; logging

;; Effect executor — the ONLY impure code
(defn execute-training-effects! [effects]
  (when-let [arrays (:eval effects)]
    (apply mx/eval! arrays))
  (when (= :clear-cache (:gc effects))
    (mx/clear-cache!))
  (when-let [r (:report effects)]
    (when callback (callback (:iteration r) (:loss r)))))

;; Training loop becomes a pure reduce
(defn train [opts loss-grad-fn init-params]
  (reduce
    (fn [state i]
      (let [result (training-step state i)
            effects (:effects result)]
        (execute-training-effects! effects)
        (dissoc result :effects)))
    {:params init-params :adam-state (adam-init init-params) :step 0}
    (range (:iterations opts))))
```

**The same pattern applies to `nn/step!`:**

```clojure
;; Pure: returns effect description
(defn nn-training-step [weights optimizer-state inputs loss-fn]
  (let [[loss grads] (loss-fn weights inputs)
        new-weights (apply-gradients weights grads optimizer-state)]
    {:weights         new-weights
     :optimizer-state (update-optimizer optimizer-state grads)
     :effects         {:apply-weights new-weights
                       :eval          [loss new-weights]}}))

;; Impure executor
(defn execute-nn-effects! [module optimizer effects]
  (when-let [w (:apply-weights effects)]
    (.update optimizer module w))
  (when-let [a (:eval effects)]
    (apply mx/eval! a)))
```

**Benefits:**
- Training loops testable without a GPU — just check the effects map
- Effect descriptions are data: serializable, inspectable, replayable
- Memory management becomes declarative, not scattered
- Same pattern works for `train-proposal!` in `amortized.cljs`

**Files changed:** `nn.cljs`, `learning.cljs`, `inference/amortized.cljs`

---

### Change 4: Effects-as-data for inference memory management

**Effort:** Medium — **Purity gain:** High — **Priority:** High

**Current:** `mx/tidy`, `mx/eval!`, `dispose-trace!`, and `force-gc!` are called
imperatively throughout inference loops:

```clojure
;; mcmc.cljs — scattered GPU/memory ops
(loop [trace init-trace, samples [], i 0]
  (let [new-trace (mx/tidy                          ;; memory mgmt
                    #(let [{:keys [trace]} (mh-step trace selection)]
                       (mx/eval! (:score trace))    ;; GPU sync
                       trace))]
    (when (> i burn)
      (dispose-trace! trace))                        ;; memory mgmt
    (recur new-trace (conj samples new-trace) (inc i))))
```

**Proposed:** Inference steps return effects alongside results:

```clojure
;; Pure inference step — returns result + effects
(defn mh-step-pure [trace selection key]
  (let [{:keys [trace weight]} (regenerate trace selection key)
        accept? (< (log-uniform key) (min 0 weight))]
    {:trace   (if accept? trace trace)
     :effects {:eval    [(:score trace)]           ;; GPU sync
               :dispose (when accept? [trace])     ;; old trace
               :tidy    true}}))                   ;; wrap in mx/tidy

;; Generic inference loop with effect executor
(defn run-inference [step-fn init-state opts]
  (reduce
    (fn [state i]
      (let [result (step-fn (:trace state) (:key state))
            effects (:effects result)]
        (execute-inference-effects! effects)
        (-> state
            (assoc :trace (:trace result))
            (update :samples conj-if-past-burn (:trace result) i opts))))
    init-state
    (range (:samples opts))))

;; Effect executor for inference
(defn execute-inference-effects! [effects]
  (when (:tidy effects)
    ;; batch eval inside tidy for memory safety
    (mx/tidy #(when-let [a (:eval effects)] (apply mx/eval! a))))
  (when-let [traces (:dispose effects)]
    (doseq [t traces] (dispose-trace! t)))
  (when (:gc effects)
    (force-gc!)))
```

**Benefits:**
- Inference steps become pure and testable
- Memory management strategy is declarative and swappable
- Easy to add different memory strategies (aggressive vs lazy disposal)
- The ~30 `mx/tidy` and ~40 `mx/eval!` calls collapse into a few executor functions

**Files changed:** `inference/mcmc.cljs`, `inference/smc.cljs`, `inference/vi.cljs`,
`inference/adev.cljs`, `inference/util.cljs`

---

### Change 5: Interceptor chain for composable inference

**Effort:** Large — **Purity gain:** Very high — **Priority:** Medium (ambitious but transformative)

**Current:** Inference algorithms are monolithic functions that mix proposal logic,
scoring, acceptance, GPU sync, and diagnostics:

```clojure
;; mcmc.cljs — monolithic MH
(defn mh [opts model args observations]
  (let [init-trace (init-trace model args observations)]
    (loop [trace init-trace, samples [], i 0]
      ;; proposal, scoring, acceptance, eval!, tidy, dispose
      ;; all interleaved in one function
      ...)))
```

**Proposed:** Inference operations as interceptor chains:

```clojure
;; Interceptor: a data map with :before and :after
(def propose-interceptor
  {:id     :propose
   :before (fn [ctx]
             (let [{:keys [trace selection key]} (:coeffects ctx)
                   result (p/regenerate (:gen-fn trace) trace selection)]
               (-> ctx
                   (assoc-in [:effects :proposed] (:trace result))
                   (assoc-in [:effects :log-alpha] (:weight result)))))})

(def accept-reject-interceptor
  {:id     :accept-reject
   :before (fn [ctx]
             (let [log-alpha (get-in ctx [:effects :log-alpha])
                   key       (get-in ctx [:coeffects :key])
                   accept?   (accept-mh? log-alpha key)]
               (assoc-in ctx [:effects :accepted?] accept?)))})

(def gpu-sync-interceptor
  {:id    :gpu-sync
   :after (fn [ctx]
            (let [trace (get-in ctx [:effects :trace])]
              (mx/eval! (:score trace) (:choices trace))
              ctx))})

(def diagnostics-interceptor
  {:id    :diagnostics
   :after (fn [ctx]
            (let [accepted? (get-in ctx [:effects :accepted?])]
              (update-in ctx [:effects :diagnostics :accept-rate]
                         running-mean accepted?)))})

;; Compose into a chain
(def mh-chain
  [propose-interceptor
   accept-reject-interceptor
   gpu-sync-interceptor
   diagnostics-interceptor])

;; Execute: walk :before functions forward, :after functions backward
(defn run-chain [chain coeffects]
  (let [init-ctx {:coeffects coeffects :effects {} :queue chain :stack []}]
    (reduce (fn [ctx interceptor]
              (if-let [before (:before interceptor)]
                (before ctx)
                ctx))
            init-ctx
            chain)))
```

**Users compose custom inference by combining interceptors:**

```clojure
;; Standard MH
(def my-mh [propose accept-reject gpu-sync])

;; MH with gradient diagnostics
(def my-mh-debug [propose accept-reject gradient-check gpu-sync diagnostics])

;; MH with adaptive step size
(def my-adaptive-mh [propose accept-reject adapt-step-size gpu-sync])

;; HMC: just swap the proposal interceptor
(def my-hmc [leapfrog-propose accept-reject gpu-sync])
```

**Benefits:**
- Composable inference middleware — add tracing, undo, convergence checks as interceptors
- Each interceptor independently testable as `context -> context'`
- Cross-cutting concerns (logging, diagnostics, memory) orthogonal to algorithm logic
- Users can build custom inference by assembling interceptors
- Matches re-frame's pattern exactly

**Files changed:** New `interceptor.cljs` module, refactor of `inference/mcmc.cljs`,
`inference/smc.cljs`, etc.

---

### Change 6: Flows for derived inference quantities

**Effort:** Medium-Large — **Purity gain:** High — **Priority:** Low (nice-to-have)

**Current:** Derived quantities (ESS, log-ML, acceptance rate) are computed manually
at specific points in inference loops.

**Proposed:** Declarative flows that auto-maintain derived values:

```clojure
(def inference-flows
  [{:id     :ess
    :inputs {:weights [:particles :log-weights]}
    :output (fn [{:keys [weights]}] (compute-ess weights))
    :path   [:diagnostics :ess]}

   {:id     :should-resample?
    :inputs {:ess (flow<- :ess)
             :n   [:particles :count]}
    :output (fn [{:keys [ess n]}] (< ess (* 0.5 n)))
    :path   [:diagnostics :should-resample?]}

   {:id     :log-ml-estimate
    :inputs {:weights [:particles :log-weights]}
    :output (fn [{:keys [weights n]}]
              (mx/subtract (mx/logsumexp weights)
                           (mx/log (mx/scalar n))))
    :path   [:diagnostics :log-ml]}

   {:id     :convergence
    :inputs {:samples [:chain :samples]}
    :output (fn [{:keys [samples]}]
              {:r-hat (r-hat samples)
               :ess   (ess-from-autocorrelation samples)})
    :path   [:diagnostics :convergence]
    :live?  (fn [state] (:compute-diagnostics? state))}])
```

When particle weights change, ESS auto-recomputes. When ESS changes, the resample
decision updates. The `:live?` predicate activates expensive diagnostics only when
requested. Change propagation stops when outputs haven't changed (deduplication via
ClojureScript `=`).

**Benefits:**
- No manual bookkeeping for derived quantities
- Dependency graph handles change propagation
- Deduplication prevents unnecessary recomputation
- `:live?` controls expensive diagnostics

**Files changed:** New `flows.cljs` module, integration with inference loops

---

### Change 7: Remove the Wishart atom

**Effort:** Tiny — **Purity gain:** Tiny — **Priority:** Low (completionism)

**Current:** `dist.cljs:1074` uses an atom as a counter to index into pre-split keys:

```clojure
(let [ki (atom 0)
      next-key! (fn [] (let [i @ki] (swap! ki inc) (nth keys i)))]
  ...)
```

**Proposed:** Replace with `loop` + index threading or `reduce-kv`:

```clojure
(loop [i 0, result initial]
  (if (>= i n)
    result
    (let [k (nth keys i)]
      (recur (inc i) (update result ...)))))
```

**Files changed:** `dist.cljs`

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PURE WORLD                               │
│                                                             │
│  gen bodies ──────── pure: state -> computation graph       │
│  transitions ─────── pure: (state, addr, dist) -> [v, s']  │
│  training steps ──── pure: (weights, batch) -> effects-map  │
│  inference steps ─── pure: interceptor chain ctx -> ctx'    │
│  flows ───────────── pure: derived quantities, auto-maint.  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                 COEFFECT BOUNDARY                           │
│                                                             │
│  inject :prng-key ── provide deterministic PRNG key         │
│  inject :params ──── provide parameter store                │
│  inject :trace ───── provide current trace                  │
│  inject :gpu ─────── provide device info / memory state     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                  EFFECT BOUNDARY                            │
│                                                             │
│  :state-update ───── single vreset! point                   │
│  :eval ───────────── mx/eval! (GPU commit)                  │
│  :dispose ────────── mx/dispose! (free buffers)             │
│  :weights ────────── nn module weight update                │
│  :gc ─────────────── mx/clear-cache!, force-gc!             │
│  :seed ───────────── seed-from-key!                         │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                GPU / METAL (the "DOM")                      │
│                                                             │
│  MLX computation graph execution                            │
│  Metal command buffer submission                            │
│  Buffer allocation / deallocation                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Order

| Phase | Changes | Effort | Impact |
|-------|---------|--------|--------|
| **Phase 1** | Single `vreset!` point (Change 1) | 1 day | Enables all subsequent changes |
| **Phase 2** | Effects-as-data for training (Change 3) | 2-3 days | Testable training loops |
| **Phase 3** | Effects-as-data for inference memory (Change 4) | 3-4 days | Cleaner inference, fewer scattered `eval!`/`tidy` |
| **Phase 4** | Explicit coeffects (Change 2) | 1-2 days | Easier testing, fewer dynamic vars |
| **Phase 5** | Interceptor chain (Change 5) | 5-7 days | Composable inference middleware |
| **Phase 6** | Flows (Change 6) | 3-5 days | Auto-maintained diagnostics |
| **Phase 7** | Remove Wishart atom (Change 7) | 15 min | Completionism |

**Total estimated effort:** 15-22 days

**Phases 1-4 are the high-value, moderate-effort core.** They eliminate scattered
mutation from training and inference loops while keeping the user-facing API stable.

**Phase 5 (interceptors) is the most transformative** but requires careful API design
to ensure the interceptor abstraction is natural for probabilistic programming, not
just a re-frame port.

**Phase 6 (flows) is a nice-to-have** that becomes natural once interceptors exist —
flows can run as an `:after` interceptor.

---

## What Cannot Be Made Pure

Two dynamic vars are structurally required and cannot become coeffects:

1. **`*handler*`** — `dyn/trace` is called mid-execution inside `gen` bodies.
   The handler must be available via dynamic scope because user code doesn't
   (and shouldn't) thread it explicitly. This is the fundamental tension:
   re-frame handlers fire-and-forget, but `dyn/trace` must return a value for
   the model body to continue.

2. **`*state*`** — Same reason. The volatile holds the handler state that
   `dyn/trace` reads and writes during imperative model execution.

These are GenMLX's equivalent of re-frame's `app-db` atom — the minimal mutable
shell that the pure functional core lives inside. They exist because ClojureScript
lacks algebraic effects or monadic do-notation that would allow implicit state
threading through imperative code.

GPU operations (`mx/eval!`, `mx/dispose!`) are fundamentally stateful — they mutate
Metal buffer state. This is GenMLX's DOM: the external world that the framework
manages through a layer of indirection. The goal is not to eliminate these mutations
but to **push them to the boundary** so all code above the effect executor is pure.

---

## Relation to Other Papers

This work connects to several planned publications:

- **System paper** (`plan-system.md`): The purity architecture is a key
  differentiator vs Gen.jl (Julia, imperative) and GenJAX (JAX, functional but
  vmap-heavy). A section on "effects-as-data for GPU-accelerated inference" would
  strengthen the systems contribution.

- **Formal paper** (`plan-formal.md`): The interceptor chain formalizes inference
  as composable pure transformations on a context map, which connects to the
  measure-theoretic GFI contracts.

- **Vectorization paper** (`plan-vectorization.md`): The effects boundary naturally
  supports batched effect execution — multiple `eval!` calls can be fused into one
  at the executor level.

---

## References

- [re-frame documentation](https://day8.github.io/re-frame/re-frame/)
- [re-frame: A Data Loop](https://github.com/day8/re-frame/blob/master/docs/a-loop.md)
- [re-frame: Effectful Handlers](https://day8.github.io/re-frame/EffectfulHandlers/)
- [re-frame: Coeffects](https://day8.github.io/re-frame/Coeffects/)
- [re-frame: Interceptors](https://github.com/day8/re-frame/blob/master/docs/Interceptors.md)
- [re-frame: Flows](https://github.com/day8/re-frame/blob/master/docs/Flows.md)
- [re-frame API Reference](https://day8.github.io/re-frame/api-re-frame.core/)
