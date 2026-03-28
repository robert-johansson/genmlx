# Malli Integration in GenMLX

*Malli serves as the structural type system for GenMLX's data boundaries. It sits at the seams between components, making implicit contracts explicit, composable, and checkable. This document specifies where Malli schemas are defined, what they validate, and how they integrate with the ARCHITECTURE.md refactoring.*

---

## Design Principle

Malli validates **shapes at component boundaries**, not values in the hot path. The handler transitions (`handler.cljs`) are pure functions called hundreds of thousands of times per inference step. Malli never runs inside them. Instead, Malli validates the data that flows *between* components: the initial state entering `run-handler`, the sub-result entering `merge-sub-result`, the transition-spec returned by a dispatcher, the return value leaving a GFI method.

Validation is gated by a development flag:

```clojure
(def ^:dynamic *validate?* false)  ;; true during dev/test, false in production

(defn validated
  "Validate value against schema in dev mode. No-op in production."
  [schema value context]
  (when *validate?*
    (when-not (m/validate schema value)
      (throw (ex-info (str "Schema violation: " context)
                      {:errors (me/humanize (m/explain schema value))
                       :context context})))))
```

At 39ns per compiled validation (Malli benchmark), the overhead is negligible during development. In production inference, `*validate?*` is false and the check is a single boolean test.

---

## 1. Handler State Schemas

Lines 15-28 of `handler.cljs` document the handler state schemas as a markdown table in a docstring. This table is the most important structural contract in GenMLX -- every transition reads from and writes to these keys, every `run-*` helper constructs an initial state with these keys, and `merge-sub-result` assumes this shape. Today, misspelling `:constraints` as `:constraint` produces a silent nil lookup and wrong importance weights.

Malli makes the table a value. Each mode is a separate `def`, composed via `mu/merge`:

```clojure
(def BaseState
  "Keys common to all handler modes."
  [:map
   [:key some?]
   [:choices [:fn {:error/message "should be a ChoiceMap"} choicemap?]]
   [:score some?]
   [:executor {:optional true} fn?]])

(def SimulateState  BaseState)
(def GenerateState  (mu/merge BaseState [:map [:weight some?] [:constraints ...]]))
(def AssessState    GenerateState)
(def UpdateState    (mu/merge GenerateState [:map [:old-choices ...] [:discard ...]]))
(def RegenerateState (mu/merge BaseState [:map [:weight some?] [:old-choices ...] [:selection some?]]))
(def ProjectState   (mu/merge BaseState [:map [:weight some?] [:old-choices ...] [:selection some?] [:constraints ...]]))
```

The `mu/merge` composition mirrors the code: generate-state is simulate-state plus `:weight` and `:constraints`. Update-state is generate-state plus `:old-choices` and `:discard`. The inheritance is data composition, not class hierarchy.

**Validation point:** Entry to `run-handler` in `runtime.cljs` validates against `BaseState` (the common subset). Mode-specific validation is not yet implemented — the dispatcher knows the mode but does not currently pass it to `run-handler`.


## 2. Sub-Result Shape

`merge-sub-result` (`handler.cljs:260`) merges a sub-generative-function's result into the parent state. It is the most fragile junction in GenMLX. Different execution paths produce slightly different shapes: ExactGF returns `{:choices cm/EMPTY :score log-ml :retval probs}`, standard `execute-sub` returns `{:choices ... :score ... :retval ... :weight ...}`, compiled paths return `{:values ... :score ... :retval ...}`.

If a new domain integration returns a map missing `:score`, `merge-sub-result` calls `(mx/add sc nil)` and crashes deep in MLX with an opaque error.

```clojure
(def SubResult
  "Shape of sub-generative-function results for merge-sub-result."
  [:map
   [:choices [:fn {:error/message "should be a ChoiceMap"} choicemap?]]
   [:score some?]
   [:retval {:optional true} any?]
   [:weight {:optional true} any?]
   [:discard {:optional true} [:or nil? [:fn {:error/message "should be a ChoiceMap"} choicemap?]]]
   [:splice-scores {:optional true} map?]])
```

**Validation point:** Entry to `merge-sub-result`, before any state manipulation.


## 3. GFI Return Contracts

ClojureScript protocols define method signatures but not return types. The actual return shapes are constructed by helpers in `dynamic.cljs` (`make-generate-result`, `make-update-result`, `make-regen-result`). These shapes are the GFI's external contract -- what callers depend on.

```clojure
(def Trace
  "A valid GenMLX trace."
  [:fn #(instance? tr/Trace %)])

(def SimulateReturn Trace)

(def GenerateReturn
  [:map
   [:trace Trace]
   [:weight some?]
   [:unused-constraints {:optional true} set?]])

(def UpdateReturn
  [:map
   [:trace Trace]
   [:weight some?]
   [:discard [:fn cm/choicemap?]]
   [:unused-constraints {:optional true} set?]])

(def RegenerateReturn
  [:map
   [:trace Trace]
   [:weight some?]])

(def AssessReturn
  [:map
   [:retval some?]
   [:weight some?]])

(def ProposeReturn
  [:map
   [:choices [:fn cm/choicemap?]]
   [:weight some?]
   [:retval some?]])

(def ProjectReturn some?)  ;; scalar
```

**Validation point:** Exit from each GFI method in `DynamicGF`, after `mx/gfi-cleanup!`.


## 4. Dispatcher Transition-Spec

The ARCHITECTURE.md refactoring introduces transition-specs returned by dispatchers. This is a new data shape. Defining it in Malli from the start means it is born validated:

```clojure
(def ScoreType
  "Score encoding declared by transitions."
  [:enum :joint :marginal :collapsed :beam-marginal])

(def TransitionSpec
  "Shape returned by IDispatcher/resolve-transition.
   Note: the actual key is :run, not :transition — the :run function
   encapsulates the full execution path for a given GFI operation."
  [:map
   [:run fn?]
   [:score-type ScoreType]])
```

**Validation point:** Return value of `dispatch/resolve`, before `execute-spec` processes it.


## 5. Model Schema Output

`schema.cljs` extracts structural metadata from gen body source forms at construction time. The resulting schema map drives Level 1 compilation, Level 3 conjugacy detection, and Level 4 method selection. Its shape is currently documented only by reading the code:

```clojure
(def TraceSite
  [:map
   [:addr some?]
   [:addr-form some?]
   [:dist-type keyword?]
   [:dist-args vector?]
   [:deps set?]
   [:static? boolean?]])

(def SpliceSite
  [:map
   [:addr some?]
   [:gf-sym some?]
   [:deps set?]])

(def ModelSchema
  "Output of schema/extract-schema, augmented by conjugacy and compilation."
  [:map
   [:trace-sites [:vector TraceSite]]
   [:static? boolean?]
   [:dynamic-addresses? boolean?]
   [:has-branches? boolean?]
   ;; Optional keys added by various augmentation passes
   [:splice-sites {:optional true} vector?]
   [:param-sites {:optional true} vector?]
   [:loop-sites {:optional true} vector?]
   [:dep-order {:optional true} vector?]
   [:return-form {:optional true} some?]
   [:has-loops? {:optional true} boolean?]
   [:params {:optional true} some?]
   ;; Added by conjugacy detection
   [:conjugate-pairs {:optional true} [:or vector? map?]]
   [:has-conjugate? {:optional true} boolean?]
   [:analytical-plan {:optional true} some?]
   ;; Added by compilation
   [:compiled-simulate {:optional true} fn?]
   [:compiled-generate {:optional true} fn?]
   [:compiled-update {:optional true} fn?]
   [:compiled-regenerate {:optional true} fn?]
   [:compiled-assess {:optional true} fn?]
   [:compiled-project {:optional true} fn?]
   ;; Added by auto-analytical
   [:auto-handlers {:optional true} map?]
   [:auto-regenerate-transition {:optional true} fn?]
   [:auto-regenerate-handlers {:optional true} map?]])
```

This makes the schema system's output self-documenting. When a new compilation level or domain integration reads the schema, the Malli definition tells it exactly what fields exist and what types they carry.

**Validation point:** Output of `schema/extract-schema` in `dynamic.cljs`'s `make-gen-fn`.


## 6. Conjugacy Table Entries

The conjugacy table in `conjugacy.cljs` is already a plain Clojure map. Its values have a specific structure that varies by conjugate family:

```clojure
(def ConjugacyEntry
  "One entry in the conjugacy table. All families have :family and
   :natural-param-idx. Additional keys are family-specific param accessors
   (e.g., :prior-mean-key, :obs-noise-key) — validated as a plain map
   rather than per-family :multi dispatch, since new families are added
   infrequently and the additional keys are all keywords."
  [:map
   [:family keyword?]
   [:natural-param-idx int?]])
```

The schema validates the common structure. Family-specific keys (`:prior-mean-key`, `:obs-noise-key`, etc.) are open map entries — Malli's open map semantics allow them without listing each one.

**Validation point:** Not currently implemented at runtime. The schema is available for development-time REPL checks.


## 7. Domain Extension via mu/merge

This is where Malli's data-driven composition pays off for the ARCHITECTURE.md domain integration pattern. When the LLM integration, grammar engine, or any future domain adds state keys, it extends the base schema without modifying it:

```clojure
;; LLM domain: adds KV cache and token context
(def LLMGenerateState
  (mu/merge GenerateState
    [:map
     [:kv-cache some?]
     [:context vector?]]))

;; Beam domain: adds beam hypotheses for byte-level marginalization
(def BeamGenerateState
  (mu/merge GenerateState
    [:map
     [:beam-states vector?]
     [:beam-logZ number?]]))

;; Grammar domain: adds parser state for constrained decoding
(def GrammarGenerateState
  (mu/merge GenerateState
    [:map
     [:parser-chart some?]
     [:grammar some?]]))
```

Each domain's state schema is a data composition of the base schema with domain-specific extensions:

- Base validation still works (`:key`, `:choices`, `:score` are always checked)
- Domain-specific keys are validated when the domain's schema is used
- New domains compose without touching existing schemas
- The schemas document what state each domain's transition expects

This is the same pattern as `mu/merge` for web request schemas in Reitit/Ring -- extend the base with route-specific keys. The difference is that GenMLX's "routes" are handler modes and domain integrations.


## 8. Distribution Records

The `Distribution` record in `dist/core.cljs` has `{:type keyword :params map}`. The `:params` shape varies by distribution type. Malli's `:multi` dispatch validates per-type:

```clojure
(def DistRecord
  "A GenMLX Distribution record: {:type keyword :params map}."
  [:map
   [:type keyword?]
   [:params map?]])
```

The schema validates the common structure of all distributions. Per-type param validation (e.g., gaussian requires `:mu` and `:sigma`) was considered but skipped — it would break when new distributions are added via `defdist`, and the structural check is too basic to catch real param errors at the constructor level.

**Validation point:** Not currently implemented at constructors. Available for development-time REPL checks.


## 9. What Malli Does NOT Replace

**`contracts.cljs`** — Malli validates shapes. Contracts verify measure-theoretic properties: "simulate score equals assess weight on the same choices", "update weight equals log density ratio", "project all equals score". These are numerical invariants, not structural ones. Both are needed; they operate at different levels.

**`verify.cljs`** — Static source analysis: detecting `mx/eval!` calls in model bodies, duplicate trace addresses, non-finite scores. Malli validates runtime data shapes, not source form properties.

**`schema.cljs`** — Source form analysis at macro time. `schema.cljs` walks quoted ClojureScript to extract model structure. Malli validates the *output* of schema extraction (the `ModelSchema` defined above), not the extraction process itself.


## 10. Validation Points Summary

| Component | Schema | When | Catches |
|---|---|---|---|
| `run-handler` entry | `BaseState` | Init state construction | **Implemented.** Missing `:key`, `:choices`, or `:score` |
| `merge-sub-result` entry | `SubResult` | Each splice | **Implemented.** Missing `:choices` or `:score` from sub-models |
| GFI method exit | `GenerateReturn` etc. | Each GFI call | **Implemented.** All 7 methods validated (simulate, generate, update, regenerate, assess, propose, project) |
| Dispatcher resolution | `TransitionSpec` | Each dispatch | *Not yet implemented.* Schema defined but not wired into `run-dispatched`. |
| Schema extraction output | `ModelSchema` | Gen-fn construction | *Not yet implemented.* Schema defined but not wired into `make-gen-fn`. |
| Distribution construction | `DistRecord` | Each constructor | *Not yet implemented.* Schema defined for REPL use only. |
| Conjugacy table | `ConjugacyEntry` | Static, one-time | *Not yet implemented.* Schema defined for REPL use only. |
| Domain state extensions | `mu/merge` compositions | Domain-specific init | *Mechanism verified.* Real schemas created when LLM integration is built. |

Implemented validation points are gated behind `*validate?*` and have zero cost in production. The remaining schemas are defined and available for development-time REPL validation but not wired into automatic runtime checks.


## 11. File Placement

The Malli schemas live in a single namespace that the rest of the codebase can require:

```
src/genmlx/
  schemas.cljs    ;; All Malli schema definitions
                  ;; Requires: malli.core, malli.util, malli.error
                  ;; Requires: genmlx.choicemap, genmlx.trace (for predicates)
                  ;; Required by: handler.cljs (optional), dynamic.cljs, dispatch.cljs
```

Alternatively, schemas can be co-located with their components:

```
src/genmlx/
  handler.cljs         ;; imports schemas for HandlerState validation
  handler/schemas.cljs  ;; HandlerState, SubResult schemas
  dispatch.cljs        ;; imports schemas for TransitionSpec validation
  dispatch/schemas.cljs ;; TransitionSpec, ScoreType schemas
  dynamic.cljs         ;; imports schemas for GFI return validation
  dynamic/schemas.cljs  ;; GenerateReturn, UpdateReturn, etc.
```

The single-file approach is simpler and appropriate for the current codebase size. The co-located approach scales better if GenMLX grows significantly.


## 12. Interaction with the Dispatcher Refactoring

The dispatcher stack from ARCHITECTURE.md and the Malli schemas reinforce each other:

1. **Each dispatcher returns a `TransitionSpec`.** Malli validates this shape, catching malformed specs from new dispatchers during development.

2. **The `TransitionSpec` carries `:score-type`.** This is a Malli `:enum` -- the set of valid score encodings is declared as data and validated, not as a convention documented in comments.

3. **`with-handler` attaches a transition to metadata.** The custom-transition-dispatcher extracts it. Malli can validate that the attached value is a function (not, say, a keyword accidentally passed as the transition argument).

4. **Domain extensions compose via `mu/merge`.** When a new domain adds state keys, the merged schema validates both the base keys (catching regressions in existing code) and the new keys (catching typos in new code). The merge is pure data composition -- no modification to any existing schema.

The Malli integration does not change any algorithms, any protocol signatures, or any data structures. It adds a validation layer at component boundaries that is invisible in production and catches structural bugs during development.
