# Traces

Traces are immutable records of generative function execution. Every GFI operation that runs a model produces a trace capturing the random choices made, their joint log-probability, the return value, and a reference back to the model and arguments. Traces are the primary data structure flowing through inference algorithms.

Source: `src/genmlx/trace.cljs`, `src/genmlx/diff.cljs`

## Trace Record

A `Trace` is a ClojureScript record with five fields:

```clojure
(defrecord Trace [gen-fn args choices retval score])
```

| Field | Type | Description |
|-------|------|-------------|
| `gen-fn` | Generative function | The model that produced this trace |
| `args` | Vector | Arguments passed to the model |
| `choices` | [ChoiceMap](choicemap.md) | All random choices made during execution |
| `retval` | Any | Return value of the model body |
| `score` | MLX scalar | Log-joint probability: log p(choices \| args) |

All field values are accessible via keyword lookup on the trace record.

## Creating Traces

### `make-trace`

```clojure
(make-trace {:gen-fn gf :args args :choices cm :retval rv :score s})
```

Create a `Trace` from a map. All five fields are required.

| Parameter | Type | Description |
|-----------|------|-------------|
| `m` | Map | Map with keys `:gen-fn`, `:args`, `:choices`, `:retval`, `:score` |

**Returns:** `Trace` record.

In practice, you rarely call `make-trace` directly. Traces are produced by GFI operations:

```clojure
(require '[genmlx.protocols :as p]
         '[genmlx.choicemap :as cm])

;; simulate: unconstrained forward sampling
(def trace (p/simulate model [x]))

;; generate: constrained execution, returns {:trace :weight}
(def result (p/generate model [x] (cm/choicemap :y 2.5)))
(def trace (:trace result))

;; update: modify an existing trace, returns {:trace :weight :discard}
(def updated (p/update model trace (cm/choicemap :slope 1.0)))
(def new-trace (:trace updated))

;; regenerate: resample selected addresses, returns {:trace :weight}
(def regen (p/regenerate model trace (sel/select :slope)))
(def new-trace (:trace regen))
```

## Accessing Trace Data

Trace fields are accessed directly via keyword lookup:

```clojure
(def trace (p/simulate model [x]))

;; Direct field access
(:retval trace)    ;; return value of the model
(:score trace)     ;; log p(choices | args) as MLX scalar
(:choices trace)   ;; ChoiceMap of all random choices
(:gen-fn trace)    ;; the generative function that produced this trace
(:args trace)      ;; the arguments vector, e.g. [x]

;; Access a specific choice value
(cm/get-value (cm/get-submap (:choices trace) :slope))

;; Or use get-choice with a path
(cm/get-choice (:choices trace) [:slope])
```

The score is an MLX scalar (stays on GPU). Use `mx/item` only at inference boundaries to extract a JavaScript number:

```clojure
(mx/item (:score trace))  ;; => -4.237 (JS number)
```

## Trace Metadata

Traces carry Clojure metadata (via `with-meta`) for internal bookkeeping by combinators and the dynamic execution engine. This metadata is not part of the trace record itself but is preserved across GFI operations.

### `::splice-scores`

Attached by `DynamicGF` when the model body contains `splice` calls (calls to sub-generative-functions). Maps splice addresses to their individual log-probability contributions.

```clojure
;; Produced internally by DynamicGF simulate/generate/update
(::genmlx.dynamic/splice-scores (meta trace))
;; => {:sub-model -3.21, :other-model -1.05}
```

Used by `update` and `regenerate` to correctly recompute scores when sub-model choices change.

### `::element-scores`

Attached by the `MapCombinator` and `VmapGF` to cache per-element log-probabilities. This is a vector of MLX scalars, one per element in the mapped collection.

```clojure
;; After simulating a Map combinator
(::genmlx.combinators/element-scores (meta trace))
;; => [<mx scalar> <mx scalar> <mx scalar> ...]
```

Used by `update-with-diffs` to skip recomputation of unchanged elements, and by `regenerate` to correctly compute proposal weights.

### `::step-scores`

Attached by the `UnfoldCombinator` and `ScanCombinator` to cache per-step log-probabilities across the unfolded sequence.

```clojure
(::genmlx.combinators/step-scores (meta trace))
;; => [<mx scalar> <mx scalar> ...]
```

### `::compiled-path`

Boolean flag indicating that this trace was produced via a compiled execution path rather than the handler path. Both paths produce identical trace data; this flag is used internally for optimization decisions.

```clojure
(::genmlx.combinators/compiled-path (meta trace))
;; => true
```

## Diffs

Diffs represent incremental changes to arguments or data structures. They are used by `update-with-diffs` and combinators to skip unchanged sub-computations -- critical for MCMC performance where each step typically changes only one or a few addresses.

Source: `src/genmlx/diff.cljs`

### `no-change`

```clojure
diff/no-change
```

Constant indicating a value has not changed. This is a map `{:diff-type :no-change}`.

**Type:** Map

### `no-change?`

```clojure
(no-change? d)
```

Returns true if the diff indicates no change.

| Parameter | Type | Description |
|-----------|------|-------------|
| `d` | Map | A diff value |

**Returns:** Boolean.

### `changed?`

```clojure
(changed? d)
```

Returns true if the diff indicates any change (i.e., not `no-change`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `d` | Map | A diff value |

**Returns:** Boolean.

### Diff types in practice

Combinators use diff-type keywords internally to express structured changes:

| Diff type | Description | Used by |
|-----------|-------------|---------|
| `:no-change` | Value has not changed | All combinators |
| `:vector-diff` | Changes at specific vector indices | `MapCombinator` `update-with-diffs` |

To trigger diff-aware updates in combinators, use `update-with-diffs`:

```clojure
(require '[genmlx.protocols :as p]
         '[genmlx.diff :as diff])

;; Tell the Map combinator that arguments haven't changed
(p/update-with-diffs map-model trace new-constraints diff/no-change)

;; Unchanged elements are skipped entirely -- only constrained
;; elements are re-executed, using cached ::element-scores
```
