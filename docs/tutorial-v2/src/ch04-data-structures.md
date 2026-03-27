# Choice Maps and Traces

GenMLX's core data structures are simple: two record types for choice maps, one record type for traces, and a protocol for selections. All are immutable, all are inspectable as plain Clojure data, and all support the operations that inference algorithms need. This chapter covers them in detail.

## Choice maps: Value and Node

A choice map has exactly two types:

- **Value** — a leaf that wraps a single random choice (an MLX array).
- **Node** — a branch that maps addresses (keywords) to sub-choice-maps.

```clojure
;; A leaf
(def v (cm/->Value (mx/scalar 3.0)))
(cm/has-value? v)           ;; => true
(mx/item (cm/get-value v))  ;; => 3.0

;; A branch
(def n (cm/->Node {:x (cm/->Value (mx/scalar 1.0))
                    :y (cm/->Value (mx/scalar 2.0))}))
(cm/has-value? n)                                ;; => false
(cm/has-value? (cm/get-submap n :x))             ;; => true
(mx/item (cm/get-value (cm/get-submap n :x)))    ;; => 1.0
```

The empty choice map `cm/EMPTY` is a `Node` with an empty map. Addresses that don't exist return `EMPTY`:

```clojure
(= cm/EMPTY (cm/get-submap n :z))  ;; => true (no :z address)
```

This two-type design gives you structural sharing, equality, and serialization for free — they're persistent Clojure data structures underneath.

## Building choice maps

The `cm/choicemap` constructor takes keyword-value pairs:

```clojure
(def obs (cm/choicemap :slope (mx/scalar 2.0)
                       :intercept (mx/scalar 1.0)
                       :y0 (mx/scalar 5.0)))
```

For nested structures, you can nest `choicemap` calls or use `cm/from-map`:

```clojure
;; Nested choicemap
(def nested (cm/choicemap :params (cm/choicemap :slope (mx/scalar 2.0)
                                                :intercept (mx/scalar 1.0))))

;; Or from a plain Clojure map
(def from-map (cm/from-map {:params {:slope 2.0 :intercept 1.0}
                             :obs {:y0 5.0}}))
```

To convert back to a plain map for inspection:

```clojure
(cm/to-map obs)
;; => {:slope #mlx[2.0], :intercept #mlx[1.0], :y0 #mlx[5.0]}
```

## Reading choice maps

Three access operations cover all common patterns:

```clojure
(def c (cm/choicemap :params (cm/choicemap :slope (mx/scalar 2.0)
                                            :intercept (mx/scalar 1.0))
                     :y0 (mx/scalar 5.0)))

;; get-submap: returns the sub-choicemap at an address
(def params (cm/get-submap c :params))  ;; => Node{:slope ..., :intercept ...}

;; get-value: extracts the raw value from a Value leaf
(mx/item (cm/get-value (cm/get-submap c :y0)))  ;; => 5.0

;; get-choice: path-based access (combines get-submap + get-value)
(mx/item (cm/get-choice c [:params :slope]))  ;; => 2.0
(mx/item (cm/get-choice c [:y0]))              ;; => 5.0
```

## Modifying choice maps

Choice maps are immutable — modification returns a new choice map:

```clojure
;; set-value: set a single address
(def c1 (cm/set-value cm/EMPTY :x (mx/scalar 7.0)))
(mx/item (cm/get-choice c1 [:x]))  ;; => 7.0

;; set-choice: set at a path (creates intermediate nodes)
(def c2 (cm/set-choice cm/EMPTY [:params :slope] (mx/scalar 3.0)))
(mx/item (cm/get-choice c2 [:params :slope]))  ;; => 3.0

;; merge-cm: values in b override values in a
(def a (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0)))
(def b (cm/choicemap :y (mx/scalar 99.0) :z (mx/scalar 3.0)))
(def merged (cm/merge-cm a b))
(mx/item (cm/get-choice merged [:x]))  ;; => 1.0  (from a)
(mx/item (cm/get-choice merged [:y]))  ;; => 99.0 (b overrides a)
(mx/item (cm/get-choice merged [:z]))  ;; => 3.0  (from b)
```

## Enumerating addresses

`cm/addresses` returns all leaf address paths as a vector of vectors:

```clojure
(def c (cm/choicemap :slope (mx/scalar 2.0)
                     :intercept (mx/scalar 1.0)
                     :y0 (mx/scalar 5.0)))
(cm/addresses c)
;; => [[:slope] [:intercept] [:y0]]

;; Nested addresses include the full path
(def nested (cm/choicemap :params (cm/choicemap :slope (mx/scalar 2.0)
                                                :intercept (mx/scalar 1.0))))
(cm/addresses nested)
;; => [[:params :slope] [:params :intercept]]
```

## Stacking and unstacking for batching

For vectorized inference ([Chapter 8](./ch08-vectorization.md)), GenMLX runs N particles through a model simultaneously using \\([N]\\)-shaped arrays. `stack-choicemaps` combines N scalar choice maps into one choice map with \\([N]\\)-shaped leaves:

```clojure
(def cm1 (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 10.0)))
(def cm2 (cm/choicemap :x (mx/scalar 2.0) :y (mx/scalar 20.0)))
(def cm3 (cm/choicemap :x (mx/scalar 3.0) :y (mx/scalar 30.0)))

(def stacked (cm/stack-choicemaps [cm1 cm2 cm3] mx/stack))
;; :x is now a [3]-shaped array: [1.0, 2.0, 3.0]
;; :y is now a [3]-shaped array: [10.0, 20.0, 30.0]
```

The inverse, `unstack-choicemap`, splits back into N scalar choice maps:

```clojure
(def unstacked (cm/unstack-choicemap stacked 3 mx/index
                 (fn [v] (= [] (mx/shape v)))))
;; => [cm1', cm2', cm3'] — each with scalar values
```

## Traces

A trace is an immutable record of one complete execution:

```clojure
;; Trace fields
{:gen-fn    ;; the generative function that produced this trace
 :args      ;; the arguments passed to the function
 :choices   ;; a choice map of all random choices made
 :retval    ;; the return value
 :score}    ;; log p(all choices) — the log joint probability
```

You typically get traces from `simulate` or `generate`, not by constructing them manually:

```clojure
(def model (dyn/auto-key (gen [x]
             (let [v (trace :v (dist/gaussian x 1))]
               (mx/multiply v v)))))

(let [t (p/simulate model [0])]
  (println "args:" (:args t))              ;; [0]
  (println "retval:" (mx/item (:retval t))) ;; v^2 (non-negative)
  (println "score:" (mx/item (:score t))))  ;; log p(v)
```

Traces from composed models (using `splice`) carry metadata with splice scores — the per-sub-function score contributions:

```clojure
(def sub-model (gen [] (trace :z (dist/gaussian 0 1))))
(def parent-model (dyn/auto-key (gen [] (splice :child sub-model []))))

(let [t (p/simulate parent-model [])]
  ;; The child's choices live under :child
  (cm/has-value? (cm/get-submap (cm/get-submap (:choices t) :child) :z))
  ;; => true
  )
```

## Selections

Selections identify subsets of addresses for operations like `regenerate` (resample selected addresses) and `project` (score selected addresses). They form a Boolean algebra:

```clojure
;; Select specific addresses
(def s (sel/select :x :y))
(sel/selected? s :x)  ;; => true
(sel/selected? s :z)  ;; => false

;; Select everything
(sel/selected? sel/all :anything)  ;; => true

;; Select nothing
(sel/selected? sel/none :x)  ;; => false
```

### Complement

The complement selects everything the original doesn't:

```clojure
(def s (sel/select :x :y))
(def c (sel/complement-sel s))
(sel/selected? c :x)        ;; => false
(sel/selected? c :z)        ;; => true
(sel/selected? c :anything)  ;; => true
```

### Hierarchical selections

For models with `splice`, hierarchical selections target addresses within sub-models:

```clojure
;; Select only :slope within the :params sub-model
(def s (sel/hierarchical :params (sel/select :slope)))
(sel/selected? s :params)  ;; => true
(sel/selected? s :other)   ;; => false

;; The sub-selection under :params
(def sub (sel/get-subselection s :params))
(sel/selected? sub :slope)      ;; => true
(sel/selected? sub :intercept)  ;; => false
```

### Selections as Boolean algebra

Selections satisfy the laws of a Boolean algebra:

- `all` is the top element (selects everything)
- `none` is the bottom element (selects nothing)
- `complement-sel` is negation
- `complement(all) = none` (functionally)
- `complement(none) = all` (functionally)

```clojure
;; complement(all) selects nothing
(sel/selected? (sel/complement-sel sel/all) :x)   ;; => false

;; complement(none) selects everything
(sel/selected? (sel/complement-sel sel/none) :x)  ;; => true
```

This algebraic structure matters because selections compose — you can build complex address patterns from simple ones, and the GFI operations work correctly with any selection.

## What we've learned

- Choice maps have two types: **Value** (leaf) and **Node** (branch). Both are immutable persistent data.
- `choicemap`, `from-map`, `to-map` build and convert choice maps.
- `get-submap`, `get-value`, `get-choice` read them; `set-value`, `set-choice`, `merge-cm` produce new ones.
- `stack-choicemaps` / `unstack-choicemap` convert between scalar and batched representations.
- Traces are immutable records with five fields: gen-fn, args, choices, retval, score.
- Selections form a **Boolean algebra**: `select`, `all`, `none`, `complement-sel`, `hierarchical`.

In the next chapter, we'll use these data structures with `update` and `regenerate` — the operations that make MCMC possible.
