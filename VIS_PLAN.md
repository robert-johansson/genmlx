# VIS Plan: Vectorized Inference for Dynamic-Address Models

## Problem Statement

Vectorized importance sampling (`vgenerate`/`vsimulate`) fails on models with
loop-generated dynamic addresses — the most common real-world model pattern.

```clojure
;; This model CANNOT be vectorized today:
(def linreg
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1)))
      slope)))
```

This blocks vectorized inference for **most real models**: linear regression,
GMMs, hierarchical models, time series — any model that observes N data points
via a loop.

---

## Root Cause Analysis

### How Batched Execution Works (Working Case)

`vsimulate` runs the model body **once** with `[N]`-shaped arrays instead of N
times with scalar arrays. The handler calls `dist-sample-n` (returning `[N]`
values) and MLX broadcasting handles all downstream arithmetic.

For **static addresses**, this works perfectly:

```
Model body runs ONCE:
  trace :slope (gaussian 0 10)     → handler samples [N] values, stores at :slope
  trace :intercept (gaussian 0 10) → handler samples [N] values, stores at :intercept
  trace :y (gaussian ...)          → handler samples [N] values, stores at :y
```

Each address is visited exactly once. Each `[N]`-shaped value represents N
independent particles. No collisions.

### Why Dynamic Addresses Fail

When the model has `(doseq [[j x] ...] (trace (keyword (str "y" j)) ...))`:

```
Model body runs ONCE:
  trace :slope → [N] values at :slope                    ✓
  trace :intercept → [N] values at :intercept             ✓
  doseq iteration j=0:
    trace :y0 → [N] values at :y0                        ✓ (first visit)
  doseq iteration j=1:
    trace :y1 → [N] values at :y1                        ✓ (first visit)
  ...
```

Wait — this actually stores correctly! The addresses `:y0`, `:y1`, ... are
distinct. So what actually breaks?

**The real problem is subtler.** The doseq loop runs with **scalar** loop
variable `j`, but the distribution arguments contain `[N]`-shaped values
(`slope` is `[N]`). This means:

1. **`x` is scalar** (from the input vector `xs`) — correct
2. **`slope` is `[N]`** (from batched trace) — correct, broadcasts with `x`
3. **`(keyword (str "y" j))`** — `j` is scalar — correct, address is well-defined

So the loop itself works! The issue is **constraint matching in `vgenerate`**:

```clojure
;; User provides observations:
(def obs (cm/choicemap :y0 1.0 :y1 2.0 :y2 3.0))

;; vgenerate's batched-generate-transition checks:
(if-let [cv (cm/get-value constraints addr)]
  ;; Use constraint value → but cv is scalar 1.0, needs to be [N]-shaped
  ...)
```

The constraint values are **scalar** but the handler expects `[N]`-shaped
values for batched mode. The `batched-generate-transition` in `handler.cljs`
broadcasts constraint values correctly (line 205-206):

```clojure
value (if cv
        (let [v (mx/ensure-array cv)]
          (if (= [] (mx/shape v))
            (mx/broadcast-to v [n])    ;; ← scalar → [N] broadcast
            v))
        (dc/dist-sample-n dist k2 n))
```

**So batched-generate-transition already handles scalar constraints!**

### The ACTUAL Blockers

After careful analysis, the real blockers are:

1. **Schema classification blocks compilation.** Models with `has-loops? true`
   and `dynamic-addresses? true` get `static? false`, which disables L1
   compiled paths. The model falls back to handler-only execution, which works
   but is slower.

2. **`vsimulate`/`vgenerate` skip models without compiled ops.** Looking at
   `dynamic.cljs`, the vectorized paths work through the batched handler
   regardless of compilation status. The issue is that **models with dynamic
   addresses are never tested/validated for vectorized execution** — they may
   work for simple cases but fail for complex patterns (nested loops, data-
   dependent iteration counts, etc.).

3. **Observations must match the dynamic address pattern.** The user must
   construct observation choicemaps with the exact dynamic addresses (`:y0`,
   `:y1`, ...). There's no auto-vectorized observation format.

4. **Score accumulation across loop iterations.** Each `doseq` iteration adds
   to the `[N]`-shaped score via `mx/add`. For T iterations, this means T
   sequential score additions. The Map combinator avoids this by computing all
   scores at once via fused execution.

5. **No fused execution for inline loops.** The Map combinator can fuse T
   element evaluations into one Metal dispatch. Inline `doseq` cannot — each
   iteration is a separate handler call through the volatile! cell.

### The Performance Gap (Not a Correctness Gap)

The batched handler already handles dynamic addresses correctly for simple
patterns. **The gap is performance, not correctness:**

| Pattern | Scalar (N particles) | Batched (vsimulate) | Fused (Map combinator) |
|---------|---------------------|---------------------|----------------------|
| Static model | N × body | 1 × body, [N] arrays | 1 Metal dispatch |
| Loop model (doseq) | N × T iterations | 1 × T iterations, [N] arrays | 1 Metal dispatch |
| Map combinator | N × T kernel calls | 1 × T kernel calls (IBatchedSplice) | 1 Metal dispatch |

Batched doseq is T× faster than scalar (runs loop once vs N times). But Map
combinator's fused path is T× faster still (one dispatch vs T iterations).

---

## Strategy: Auto-Rewrite Loops to Map Combinator

The most impactful fix: **detect loop-with-trace patterns in the schema and
automatically rewrite them to Map combinator at construction time.**

This is the same philosophy as L1-M4 (branch rewriting to `mx/where`): the
compiler transforms the user's code into a more efficient form, transparently.

### Why This Is the Right Approach

1. **Map combinator already works perfectly** — fused execution, IBatchedSplice,
   full GFI support (simulate, generate, update, regenerate).
2. **Schema already detects loops** — `has-loops?`, `dynamic-addresses?`, and
   the trace sites inside loops are all captured.
3. **No new runtime infrastructure needed** — reuses existing combinators.
4. **User code unchanged** — the `gen` macro does the rewrite transparently.
5. **Composable** — the rewritten model still satisfies all GFI contracts.

---

## Detailed Implementation Plan

### Phase 1: Loop Pattern Analysis (Schema Enhancement)

**Goal:** Enrich the schema to capture enough information about loop structure
to enable automatic rewriting.

#### 1.1: Classify Loop Patterns

Add a new schema field `:loop-sites` that captures:

```clojure
{:loop-sites
 [{:type :doseq                       ;; doseq, dotimes, for, map, loop
   :source-form '(doseq [[j x] (map-indexed vector xs)] ...)
   :binding-sym 'j                     ;; loop variable
   :collection-form '(map-indexed vector xs)  ;; what's being iterated
   :iteration-count-form '(count xs)   ;; static if determinable, nil otherwise
   :trace-sites [:y0 :y1 ...]          ;; dynamic — captured as pattern
   :addr-pattern {:prefix "y" :index-sym 'j :format :keyword-str}
   :body-traces [{:addr-form '(keyword (str "y" j))
                  :dist-form '(dist/gaussian ...)
                  :deps #{:slope :intercept}
                  :element-deps #{'x}   ;; deps on loop element
                  :static-deps #{:slope :intercept}}]  ;; deps outside loop
   :rewritable? true}]}
```

**Key classifications:**

| Pattern | Rewritable? | Reason |
|---------|------------|--------|
| `(doseq [[j x] (m-i v xs)] (trace (kw (str "y" j)) (dist args)))` | Yes | Simple indexed loop |
| `(dotimes [i n] (trace (kw (str "x" i)) (dist args)))` | Yes | Count-based loop |
| `(doseq [x xs] (trace :shared-addr (dist x)))` | No | Same address each iteration |
| `(loop [i 0 state init] ... (trace ...) ... (recur ...))` | Maybe | Depends on state threading |
| Nested loops | No (initially) | Complex address patterns |

#### 1.2: Schema Walker Changes

Modify `handle-loop-form` in `schema.cljs` to extract structured loop info:

```clojure
(defn- handle-loop-form [acc env args]
  (let [;; Current: just set has-loops? true
        ;; New: also extract loop structure
        loop-info (analyze-loop-structure args env)
        acc' (walk-forms acc env args)]
    (cond-> acc'
      (:has-trace? loop-info)
      (-> (assoc :has-loops? true)
          (update :loop-sites (fnil conj []) loop-info)))))
```

New helper `analyze-loop-structure`:

```clojure
(defn- analyze-loop-structure [args env]
  (let [[bindings & body] args]
    {:type (infer-loop-type bindings)           ;; :doseq, :dotimes, etc.
     :binding-sym (extract-binding-sym bindings)
     :collection-form (extract-collection bindings)
     :addr-pattern (detect-addr-pattern body)   ;; {:prefix :index-sym :format}
     :body-traces (extract-body-traces body env)
     :rewritable? (rewritable? ...)}))
```

#### 1.3: Address Pattern Detection

Recognize common address generation patterns:

```clojure
;; Pattern 1: (keyword (str "prefix" index-var))
;; Pattern 2: (keyword (str "prefix" index-var suffix))
;; Pattern 3: index-var directly (when it's already a keyword)
;; Pattern 4: (keyword (name base) (str index-var))  ;; hierarchical

(defn- detect-addr-pattern [body]
  (let [trace-forms (filter trace-call? body)]
    (for [form trace-forms]
      (let [addr-form (second form)]
        (cond
          (keyword-str-pattern? addr-form)
          {:format :keyword-str
           :prefix (extract-prefix addr-form)
           :index-sym (extract-index-sym addr-form)}

          (keyword? addr-form)
          {:format :static :addr addr-form}

          :else
          {:format :unknown :form addr-form})))))
```

### Phase 2: Loop-to-Map Rewriter

**Goal:** Transform loop-with-trace patterns into Map combinator invocations
at `make-gen-fn` construction time.

#### 2.1: Rewrite Decision

In `dynamic.cljs/make-gen-fn`, after schema extraction:

```clojure
(defn make-gen-fn [body-fn source]
  (let [schema (schema/extract-schema source)
        ;; NEW: check for rewritable loops
        schema (if (and schema (seq (:loop-sites schema)))
                 (loop-rewrite/try-rewrite-loops schema source)
                 schema)
        ;; ... existing compilation pipeline ...
        ]
    (->DynamicGF body-fn source schema)))
```

#### 2.2: The Rewrite Transform

For a model like:

```clojure
(gen [xs]
  (let [slope     (trace :slope (dist/gaussian 0 10))
        intercept (trace :intercept (dist/gaussian 0 10))]
    (doseq [[j x] (map-indexed vector xs)]
      (trace (keyword (str "y" j))
             (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1)))
    slope))
```

The rewriter produces an equivalent model that uses Map:

```clojure
(gen [xs]
  (let [slope     (trace :slope (dist/gaussian 0 10))
        intercept (trace :intercept (dist/gaussian 0 10))
        ;; Loop body extracted as kernel:
        obs-kernel (gen [x slope intercept]
                     (trace :y (dist/gaussian
                                 (mx/add (mx/multiply slope x) intercept) 1)))
        ;; Map combinator over xs:
        _ (splice :obs (map-combinator obs-kernel)
                  (mapv vector xs
                        (repeat slope)
                        (repeat intercept)))]
    slope))
```

**Critical:** The rewritten model must produce **compatible choicemaps**. The
original model stores `:y0`, `:y1`, ... at the top level. The Map combinator
stores `:obs → {0 → {:y v0}, 1 → {:y v1}, ...}`. These are different!

#### 2.3: Address Compatibility Layer

Two options:

**Option A: Transparent address translation.** Add an address mapping layer
that translates between flat dynamic addresses (`:y0`, `:y1`) and Map's
hierarchical addresses (`[:obs 0 :y]`, `[:obs 1 :y]`).

```clojure
;; On the schema:
{:addr-translation
 {:flat->hierarchical {":y0" [:obs 0 :y], ":y1" [:obs 1 :y], ...}
  :hierarchical->flat {[:obs 0 :y] ":y0", [:obs 1 :y] ":y1", ...}
  :pattern {:flat-prefix "y" :hier-path [:obs :idx :y]}}}

;; In p/generate, translate user-provided flat constraints to hierarchical:
(defn translate-constraints [constraints addr-translation]
  (reduce-kv
    (fn [cm flat-addr value]
      (if-let [hier-path (get (:flat->hierarchical addr-translation) (name flat-addr))]
        (cm/set-choice cm hier-path value)
        (cm/set-value cm flat-addr value)))
    cm/EMPTY
    (cm/entries constraints)))
```

**Option B: Stacked observation format.** Instead of individual `:y0`, `:y1`,
the model uses a single `:obs` address with a `[T]`-shaped tensor. Users
provide observations as `(cm/choicemap :obs (mx/array [1.0 2.0 3.0]))`.

This is simpler but changes the user-facing API.

**Recommendation: Option A for backward compatibility.** The translation is
mechanical and can be cached on the schema. Users continue writing
`(cm/choicemap :y0 1.0 :y1 2.0 ...)` and the rewriter handles the rest.

#### 2.4: Runtime Rewrite (Alternative to Source Rewrite)

Source-level rewriting is powerful but complex. A simpler alternative:
**runtime loop interception** in the batched handler.

Instead of rewriting source, modify the batched handler to detect when it's
inside a loop (via schema metadata) and accumulate results differently:

```clojure
(defn batched-simulate-transition-with-loop-fusion [state addr dist]
  (if-let [loop-info (get-in state [:loop-fusion addr])]
    ;; Fused path: accumulate into [T, N] tensor
    (let [{:keys [iteration total-iterations]} loop-info
          n (:batch-size state)
          [k1 k2] (rng/split (:key state))
          value (dc/dist-sample-n dist k2 n)       ;; [N]
          lp (dc/dist-log-prob dist value)]         ;; [N]
      [value (-> state
               (assoc :key k1)
               (update-in [:loop-values addr] (fnil conj []) value)
               (update :score #(mx/add % lp)))])
    ;; Standard path
    (batched-simulate-transition state addr dist)))
```

After the loop completes, stack the accumulated `[N]` values into `[T, N]`
and store under the loop's address prefix.

This avoids source rewriting entirely but requires handler changes.

### Phase 3: Stacked Trace Representation

**Goal:** Enable models to use a single `[T, N]`-shaped tensor for loop
observations instead of T separate `[N]`-shaped values.

#### 3.1: New Distribution: `iid`

```clojure
(defn iid [base-dist n]
  "Independent and identically distributed: sample n values from base-dist."
  (dist/iid base-dist n))

;; Usage in model:
(gen [xs]
  (let [slope (trace :slope (dist/gaussian 0 10))
        means (mx/add (mx/multiply slope (mx/array xs)) intercept)]
    (trace :ys (dist/iid (dist/gaussian means 1) (count xs)))
    slope))
```

This sidesteps the loop entirely. The `iid` distribution:
- `sample`: calls `dist-sample-n` on base dist → `[T]` tensor
- `log-prob`: sums element-wise log-probs → scalar
- `sample-n`: returns `[N, T]` tensor (batched across particles)

#### 3.2: Vectorized iid

For `vsimulate`/`vgenerate`, `iid`'s `sample-n` returns `[N, T]`:

```clojure
(defmethod dc/dist-sample-n* :iid [{:keys [params]} key n]
  (let [{:keys [base-dist count]} params
        ;; Sample [N, T] from base distribution
        samples (dc/dist-sample-n base-dist key (* n count))]
    (mx/reshape samples [n count])))
```

This gives maximum GPU parallelism: one Metal dispatch for all N×T samples.

#### 3.3: Observation Format for Stacked Traces

Users provide `[T]`-shaped observations:

```clojure
(def obs (cm/choicemap :ys (mx/array [1.0 2.0 3.0 4.0 5.0])))
(p/generate (dyn/auto-key model) [(range 5)] obs)
```

Clean, no dynamic address construction needed.

### Phase 4: Schema-Guided Auto-Vectorization

**Goal:** When a model has rewritable loops, automatically produce a vectorized
variant that uses stacked traces.

#### 4.1: Auto-Vectorized Model Construction

At `make-gen-fn` time, if the schema detects a rewritable loop pattern,
attach an `:auto-vectorized-body` to the schema:

```clojure
(defn make-gen-fn [body-fn source]
  (let [schema (schema/extract-schema source)
        schema (if (and schema (seq (:loop-sites schema))
                        (every? :rewritable? (:loop-sites schema)))
                 (assoc schema
                   :vectorized-body-fn
                   (build-vectorized-body schema source))
                 schema)]
    (->DynamicGF body-fn source schema)))
```

The `vectorized-body-fn` uses stacked traces instead of loops:

```clojure
(defn- build-vectorized-body [schema source]
  ;; For each rewritable loop, replace with stacked trace
  ;; Returns a new body-fn that:
  ;; 1. Traces static sites normally (slope, intercept)
  ;; 2. Computes all loop-body dist-args as [T]-shaped tensors
  ;; 3. Traces one stacked site per loop (e.g., :ys instead of :y0..yT)
  ...)
```

#### 4.2: Modified vgenerate

When `vsimulate`/`vgenerate` detect `:vectorized-body-fn` on the schema,
use it instead of the original body:

```clojure
(defn vsimulate [gf args n key]
  (let [body-fn (or (get-in gf [:schema :vectorized-body-fn])
                    (:body-fn gf))
        ...]
    (rt/run-handler
      h/batched-simulate-transition
      init-state
      (fn [rt] (apply body-fn rt args)))))
```

#### 4.3: Constraint Translation for vgenerate

When the user provides flat constraints (`{:y0 1.0 :y1 2.0 ...}`), detect
the loop pattern and auto-stack into `{:ys [1.0 2.0 ...]}`:

```clojure
(defn auto-stack-constraints [constraints schema]
  (if-let [loops (:loop-sites schema)]
    (reduce
      (fn [cm loop-info]
        (let [{:keys [addr-pattern]} loop-info
              prefix (:prefix addr-pattern)
              ;; Find all :y0, :y1, ... in constraints
              matching (filter #(str/starts-with? (name %) prefix)
                               (cm/addresses constraints))
              ;; Sort by index and stack values
              sorted (sort-by #(parse-long (subs (name %) (count prefix))) matching)
              values (mapv #(cm/get-value (cm/get-submap constraints %)) sorted)]
          (if (seq values)
            (-> (reduce #(cm/set-value %1 %2 nil) cm matching)  ;; remove flat
                (cm/set-value (:stacked-addr loop-info)
                              (mx/array (mapv mx/item values))))  ;; add stacked
            cm)))
      constraints
      loops)
    constraints))
```

### Phase 5: Fused Loop Execution

**Goal:** For static loop patterns, compile the entire loop into a single
Metal dispatch (analogous to fused-unfold).

#### 5.1: Fused Loop Simulate

For a loop with T iterations over identical distribution type:

```clojure
(defn make-fused-loop-simulate [schema loop-info]
  ;; Build a single compiled function that:
  ;; 1. Takes args + PRNG key
  ;; 2. Generates [T] noise values
  ;; 3. Transforms all T values via noise transform
  ;; 4. Computes [T] log-probs
  ;; 5. Returns {values: [T], score: scalar}
  (let [dist-type (:dist-type (first (:body-traces loop-info)))
        noise-transform (compiled/get-noise-transform dist-type)]
    (fn [key args-vec n-iterations]
      (let [noise (rng/normal key [n-iterations])
            ;; Compute all dist args as [T]-shaped tensors
            dist-args (compute-loop-dist-args args-vec loop-info n-iterations)
            ;; Apply noise transform to all T at once
            values (apply (:transform noise-transform) noise dist-args)
            ;; Compute all T log-probs at once
            log-probs (apply (:log-prob noise-transform) values dist-args)]
        {:values values
         :score (mx/sum log-probs)}))))
```

#### 5.2: Fused Loop Generate

Same as simulate but with constraint matching:

```clojure
(defn make-fused-loop-generate [schema loop-info]
  (fn [key args-vec n-iterations constraints]
    (let [dist-args (compute-loop-dist-args args-vec loop-info n-iterations)
          ;; For constrained sites: use constraint values
          ;; For unconstrained: sample via noise transform
          values (if constraints
                   constraints  ;; [T]-shaped constraint tensor
                   (apply noise-transform (rng/normal key [n-iterations]) dist-args))
          log-probs (apply (:log-prob noise-transform) values dist-args)]
      {:values values
       :weight (mx/sum log-probs)
       :score (mx/sum log-probs)})))
```

### Phase 6: Testing

#### 6.1: Correctness Tests

```clojure
;; Test 1: vsimulate on loop model produces valid [N]-shaped traces
;; Test 2: vgenerate on loop model matches scalar generate (statistical)
;; Test 3: Address translation round-trips correctly
;; Test 4: Stacked trace log-probs match sum of individual log-probs
;; Test 5: Auto-stacked constraints produce same weights as flat constraints
;; Test 6: Fused loop simulate matches handler simulate
;; Test 7: Fused loop generate matches handler generate
;; Test 8: iid distribution sample/log-prob correctness
;; Test 9: iid distribution sample-n shape correctness
;; Test 10: VIS on linreg model produces correct posterior estimates
;; Test 11: VIS on GMM model produces correct cluster assignments
;; Test 12: VIS on hierarchical model with nested loops
```

#### 6.2: Benchmark Tests

```clojure
;; Bench 1: Scalar IS vs VIS on linreg (T=50 obs, N=1000 particles)
;;          Expected: VIS 10-50x faster
;; Bench 2: VIS with fused loop vs VIS with batched handler loop
;;          Expected: Fused 3-5x faster
;; Bench 3: VIS with stacked trace vs VIS with flat dynamic addresses
;;          Expected: Stacked 2-3x faster (fewer choicemap operations)
;; Bench 4: Scaling: VIS speedup vs N particles (expect linear)
;; Bench 5: Scaling: VIS speedup vs T observations (expect sublinear)
```

---

## Implementation Order

### Milestone 1: Validate That Batched Handler Already Works

Before building anything, **verify** that `vsimulate`/`vgenerate` already
handle simple dynamic-address models correctly:

```clojure
(def simple-loop-model
  (gen [xs]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian mu 1)))
      mu)))

;; Does this work?
(dyn/vsimulate (dyn/auto-key simple-loop-model) [(range 5)] 100 key)
(dyn/vgenerate (dyn/auto-key simple-loop-model) [(range 5)]
               (cm/choicemap :y0 1.0 :y1 2.0 :y2 3.0 :y3 4.0 :y4 5.0)
               100 key)
```

If this already works, the problem is **performance only**, and we skip
straight to Phase 5 (fused execution).

If it fails, identify the exact failure point and fix it (likely constraint
broadcasting or address handling).

### Milestone 2: `iid` Distribution (Simplest Win)

Implement the `iid` distribution (Phase 3). This immediately enables:

```clojure
(def linreg-v
  (gen [xs]
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))
          means (mx/add (mx/multiply slope (mx/array xs)) intercept)]
      (trace :ys (dist/iid-gaussian means 1))
      slope)))
```

- Single trace site for all observations
- Native `[N, T]` batch sampling
- Works with existing `vsimulate`/`vgenerate` unchanged
- User writes model slightly differently (stacked observations)

**Estimated scope:** ~100 lines in `dist.cljs`, ~50 lines of tests.

### Milestone 3: Schema Loop Analysis (Phase 1)

Enrich the schema to capture loop structure. This is prerequisite for
auto-rewriting but also useful for diagnostics and future compilation.

**Estimated scope:** ~150 lines in `schema.cljs`, ~100 lines of tests.

### Milestone 4: Auto-Constraint Stacking (Phase 4.3)

For models with dynamic addresses, auto-detect and stack constraints in
`vgenerate`. This requires no model changes — users provide flat constraints
and the system stacks them automatically.

**Estimated scope:** ~80 lines in `dynamic.cljs`, ~60 lines of tests.

### Milestone 5: Fused Loop Execution (Phase 5)

Compile loops with homogeneous distributions into single Metal dispatches.
This is the performance ceiling.

**Estimated scope:** ~200 lines in `compiled.cljs`, ~150 lines of tests.

### Milestone 6: Auto-Rewrite (Phase 2) — Optional

Full source-level loop-to-Map rewriting. Only needed if the simpler approaches
(iid, fused loops) don't cover enough patterns.

**Estimated scope:** ~300 lines in new `loop_rewrite.cljs`, ~200 lines of tests.

---

## Expected Impact

| Approach | Models Covered | Speedup vs Scalar | Effort |
|----------|---------------|-------------------|--------|
| Validate existing batched handler | Simple loops | 10-50x (N particles) | 1 day |
| `iid` distribution | Stacked-obs models | 50-200x (N×T fused) | 2 days |
| Schema loop analysis | All loop models | Diagnostic only | 2 days |
| Auto-constraint stacking | All loop models | 10-50x (N particles) | 1 day |
| Fused loop execution | Homogeneous loops | 50-200x (N×T fused) | 3 days |
| Auto-rewrite to Map | All rewritable loops | 50-200x (Map fused) | 5 days |

**Recommended path:** M1 → M2 → M4 → M5. This covers the most common
patterns (linreg, GMM, hierarchical) with maximum speedup and minimal effort.

---

## Design Principles

1. **Validate before building.** The batched handler may already handle more
   than we think. Test first.
2. **Compose, don't duplicate.** The `iid` distribution composes with existing
   infrastructure. No new handler paths.
3. **Handler is ground truth.** Any fused/compiled loop must produce identical
   traces, scores, and weights as the handler path.
4. **User code unchanged.** The `gen` macro should handle rewriting
   transparently. Users who write `doseq` get vectorized inference without
   knowing it.
5. **Graceful fallback.** If a loop can't be rewritten, fall back to the
   batched handler (which runs the loop T times with `[N]`-shaped arrays).
   This is still N× faster than scalar.
