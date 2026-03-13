# M3 Plan: Schema Loop Analysis

## Goal

Enrich the schema walker to extract structured metadata about loops that
contain trace sites. This metadata is the foundation for M4 (auto-constraint
stacking) and M5 (fused loop execution).

**Current state:** `handle-loop-form` (schema.cljs:186-191) sets a single
boolean `has-loops? true` and discards all structural information about the
loop. Trace sites inside loops are accumulated into the flat `:trace-sites`
vector with `:addr :dynamic` — indistinguishable from non-loop dynamic traces.

**After M3:** The schema carries a `:loop-sites` vector with one entry per
loop-with-traces, capturing everything M4 and M5 need: loop type, bindings,
iteration count, address pattern, per-iteration trace structure, dependencies,
and rewritability classification.

---

## Current Schema Flow (What Exists)

```
gen macro captures source form
  → extract-schema (schema.cljs:337)
    → init-acc: {:trace-sites [] :splice-sites [] :param-sites []
                 :dynamic-addresses? false :has-branches? false :has-loops? false}
    → walk-forms threads (acc, env) through body
    → handle-call dispatches on form head:
        "doseq"/"dotimes"/"for" → handle-loop-form
        "loop" → handle-loop-loop
    → handle-loop-form:
        1. checks body for contains-gen-call?
        2. walks all sub-forms (accumulates trace sites with :addr :dynamic)
        3. sets :has-loops? true if gen calls found
    → post-processing: topo-sort, static? flag
```

Key limitation: trace sites inside loops are marked `:addr :dynamic` and
dumped into the top-level `:trace-sites` vector. No way to know which
traces belong to which loop, what the address pattern is, or what the
iteration structure looks like.

---

## New Schema Field: `:loop-sites`

### Data Structure

```clojure
{:loop-sites
 [{:type :doseq                            ;; :doseq | :dotimes | :for
   :bindings-form '[j x] OR 'i             ;; raw binding form
   :element-sym 'x                         ;; element variable (doseq), nil (dotimes)
   :index-sym 'j                           ;; index variable, nil if no index
   :collection-form '(map-indexed vector xs) ;; what's iterated (doseq/for), nil (dotimes)
   :count-form '(count xs)                 ;; iteration count expr, or literal int (dotimes)
   :count-arg-idx 0                        ;; which gen-fn arg determines count (-1 if none)

   ;; Trace sites that belong to THIS loop
   :trace-sites
   [{:addr :dynamic
     :addr-form '(keyword (str "y" j))
     :addr-pattern {:type :keyword-str      ;; recognized pattern type
                    :prefix "y"             ;; string prefix before index
                    :index-sym 'j           ;; symbol used as index
                    :suffix nil}            ;; optional suffix after index
     :dist-type :gaussian
     :dist-args ['(mx/add (mx/multiply slope (mx/scalar x)) intercept)
                 '(mx/scalar 1)]
     :element-deps #{x}                    ;; deps on loop element variable
     :outer-deps #{:slope :intercept}      ;; deps on trace addrs outside loop
     :static-dist-type? true}]             ;; same dist type every iteration?

   ;; Splice/param sites in this loop (uncommon but possible)
   :splice-sites []
   :param-sites []

   ;; Classification
   :homogeneous? true                      ;; all iterations same dist type + arity?
   :rewritable? true                       ;; safe to rewrite to Map/iid?
   :rewrite-blockers []}]}                 ;; reasons if not rewritable
```

### Address Pattern Recognition

The walker recognizes these common patterns for dynamic addresses:

| Source Form | Pattern Type | Fields |
|-------------|-------------|--------|
| `(keyword (str "y" j))` | `:keyword-str` | `{:prefix "y" :index-sym j}` |
| `(keyword (str "obs-" j))` | `:keyword-str` | `{:prefix "obs-" :index-sym j}` |
| `(keyword (str "y" j "-mean"))` | `:keyword-str` | `{:prefix "y" :index-sym j :suffix "-mean"}` |
| `(keyword j)` | `:keyword-sym` | `{:index-sym j}` |
| Any other | `:unknown` | `{:form <raw-form>}` |

Only `:keyword-str` and `:keyword-sym` patterns are classified as rewritable.

### Rewritability Rules

A loop is `:rewritable? true` when ALL of these hold:

- [ ] No splice sites inside the loop
- [ ] No param sites inside the loop
- [ ] All trace addresses use a recognized pattern (`:keyword-str` or `:keyword-sym`)
- [ ] All trace sites have the same distribution type (`:homogeneous? true`)
- [ ] The loop body does not contain nested loops
- [ ] The loop body does not contain branches with trace sites
- [ ] `loop/recur` is NOT used (only `doseq`/`dotimes`/`for`)

When any condition fails, `:rewrite-blockers` captures the reason(s).

---

## Implementation Checklist

### Phase 1: Address Pattern Detection

- [ ] **1.1** Add `detect-addr-pattern` helper function
  - Input: address form (the first arg to `trace`)
  - Output: `{:type :keyword-str :prefix "y" :index-sym 'j ...}` or `{:type :unknown :form ...}`
  - Recognizes `(keyword (str "prefix" sym))` and `(keyword (str "prefix" sym "suffix"))` forms
  - Recognizes `(keyword sym)` form
  - Returns `{:type :static :addr <kw>}` for plain keyword addresses
  - Place BEFORE `handle-loop-form` in schema.cljs (forward-reference constraint)

- [ ] **1.2** Add `detect-addr-pattern` unit tests (in schema_test.cljs)
  - `(keyword (str "y" j))` → `{:type :keyword-str :prefix "y" :index-sym j}`
  - `(keyword (str "obs-" i))` → `{:type :keyword-str :prefix "obs-" :index-sym i}`
  - `(keyword (str "x" i "-val"))` → `{:type :keyword-str :prefix "x" :index-sym i :suffix "-val"}`
  - `(keyword j)` → `{:type :keyword-sym :index-sym j}`
  - `:static-addr` → `{:type :static :addr :static-addr}`
  - `(some-fn i)` → `{:type :unknown :form ...}`

### Phase 2: Loop Binding Analysis

- [ ] **2.1** Add `analyze-doseq-bindings` helper function
  - Input: bindings vector from doseq form (e.g., `[[j x] (map-indexed vector xs)]`)
  - Output: `{:element-sym 'x :index-sym 'j :collection-form '(map-indexed vector xs)}`
  - Handles: `[x coll]` (element only), `[[j x] (map-indexed vector coll)]` (indexed),
    `[[idx item] (map-indexed vector coll)]` (any destructuring of 2-tuple)
  - Place BEFORE `handle-loop-form`

- [ ] **2.2** Add `analyze-dotimes-bindings` helper function
  - Input: bindings vector from dotimes form (e.g., `[i 5]` or `[i (count xs)]`)
  - Output: `{:index-sym 'i :count-form 5}` or `{:index-sym 'i :count-form '(count xs)}`
  - Place BEFORE `handle-loop-form`

- [ ] **2.3** Add `analyze-for-bindings` helper function
  - Input: bindings vector from for form (e.g., `[item items]`)
  - Output: `{:element-sym 'item :collection-form 'items}`
  - Note: `for` has richer binding syntax (`:let`, `:when`, `:while`) — start
    with simple `[sym coll]` form only, mark others as non-rewritable
  - Place BEFORE `handle-loop-form`

- [ ] **2.4** Add `infer-count-arg-idx` helper function
  - Input: count-form expression and gen-fn params vector
  - Output: index into params vector, or -1
  - Handles: `(count xs)` where `xs` is params[0] → returns 0
  - Handles: literal int `5` → returns -1
  - Handles: symbol `n` where `n` is params[1] → returns 1
  - Place BEFORE `handle-loop-form`

- [ ] **2.5** Add binding analysis unit tests (in schema_test.cljs)
  - doseq with `[x coll]` → element only
  - doseq with `[[j x] (map-indexed vector xs)]` → indexed
  - dotimes with `[i 5]` → literal count
  - dotimes with `[i (count xs)]` → computed count
  - for with `[item items]` → element only

### Phase 3: Loop Body Analysis

- [ ] **3.1** Add `extract-loop-trace-sites` helper function
  - Input: loop body forms, outer env (binding environment), loop binding symbols
  - Output: vector of trace site maps with `:addr-pattern`, `:element-deps`, `:outer-deps`
  - Walk the loop body forms, collecting trace calls
  - For each trace: run `detect-addr-pattern` on address form
  - Separate deps into `:element-deps` (depend on loop binding symbols) vs
    `:outer-deps` (depend on trace addresses from outer scope)
  - Add `:static-dist-type?` — true if dist-type is a known keyword (not `:unknown`)
  - Does NOT modify the top-level accumulator's `:trace-sites` (these are loop-local)
  - Place BEFORE `handle-loop-form`

- [ ] **3.2** Add `classify-loop` helper function
  - Input: trace-sites from loop body, splice-sites, param-sites, loop type
  - Output: `{:homogeneous? bool :rewritable? bool :rewrite-blockers [...]}`
  - `:homogeneous?` — all trace sites have same `:dist-type` and same arity of `:dist-args`
  - `:rewritable?` — per rules in the spec above
  - `:rewrite-blockers` — vector of strings explaining why not rewritable
  - Place BEFORE `handle-loop-form`

- [ ] **3.3** Add loop body analysis unit tests (in schema_test.cljs)
  - Single gaussian trace inside doseq → homogeneous, rewritable
  - Two different dist types inside loop → not homogeneous, not rewritable
  - Splice inside loop → not rewritable, blocker: "splice in loop body"
  - Nested loop with traces → not rewritable, blocker: "nested loop"
  - Branch with trace inside loop → not rewritable, blocker: "branch in loop body"

### Phase 4: Enhanced `handle-loop-form`

- [ ] **4.1** Rewrite `handle-loop-form` to extract structured metadata
  - Current (6 lines): sets boolean flag, walks sub-forms
  - New behavior:
    1. Analyze bindings based on loop type (doseq/dotimes/for)
    2. Walk body in an extended env that includes loop binding symbols
    3. Extract loop-specific trace sites (with addr pattern + dep classification)
    4. Classify rewritability
    5. Append loop-site map to `:loop-sites` on accumulator
    6. Still set `:has-loops? true` (backward compat)
    7. Still accumulate trace sites into top-level `:trace-sites` (backward compat)
    8. Set `:dynamic-addresses? true` if any loop trace has non-static address
  - CRITICAL: the walker must still produce identical `:trace-sites`, `:has-loops?`,
    `:dynamic-addresses?` as before — `:loop-sites` is purely additive

- [ ] **4.2** Update `handle-loop-loop` for `loop/recur` forms
  - `loop/recur` is harder to analyze (arbitrary state threading)
  - For now: walk as before, set `:has-loops? true`, do NOT produce loop-site entry
  - Add blocker note: "loop/recur not analyzable"
  - Future: could detect simple counting patterns `(loop [i 0] ... (recur (inc i)))`

- [ ] **4.3** Update `extract-schema` to initialize `:loop-sites` in the accumulator
  - Add `:loop-sites []` to init-acc (line 346)
  - This is backward compatible — empty vector means no loop info

### Phase 5: Integration Fields

- [ ] **5.1** Add `:prefix-sites` and `:suffix-sites` to schema
  - `:prefix-sites` — static trace sites that appear BEFORE the first loop
  - `:suffix-sites` — static trace sites that appear AFTER the last loop
  - Computed from the existing `:trace-sites` vector + position tracking
  - These enable M5: compile prefix statically, run loop fused, compile suffix statically
  - Implementation: add position tracking to the accumulator (`:form-position` counter),
    tag each trace site with its position relative to loops

- [ ] **5.2** Add `:loop-count-param` convenience field
  - For models like `(gen [xs] ... (doseq [[j x] (m-i v xs)] ...))`,
    record that the loop count is determined by `(count xs)` where `xs` is param 0
  - This tells M5 how to determine the tensor dimension T at runtime
  - Stored at the loop-site level as `:count-form` and `:count-arg-idx`

### Phase 6: Tests

- [ ] **6.1** Test: Simple doseq with keyword-str address pattern
  ```clojure
  (gen [xs]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian mu 1)))
      mu))
  ```
  Assert:
  - `:loop-sites` has 1 entry
  - loop type is `:doseq`
  - `:index-sym` is `j`, `:element-sym` is `x`
  - trace site addr-pattern: `{:type :keyword-str :prefix "y" :index-sym j}`
  - `:outer-deps` includes `:mu`
  - `:element-deps` includes `x`
  - `:homogeneous?` true
  - `:rewritable?` true
  - backward compat: `:has-loops?` true, `:dynamic-addresses?` true

- [ ] **6.2** Test: Linear regression (3 static + 1 loop)
  ```clojure
  (gen [xs ys]
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))
          noise (trace :noise (dist/exponential 1))]
      (doseq [[j [x y]] (map-indexed vector (map vector xs ys))]
        (trace (keyword (str "obs" j))
               (dist/gaussian (+ (* slope x) intercept) noise)))
      slope))
  ```
  Assert:
  - `:loop-sites` has 1 entry
  - `:prefix-sites` has 3 entries (slope, intercept, noise)
  - loop trace `:outer-deps` includes `:slope`, `:intercept`, `:noise`
  - `:homogeneous?` true, `:rewritable?` true
  - `:count-arg-idx` 0 (count determined by first param `xs`)

- [ ] **6.3** Test: dotimes with literal count
  ```clojure
  (gen []
    (dotimes [i 5]
      (trace (keyword (str "x" i)) (dist/gaussian 0 1))))
  ```
  Assert:
  - loop type `:dotimes`
  - `:count-form` is `5` (literal)
  - `:count-arg-idx` -1
  - `:rewritable?` true

- [ ] **6.4** Test: for loop
  ```clojure
  (gen [items]
    (doall (for [item items]
      (trace (keyword (str "obs-" item)) (dist/gaussian 0 1)))))
  ```
  Assert:
  - loop type `:for`
  - `:element-sym` is `item`
  - `:rewritable?` true

- [ ] **6.5** Test: Non-rewritable — mixed distribution types
  ```clojure
  (gen [xs]
    (doseq [[j x] (map-indexed vector xs)]
      (trace (keyword (str "y" j)) (dist/gaussian x 1))
      (trace (keyword (str "z" j)) (dist/bernoulli 0.5))))
  ```
  Assert:
  - `:homogeneous?` false (gaussian + bernoulli)
  - `:rewritable?` false
  - `:rewrite-blockers` includes "heterogeneous distribution types"

- [ ] **6.6** Test: Non-rewritable — splice in loop
  ```clojure
  (gen [xs]
    (doseq [[j x] (map-indexed vector xs)]
      (splice (keyword (str "sub" j)) inner-gf x)))
  ```
  Assert:
  - `:rewritable?` false
  - `:rewrite-blockers` includes "splice in loop body"

- [ ] **6.7** Test: Non-rewritable — branch with trace in loop
  ```clojure
  (gen [xs]
    (doseq [[j x] (map-indexed vector xs)]
      (if (pos? x)
        (trace (keyword (str "y" j)) (dist/gaussian x 1))
        (trace (keyword (str "y" j)) (dist/gaussian 0 1)))))
  ```
  Assert:
  - `:rewritable?` false
  - `:rewrite-blockers` includes "branch in loop body"

- [ ] **6.8** Test: Non-rewritable — unknown address pattern
  ```clojure
  (gen [xs]
    (doseq [[j x] (map-indexed vector xs)]
      (trace (compute-addr j x) (dist/gaussian x 1))))
  ```
  Assert:
  - addr-pattern type `:unknown`
  - `:rewritable?` false
  - `:rewrite-blockers` includes "unrecognized address pattern"

- [ ] **6.9** Test: Static model — no loop-sites
  ```clojure
  (gen [x]
    (let [mu (trace :mu (dist/gaussian 0 1))]
      (trace :y (dist/gaussian mu 1))))
  ```
  Assert:
  - `:loop-sites` is `[]`
  - `:has-loops?` false
  - `:static?` true

- [ ] **6.10** Test: Backward compatibility — all existing schema_test.cljs tests still pass
  - The 174 existing tests must produce identical results
  - `:loop-sites` is a NEW additive field — no existing field changes

- [ ] **6.11** Test: Multiple loops in one model
  ```clojure
  (gen [xs ys]
    (let [mu-x (trace :mu-x (dist/gaussian 0 10))
          mu-y (trace :mu-y (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "x" j)) (dist/gaussian mu-x 1)))
      (doseq [[j y] (map-indexed vector ys)]
        (trace (keyword (str "y" j)) (dist/gaussian mu-y 1)))
      [mu-x mu-y]))
  ```
  Assert:
  - `:loop-sites` has 2 entries
  - First loop: prefix "x", outer-deps #{:mu-x}
  - Second loop: prefix "y", outer-deps #{:mu-y}
  - Both rewritable

- [ ] **6.12** Test: loop/recur — not analyzable
  ```clojure
  (gen [n]
    (loop [i 0]
      (when (< i n)
        (trace (keyword (str "x" i)) (dist/gaussian 0 1))
        (recur (inc i)))))
  ```
  Assert:
  - `:has-loops?` true (from existing behavior)
  - `:loop-sites` is `[]` (loop/recur not analyzed)

### Phase 7: Verification

- [ ] **7.1** All 174 existing schema_test.cljs tests pass unchanged
- [ ] **7.2** All new M3 tests pass (12 new test sections, ~40 assertions)
- [ ] **7.3** Full test suite still 744/744
- [ ] **7.4** No changes to any file outside schema.cljs and schema_test.cljs
- [ ] **7.5** Confirm: models with loops produce identical compiled/handler
      behavior (the schema change is metadata-only, no execution change)

---

## File Changes

| File | Change | Lines (est.) |
|------|--------|-------------|
| `src/genmlx/schema.cljs` | New helpers + enhanced handle-loop-form | +120-150 |
| `test/genmlx/schema_test.cljs` | 12 new test sections | +180-220 |

No other files modified. M3 is purely additive to the schema — no execution
paths change, no APIs change, no consumer code needs updating.

---

## Function Placement Order (Forward-Reference Safe)

```
;; In schema.cljs, new functions placed BEFORE handle-loop-form (line 186):

detect-addr-pattern          ;; used by extract-loop-trace-sites
analyze-doseq-bindings       ;; used by handle-loop-form
analyze-dotimes-bindings     ;; used by handle-loop-form
analyze-for-bindings         ;; used by handle-loop-form
infer-count-arg-idx          ;; used by handle-loop-form
extract-loop-trace-sites     ;; used by handle-loop-form
classify-loop                ;; used by handle-loop-form
handle-loop-form             ;; REPLACED (existing, line 186)
```

---

## What M3 Enables

### M4: Auto-Constraint Stacking
M4 reads `:loop-sites[].trace-sites[].addr-pattern` to detect that user-provided
flat constraints (`:y0 1.0 :y1 2.0 ...`) match a loop's address pattern
(`prefix "y"`) and auto-stacks them into a `[T]` tensor.

### M5: Fused Loop Execution
M5 reads `:loop-sites[].homogeneous?`, `:loop-sites[].trace-sites[].dist-type`,
`:loop-sites[].trace-sites[].outer-deps`, and `:loop-sites[].count-form` to
compile the loop into a single Metal dispatch using noise transforms.

### M6: Auto-Rewrite to Map Combinator
M6 reads `:loop-sites[].rewritable?` to decide whether to source-rewrite the
loop into a Map combinator invocation. Uses `:loop-sites[].trace-sites` to
build the kernel gen function.

---

## Design Principles

1. **Purely additive.** No existing schema fields change. `:loop-sites` is new.
   All 174 existing tests pass without modification.

2. **Idiomatic ClojureScript.** Pure functions, immutable data, no side effects.
   Same walker pattern (acc + env threading) used by all other handlers.

3. **Conservative classification.** When in doubt, mark as not rewritable.
   False negatives (missing optimization) are safe. False positives (incorrect
   rewrite) would break correctness.

4. **Handler is ground truth.** M3 adds metadata only. No execution paths
   change. The handler continues to run loops exactly as before.
