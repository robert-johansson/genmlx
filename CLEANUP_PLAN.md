# Cleanup Plan: From Working Refactoring to Beautiful Code

*The dispatcher mechanism is correct and all tests pass. But `dynamic.cljs` grew from 939 to 1028 lines because the dispatchers create closure wrappers around the existing run-\* helpers. This plan eliminates the boilerplate by standardizing the helpers to the `:run` signature and replacing the dispatchers' `(case op ...)` blocks with simple lookup tables.*

---

## The Problem

The compiled-dispatcher is currently 65 lines of this pattern:

```clojure
{:run (case op
        :simulate (fn [gf args key _] (run-simulate-compiled ...))
        :generate (fn [gf args key {:keys [constraints]}] (run-generate-compiled ...))
        :update   (fn [gf args key {:keys [trace constraints]}] (run-update-compiled ...))
        ...)}
```

Each closure just adapts the `:run` signature to the run-\* helper's signature. The handler-dispatcher and prefix paths have the same pattern. That's ~100 lines of pure adapter boilerplate.

## The Fix

**Standardize all run-\* helpers to the `:run` signature: `(fn [gf args key opts])`.**

The helpers currently take `body-fn` and `this` as separate parameters:
```clojure
(defn- run-simulate-handler [gf args key body-fn this] ...)
```

After standardization:
```clojure
(defn- run-simulate-handler [gf args key opts]
  (let [body-fn (:body-fn gf)] ...))
```

Then the dispatchers become lookup tables:
```clojure
(def ^:private handler-table
  {:simulate   run-simulate-handler
   :generate   run-generate-handler
   :update     run-update-handler
   :regenerate run-regen-handler
   :assess     run-assess-handler
   :project    run-project-handler})

(def ^:private handler-dispatcher
  (reify dispatch/IDispatcher
    (resolve-transition [_ op _ _]
      {:run (get handler-table op) :score-type :joint})))
```

~10 lines instead of ~30. Same for compiled-dispatcher and prefix paths.

---

## Phase A: Standardize run-\* Helper Signatures

*Goal: All 18 run-\* helpers take `[gf args key opts]`. No behavior change. Test after each group.*

### A.1 Handler helpers (L0)

- [ ] **`run-simulate-handler`** — Change from `[gf args key body-fn this]` to `[gf args key opts]`. Get `body-fn` from `(:body-fn gf)`. Remove `this` (it's the same as `gf`). Get `param-store` from `(::param-store (meta gf))` instead of `(::param-store (meta this))`.
- [ ] **`run-generate-handler`** — Same pattern. Get `constraints` from `(:constraints opts)`.
- [ ] **`run-update-handler`** — Get `trace` and `constraints` from opts.
- [ ] **`run-regen-handler`** — Get `trace`, `selection`, `old-score` from opts. Note: `old-score` is `(:score trace)`.
- [ ] **`run-assess-handler`** — Get `choices` from `(:constraints opts)` (assess passes choices as constraints).
- [ ] **`run-project-handler`** — Get `trace` and `selection` from opts.
- [ ] **Run tests:** L0 certification (68/68), Gen.clj compat (356/356).

### A.2 Compiled helpers (L1-M2)

- [ ] **`run-simulate-compiled`** — Change from `[compiled-sim gf args key]` to `[gf args key opts]`. Get `compiled-sim` from `(:compiled-simulate (:schema gf))`.
- [ ] **`run-generate-compiled`** — Get `compiled-gen` from schema, `constraints` from opts.
- [ ] **`run-update-compiled`** — Get `compiled-upd` from schema, `trace` and `constraints` from opts.
- [ ] **`run-regen-compiled`** — Get from schema and opts.
- [ ] **Compiled assess** — Currently inline in the dispatcher. Extract to `run-assess-compiled`.
- [ ] **Compiled project** — Currently inline in the dispatcher. Extract to `run-project-compiled`.
- [ ] **Run tests:** Compiled simulate (82/82), L4 certification (41/41).

### A.3 Prefix helpers (L1-M3)

- [ ] **`run-simulate-prefix`** — Change from `[compiled-pfx gf args key body-fn this]` to `[gf args key opts]`. Get `compiled-pfx` from schema.
- [ ] **`run-generate-prefix`** — Same pattern.
- [ ] **`run-update-prefix`** — Same pattern.
- [ ] **`run-regen-prefix`** — Same pattern.
- [ ] **`run-assess-prefix`** — Same pattern.
- [ ] **`run-project-prefix`** — Same pattern.
- [ ] **Run tests:** Partial compile (92/92), combinator compile (92/92).

### A.4 Analytical helpers (L3)

- [ ] **`run-generate-analytical`** — Change from `[schema gf args key constraints body-fn this]` to `[gf args key opts]`. Get `schema` from gf, `constraints` from opts.
- [ ] **`run-assess-analytical`** — Same pattern.
- [ ] **`run-regen-analytical`** — Same pattern.
- [ ] **Run tests:** GenJAX compat (73/73), full suite (1,031/1,031).

**Done criterion for Phase A:** All 18 run-\* helpers take `[gf args key opts]`. All tests pass. Dispatchers still use `(case op ...)` with closures (they'll be simplified in Phase B).

---

## Phase B: Replace Dispatcher Case Blocks with Lookup Tables

*Goal: Each dispatcher is 5-15 lines instead of 30-65 lines.*

- [ ] **Define `handler-table`** — Map from op keyword to handler run-fn.
- [ ] **Simplify `handler-dispatcher`** — `{:run (get handler-table op) :score-type :joint}`.
- [ ] **Define `compiled-table`** — Map from op keyword to compiled run-fn. Only return if schema has the compiled key.
- [ ] **Simplify `compiled-dispatcher`** — Check schema, return `{:run (get compiled-table op)}` or check prefix-table.
- [ ] **Simplify `analytical-dispatcher`** — This one keeps conditional logic (it checks constraints), but the `:run` values are direct references instead of closures.
- [ ] **Remove `:op` from `full-opts` in `run-dispatched`** — Standard helpers no longer need it. For `enumerate-run` (custom handler), inject op via the custom-transition-dispatcher: `{:run (fn [gf args key opts] (t op gf args key opts))}`.
- [ ] **Run full test suite** — 1,031/1,031.
- [ ] **Run all 26 memo examples.**

**Done criterion for Phase B:** Dispatchers are ~50 lines total (down from ~150). No `(case op ...)` blocks with closure wrappers.

---

## Phase C: Inline `analytical-applicable?`

- [ ] **Move the predicate into `analytical-dispatcher`'s `resolve-transition`** — It's a 5-line private function used in one place. Inline it.
- [ ] **Run tests.**

---

## Phase D: Verify and Measure

- [ ] **Count lines:** `dynamic.cljs` should be shorter than the original 939 lines (main branch baseline).
- [ ] **Full test suite:** 1,031/1,031.
- [ ] **All 26 memo examples.**
- [ ] **Run with `*validate?*` true:** All tests pass with Malli validation enabled.
- [ ] **Review the dispatcher code** — Each dispatcher should fit on one screen. The `(reify ...)` bodies should be obvious at a glance.
- [ ] **Review the run-\* helpers** — Each should be self-contained: get what it needs from `gf` and `opts`, do its work, return the result.

**Done criterion for Phase D:** `dynamic.cljs` < 939 lines. Code reads cleanly. No boilerplate closures. Each function does one thing.

---

## What This Does NOT Change

- `dispatch.cljs` — Already clean (63 lines).
- `schemas.cljs` — Already clean (258 lines).
- `handler.cljs` — Transitions are already pure and well-factored.
- `runtime.cljs` — The volatile! boundary is already minimal.
- `inference/exact.cljs` — `enumerate` and `enumerate-run` are already clean.
- All test files and examples (except possibly chimp.cljs which was already updated).
- The GFI protocol surface — Callers see the same interface.

## Expected Result

Before cleanup:
```
dynamic.cljs: 1028 lines (dispatchers = 150 lines of case/closure boilerplate)
```

After cleanup:
```
dynamic.cljs: ~880-920 lines (dispatchers = ~50 lines of lookup tables)
```

The dispatchers go from "I need to read 150 lines to understand the dispatch" to "I can see the entire dispatch in one table." The run-\* helpers go from "I'm called in two different ways depending on who calls me" to "I have one signature and one caller."
