# TODO: Pure Functional GenMLX

**Goal:** Zero impurity above Layer 0. All dynamic vars, volatiles, atoms,
`binding` forms, and `!`-named functions eliminated from Layers 1-8.
`mx/eval!` and `mx/tidy` calls pushed down into Layer 0 boundary helpers.

**Approach:** Pass the runtime as a parameter — no code-walking macro needed.
The `gen` macro adds a hidden runtime parameter; `trace`, `splice`, `param`
become local function bindings. The single encapsulated mutable cell lives
in `run-handler` (Layer 0 execution runtime), exactly like re-frame's app-db atom.

**Constraint:** At least as fast as current implementation. No API breakage.

**Current state (after Phase 1):** 3 dynamic vars eliminated (`*handler*`,
`*state*`, `*param-store*`), handler.cljs now purely functional, runtime.cljs
is the single mutable boundary. Remaining: 2 dynamic vars (`*prng-key*`,
`*batched-exec?*`), 1 atom, ~100 `mx/eval!` calls in Layers 2-8, ~18
`mx/tidy` calls, ~10 `js/Math.random` calls, 1 top-level side effect.

---

## Phase 1: Runtime-as-parameter ✅ COMPLETE

The core transformation. The `gen` macro passes a runtime object as a hidden
parameter. `trace`, `splice`, and `param` become local bindings — plain
functions. No code walking. No restrictions on HOFs, `map`, `for`, etc.

### 1.1 Create `runtime.cljs` (Layer 0 — execution runtime) ✅

- [x] Create `src/genmlx/runtime.cljs` (~155 lines)
- [x] Implement `run-handler` with volatile!-based mutable cell
- [x] Cell uses `volatile!` (functionally equivalent to JS object cell)
- [x] `rt` is the runtime object with `.trace`, `.splice`, `.param` closures
- [x] Implement splice logic: scalar mode (executor), batched DynamicGF
      (recursive `run-handler`), combinator fallback (`h/combinator-batched-fallback`)
- [x] Param-store propagated to sub-executions via init-state

### 1.2 Rewrite `gen` macro ✅

- [x] New macro adds `ᐩrt` runtime parameter + local bindings (~10 lines)
- [x] Zero code walking. Zero form analysis.
- [x] `trace`, `splice`, `param` are local bindings that shadow namespace imports
- [x] Works with ALL Clojure constructs: `map`, `for`, `loop`, HOFs, closures

### 1.3 Update DynamicGF (dynamic.cljs) ✅

- [x] All 7 protocol methods use `rt/run-handler` with pure transitions
- [x] All 4 vectorized functions (`vsimulate`, `vgenerate`, `vupdate`, `vregenerate`)
      use `rt/run-handler` with batched transitions
- [x] Param-store: passed in init-state as `:param-store` field, read from
      `(::param-store (meta this))` metadata on DynamicGF
- [x] Executors (`execute-sub`, `execute-sub-project`, `execute-sub-assess`)
      propagate param-store to sub-gfs via `vary-meta`
- [x] Deprecated `dyn/trace`, `dyn/splice` wrappers removed
- [x] `dyn/param` simplified to direct-mode only (outside gen bodies)
- [ ] `warn-unused-constraints` still uses `js/console.warn` (deferred to Phase 6)

### 1.4 Delete old handler dispatch machinery (handler.cljs) ✅

- [x] Delete `(def ^:dynamic *handler* nil)`
- [x] Delete `(def ^:dynamic *state* nil)`
- [x] Delete `(def ^:dynamic *param-store* nil)`
- [x] Delete all 10 handler wrapper functions
- [x] Delete `trace-choice!`, `trace-param!`, `trace-gf!`
- [x] Delete `batched-splice-transition`
- [x] Delete old `run-handler`
- [x] Keep all 10 pure transition functions (unchanged)
- [x] Keep `merge-sub-result`, `combinator-batched-fallback`, helpers

**Result:** handler.cljs is now purely functional (~300 lines). Zero dynamic
vars, zero volatile!, zero binding, zero side effects.

### 1.5 Update ADEV handler (inference/adev.cljs) ✅

- [x] `adev-transition`: kept as-is (already pure)
- [x] `adev-handler`: DELETED
- [x] `vadev-transition`: kept as-is (already pure)
- [x] `vadev-handler`: DELETED
- [x] `adev-execute`: uses `rt/run-handler` with `adev-transition`
- [x] `vadev-execute`: uses `rt/run-handler` with `vadev-transition`
- [x] Removed 3 `binding [h/*param-store* ...]` sites — param-store via `vary-meta`
- [x] Removed `genmlx.handler` import (no longer needed)

### 1.6 Update verify.cljs ✅

- [x] `validate-handler` → pure `validate-transition` with `:seen-addrs`/`:violations`
- [x] `run-validation-trial`: uses `rt/run-handler` with `validate-transition`
- [x] Removed `vreset!` site
- [x] Removed `genmlx.handler` import

### 1.7 Update learning.cljs ✅

- [x] `simulate-with-params`: `vary-meta` instead of `binding [h/*param-store*]`
- [x] `generate-with-params`: same
- [x] `make-param-loss-fn`: same
- [x] Removed `genmlx.handler` import
- [x] `wake-phase-loss`, `sleep-phase-loss`: replaced `binding [rng/*prng-key*]`
      with `vary-meta` key propagation (Phase 2)

### 1.8 Direct execution mode (no handler active)

- [ ] Not yet implemented — gen fns currently require GFI execution
      (`p/simulate`, `p/generate`, etc.). Direct `(model args)` calls
      will error with nil runtime. This is acceptable for now; users
      should use `(dyn/call model args...)` or `(p/simulate model args)`.

### 1.9 Run full test suite ✅

- [x] All 8 core tests pass
- [x] `gen_clj_compat_test.cljs`: 165/165
- [x] `genjax_compat_test.cljs`: 73/73 (pre-existing MCMC flakiness)
- [x] `vectorized_test.cljs`: all pass
- [x] Additional suites: adev, vmap, verify, recurse, custom_gradient,
      project, assess_propose, correctness, contracts, nn — all pass
- [ ] Run benchmarks — verify no regression (not yet done)

### 1.10 Migrate test files ✅

- [x] 77 test files: `dyn/trace` → `trace`, `dyn/splice` → `splice`,
      `dyn/param` → `param` inside gen bodies
- [x] `handler_test.cljs` rewritten to use `rt/run-handler` + transitions
- [x] `remaining_fixes_test.cljs` param tests updated to use
      `learn/simulate-with-params`
- [x] `lowering_ground_truth_2.cljs` updated to use `rt/run-handler`

---

## Phase 2: Eliminate `*prng-key*` dynamic var ✅

Eliminated the `*prng-key*` dynamic var and `next-key` function. PRNG keys
are now threaded via metadata on gen-fns (same pattern as `param-store`).

### 2.1 Thread PRNG keys through metadata ✅

- [x] DynamicGF GFI methods read key from `(::key (meta this))`, fallback `fresh-key`
- [x] `with-key` changed from `[key f]` thunk wrapper to `[gf key]` metadata setter
- [x] `execute-sub`, `execute-sub-project`, `execute-sub-assess` propagate key via metadata
- [x] Scalar splice in `runtime.cljs` splits key before calling executor

### 2.2 Remove `seed-from-key!` from inference utilities ✅

- [x] `inference/util.cljs` `systematic-resample`: removed `seed-from-key!` call
- [x] `inference/util.cljs` `accept-mh?`: removed `seed-from-key!` call
- [x] Replaced `seed-from-key!` with minimal `seed!` in `random.cljs`

### 2.3 Update `dist-simulate` / `dist-propose` ✅

- [x] `dist/core.cljs` `dist-simulate`: reads key from metadata, calls `seed!`
- [x] `dist/core.cljs` `dist-propose`: reads key from metadata, calls `seed!`

### 2.4 Delete `*prng-key*` and `next-key` ✅

- [x] Deleted `(def ^:dynamic *prng-key* nil)` from `mlx/random.cljs`
- [x] Deleted `next-key` function from `mlx/random.cljs`
- [x] Deleted old `seed-from-key!` (replaced with `seed!`)
- [x] Added `seed!` calls in handler transitions (before `dist-sample`/`dist-sample-n`)
      Note: MLX requires global PRNG seeding alongside key passing for determinism

### 2.5 Update callers ✅

- [x] `inference/mcmc.cljs`: `dyn/with-key` call sites updated to new API
- [x] `inference/importance.cljs`: same
- [x] `learning.cljs`: 4 `binding [rng/*prng-key*]` sites → `vary-meta` key propagation
- [x] `inference/amortized.cljs`: `rng/next-key` → `rng/fresh-key`
- [x] `vmap.cljs`: `rng/next-key` → `rng/fresh-key` (2 sites)
- [x] `verify.cljs`: `rng/next-key` → `rng/fresh-key`

### 2.6 Eliminate `js/Math.random` calls ✅

- [x] `rng/fresh-key` no-arg arity: single entropy injection point (kept)
- [x] `vectorized.cljs`: removed `js/Math.random` fallback, uses `rng/ensure-key`
- [x] `inference/mcmc.cljs` NUTS main loop: replaced 3 `js/Math.random` with key-based
- [x] `inference/mcmc.cljs` NUTS warmup: replaced 3 `js/Math.random` with key-based
- [x] `inference/mcmc.cljs` `build-tree`: replaced `js/Math.random` with key-based

### 2.7 Run full test suite ✅

- [x] PRNG key reproducibility: 13/13
- [x] `gen_clj_compat_test.cljs`: 165/165
- [x] `genjax_compat_test.cljs`: 73/73
- [x] Core tests (gen, handler, inference, combinators, dist, verify): all pass
- [x] Vectorized, ADEV, vmap, amortized, adaptive NUTS: all pass

---

## Phase 3: Push `mx/eval!` down to Layer 0 boundary

~100 `mx/eval!` calls scattered across Layers 2-8. Move them all into
Layer 0 boundary helpers.

### 3.1 Create Layer 0 boundary helpers in `mlx.cljs`

- [ ] **`materialize!`** — `(defn materialize! [& arrays] (apply eval! arrays))`
      For explicit graph materialization at loop boundaries
- [ ] **`tidy-step`** — move from `inference/util.cljs` to `mlx.cljs`.
      Wraps a step function in `mx/tidy`, smuggling result out via volatile
      (the volatile is a Layer 0 implementation detail)
- [ ] **`realize-clj`** — `(defn realize-clj [x] (eval! x) (->clj x))`
      For extracting values to ClojureScript
- [ ] **`eval-state!`** — move from `inference/util.cljs` to `mlx.cljs`.
      Evaluates all arrays in a trace/state

### 3.2 Refactor training (learning.cljs, nn.cljs)

- [ ] `adam-step` line 92: move `mx/eval!` into a separate boundary call at
      the loop level in `train`
- [ ] `train` loop line 125: single `materialize!` per iteration
- [ ] `wake-sleep` lines 305, 315: `materialize!` at loop boundary
- [ ] `nn/step!`: move to `mlx.cljs` — it wraps MLX nn.Module mutation,
      which is inherently Layer 0

### 3.3 Refactor inference (mcmc.cljs — ~38 eval sites)

- [ ] `mh` / `mh-step`: pure step logic + 1 `materialize!` per iteration
- [ ] `compiled-mh`: same
- [ ] `mala` / `mala-step`: same
- [ ] `hmc` / `hmc-step`: same
- [ ] `nuts` / `nuts-step`: batch evals in tree recursion
- [ ] `gibbs` / `enumerative-gibbs`: same
- [ ] `elliptical-slice-step`: same
- [ ] `map-optimize`: same
- [ ] Vectorized variants: same pattern
- [ ] Target: 1 `materialize!` call per loop iteration

### 3.4 Refactor inference (vi.cljs — ~16 eval sites)

- [ ] `advi`: 1 boundary per iteration
- [ ] `compiled-advi`: same
- [ ] `programmable-vi`: same
- [ ] `compiled-programmable-vi`: same

### 3.5 Refactor inference (smc.cljs, smcp3.cljs)

- [ ] `smc` / `smc-step`: 1 boundary per step
- [ ] `csmc`: same
- [ ] `smc-rejuvenate`: same
- [ ] `smcp3-step`: same

### 3.6 Refactor inference utilities (inference/util.cljs)

- [ ] `materialize-weights`: keep (already a boundary helper)
- [ ] `normalize-log-weights`: reduce to 1 eval call
- [ ] Move `eval-state!`, `tidy-step`, `dispose-trace!`, `force-gc!`,
      `with-resource-guard` to `mlx.cljs`
- [ ] Top-level side effect line 221 `(mx/set-cache-limit!)`: move to
      `mlx.cljs` module init

### 3.7 Refactor distributions (dist.cljs)

- [ ] Categorical line 408: `mx/eval!` for shape — use Layer 0 helper
- [ ] Multivariate normal lines 915, 923: Cholesky — use Layer 0 helper
- [ ] Wishart/Inverse-Wishart lines 1058-1155: matrix ops — use Layer 0 helpers

### 3.8 Refactor remaining files

- [ ] `gradients.cljs` lines 38, 67: callers do the eval, or use `materialize!`
- [ ] `contracts.cljs` lines 19, 177: use `mx/realize` helper
- [ ] `vectorized.cljs` lines 25, 60, 97, 151: use boundary helpers

### 3.9 Rename/move `!` functions out of Layers 1-8

- [x] `trace-choice!` → eliminated (Phase 1)
- [x] `trace-param!` → eliminated (Phase 1)
- [x] `trace-gf!` → eliminated (Phase 1)
- [ ] `nn/step!` → move to `mlx.cljs`
- [ ] `eval-state!` → move to `mlx.cljs`
- [ ] `force-gc!` → move to `mlx.cljs`
- [ ] `dispose-trace!` → move to `mlx.cljs`
- [ ] `train-proposal!` → rename to `train-proposal` (calls Layer 0 boundary)

### 3.10 Run full test suite

- [ ] All tests pass
- [ ] Run benchmarks — verify no regression
- [ ] Memory profiling — verify no Metal buffer leaks

---

## Phase 4: Remove Wishart atom

- [ ] `dist.cljs` line 1074-1075: replace `(atom 0)` / `(swap! ki inc)` with
      indexed `loop` or `reduce` over pre-split keys
- [ ] Zero atoms in codebase
- [ ] Run distribution tests

---

## Phase 5: Remove `*batched-exec?*` dynamic var

- [ ] This var is only used for dev warnings in `mx/eval!` and `mx/item`
- [ ] After Phase 3, Layers 1-8 never call `mx/eval!` directly — warning is moot
- [ ] Delete `(def ^:dynamic *batched-exec?* false)` from `mlx.cljs`
- [ ] Delete the `when *batched-exec?*` checks from `mx/eval!` and `mx/item`
- [ ] Remove `binding [mx/*batched-exec?*]` from `runtime.cljs`

---

## Phase 6: Remove `js/console.warn`

- [ ] `mlx.cljs` lines 74, 82: removed with Phase 5
- [ ] `dynamic.cljs` line 35: replace with pure validation (return unused
      constraint addresses as metadata, or separate validation function)

---

## Phase 7: Remove top-level side effect

- [ ] `inference/util.cljs` line 221: `(mx/set-cache-limit! DEFAULT-CACHE-LIMIT)`.
      Move to `mlx.cljs` module init or explicit `init!` function.

---

## Phase 8: Verify and document

### 8.1 Final audit

- [ ] `grep 'vreset!\|vswap!\|volatile!' src/genmlx/` — only in `runtime.cljs`
- [ ] `grep 'set!' src/genmlx/` — only in `runtime.cljs`
- [ ] `grep 'def.*\^:dynamic' src/genmlx/` — zero hits
- [ ] `grep 'binding \[' src/genmlx/` — zero hits
- [ ] `grep 'atom\|swap!\|reset!' src/genmlx/` — zero hits
- [ ] `grep 'defn.*!' src/genmlx/ --include='*.cljs'` — only in `mlx.cljs`,
      `mlx/random.cljs`, `runtime.cljs`
- [ ] `grep 'mx/eval!' src/genmlx/` — only in `mlx.cljs` and `runtime.cljs`
- [ ] `grep 'mx/tidy' src/genmlx/` — only in `mlx.cljs` and `runtime.cljs`
- [ ] `grep 'js/Math.random' src/genmlx/` — only in `rng/fresh-key`
- [ ] `grep 'js/console' src/genmlx/` — zero hits

### 8.2 Performance verification

- [ ] `vectorized_benchmark.cljs` — compare throughput before/after
- [ ] `gpu_benchmark.cljs` — compare GPU operation timing
- [ ] `benchmark.cljs` — compare end-to-end inference timing
- [ ] Micro-benchmark: JS property access (rt.trace) vs dynamic var lookup
- [ ] Verify Metal buffer memory profile unchanged

### 8.3 Test suite

- [ ] All core tests pass
- [ ] `gen_clj_compat_test.cljs`: 165/165
- [ ] `genjax_compat_test.cljs`: 73/73
- [ ] `vectorized_test.cljs`: all pass
- [ ] `vectorized_benchmark.cljs`: no regression

### 8.4 Update documentation

- [ ] Update CLAUDE.md architecture layers:
      ```
      Layer 0: MLX + Runtime    (mlx.cljs, mlx/random.cljs, runtime.cljs)
      Layer 1: Core Data        (choicemap, trace, selection — pure)
      Layer 2: GFI & Execution  (handler transitions, dynamic — pure)
      Layer 3: DSL              (gen macro — pure)
      Layer 4: Distributions    (27 distributions — pure)
      Layer 5: Combinators      (Map, Unfold, Switch, etc. — pure)
      Layer 6: Inference        (IS, MCMC, SMC, VI, ADEV — pure)
      Layer 7: Vectorized       (batched execution — pure)
      Layer 8: Verification     (contracts, verify — pure)
      ```
- [ ] Document `runtime.cljs` as the execution runtime (the re-frame analog)
- [ ] Document `fresh-key` as the single entropy injection point
- [x] Update handler.cljs docstring: pure transitions only, no dispatch machinery

---

## Summary: What gets eliminated

| What | Before | After Phase 1 | After All | Where |
|------|--------|---------------|-----------|-------|
| Dynamic vars | 5 | **2** | 0 | `*prng-key*`, `*batched-exec?*` remain |
| `volatile!` / `set!` | 8+17 sites | **1** | 1 | `runtime.cljs` (encapsulated) |
| `vreset!` / `vswap!` | 21 sites | **2** | 0 | `runtime.cljs` only |
| `atom` / `swap!` | 1 each | 1 each | 0 | Wishart (Phase 4) |
| `binding` forms | 12 | **3** | 0 | `runtime.cljs` (1), `dynamic.cljs` (1), `learning.cljs` (1) |
| `!` functions (Layers 1-8) | 8 | **5** | 0 | Phase 3 eliminates rest |
| `mx/eval!` (Layers 1-8) | ~100 | ~100 | 0 | Phase 3 |
| `mx/tidy` (Layers 1-8) | ~18 | ~18 | 0 | Phase 3 |
| `js/Math.random` | ~10 | ~10 | 1 | Phase 2 |
| `js/console.warn` | 3 | 3 | 0 | Phase 6 |
| Top-level side effects | 1 | 1 | 0 | Phase 7 |
| Gen macro complexity | 1 line | **~10 lines** | ~10 lines | zero code walking |

**Phase 1 eliminated:** 3 dynamic vars, 10 handler wrappers, 3 `!`-dispatch
functions, ~15 `vreset!/vswap!` sites, ~9 `binding` forms. handler.cljs is
now purely functional.

## Estimated effort

| Phase | Effort | Risk | Status |
|-------|--------|------|--------|
| 1. Runtime-as-parameter | 2-3 days | Low | **✅ COMPLETE** |
| 2. Eliminate PRNG vars | 2-3 days | Medium (audit sampling paths) | |
| 3. Push eval to Layer 0 | 3-5 days | Medium (many files, same pattern) | |
| 4. Remove Wishart atom | 15 min | None | |
| 5. Remove batched var | 30 min | None | |
| 6. Remove console.warn | 30 min | None | |
| 7. Remove top-level effect | 30 min | None | |
| 8. Verify and document | 1-2 days | None | |
| **Total** | **9-14 days** | | |
