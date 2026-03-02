# TODO: Pure Functional GenMLX

**Goal:** Zero impurity above Layer 0. All dynamic vars, volatiles, atoms,
`binding` forms, and `!`-named functions eliminated from Layers 1-8.
`mx/eval!` and `mx/tidy` calls pushed down into Layer 0 boundary helpers.

**Approach:** Pass the runtime as a parameter — no code-walking macro needed.
The `gen` macro adds a hidden runtime parameter; `trace`, `splice`, `param`
become local function bindings. The single encapsulated mutable cell lives
in `run-handler` (Layer 0 execution runtime), exactly like re-frame's app-db atom.

**Constraint:** At least as fast as current implementation. No API breakage.

**Current state (after Phase 3):** All dynamic vars eliminated. All `mx/eval!`
and `mx/tidy` calls confined to Layer 0 boundary helpers. Zero atoms, zero
`js/console.warn`, zero top-level side effects in Layers 1-8. The only mutable
boundary is `volatile!` in `runtime.cljs`.

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
- [x] `warn-unused-constraints` replaced with pure `find-unused-constraints`
      (returns data in `:unused-constraints` key — Phase 3)

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

### 1.8 Direct execution mode (no handler active) ✅

- [x] `IFn` on defrecords is not possible in nbb/SCI (JS objects aren't callable).
      `(dyn/call model args...)` is the canonical way to call gen fns as
      regular functions. This delegates to `(:retval (p/simulate model [args]))`.
      Direct `(model args)` syntax would require compiled ClojureScript.

### 1.9 Run full test suite ✅

- [x] All 8 core tests pass
- [x] `gen_clj_compat_test.cljs`: 165/165
- [x] `genjax_compat_test.cljs`: 73/73 (pre-existing MCMC flakiness)
- [x] `vectorized_test.cljs`: all pass
- [x] Additional suites: adev, vmap, verify, recurse, custom_gradient,
      project, assess_propose, correctness, contracts, nn — all pass
- [x] Run benchmarks — no regression observed

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

## Phase 3: Push `mx/eval!` and `mx/tidy` down to Layer 0 ✅ COMPLETE

~73 `mx/eval!` calls and ~14 `mx/tidy` calls pushed down into Layer 0
boundary helpers. Also eliminated Wishart atom, `*batched-exec?*` dynamic
var, `js/console.warn` calls, and top-level side effect.

### 3.1 Create Layer 0 boundary helpers in `mlx.cljs` ✅

- [x] **`materialize!`** — variadic, calls `(apply eval! arrs)`
- [x] **`realize-clj`** — `(eval! x) (->clj x)`
- [x] **`tidy-materialize`** — `(let [r (tidy f)] (eval! r) r)`
- [x] **`tidy-run`** — generic tidy wrapper with volatile + collect-fn
- [x] Moved `force-gc!`, `with-resource-guard`, `DEFAULT-CACHE-LIMIT`
      from `inference/util.cljs`
- [x] Moved `training-step!` from `nn.cljs` (was `step!`)
- [x] Removed `*batched-exec?*` dynamic var (Phase 5)
- [x] Removed `js/console.warn` guards from `eval!`/`item` (Phase 6)
- [x] Top-level `(set-cache-limit! DEFAULT-CACHE-LIMIT)` moved here (Phase 7)

### 3.2 Refactor training (learning.cljs, nn.cljs) ✅

- [x] `adam-step`: `mx/eval!` → `mx/materialize!`
- [x] `train` loop: `mx/eval!` → `mx/materialize!`
- [x] `wake-sleep` (2 sites): `mx/eval!` → `mx/materialize!`
- [x] `nn/step!` → `def step! mx/training-step!` (alias to Layer 0)

### 3.3 Refactor inference (mcmc.cljs — ~20 eval + ~8 tidy sites) ✅

- [x] All ~20 `mx/eval!` → `mx/materialize!`
- [x] `leapfrog-step`: `mx/tidy` → `mx/tidy-run`
- [x] `hmc-step`: `mx/tidy` → `mx/tidy-run`
- [x] `run-loop-compiled-mh` tidy: → `mx/tidy-materialize`
- [x] `run-loop-compiled-mala` tidy: → `mx/tidy-run`
- [x] `run-loop-compiled-hmc` tidy: → `mx/tidy-materialize`
- [x] `vectorized-map-optimize` tidy: → `mx/tidy-run`
- [x] `doto ... mx/eval!` patterns → explicit `let` + `materialize!`

### 3.4 Refactor inference (vi.cljs — ~12 eval + ~8 tidy sites) ✅

- [x] All `mx/eval!` → `mx/materialize!`
- [x] `vi`/`compiled-vi` init: `mx/tidy` → `mx/tidy-materialize`
- [x] Gradient computation: `doto (mx/tidy ...) mx/eval!` → `mx/tidy-materialize`
- [x] ELBO evaluation: `mx/tidy` + eval + item → `mx/realize` + `mx/tidy-materialize`
- [x] `compiled-programmable-vi`: `mx/tidy` → `mx/tidy-materialize`

### 3.5 Refactor inference (smc.cljs, smcp3.cljs, adev.cljs) ✅

- [x] `smc.cljs`: 8 `mx/eval!` → `mx/materialize!`
- [x] `smcp3.cljs`: 4 `mx/eval!` → `mx/materialize!`
- [x] `adev.cljs`: 2 `mx/eval!` → `mx/materialize!`, 1 tidy → `mx/tidy-run`

### 3.6 Refactor inference utilities (inference/util.cljs) ✅

- [x] `materialize-weights`: `mx/eval!` → `mx/materialize!`
- [x] `normalize-log-weights`: `mx/eval!` → `mx/materialize!`
- [x] `eval-state!` → renamed `materialize-state`, `mx/eval!` → `mx/materialize!`
- [x] `dispose-trace!` → renamed `dispose-trace`
- [x] `tidy-step` → rewritten using `mx/tidy-run`
- [x] Moved `force-gc!`, `with-resource-guard`, `DEFAULT-CACHE-LIMIT`,
      top-level `(mx/set-cache-limit!)` to `mlx.cljs`
- [x] Backwards-compatible aliases: `(def force-gc! mx/force-gc!)`,
      `(def with-resource-guard mx/with-resource-guard)`

### 3.7 Refactor distributions (dist.cljs) ✅

- [x] All ~11 `mx/eval!` → `mx/materialize!`
- [x] Wishart atom eliminated — replaced with indexed `reduce` over pre-split keys

### 3.8 Refactor remaining files ✅

- [x] `gradients.cljs` (2 sites): `mx/eval!` → `mx/materialize!`
- [x] `contracts.cljs`: `ev` helper → `mx/realize`, broadcast eval → `mx/materialize!`
- [x] `verify.cljs` (1 site): `mx/eval!` → `mx/materialize!`
- [x] `vectorized.cljs` (4 sites): `mx/eval!` → `mx/materialize!`
- [x] `importance.cljs` (1 site): `mx/eval!` → `mx/materialize!`
- [x] `diagnostics.cljs` (4 sites): `mx/eval!` → `mx/materialize!`
- [x] `kernel.cljs` (1 site): `mx/eval!` → `mx/materialize!`

### 3.9 Rename/move `!` functions out of Layers 1-8 ✅

- [x] `trace-choice!` → eliminated (Phase 1)
- [x] `trace-param!` → eliminated (Phase 1)
- [x] `trace-gf!` → eliminated (Phase 1)
- [x] `nn/step!` → `def step! mx/training-step!` (delegates to Layer 0)
- [x] `eval-state!` → renamed `materialize-state` (no `!` suffix)
- [x] `force-gc!` → moved to `mlx.cljs` (Layer 0)
- [x] `dispose-trace!` → renamed `dispose-trace` (no `!` suffix)
- [x] `train-proposal!` → renamed `train-proposal`

### 3.10 Additional cleanups (Phases 4-7) ✅

- [x] **Phase 4 — Wishart atom:** Replaced `(atom 0)` + `swap!` with indexed
      `reduce` over pre-split keys. Zero atoms in codebase.
- [x] **Phase 5 — `*batched-exec?*`:** Deleted dynamic var from `mlx.cljs`,
      removed `binding` from `runtime.cljs`. Zero dynamic vars.
- [x] **Phase 6 — `js/console.warn`:** Removed from `eval!`/`item` (Phase 5).
      `dynamic.cljs` `warn-unused-constraints` → pure `find-unused-constraints`
      (returns data in `:unused-constraints` key). Zero console output.
- [x] **Phase 7 — Top-level side effect:** `(mx/set-cache-limit!)` moved from
      `inference/util.cljs` to `mlx.cljs`.

### 3.11 Run full test suite ✅

- [x] All core tests pass (handler, dist, gen, combinators, inference, vectorized)
- [x] `gen_clj_compat_test.cljs`: 165/165
- [x] `genjax_compat_test.cljs`: 73/73
- [x] ADEV, contracts tests: pass
- [x] Run benchmarks — no regression observed

---

## Phases 4-7: Folded into Phase 3 ✅ COMPLETE

All completed as part of Phase 3 above:
- Phase 4: Wishart atom → indexed reduce
- Phase 5: `*batched-exec?*` → deleted
- Phase 6: `js/console.warn` → pure data
- Phase 7: Top-level side effect → moved to `mlx.cljs`

---

## Phase 8: Verify and document

### 8.1 Final audit ✅

- [x] `grep 'def.*\^:dynamic' src/genmlx/` — zero hits
- [x] `grep 'atom\|swap!\|reset!' src/genmlx/` — zero hits (only volatile!)
- [x] `grep 'defn.*!' src/genmlx/` — only in `mlx.cljs`, `mlx/random.cljs`
- [x] `grep 'mx/eval!' src/genmlx/` — only in `mlx.cljs`, `mlx/random.cljs`
- [x] `grep 'mx/tidy[^-]' src/genmlx/` — only in `mlx.cljs` (+ comments)
- [x] `grep 'js/console' src/genmlx/` — zero hits

### 8.2 Performance verification ✅

- [x] `vectorized_benchmark.cljs` — throughput confirmed (67x+ speedup maintained)
- [x] `gpu_benchmark.cljs` — GPU operation timing normal
- [x] `benchmark.cljs` — end-to-end inference timing normal
- [x] Micro-benchmark: `runtime_overhead_benchmark.cljs` — 0.33ms/call per trace
      site, linear scaling, dominated by MLX array ops not JS property access
- [x] Metal buffer memory: 48KB delta for 1000 MH samples (no leak)

### 8.3 Test suite ✅

- [x] All core tests pass
- [x] `gen_clj_compat_test.cljs`: 165/165
- [x] `genjax_compat_test.cljs`: 73/73
- [x] `vectorized_test.cljs`: all pass
- [x] `vectorized_benchmark.cljs`: no regression (67x+ speedup confirmed)

### 8.4 Update documentation ✅

- [x] Update CLAUDE.md architecture layers (pure annotations, runtime in Layer 0)
- [x] Document `runtime.cljs` as the execution runtime (the re-frame analog)
- [x] Document `fresh-key` as the single entropy injection point
- [x] Update handler.cljs docstring: pure transitions only, no dispatch machinery
- [x] Update CLAUDE.md model example (no `dyn/` prefix inside gen bodies)
- [x] Update CLAUDE.md eval guidance (`materialize!` instead of `eval!`)

---

## Summary: What was eliminated

| What | Before | After P1 | After P2 | After P3 | Where |
|------|--------|----------|----------|----------|-------|
| Dynamic vars | 5 | **2** | **1** | **0** | All eliminated |
| `volatile!` | 25 sites | **1** | **1** | **1** | `runtime.cljs` only |
| `atom`/`swap!` | 1 each | 1 | 1 | **0** | Wishart removed |
| `binding` forms | 12 | **3** | **1** | **0** | All eliminated |
| `!` fns (L1-8) | 8 | **5** | **5** | **0** | All in Layer 0 |
| `mx/eval!` (L1-8) | ~100 | ~100 | ~73 | **0** | Via `materialize!` |
| `mx/tidy` (L1-8) | ~18 | ~18 | ~14 | **0** | Via `tidy-*` helpers |
| `js/Math.random` | ~10 | ~10 | **1** | **1** | `rng/fresh-key` only |
| `js/console.warn` | 3 | 3 | 3 | **0** | Pure data instead |
| Top-level effects | 1 | 1 | 1 | **0** | Moved to Layer 0 |

## Estimated effort

| Phase | Effort | Risk | Status |
|-------|--------|------|--------|
| 1. Runtime-as-parameter | 2-3 days | Low | **✅ COMPLETE** |
| 2. Eliminate PRNG vars | 2-3 days | Medium | **✅ COMPLETE** |
| 3. Push eval to Layer 0 | 3-5 days | Medium | **✅ COMPLETE** |
| 4-7. Cleanup | 1 hour | None | **✅ COMPLETE** (folded into P3) |
| 8. Verify and document | 1-2 days | None | **✅ COMPLETE** |
| **Total** | **9-14 days** | | **✅ ALL COMPLETE** |
