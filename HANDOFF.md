# GenMLX Optimization Handoff

> Last updated: 2026-02-24 (night — post loop-compilation breakthrough)
> This document provides background context for someone picking up the optimization work.
> For the actual plan and step-by-step instructions, read **TODO_OPTIMIZATION.md**.

---

## Start Here

1. Read **TODO_OPTIMIZATION.md** — it's the plan. Steps 0-1 done, Step 2 skipped, **Step 2.5 (ground truth) is the critical finding**, Steps 3-4 skipped (compile-fn already handles them), **Step 5.4 (loop compilation) is the breakthrough: 5.6x on compiled-mh**, Step 6 and production integration are next.
2. Read this file for background: what each source file does, how to run things, what's been tried, and **what we now know about the real performance model**.
3. Read **CLAUDE.md** for project conventions and test patterns.
4. Ignore OPTIMIZATION_MANIFESTO.md and TODO_SPEED.md — they contain estimates that turned out to be wrong. All accurate numbers are in TODO_OPTIMIZATION.md.

---

## The Critical Finding (Step 2.5)

**Previous assumption (WRONG):** `mx/compile-fn` re-executes the JS/SCI body on every call for functions with side effects (like `make-score-fn` which calls `p/generate`). Therefore, model lowering (eliminating SCI from the score function) would give ~8x speedup.

**Reality (VERIFIED):** `mx/compile-fn` caches the Metal kernel and skips the JS body on ALL subsequent same-shape calls, regardless of side effects (volatile!, dynamic vars, SCI, p/generate — none of it matters). The SCI body runs exactly once (the trace call), then the cached Metal program replays.

**Evidence:**
- `test/genmlx/lowering_ground_truth.cljs` — compile-fn skips body for make-score-fn (0 re-executions out of 2 subsequent calls)
- Compiled original score-fn (0.202ms) ≈ Compiled hand-written pure MLX (0.215ms) — **zero gap**
- Compiled gradient: 0.285ms original ≈ 0.292ms hand-written — **zero gap**
- compile-fn skips body even for functions with volatile!, SCI closures with reduce, etc.

**Consequence:** Steps 3-4 (model lowering for score/gradient functions) are zero-value and have been skipped.

---

## The Real Performance Model

The ground truth tests (`lowering_ground_truth.cljs`, `lowering_ground_truth_2.cljs`, `eval_cost_model.cljs`, `eval_cost_model_2.cljs`) established the actual cost model:

```
eval! cost = 0.16ms (fixed dispatch) + N_ops × 0.013ms (per graph op)

Component decomposition (compiled 7-site score-fn):
──────────────────────────────────────────────────────
Graph construction (JS/SCI):    0.012ms  (4%)
Metal dispatch + sync:          0.287ms  (96%)
──────────────────────────────────────────────────────
Total:                          0.300ms
```

**The bottleneck is Metal dispatch, not SCI.** Each eval! call costs ~0.16ms fixed overhead (kernel launch + GPU synchronization) plus ~0.013ms per graph operation. For compiled inference, SCI graph construction is 4% of time.

**What this means for optimization:**
- Eliminating SCI from the score path: **saves ~0.012ms** (negligible)
- Eliminating one eval! call: **saves ~0.16ms** (significant)
- Batching N chains into one eval!: **saves (N-1) × 0.16ms** (the winning strategy)

---

## What's Been Done

**Steps 0-1 complete.** Four bugs fixed, benchmark baselines established, Gen.jl comparison measured.

**Step 2.5 complete (ground truth).** Corrected the performance model. Key findings:
1. compile-fn already caches score/gradient functions — no lowering needed
2. eval! dispatch (0.16ms fixed) is the real bottleneck
3. Micro-batched lazy MH is a dead end (per-op cost scales linearly)
4. The triple transform `compile(vmap(grad(score-fn)))` works on the existing SCI-based functions
5. vmap composes directly with make-score-fn — no lowering needed

**Step 5 mostly complete.** eval! cost model established, micro-batching tested and ruled out, triple transform validated, **loop compilation achieves 5.6x** (Step 5.4). Remaining: 5.3 (collect-samples overhead) deprioritized since loop compilation bypasses it.

**Bugs fixed (Step 0):**
1. `mcmc.cljs:299` — `mx/random-normal` → `mx/random-uniform` in vectorized-mh accept/reject
2. `vi.cljs:323,327` — REINFORCE baseline uses `obj-val` instead of `log-q`
3. `learning.cljs:201-206` — Wake-phase ELBO variable shadowing fixed
4. `vectorized.cljs:94-99` — `mx/eval!` after resampling to prevent unbounded graph growth

**All tests pass:** 165/165 Gen.clj compat, 73/73 GenJAX compat, all core tests.

---

## What's Been Done (Since Last Handoff)

**Step 5.4: Loop compilation — THE HEADLINE RESULT (2026-02-24)**

compile-fn can cache an entire K-step MH chain as one Metal dispatch. **5.6x speedup.**

```
Compiled chain (200 steps): 13.8 ms  (0.069 ms/step)
Eager MH (200 steps):       77.9 ms  (0.390 ms/step)
Speedup:                     5.6x
```

Key challenge: compile-fn freezes random ops (both stateful and key-based).
Fix: pre-generate `[K,D]` noise + `[K]` uniforms OUTSIDE compile-fn, pass as inputs.
The compiled function indexes into pre-generated arrays at each step.

**Where it helps:** burn-in (5x), thinned collection with thin > 1 (proportional).
**Where it doesn't:** thin=1 sample collection — 1-step compiled chains are slower
than eager due to 2D array generation overhead. Falls back to eager per-step.
For `{:burn 0 :thin 1}`, no speedup over the old code.

Test: `test/genmlx/loop_compilation_test.cljs` — correctness, scaling, statistical validity.

---

## What's Left to Do

| Step | What | Expected gain | Status |
|---|---|---|---|
| **5.4→prod** | Integrate loop compilation into `compiled-mh` | 5.6x on compiled-mh | **NEXT** |
| **5.4→mala** | Apply loop compilation to MALA | ~5x on mala | Ready |
| **5.4→hmc** | Apply loop compilation to HMC leapfrog | ~3-4x on hmc | Ready |
| **5.3** | Optimize `collect-samples` loop overhead | 1.5-1.76x (lower priority now) | Deprioritized |
| **6** | Remove lazy variants + tidy discipline | Correctness, long-chain stability | Ready |
| **7** | Bun docs | Documentation only | Nearly done |
| **8** | Distribution optimizations (mlx-log-gamma) | Enables beta/gamma gradients | Ready |
| **9** | Lowered simulate/generate | 2-3x on simulate | Reassess |
| **10** | Lowered vectorized sim/gen | Blocked on 9 | Later |
| **11** | Final benchmark | Blocked on all | Later |

**Skipped (zero-value):** Steps 3 (score lowering), 4 (gradient lowering) — compile-fn handles these.
**Dead end:** Micro-batched lazy MH (Step 5.1) — measured, doesn't help.
**Done:** Loop compilation (Step 5.4) — validated, ready for production integration.

---

## What Each Source File Does (Optimization-Relevant)

### `inference/util.cljs` — Score Function Construction

- `make-score-fn` (line 30): Takes model+args+observations+addresses → returns `(fn [params] -> MLX scalar)`. Calls `p/generate` on every invocation, but **this doesn't matter** because `mx/compile-fn` caches the Metal kernel and skips the body.
- `make-vectorized-score-fn` (line 43): Same but for `[N,D]`-shaped params → `[N]`-shaped scores.
- `make-differentiable-vectorized-score-fn` (line 85): Uses matmul+one-hot for differentiable column extraction.
- `make-compiled-score-fn` (line 59): Wraps `make-score-fn` in `mx/compile-fn`. **Already at the Metal floor** (0.202ms, matching handwritten pure MLX at 0.215ms).
- `make-compiled-grad-score` (line 65): `mx/compile-fn(mx/grad(make-score-fn(...)))`. **Already at the Metal floor** (0.285ms).
- `make-compiled-val-grad` (line 71): `mx/compile-fn(mx/value-and-grad(make-score-fn(...)))`. **Already at the Metal floor** (0.325ms).
- `extract-params` (line 77): Extracts parameter values from trace as flat MLX array.
- `init-vectorized-params` (line 136): Creates `[N,D]` param matrix from N independent generates.

### `inference/mcmc.cljs` — All MCMC Variants (~1467 lines)

**MH variants:**
- `mh` (line 84): GFI-based. Calls `p/regenerate` per step. 394x slower than Gen.jl.
- `compiled-mh` (line 187): Parameter-space. Uses compiled score-fn. **0.375ms/step manual, 0.66ms/step actual** (gap is collect-samples overhead).
- `compiled-mh-lazy` (line 242): Deferred eval. **REGRESSION** (1.81x slower). To be removed (Step 6).
- `vectorized-compiled-mh` (line 307): N parallel chains. Bug 0.1 fixed.

**MALA variants:**
- `mala` (line 507): Uses `value-and-grad`. ~0.64ms/step.
- `mala-lazy` (line 570): Deferred eval. **REGRESSION** (1.15-1.18x slower). To be removed (Step 6).
- `vectorized-mala` (line 669): N parallel chains. Working, 12x effective at N=50.

**HMC variants:**
- `hmc` (line 853): Uses `mx/tidy` around leapfrog. 2.2-3.6x slower than Gen.jl.
- `hmc-lazy` (line 925): Deferred eval. **REGRESSION** (~0.5x). To be removed (Step 6).
- `vectorized-hmc` (line 1051): N parallel chains. Working, 2.5x effective at N=10.

**Other:** NUTS (line 1097), Gibbs (line 380), elliptical slice (line 412), involutive MH (line 449), MAP (line 1378).

### `mlx.cljs` — MLX Wrapper (~400 lines)

Key functions:
- `compile-fn` (line 350): Wraps `core.compile()`. **Traces JS body once, replays cached Metal program. Works regardless of side effects (volatile!, SCI, p/generate).** Can cache entire K-step loops (200-step MH chain = one Metal dispatch, 5.6x speedup). **Freezes random ops** — must pre-generate noise outside and pass as inputs. Validated 2026-02-24.
- `grad` (line 355): Wraps `core.grad()`. Automatic differentiation.
- `value-and-grad` (line 357): Both score and gradient in one call.
- `vmap` (line 360): Vectorized map. **Composes with make-score-fn directly** (validated). `compile(vmap(grad(f)))` triple transform validated on SCI-based functions.
- `tidy` (line 370): Scope-based memory management. Required to prevent Metal resource leak.
- `eval!` (line 72): Forces materialization. **THE performance bottleneck** — 0.16ms fixed dispatch + 0.013ms per graph op.

### `handler.cljs` — The Handler System (~450 lines)

Pure state transitions `(state, addr, dist) -> (value, state')`. The only mutable boundary is `volatile!` in `run-handler`. Each handler variant (simulate, generate, update, regenerate) implements a different transition function.

**For optimization:** The handler is NOT the bottleneck. Neither is SCI for compiled inference (compile-fn caches). The bottleneck is eval! dispatch.

### `dynamic.cljs` — DynamicGF (~300 lines)

Implements all GFI protocols via handlers. `simulate` = run model with simulate-handler. `generate` = run model with generate-handler.

**simulate/generate are still slow** (38x vs Gen.jl for 7-site) because compile-fn can't cache the choicemap construction (JS side effects that produce the Trace data structure). Model lowering IS still relevant for these operations.

`vsimulate` and `vgenerate` (batched variants): Change array shapes to `[N]` instead of `[]`, enabling MLX broadcasting for free parallelism.

### `vectorized.cljs` — Vectorized Trace (~150 lines)

`VectorizedTrace`: choices where leaves hold `[N]`-shaped arrays.
`systematic-resample-indices`: Two implementations — CPU (line 19) and GPU-native (line 41).
`resample-vtrace`: Reindexes all leaves by ancestor indices. Now includes `mx/eval!` after reindexing (bug 0.4 fix).

---

## Metal Resource Leak

The 499,000 Metal buffer object limit is a **hard cap** in MLX's allocator — it's a count of concurrent `MTL::Buffer` objects, not bytes. **Same on all Apple Silicon chips regardless of RAM.**

Tested on both MLX 0.30.6 (local fork) and MLX 0.25.0 (official `@frost-beta/mlx@0.4.0`):
- Both versions hit the same 499K limit
- Local fork (0.30.6) is strictly better — handles eval-every-iteration that 0.25.0 cannot
- Only reliable fix: `mx/tidy` + `mx/eval!` per iteration

**Impact:** Without tidy discipline, inference crashes at ~2K-25K iterations depending on model size. This invalidates lazy chain variants (they accumulate graph nodes by design). Confirmed by eval_cost_model.cljs Test 2 (hit 499K limit at K=20 chained compiled-fn calls × 500 trials).

---

## node-mlx Status

GenMLX uses a **local fork** of `@frost-beta/mlx` with MLX bumped from 0.25.0 to 0.30.6.

- Location: `./node-mlx/` (referenced via `"file:./node-mlx"` in package.json)
- Built locally with `npx cmake-js build`
- MLX submodule at `deps/mlx` pinned to `v0.30.6`
- Upstream (`frost-beta/node-mlx`) dormant since April 2025 (last release: v0.4.0 with MLX 0.25.0)

**No further node-mlx work needed.** N-API overhead is negligible. All transforms (grad, vmap, compile-fn, value-and-grad) compose correctly — including on SCI-based functions.

---

## Benchmark & Test Files

### Ground Truth Tests (2026-02-24) — READ THESE FIRST

| File | What it tests | Key finding |
|---|---|---|
| `test/genmlx/lowering_ground_truth.cljs` | Does compile-fn cache make-score-fn? Handwritten vs original. | compile-fn caches. Zero gap. |
| `test/genmlx/lowering_ground_truth_2.cljs` | MH step decomposition, gradient floor, simulate/generate gap. | eval! is 96% of cost. Gradients at floor. |
| `test/genmlx/eval_cost_model.cljs` | eval! cost vs graph depth. | 0.16ms fixed + 0.013ms/op. |
| `test/genmlx/eval_cost_model_2.cljs` | Micro-batched MH, vmap, triple transform, dispatch cost. | Micro-batch dead end. Triple transform works. |

### Loop Compilation Test (2026-02-24) — THE BREAKTHROUGH

| File | What it tests | Key finding |
|---|---|---|
| `test/genmlx/loop_compilation_test.cljs` | compile-fn around entire K-step MH chain | **5.6x speedup.** Pre-generated noise fixes frozen randomness. |

### Earlier Benchmarks

| File | What it tests |
|---|---|
| `test/genmlx/optimization_benchmark.cljs` | All GenMLX operations, all model sizes |
| `test/genmlx/genjl_comparison_benchmark.cljs` | Apples-to-apples vs Gen.jl |
| `test/genmlx/compile_fn_test.cljs` | compile-fn trace-once behavior (valid but incomplete — didn't test make-score-fn) |
| `test/reference/gen_jl_benchmark.jl` | Julia script for Gen.jl reference numbers |

**Run:** `bun run --bun nbb test/genmlx/<file>.cljs`

---

## How to Run Things

```bash
# Run with Bun (recommended — 1.5x faster than Node.js)
bun run --bun nbb <file.cljs>

# Run all core tests
for f in choicemap_test trace_test selection_test handler_test dist_test gen_test combinators_test inference_test; do
  bun run --bun nbb "test/genmlx/${f}.cljs"
done

# Compatibility suites (must pass 165/165 and 73/73)
bun run --bun nbb test/genmlx/gen_clj_compat_test.cljs
bun run --bun nbb test/genmlx/genjax_compat_test.cljs
```

---

## Documents in This Repo

| Document | Status | Purpose |
|---|---|---|
| **TODO_OPTIMIZATION.md** | **ACTIVE — the plan** | Step-by-step optimization plan with validated data. Steps 3-4 marked superseded, Step 2.5 is the critical finding. |
| **HANDOFF.md** | Active — background | Context for someone picking up the work (this file) |
| **CLAUDE.md** | Active — conventions | Project structure, test patterns, how to edit |
| OPTIMIZATION_MANIFESTO.md | **Outdated** | Contains wrong estimates. The architecture vision (triple transform pattern) is validated but the numbers are wrong. |
| TODO_SPEED.md | **Outdated** | Contains wrong estimates. Superseded by TODO_OPTIMIZATION.md |
| TODO.md | Mixed | General project TODO, some items marked done |

---

## Key Insight for Future Work

The eval boundary principle is correct but the boundary is **not** between SCI and MLX. It's between JS and Metal:

```
                                Cost        Optimization lever
──────────────────────────────────────────────────────────────────
JS/SCI graph construction:      0.012ms     Negligible (compile-fn handles it)
Metal dispatch (per eval!):     0.160ms     Reduce eval! count (vectorize, batch)
Metal execution (per graph op): 0.013ms     Scale with model size (GPU parallelism)
```

**Two proven strategies for amortizing dispatch:**

1. **Loop compilation** (Step 5.4): compile-fn around entire K-step chain → one dispatch for K steps. 5.6x on compiled-mh. Pre-generate noise outside, pass as input arrays.

2. **Vectorized inference** (existing): N chains per eval! → one dispatch for N chains. 12x at N=50 for MALA.

**Critical detail:** compile-fn freezes ALL random operations (stateful AND key-based). Solution: pre-generate noise arrays outside compile-fn and pass them as inputs. The compiled function indexes into pre-generated arrays at each step. This is the pattern for all future loop-compiled algorithms.

The triple transform `compile(vmap(grad(f)))` works on existing SCI-based functions. This is GenMLX's path to competitive performance: not by eliminating SCI, but by amortizing Metal dispatch.
