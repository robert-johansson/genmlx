# TODO: Metal GPU Resource Management

GenMLX hits a hard macOS kernel-level limit of **499,000 simultaneous Metal
buffer objects** during inference. This document catalogs the root causes,
every affected code path, and a prioritized plan to make GenMLX run reliably
on any Apple Silicon system.

## Table of Contents

1. [Background: The 499K Limit](#1-background-the-499k-limit)
2. [Root Cause Analysis](#2-root-cause-analysis)
3. [Available MLX Memory APIs](#3-available-mlx-memory-apis)
4. [Audit: Every Inference Loop](#4-audit-every-inference-loop)
5. [Fix Plan](#5-fix-plan)
6. [Implementation Details](#6-implementation-details)
7. [Verification](#7-verification)

---

## 1. Background: The 499K Limit

### What is it?

A macOS IOKit GPU limit on total simultaneous Metal resources (buffers, textures,
heaps) per process. MLX reads it from `iogpu.rsrc_limit` sysctl; when that key
doesn't exist (which is the case on all tested machines), MLX falls back to a
hardcoded 499,000.

**Source:** `node-mlx/deps/mlx/mlx/backend/metal/device_info.cpp:32-36`
```cpp
size_t rsrc_limit = 0;
sysctlbyname("iogpu.rsrc_limit", &rsrc_limit, &length, NULL, 0);
if (rsrc_limit == 0) {
  rsrc_limit = 499000;
}
```

### Key facts

| Question | Answer |
|----------|--------|
| Is it MLX-imposed? | No — it's Apple's Metal driver limit (Tier 2 Argument Buffer cap ~500K). |
| Can users increase it? | No. The sysctl key doesn't exist on current macOS. No boot-args or SIP settings affect it. |
| Does it vary by chip? | No. Same 499K on M1, M2, M3, M4, and all Pro/Max/Ultra variants. |
| What does it count? | Number of Metal buffer **objects**, not bytes. 499K tiny buffers or 499K huge buffers — same limit. |
| What happens when hit? | `[metal::malloc] Resource limit (499000) exceeded` — process crashes. |

### How MLX enforces it

**Source:** `node-mlx/deps/mlx/mlx/backend/metal/allocator.cpp:134-145`

When `num_resources_ >= resource_limit_`:
1. Try to release cached buffers via `buffer_cache_.release_cached_buffers()`
2. If still at limit → throw `std::runtime_error`

### Buffer lifecycle

```
JS code: mx/add, mx/multiply, etc.
  → Creates lazy mx::array (shared_ptr<ArrayDesc>) — NO Metal buffer yet
  → Graph node holds reference to inputs (keeps them alive)

mx/eval! (or implicit eval via mx/item, mx/realize):
  → Walks BFS tape, calls primitive.eval_gpu() for each node
  → ALLOCATES Metal buffers for each node's output
  → After eval, calls detach() — drops input references
  → Intermediate arrays' refcounts may drop to zero → allocator::free()

allocator::free():
  → If cache has room: RECYCLES buffer to cache (still a Metal resource!)
  → If cache is full: RELEASES buffer to Metal (decrements num_resources_)

JS garbage collection:
  → When JS wrapper is collected, calls C++ delete on mx::array
  → shared_ptr refcount drops → may trigger allocator::free()
  → GC timing is NON-DETERMINISTIC — may leave dead arrays alive
```

**The fundamental problem:** JS GC doesn't know about Metal resource pressure.
Dead arrays linger in JS heap, keeping Metal buffers alive. Even after eval,
the buffer cache retains freed buffers as Metal resources.

---

## 2. Root Cause Analysis

### 2.1 Log-gamma buffer cost ✅ RESOLVED

~~The original Lanczos approximation (`mlx-log-gamma`) created ~21-34 Metal
buffers per call.~~ **Resolved:** Native `mx/lgamma` (contributed upstream in
[MLX PR #3181](https://github.com/ml-explore/mlx/pull/3181)) reduces each
log-gamma call to **1 Metal buffer** — a single GPU kernel dispatch.

**Current impact on distribution log-prob (with native lgamma):**

| Distribution | `mx/lgamma` calls | Additional ops | **Total buffers per log-prob** |
|-------------|-------------------|----------------|-------------------------------|
| Gaussian | 0 | ~8 | **~8** |
| Poisson | 1 | ~6 | **~7** |
| Gamma | 1 | ~10 | **~11** |
| Beta | 3 | ~10 | **~13** |
| Dirichlet | k+1 | ~10+k | **(k+1) + 10 + k** |
| Inv-Gamma | 1 | ~10 | **~11** |
| Neg-Binomial | 3 | ~8 | **~11** |
| Binomial | 3 | ~8 | **~11** |
| Student-t | 2 | ~10 | **~12** |

### 2.2 Handler score accumulation creates buffers per trace site

**Source:** `src/genmlx/handler.cljs`

Each `dyn/trace` call in a model dispatches to the handler, which:

```clojure
;; simulate-transition (line 77-86):
(update :score #(mx/add % lp))        ;; 1 new buffer

;; generate-transition (line 88-99):
(update :score  #(mx/add % lp))       ;; 1 new buffer
(update :weight #(mx/add % lp))       ;; 1 new buffer
```

For a model with S trace sites, score accumulation alone creates S buffers
(simulate) or 2S buffers (generate with constraints).

### 2.3 Inference loops don't clean up between iterations

The generic `collect-samples` loop (`kernel.cljs:100-124`) has **no** `mx/tidy`
or `mx/eval!` calls. Every iteration's arrays accumulate.

**Example: MH on 5-site Beta-Bernoulli model**

Per MH iteration (with native `mx/lgamma` — Phase 3):
- `regenerate` runs the model → 5 sites × ~13 buffers/Beta-log-prob = 65 buffers
- Score accumulation: 5 × `mx/add` = 5 buffers
- Weight computation: ~5 buffers
- **Total: ~75 Metal buffers per iteration**

Without Phase 2 `mx/eval!` + `mx/clear-cache!`: 75 × 500 = 37,500 buffers
(previously 187,500 with JS Lanczos). With Phase 2 cleanup each iteration,
buffers are recycled and never accumulate.

For Gaussian models (~8 buffers/log-prob): ~65 buffers/iteration — comparable.
Phase 2 cleanup makes iteration count effectively unlimited for all models.

### 2.4 Buffer cache retains freed buffers

Even when MLX's C++ layer calls `allocator::free()`, the buffer goes to a
**recycling cache** (not released to Metal) unless the cache is full:

```cpp
// allocator.cpp — MetalAllocator::free()
if (get_cache_memory() < max_pool_size_) {
    buffer_cache_.recycle_to_cache(buf);   // Still counts as a Metal resource!
} else {
    num_resources_--;
    buf->release();
}
```

The cache is only cleaned when:
1. A new allocation triggers pressure check (`num_resources_ >= resource_limit_`)
2. `clearCache()` is called explicitly
3. `setCacheLimit()` is set low enough to trigger eviction

---

## 3. Available MLX Memory APIs

### Already exposed in GenMLX (`src/genmlx/mlx.cljs`)

| ClojureScript | Purpose |
|---------------|---------|
| `(mx/tidy f)` | Runs `f`, deletes all MLX arrays created during `f` **except** the return value |
| `(mx/dispose! a)` | Immediately deletes the JS-side array wrapper (triggers C++ destructor) |
| `(mx/eval! & arrs)` | Forces computation graph materialization; detaches graph nodes, allowing intermediate buffers to be freed |

### Available in node-mlx but **NOT yet exposed** in GenMLX

**Source:** `node-mlx/src/memory.cc`

| node-mlx JS API | Purpose | Priority |
|-----------------|---------|----------|
| `core.getActiveMemory()` | Bytes of Metal buffers currently in use (not cached) | High |
| `core.getCacheMemory()` | Bytes of Metal buffers in recycling cache | High |
| `core.getPeakMemory()` | High-water mark of active memory | Medium |
| `core.resetPeakMemory()` | Reset high-water mark | Low |
| `core.setMemoryLimit(n)` | Max active memory (bytes) before eval starts waiting | Medium |
| `core.setCacheLimit(n)` | Max cache size (bytes). **Set to 0 to disable caching entirely.** | High |
| `core.clearCache()` | Release all cached buffers immediately. **Directly reduces `num_resources_`.** | High |
| `core.setWiredLimit(n)` | Pin buffers in physical memory (macOS 15+) | Low |

**Source:** `node-mlx/src/metal.cc`

| node-mlx JS API | Purpose | Priority |
|-----------------|---------|----------|
| `core.metal.deviceInfo()` | Returns map including `resource_limit` (499000) | Medium |

**Source:** `node-mlx/src/array.cc`

| node-mlx JS API | Purpose | Priority |
|-----------------|---------|----------|
| `core.getWrappersCount()` | Number of live JS-wrapped array objects (debugging) | Low |

---

## 4. Audit: Every Inference Loop

Comprehensive audit of every loop that creates MLX arrays, whether it uses
`mx/tidy` or `mx/eval!`, and estimated Metal buffers per iteration.

Legend: **+** = present, **-** = absent

### 4.1 Importance Sampling (`inference/importance.cljs`)

| Function | Loop type | Iters | tidy | eval! | Buffers/iter | Risk |
|----------|-----------|-------|------|-------|-------------|------|
| `importance-sampling` (line 23) | `mapv` over N generates | N | - | - | ~8-73 (model-dependent) | **HIGH** |
| `importance-resampling` (line 36) | `mapv` over resample keys | N | - | - | ~2-4 | Medium |
| `vectorized-importance-sampling` (line 63) | Single batched call | 1 | - | - | ~8-12 | Low |

**Problem:** `importance-sampling` runs N sequential `generate` calls with no
cleanup. For 500 samples on a Beta model: 500 × ~73 × 5 sites ≈ 182K buffers.

### 4.2 SMC (`inference/smc.cljs`)

| Function | Loop type | Iters | tidy | eval! | Buffers/iter | Risk |
|----------|-----------|-------|------|-------|-------------|------|
| `smc-init-step` (line ~60) | `mapv` N generates | N | - | - | ~8-12 | High |
| `smc-rejuvenate` (line ~80) | `mapv`→`reduce` MH steps | N×K | - | + (line 104) | ~4-6 | Medium |
| `smc-step` (line ~120) | `mapv` update particles | N | - | - | ~8-12 | High |
| `smc` main loop (line ~170) | `loop` over T timesteps | T | - | - | ~20-40 | **CRITICAL** |
| `csmc` main loop (line ~230) | `loop` over T timesteps | T | - | - | ~20-40 | **CRITICAL** |
| `vsmc-rejuvenate` (line ~310) | `loop` MH steps | K | + (line 317) | + (line 317) | ~8-10 | Low |
| `vsmc` main loop (line ~350) | `loop` over T timesteps | T | - | + (line 367) | ~12-16 | Low |

**Problem:** Scalar SMC (`smc`, `csmc`) creates T × N traces without cleanup.
T=100, N=100 = 10,000 traces → 100K+ buffers minimum.

### 4.3 SMCP3 (`inference/smcp3.cljs`)

| Function | Loop type | Iters | tidy | eval! | Buffers/iter | Risk |
|----------|-----------|-------|------|-------|-------------|------|
| `smcp3-init` (line ~60) | `mapv` N generates | N | - | - | ~8-12 | High |
| `smcp3-step` (line ~90) | `mapv` N edits | N | - | - | ~8-12 | High |
| `smcp3` main loop (line ~150) | `loop` over T timesteps | T | - | - | ~20-40 | **CRITICAL** |

### 4.4 MCMC (`inference/mcmc.cljs`)

| Function | Loop type | Iters | tidy | eval! | Buffers/iter | Risk |
|----------|-----------|-------|------|-------|-------------|------|
| `mh-step` (line 44) | Single step | 1 | - | - | ~2-3 | - |
| `mh` (line 57) | via `collect-samples` | burn+thin×N | - | - | ~3-4 | **HIGH** |
| `mh-custom-step` (line 80) | Single step | 1 | - | + (line 113) | ~4-6 | Low |
| `mh-custom` (line 120) | via `collect-samples` | burn+thin×N | - | - | ~4-6 | **HIGH** |
| `compiled-mh-step` (line ~150) | Single step | 1 | - | + (line 155) | ~2-3 | Low |
| `run-loop-compiled-mh` burn (line ~195) | `loop` block iters | ⌈burn/block⌉ | - | + (line 201) | ~50-80 | Low |
| `run-loop-compiled-mh` collect (line ~205) | `loop` samples | N | + (line 208) | + (line 215) | ~10-15 | **Low** |
| `vectorized-mh-step` (line ~280) | Single step | 1 | - | + (lines 290-298) | ~20-30 | Low |
| `vectorized-compiled-mh` (line ~330) | Main loop | burn+thin×N | - | - | ~80-120 | **HIGH** |
| `gibbs-step-with-support` (line ~360) | Single address | 1 | - | + (line 365) | ~4-6 | Low |
| `gibbs` (line ~390) | via `collect-samples` | burn+thin×N | - | - | ~4-6 | **HIGH** |
| `involutive-mh-step` (line ~430) | Single step | 1 | - | + (line 435) | ~4-6 | Low |
| `leapfrog-step` (line ~800) | Single step | 1 | + (line 802) | + (line 809) | ~6-8 | **Low** |
| `leapfrog-trajectory` (line ~820) | `loop` L steps | L | - | - | ~6-8 | Low (tidy per step) |
| `leapfrog-trajectory-fused` (line ~830) | `loop` L-1 interior | L-1 | - | - | ~12-16 | Medium |
| `hmc-step` (line ~850) | Single step | 1 | + (line 857) | + (line 862) | ~20-30 | **Low** |
| `make-compiled-hmc-chain` (line ~870) | Compiled K steps | K | - | + (line 918) | ~100+ | Low |
| `run-loop-compiled-hmc` burn (line ~930) | `loop` block iters | ⌈burn/block⌉ | - | + (line 938) | ~100+ | Low |
| `run-loop-compiled-hmc` collect (line ~940) | `loop` samples | N | + (line 945) | + (line 952) | ~30-50 | **Low** |
| `hmc` (line ~960) | Main entry | burn+thin×N | - | - | varies | Low (delegates) |
| `find-reasonable-epsilon` (line ~1000) | `loop` doubling | ~100 max | - | - | ~6-8 | Medium |
| `dual-averaging-warmup` (line ~1050) | `loop` warmup | n-warmup | - | - | ~6-8 | Medium |
| `mala-step` (line ~550) | Single step | 1 | - | + (line ~560) | ~10-15 | Low |
| `run-loop-compiled-mala` collect (line ~570) | `loop` samples | N | + (line 574) | + (line ~580) | ~10-15 | **Low** |
| `vectorized-mala-step` (line ~660) | Single step | 1 | - | + (lines 662-686) | ~20-30 | Low |
| `vectorized-mala` (line ~700) | Main loop | burn+thin×N | - | - | ~80-120 | **HIGH** |
| `nuts-step` (line ~1100) | Single step | 1 | + (via leapfrog) | + | ~30-50 | Low |
| Elliptical slice (line ~1680) | via `collect-samples` | burn+thin×N | + (line 1691) | + | ~4-6 | Low |

### 4.5 VI (`inference/vi.cljs`)

| Function | Loop type | Iters | tidy | eval! | Buffers/iter | Risk |
|----------|-----------|-------|------|-------|-------------|------|
| `vi` main loop (line ~78) | `loop` iterations | N | + (line 99) | + (line 99) | ~10-15 | **Low** |
| `compiled-vi` main loop (line ~145) | `loop` iterations | N | + (line 182) | + (line 182) | ~10-15 | **Low** |
| `programmable-vi` loop (line ~380) | `loop` iterations | N | - | + (line 393) | ~6-10 | Medium |
| `compiled-programmable-vi` loop (line ~430) | `loop` iterations | N | + (line 440) | + (line 440) | ~10-15 | **Low** |

### 4.6 ADEV (`inference/adev.cljs`)

| Function | Loop type | Iters | tidy | eval! | Buffers/iter | Risk |
|----------|-----------|-------|------|-------|-------------|------|
| `adev-gradient` (line ~130) | `mapv` n-samples | n-samples | - | - | ~4-6 | Medium |
| `adev-optimize` loop (line ~150) | `loop` iterations | N | - | + (line 159) | ~4-6 | Medium |
| `compiled-adev-optimize` loop (line ~280) | `loop` iterations | N | + (line 288) | + (line 289) | ~8-12 | **Low** |

### 4.7 Learning (`learning.cljs`)

| Function | Loop type | Iters | tidy | eval! | Buffers/iter | Risk |
|----------|-----------|-------|------|-------|-------------|------|
| `adam-step` (line ~80) | Single step | 1 | - | + (line 92) | ~3-4 | Low |
| `train` loop (line ~120) | `loop` iterations | N | - | + (line 125) | ~3-4 | Low |
| `wake-sleep` outer loop (line ~270) | `loop` cycles | iterations | - | - | ~20-30 | Medium |
| `wake-sleep` wake inner (line ~290) | `loop` steps | wake-steps | - | + (line 296) | ~6-8 | Low |
| `wake-sleep` sleep inner (line ~300) | `loop` steps | sleep-steps | - | + (line 306) | ~4-6 | Low |

### 4.8 Kernel (`inference/kernel.cljs`)

| Function | Loop type | Iters | tidy | eval! | Buffers/iter | Risk |
|----------|-----------|-------|------|-------|-------------|------|
| `collect-samples` (line 100) | `loop` | burn+thin×N | **+** (Phase 4) | **+** (Phase 4) | varies | **Low** ✅ |
| `run-kernel` (line 126) | Delegates to `collect-samples` | burn+thin×N | + | + | varies | **Low** ✅ |

**`collect-samples` fixed in Phase 4** via `tidy-step` + `with-resource-guard`.
Each step's intermediates are disposed by `mx/tidy`; trace arrays are preserved.
Wrapper growth: ~0.3/iter. Verified at 15,000 iterations (10K samples + 5K burn).

### 4.9 Amortized Inference (`inference/amortized.cljs`)

| Function | Loop type | Iters | tidy | eval! | Buffers/iter | Risk |
|----------|-----------|-------|------|-------|-------------|------|
| `train-proposal!` (line ~80) | `mapv` iterations | N | - | - | ~6-8 | Medium |
| `neural-importance-sampling` (line ~110) | `mapv` N samples | N | - | - | ~8-12 | High |

---

## 5. Fix Plan

Prioritized by impact and difficulty.

### Phase 1: Foundation — Expose Memory APIs (Easy, Immediate) ✅ DONE

**Goal:** Give users and inference code the ability to monitor and control Metal
resource usage.

**Completed** in commit `587f5dc` on `gpu-resource-management` branch.

All 14 functions added to `src/genmlx/mlx.cljs` (lines 108-139), with tests in
`test/genmlx/memory_test.cljs` (25 assertions, all passing).

#### Task 1.1: Add memory management functions to `mlx.cljs` ✅

```clojure
;; Monitoring
(mx/get-active-memory)    ;; bytes of Metal buffers in use
(mx/get-cache-memory)     ;; bytes in recycling cache
(mx/get-peak-memory)      ;; high-water mark
(mx/reset-peak-memory!)   ;; reset high-water mark
(mx/get-wrappers-count)   ;; live JS-wrapped array objects

;; Control
(mx/set-memory-limit! n)  ;; max active memory before eval waits (returns prev)
(mx/set-cache-limit! n)   ;; max cache size; 0 disables caching (returns prev)
(mx/set-wired-limit! n)   ;; pin buffers in physical memory, macOS 15+ (returns prev)
(mx/clear-cache!)         ;; release all cached Metal buffers immediately
```

#### Task 1.2: Add Metal device info ✅

```clojure
(mx/metal-is-available?)  ;; => true on Apple Silicon
(mx/metal-device-info)    ;; => {:architecture "applegpu_g14g"
                          ;;     :device-name "Apple M2"
                          ;;     :memory-size 17179869184
                          ;;     :max-buffer-length 9534832640
                          ;;     :max-recommended-working-set-size 12713115648
                          ;;     :resource-limit 499000}
```

Note: keys are kebab-case (`:resource-limit` not `:resource_limit`).

#### Task 1.3: Add a resource pressure diagnostic ✅

```clojure
(mx/memory-report)        ;; => {:active-bytes 0
                          ;;     :cache-bytes 0
                          ;;     :peak-bytes 0
                          ;;     :wrappers 14
                          ;;     :resource-limit 499000}
```

### Phase 2: Fix All Inference Loops (Medium, Highest Impact) ✅ DONE

**Goal:** Every inference loop properly cleans up Metal buffers between
iterations. No inference algorithm should ever hit the 499K limit for
reasonable problem sizes.

**Design insight:** The original plan proposed `mx/tidy` wrapping step functions,
but tidy only protects the *direct return value* — it can't protect MLX arrays
nested inside ClojureScript maps (Trace records, `{:trace :weight}` results).
Since most inference loops deal with Traces (CLJS maps), the actual strategy is:

- **Trace-based loops** → `mx/eval!` on key arrays + periodic `mx/clear-cache!`
- **Param-array loops** → already have `mx/tidy` (compiled-mh, compiled-mala, HMC)

After eval, the computation graph detaches. Intermediate arrays' JS wrappers
become GC-eligible when step-fn returns. V8 collects them under allocation
pressure. `clear-cache!` then releases Metal buffers from the cache.

**Verified:** All tests pass — core tests, 165/165 Gen.clj compat, 73/73 GenJAX
compat, plus new resource stress test (`test/genmlx/resource_test.cljs`).

#### Task 2.1: Add `eval-state!` + periodic `clear-cache!` to `collect-samples` ✅

Added `u/eval-state!` helper to `inference/util.cljs` that evaluates key arrays
in both MLX arrays (param vectors) and Trace records (eval score + retval).

Added to `collect-samples` loop body (`kernel.cljs`):
```clojure
_  (u/eval-state! state)
_  (when (zero? (mod i 50)) (mx/clear-cache!))
```

This single change fixes: `mh`, `mh-custom`, `gibbs`, `elliptical-slice`,
`involutive-mh`, and all kernel combinators (`chain`, `repeat`, `cycle`, `mix`).

#### Task 2.2: Fix `importance-sampling` ✅

Added `mx/eval!` after each generate in the `mapv` (`importance.cljs`):
```clojure
(let [r (p/generate model args observations)]
  (mx/eval! (:weight r) (:score (:trace r)))
  r)
```

#### Task 2.3: Fix SMC loops ✅

- `smc-init-step`: eval per generate
- `smc-step`: eval per update
- `smc` main loop: periodic `clear-cache!` every 10 timesteps
- `csmc` init step: eval per generate (both other-results and ref-result)
- `csmc` subsequent step: eval per update
- `csmc` main loop: periodic `clear-cache!` every 10 timesteps

#### Task 2.4: Fix SMCP3 loops ✅

- `smcp3-init`: eval per particle (both proposal and standard paths)
- `smcp3-step`: eval per particle (both forward-kernel and standard paths)
- `smcp3` main loop: periodic `clear-cache!` every 10 timesteps

#### Task 2.5: Fix vectorized MCMC loops ✅

- `vectorized-compiled-mh`: periodic `clear-cache!` every 50 iterations
  (step functions already eval internally)
- `vectorized-mala`: periodic `clear-cache!` every 50 iterations
  (step functions already eval internally)

#### Task 2.6 (deferred): Fix step-size adaptation loops

`find-reasonable-epsilon` and `dual-averaging-warmup` have bounded iterations
(~100 max) and leapfrog-step already uses tidy. Medium risk — deferred to
future phase.

#### Task 2.7 (deferred): Fix `programmable-vi`

Already has `mx/eval!` on loss/grad. Medium risk — deferred to future phase.

### Phase 3: Reduce Per-Operation Buffer Count ✅ DONE

**Goal:** Reduce the number of Metal buffers created per distribution log-prob
call, making all inference algorithms fundamentally cheaper.

**Result:** Native `lgamma` kernel contributed upstream to MLX, eliminating the
Lanczos approximation entirely. Each log-gamma call now creates **1 buffer**
instead of **~23**, a **~23x reduction** per call. For Beta (3 calls):
~73 → ~13 buffers. For all 16 calls across 8 distributions: ~752 → ~16.

#### Task 3.1: Contribute `lgamma` to MLX ✅

**PR:** [ml-explore/mlx#3181](https://github.com/ml-explore/mlx/pull/3181)

Added native `lgamma` and `digamma` as unary ops across all MLX backends:
- **Metal:** Lanczos g=5 kernel (`lgamma.h`) + asymptotic digamma
- **CPU:** `std::lgamma` + custom digamma
- **CUDA:** built-in `::lgamma` + custom digamma
- **Autograd:** `grad(lgamma) = digamma`
- **Python:** `mx.lgamma()`, `mx.digamma()` with docstrings

19 files, ~510 lines. All 236 MLX tests pass (3369 assertions).

Node-mlx bindings updated in `robert-johansson/node-mlx` (commit `e4aeb03`).

#### Task 3.2: Reduce Lanczos coefficients — SUPERSEDED

~~Reduce from g=7/8-term to g=5/6-term.~~ No longer needed — native `mx/lgamma`
replaces the entire ClojureScript Lanczos approximation.

The JS-side `log-gamma` (used only for Wishart scalar computation) was updated
to g=5/6-term Numerical Recipes coefficients as part of the interim optimization.

#### Task 3.3: Cache Lanczos constant arrays — SUPERSEDED

~~Cache `mx/scalar` constants at module level.~~ No longer needed — the cached
constants (`LANCZOS-G`, `LANCZOS-T-OFFSET`, `LANCZOS-C0`, `LANCZOS-COEFFS`)
and the `mlx-log-gamma` function were removed from `dist.cljs` entirely.

All 16 call sites now use `(mx/lgamma ...)` directly.

#### Task 3.4: Explore fused log-gamma via `mx/compile-fn` — NOT NEEDED

Native `lgamma` is already a single Metal kernel dispatch. `compile-fn` would
add no benefit.

### Phase 4: Global Safety Mechanisms ✅ DONE

**Goal:** Even if individual loops have bugs, the system should not crash.
Long MH chains (7000+ iterations) must complete without hitting the 499K limit.

**Key discoveries during implementation:**

1. **`mx/array?` was broken in nbb.** ClojureScript's `object?` returns `false`
   for N-API objects in Bun's JavaScriptCore. This caused `eval-state!` (Phase 2)
   to silently skip evaluating trace arrays — computation graphs never detached.
2. **`Bun.gc()` does NOT trigger N-API destructor callbacks.** MLX array wrappers
   are never freed by garbage collection in Bun. Only `mx/tidy` and `mx/dispose!`
   can release Metal buffers.
3. **`mx/tidy` cannot walk CLJS persistent data structures.** It only sees arrays
   in plain JS objects/arrays. Wrapping a step function directly in `tidy` causes
   segfaults because it disposes arrays it can't find in the CLJS return value.

**Solution:** `tidy-step` — a wrapper that runs the step inside `mx/tidy`, but
first evaluates all trace arrays (detaching computation graphs) and returns them
as a JS array that tidy CAN walk. This reduces wrapper growth from ~200/iter to
~0.3/iter, making iteration count effectively unlimited.

#### Task 4.1: Fix `mx/array?` for N-API objects ✅

Removed the `object?` check from `mx/array?` (`mlx.cljs`). N-API objects report
`typeof === "object"` in V8 but ClojureScript's `object?` uses `identical?` which
fails for these objects. The fix matches the existing workaround in `vmap.cljs`.

#### Task 4.2: Enhanced `eval-state!` with `collect-trace-arrays` ✅

`collect-trace-arrays` walks the choicemap tree and collects ALL MLX arrays
(choices + score + retval). `eval-state!` now evaluates them all in one
`(apply mx/eval! arrays)` call, detaching all computation graphs.

#### Task 4.3: `tidy-step` for automatic intermediate disposal ✅

```clojure
(defn tidy-step [step-fn state key]
  ;; 1. Run step inside mx/tidy
  ;; 2. Eval all trace arrays (detaches graphs from intermediates)
  ;; 3. Return arrays as JS array for tidy to preserve
  ;; Result: tidy disposes all intermediates, trace arrays survive
  ...)
```

Applied to `collect-samples` in `kernel.cljs`. This fixes all algorithms that
use `collect-samples`: `mh`, `mh-custom`, `gibbs`, `elliptical-slice`,
`involutive-mh`, and all kernel combinators.

#### Task 4.4: `with-resource-guard` + cache limit ✅

```clojure
(mx/set-cache-limit! (* 256 1024 1024))  ;; Set at module load

(defn with-resource-guard [f]
  ;; Sets cache-limit=0 during f, restores afterward.
  ;; Freed buffers release immediately instead of being cached.
  ...)
```

`collect-samples` is wrapped in `with-resource-guard`.

#### Task 4.5: `dispose-trace!` for explicit cleanup ✅

```clojure
(defn dispose-trace! [trace-or-traces]
  ;; Collects all MLX arrays, deduplicates by identity (handles shared
  ;; observation arrays), disposes each. Safe for collections of traces.
  ...)
```

#### Task 4.6: `force-gc!` helper ✅

Calls `Bun.gc(true)` or `global.gc()` if available. Note: ineffective for
freeing MLX wrappers in Bun (see discovery #2), but retained for Node.js
compatibility and future Bun fixes.

**Verification:** Stress test (`test/genmlx/stress_test.cljs`) passes all 3 sections:
- §7.2: MH 5000 samples + 2000 burn on Beta-Bernoulli (10 obs) — PASS
- §7.5: MH 10000 samples + 5000 burn on Beta — PASS
- §7.3: SMC 100 particles × 20 timesteps — PASS

All core tests pass, 165/165 Gen.clj compat, 73/73 GenJAX compat (1 flaky).

### Phase 5: Documentation and User Guidance (Easy, Important)

#### Task 5.1: Document the resource limit in README

Add a section explaining:
- The 499K limit exists on all Apple Silicon
- How to monitor: `(mx/memory-report)`
- How to mitigate: `(mx/set-cache-limit! ...)`, `(mx/clear-cache!)`
- Which algorithms are resource-safe vs. which need care

#### Task 5.2: Add a troubleshooting guide

```
Q: I'm getting "[metal::malloc] Resource limit (499000) exceeded"
A: This means too many Metal buffers are alive simultaneously.
   Solutions:
   1. Use vectorized inference (vectorized-importance-sampling, vsmc)
   2. Reduce sample count
   3. Add (mx/set-cache-limit! (* 128 1024 1024)) at program start
   4. Split long inference runs across multiple calls with (mx/clear-cache!)
   5. Use compiled inference (compiled-mh, hmc, nuts) which manage resources
```

---

## 6. Implementation Details

### 6.1 The `mx/tidy` + `mx/eval!` pattern

This is the established pattern used in compiled-MH, HMC, MALA, and VI:

```clojure
;; CORRECT: tidy wraps the computation, eval materializes before tidy cleans up
(let [result (mx/tidy
               (fn []
                 (let [r (expensive-computation)]
                   (mx/eval! r)  ;; Force evaluation INSIDE tidy
                   r)))]         ;; Return r; tidy deletes everything else
  ;; result's buffers survive; all intermediates are deleted
  ...)
```

**Why both are needed:**
- `mx/eval!` alone materializes the graph and allows intermediate nodes to be
  freed via refcount, but **JS wrappers for intermediate arrays still exist**
  until GC runs.
- `mx/tidy` alone would try to delete arrays, but **unevaluated arrays can't
  safely be deleted** (they might still be needed as inputs to other nodes).
- Together: `eval!` materializes and detaches the graph, then `tidy` immediately
  deletes the JS wrappers for all arrays created during the function **except**
  the return value.

### 6.2 What can go wrong with tidy

**Risk 1: Deleting arrays that are still referenced outside the tidy scope.**
This happens if you create an array inside tidy, store it in a data structure
outside tidy (e.g., an atom), but don't return it. The tidy scope doesn't know
about external references.

**Mitigation:** Only use tidy for self-contained computations where all outputs
are in the return value.

**Risk 2: Deleting module-level constant arrays.**
If Lanczos coefficients are cached as `def`s, and a tidy scope calls
`mlx-log-gamma`, the tidy scope might delete the constants.

**Mitigation:** Test this. Node-mlx's tidy only tracks arrays created **during**
the tidy function via `g_tidy_arrays`. Module-level arrays were created before
the tidy scope, so they should be safe. Verify by running a tidy-wrapped
log-gamma and checking that the constants are still valid afterward.

### 6.3 How `collect-samples` tidy should work for MH

The key subtlety: when MH rejects a proposal, `step-fn` returns the **same**
trace object (identity check). That trace was created in a previous iteration
(or the initial generate). Its arrays were created **outside** the current tidy
scope, so tidy won't delete them. This is correct behavior.

When MH accepts, `step-fn` returns a **new** trace with new arrays. These were
created inside the tidy scope. We need to `eval!` them before tidy cleans up.
After tidy, only the new trace's arrays survive; the rejected alternatives are
deleted.

### 6.4 Estimated impact of Phase 2 + Phase 3 fixes

| Algorithm | Original (no fixes) | Phase 2 (eval/clear) | Phase 2+3 (native lgamma) |
|-----------|--------------------|-----------------------|--------------------------|
| MH (500 iter, 5-site Gaussian) | ~32,500 | ~65/iter | ~65/iter (unchanged) |
| MH (500 iter, 5-site Beta) | ~187,500 | ~375/iter | **~65/iter** |
| IS (500 samples, 5-site Gaussian) | ~20,000 | ~40/sample | ~40/sample (unchanged) |
| IS (500 samples, 5-site Beta) | ~182,500 | ~365/sample | **~65/sample** |
| SMC (T=100, N=100, Gaussian) | ~400,000 | ~4,000/step | ~4,000/step (unchanged) |
| collect-samples (generic) | O(N×S×C) | O(S×C) | O(S×C) with C ~5x smaller |

Phase 3 (native lgamma) has no effect on Gaussian models (they don't use
log-gamma) but dramatically reduces buffer counts for gamma-family distributions
(Beta, Gamma, Poisson, Student-t, Dirichlet, Inv-Gamma, Neg-Binomial, Binomial).

Where S = trace sites, C = buffers per log-prob, N = iterations.

---

## 7. Verification

### 7.1 Test: Resource usage monitoring ✅

Phase 1 APIs verified via `test/genmlx/memory_test.cljs` (25/25 pass):

```clojure
(println "Resource limit:" (:resource-limit (mx/metal-device-info)))
(println "Active memory:" (mx/get-active-memory))
(println "Cache memory:" (mx/get-cache-memory))
(println "Wrappers:" (mx/get-wrappers-count))
(println "Full report:" (mx/memory-report))
```

### 7.2 Test: MH doesn't hit resource limit

After Phase 2.1, verify MH runs indefinitely on a Beta-Bernoulli model:

```clojure
(def model (gen [data] ...))  ;; Beta-Bernoulli, 10 sites
(def traces (mcmc/mh {:samples 5000 :burn 2000
                       :selection (sel/select :p)}
                      model [data] observations))
;; Should complete without "[metal::malloc] Resource limit" error
```

### 7.3 Test: SMC doesn't hit resource limit

After Phase 2.3, verify scalar SMC runs with many timesteps:

```clojure
(smc/smc {:particles 100 :timesteps 50}
         model timestep-args timestep-observations)
;; Should complete without resource error
```

### 7.4 Test: Native lgamma accuracy ✅

Verified via `test/genmlx/lanczos_test.cljs` (22/22 pass):
- 6 log-gamma accuracy tests against reference values (tolerance 1e-5)
- 4 distribution log-prob tests (Beta, Gamma, Poisson, Student-t)
- 10 vectorized correctness tests (Gamma + Beta on `[5]`-shaped inputs)
- 2 shape checks

All core tests pass, 165/165 Gen.clj compat, GenJAX compat unchanged.

### 7.5 Stress test: Long inference chains

```clojure
;; This should work after all fixes:
(let [model (gen [] (let [x (dyn/trace :x (dist/beta-dist 2 2))] x))
      obs (cm/choicemap)
      traces (mcmc/mh {:samples 10000 :burn 5000 :selection (sel/select :x)}
                       model [] obs)]
  (println "Completed" (count traces) "MH iterations on Beta model"))
```

### 7.6 Run existing test suites

All existing tests must continue to pass:

```bash
# Core tests
for f in choicemap_test trace_test selection_test handler_test dist_test gen_test combinators_test inference_test; do
  bun run --bun nbb "test/genmlx/${f}.cljs"
done

# Compatibility suites
bun run --bun nbb test/genmlx/gen_clj_compat_test.cljs   # 165/165
bun run --bun nbb test/genmlx/genjax_compat_test.cljs    # 73/73

# Conjugate posterior correctness
bun run --bun nbb test/genmlx/conjugate_posterior_test.cljs
bun run --bun nbb test/genmlx/conjugate_bb_test.cljs
bun run --bun nbb test/genmlx/conjugate_gp_test.cljs
```
