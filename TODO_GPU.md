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

### 2.1 Lanczos gamma approximation creates ~21 Metal buffers per call

**Source:** `src/genmlx/dist.cljs:57-74` — `mlx-log-gamma`

Every call to `mlx-log-gamma` creates these MLX operations:

| Step | Operation | Buffers |
|------|-----------|---------|
| `x' = x - 1` | `mx/subtract` | 1 |
| `t = x' + 7.5` | `mx/add` + `mx/scalar` | 2 |
| 8 Lanczos coefficients | 8 × (`mx/scalar` + `mx/divide` + `mx/add`) | 24 |
| Final: `log(2π)/2 + (x'+0.5)*log(t) - t + log(s)` | `mx/add` × 3 + `mx/multiply` + `mx/log` × 2 + `mx/negative` | 7 |
| **Total** | | **~34** |

(Some `mx/scalar` calls for constants may be cached, actual count is ~21-34.)

**Impact on distribution log-prob:**

| Distribution | `mlx-log-gamma` calls | Additional ops | **Total buffers per log-prob** |
|-------------|----------------------|----------------|-------------------------------|
| Gaussian | 0 | ~8 | **~8** |
| Poisson | 1 | ~6 | **~27** |
| Gamma | 1 | ~10 | **~31** |
| Beta | 3 | ~10 | **~73** |
| Dirichlet | k+1 | ~10+k | **~21(k+1) + 10 + k** |
| Inv-Gamma | 1 | ~10 | **~31** |
| Fisher-F | 3 | ~12 | **~75** |
| Beta-Binomial | 4 | ~10 | **~94** |
| Beta-Neg-Binomial | 4 | ~10 | **~94** |

MLX does **not** have a native `lgamma` operation. The Lanczos approximation is
the only available path for log-gamma on MLX arrays. (Confirmed: searched all of
`mlx/ops.h` and `node_mlx.node.d.ts` — no lgamma, digamma, or polygamma.)

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

Per MH iteration:
- `regenerate` runs the model → 5 sites × ~73 buffers/Beta-log-prob = 365 buffers
- Score accumulation: 5 × `mx/add` = 5 buffers
- Weight computation: ~5 buffers
- **Total: ~375 Metal buffers per iteration**

At 500 iterations: 375 × 500 = **187,500 buffers**
With 200 burn-in: 375 × 700 = **262,500 buffers** → approaching limit

For Gaussian models (~8 buffers/log-prob): 375 → ~65 buffers/iteration,
allowing ~7,600 iterations. Still limited for serious inference.

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
| `collect-samples` (line 100) | `loop` | burn+thin×N | **-** | **-** | varies | **CRITICAL** |
| `run-kernel` (line 126) | Delegates to `collect-samples` | burn+thin×N | - | - | varies | **CRITICAL** |

**`collect-samples` is the single most impactful function to fix.** It's the
generic loop used by `mh`, `mh-custom`, `gibbs`, `involutive-mh`, `elliptical-slice`,
and all kernel combinators (`chain`, `repeat-kernel`, `cycle-kernels`, `mix-kernels`).
Adding `mx/tidy` + `mx/eval!` here fixes all of them at once.

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

### Phase 2: Fix All Inference Loops (Medium, Highest Impact)

**Goal:** Every inference loop properly cleans up Metal buffers between
iterations. No inference algorithm should ever hit the 499K limit for
reasonable problem sizes.

#### Task 2.1: Add `mx/tidy` + `mx/eval!` to `collect-samples` (CRITICAL)

This single change fixes: `mh`, `mh-custom`, `gibbs`, `involutive-mh`,
`elliptical-slice`, and all kernel combinators.

**Current code** (`kernel.cljs:100-124`):
```clojure
(loop [i 0, state init-state, acc (transient []), n 0, n-accepted 0, rk key]
  (if (>= n samples)
    ...
    (let [[step-key next-key] (rng/split-or-nils rk)
          {:keys [state accepted?]} (step-fn state step-key)  ;; <-- No cleanup!
          ...]
      (recur (inc i) state ...))))
```

**Fix:** Wrap step-fn in `mx/tidy`, eval the state, and periodically
`clear-cache!`:

```clojure
(loop [i 0, state init-state, acc (transient []), n 0, n-accepted 0, rk key]
  (if (>= n samples)
    ...
    (let [[step-key next-key] (rng/split-or-nils rk)
          {:keys [state accepted?]}
          (mx/tidy
            (fn []
              (let [result (step-fn state step-key)]
                ;; Materialize the trace's score/weight/choices
                (when-let [s (:score (:state result))]
                  (mx/eval! s))
                result)))
          ...]
      (recur (inc i) state ...))))
```

**Design considerations:**
- The `state` returned from `step-fn` is a Trace record containing choice maps
  with MLX arrays. We need to eval the arrays we want to keep **before** tidy
  deletes intermediates.
- For MH, if the proposal is rejected, `state` is `identical?` to the previous
  trace — `mx/tidy` must not delete arrays that belong to the retained trace.
  This is safe because `tidy` only deletes arrays **created during** the wrapped
  function. If the trace is unchanged (rejection), no new arrays were added to it.
- Consider adding periodic `clear-cache!` every N iterations (e.g., every 100)
  to prevent the buffer cache from consuming resource slots.

#### Task 2.2: Fix `importance-sampling` (HIGH)

**Current:** `mapv` with no cleanup — all N traces accumulate.

**Fix option A:** Add `mx/eval!` after each generate:
```clojure
(let [results (mapv (fn [_]
                      (let [r (p/generate model args observations)]
                        (mx/eval! (:weight r) (:score (:trace r)))
                        r))
                    (range samples))]
  ...)
```

**Fix option B:** Use `mx/tidy` per sample (more aggressive cleanup):
```clojure
(let [results (mapv (fn [_]
                      (mx/tidy
                        (fn []
                          (let [r (p/generate model args observations)]
                            (mx/eval! (:weight r) (:score (:trace r)))
                            r))))
                    (range samples))]
  ...)
```

**Fix option C:** Recommend `vectorized-importance-sampling` as default.
The vectorized version runs the model body ONCE for N particles using `[N]`-shaped
arrays, creating ~12 buffers total instead of N×12.

#### Task 2.3: Fix SMC loops (CRITICAL)

`smc` and `csmc` are the most resource-hungry: T timesteps × N particles × model
cost per particle. No cleanup anywhere.

**Fix `smc-init-step`:** Wrap each generate in `mx/tidy`:
```clojure
(mapv (fn [_]
        (mx/tidy (fn []
          (let [r (p/generate model args observations)]
            (mx/eval! (:weight r))
            r))))
      (range n))
```

**Fix `smc-step`:** Wrap each update in `mx/tidy`:
```clojure
(mapv (fn [trace]
        (mx/tidy (fn []
          (let [r (p/update model trace constraints)]
            (mx/eval! (:weight r))
            r))))
      traces)
```

**Fix `smc` main loop:** Add periodic `clear-cache!` between timesteps:
```clojure
(loop [t 1, particles particles, ...]
  (when (zero? (mod t 10)) (mx/clear-cache!))  ;; Periodic cache flush
  ...)
```

**Fix `csmc`:** Same pattern as `smc`.

**Fix `smcp3`:** Same patterns for init, step, and main loop.

**Prefer `vsmc`:** The vectorized SMC (`vsmc`) already has proper cleanup
(tidy in rejuvenate, eval in main loop). Recommend it as default for new code.

#### Task 2.4: Fix `vectorized-compiled-mh` and `vectorized-mala` (HIGH)

Both run vectorized chains (`[N,D]` arrays per step) with no per-step tidy.
N chains × samples steps × D parameters = massive buffer accumulation.

**Fix:** Add `mx/tidy` + `mx/eval!` to the main loop, matching the pattern
already used in `run-loop-compiled-mh` and `run-loop-compiled-hmc`.

#### Task 2.5: Fix step-size adaptation loops (MEDIUM)

`find-reasonable-epsilon` and `dual-averaging-warmup` iterate up to ~100 times
with gradient computations. No cleanup.

**Fix:** Add `mx/tidy` per iteration in the main loop.

#### Task 2.6: Fix `programmable-vi` (MEDIUM)

Has `mx/eval!` but no `mx/tidy`. Intermediates from gradient computation
accumulate.

**Fix:** Wrap gradient computation in `mx/tidy` (matching the pattern in
`compiled-programmable-vi` which already does this).

### Phase 3: Reduce Per-Operation Buffer Count (Hard, Structural)

**Goal:** Reduce the number of Metal buffers created per distribution log-prob
call, making all inference algorithms fundamentally cheaper.

#### Task 3.1: Contribute `lgamma` to MLX (HARD, upstream)

MLX has no native `lgamma` operation. If one were added:
- Beta log-prob: 73 buffers → ~13 buffers (6x reduction)
- Gamma log-prob: 31 buffers → ~11 buffers (3x reduction)

This is an upstream contribution to `ml-explore/mlx`. It would require:
1. C++ primitive implementing `lgamma` (wrapping `std::lgamma` for CPU, Metal
   shader for GPU)
2. VJP (backward) implementation using `digamma`
3. Node-mlx binding

**Complexity:** High. Requires Metal shader development, grad rules, testing.
But it would permanently solve the problem for all gamma-family distributions.

#### Task 3.2: Reduce Lanczos coefficients (MEDIUM)

The current implementation uses g=7 with 8 coefficients (~15 digits accuracy).
For log-prob computation in inference, ~7 digits is plenty.

A g=4 approximation with 5 coefficients would:
- Cut Lanczos loop from 8 to 5 iterations
- Reduce `mlx-log-gamma` from ~21 to ~14 buffers
- Reduce Beta log-prob from ~73 to ~52 buffers (30% reduction)

```clojure
;; g=4 Lanczos coefficients (Godfrey's method)
;; Accurate to ~7 decimal places — sufficient for log-prob scoring
[[1 76.18009172947146]
 [2 -86.50532032941677]
 [3 24.01409824083091]
 [4 -1.231739572450155]
 [5 0.1208650973866179e-2]]
```

**Trade-off:** Slightly less numerical precision in log-prob. For inference this
is irrelevant — MCMC acceptance ratios involve log-prob differences, and 7 digits
of precision far exceeds the Monte Carlo noise.

#### Task 3.3: Cache Lanczos constant arrays (EASY)

The current code creates `(mx/scalar ci)` and `(mx/scalar (double i))` on every
call. These could be module-level constants:

```clojure
(def ^:private LANCZOS-COEFFS
  (mapv (fn [[i c]] [(mx/scalar (double i)) (mx/scalar c)])
        [[1 676.52...] [2 -1259.13...] ...]))

(defn- mlx-log-gamma [x]
  (let [x' (mx/subtract x ONE)
        t  (mx/add x' (mx/scalar 7.5))
        s  (reduce (fn [acc [i-arr ci-arr]]
                     (mx/add acc (mx/divide ci-arr (mx/add x' i-arr))))
                   (mx/scalar 0.99999999999980993)
                   LANCZOS-COEFFS)]
    ...))
```

This saves ~16 `mx/scalar` allocations per call (8 for coefficients + 8 for
indices). Combined with the g=4 reduction, saves ~22 buffers per call.

**Caution:** Must ensure these cached scalars aren't invalidated by `mx/tidy`
in inference loops. Since they're module-level `def`s, they exist outside any
tidy scope and should be safe.

#### Task 3.4: Explore fused log-gamma via `mx/compile-fn` (MEDIUM)

MLX's `compile-fn` can fuse a sequence of operations into a single Metal
dispatch. If `mlx-log-gamma` is wrapped in `compile-fn`, the intermediate
buffers might be fused away:

```clojure
(def compiled-mlx-log-gamma (mx/compile-fn mlx-log-gamma))
```

**Unknown:** Whether `compile-fn` actually reduces the buffer count or just the
dispatch count. Needs benchmarking. The compiled function might still allocate
intermediate buffers for the graph topology, just execute them faster.

### Phase 4: Global Safety Mechanisms (Easy, Defense in Depth)

**Goal:** Even if individual loops have bugs, the system should not crash.

#### Task 4.1: Add a global resource guard

```clojure
(defn with-resource-guard
  "Runs f with periodic cache clearing to prevent resource exhaustion.
   Options:
   - :check-interval  — how often to check (default: every 50 iterations)
   - :cache-limit     — max cache size in bytes (default: 100MB)
   - :clear-threshold — clear cache when active memory exceeds this fraction
                         of the resource limit (default: 0.7)"
  [opts f]
  (let [{:keys [cache-limit]} opts]
    (when cache-limit (mx/set-cache-limit! cache-limit))
    (try (f)
      (finally
        (mx/clear-cache!)
        (when cache-limit (mx/set-cache-limit! (* 2 1024 1024 1024)))))))
```

#### Task 4.2: Add resource-aware inference wrapper

Inference functions could optionally accept a `:resource-safe?` flag that
enables conservative memory management:

```clojure
(defn importance-sampling
  [{:keys [samples resource-safe?] :or {samples 100}} model args observations]
  (if resource-safe?
    ;; Evaluate and clear cache every 50 samples
    (let [batch-size 50]
      (loop [remaining samples, all-results []]
        (if (<= remaining 0)
          (finalize all-results)
          (let [n (min remaining batch-size)
                batch (run-batch n model args observations)]
            (mx/clear-cache!)
            (recur (- remaining n) (into all-results batch))))))
    ;; Original behavior
    (mapv (fn [_] (p/generate model args observations)) (range samples))))
```

#### Task 4.3: Set conservative cache limit at startup

Add to GenMLX initialization:

```clojure
;; Default: limit cache to 256MB to keep buffer count low
(mx/set-cache-limit! (* 256 1024 1024))
```

This forces more aggressive buffer release, trading some allocation performance
for staying well within the 499K resource limit.

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

### 6.4 Estimated impact of Phase 2 fixes

| Algorithm | Before (buffers held) | After (buffers held) | Improvement |
|-----------|----------------------|---------------------|-------------|
| MH (500 iter, 5-site Gaussian) | ~32,500 | ~65 (per iter) | 500x |
| MH (500 iter, 5-site Beta) | ~187,500 | ~375 (per iter) | 500x |
| IS (500 samples, 5-site Gaussian) | ~20,000 | ~40 (per sample) | 500x |
| SMC (T=100, N=100, Gaussian) | ~400,000 | ~4,000 (per step) | 100x |
| collect-samples (generic) | O(N×S×C) | O(S×C) | Nx |

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

### 7.4 Test: Lanczos optimization

After Phase 3.2-3.3, verify Beta/Gamma log-prob accuracy is preserved:

```clojure
(let [d (dist/beta-dist 2 5)
      v (mx/scalar 0.3)
      lp (dc/dist-log-prob d v)]
  (mx/eval! lp)
  (assert-close "Beta log-prob" -0.832 (mx/item lp) 0.001))
```

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
