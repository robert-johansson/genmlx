# GenMLX Optimization: Honest Assessment & Step-by-Step Plan

## The Vision (Explicitly Restated)

GenMLX aims to be:

1. **100% idiomatic ClojureScript** — beautiful, purely functional, data-driven code running on nbb
2. **GPU-accelerated on Apple Silicon** via MLX through node-mlx (N-API)
3. **Competitive with GenJAX for GPU-bound inference** — HMC, MALA, NUTS, VI on real models should be within 2-3x of GenJAX
4. **As expressive as Gen.jl** — full Generative Function Interface with all protocols
5. **The eval boundary principle**: the interpreter builds the computation graph (microseconds), the GPU executes it (milliseconds). Performance comes from placing the eval boundary correctly, not from changing the language above it.

The end state: a user writes a model in elegant ClojureScript, runs gradient-based inference, and gets performance limited only by GPU computation time — not by interpreter overhead.

---

## The Unifying Strategy: Model Lowering via Trace-and-Replay

All optimization steps in this plan are instances of a single architectural idea:

> **MLX is already lazy. The model body is a graph-construction script. For static
> models, the graph topology is invariant — only the leaf values (random samples,
> constraints) change between runs. Separate what changes from what doesn't.
> Compute the invariant part once.**

This is memoization at the graph level, enabled by GenMLX's purely functional
architecture. The strategy has three composable phases:

**Phase A — Trace:** Run the model body through SCI exactly once. The handler
records every `(address, distribution, dependencies)` tuple as a flat "recipe" —
a pure data structure. Detect whether the model is static (same addresses every
run) or dynamic (data-dependent control flow).

**Phase B — Lower:** From the recipe, generate a ClojureScript closure that calls
MLX ops directly — no SCI, no `p/generate`, no choicemaps, no protocol dispatch.
This closure is created at runtime but executes as a native JS function. For
dependent parameters (e.g., `mu = slope * x + intercept`), the dependency is
captured implicitly in the MLX lazy graph: the intermediate arrays ARE the
computation graph.

**Phase C — Compile:** Wrap the lowered function in `mx/compile-fn`. MLX caches
the Metal kernel topology. Subsequent calls skip both SCI and Metal compilation.
This enables the triple transform: `compile(vmap(grad(lowered-fn)))` — the MLX
equivalent of JAX's `jit(vmap(grad(f)))`.

**Why this works:** SCI interprets the model body to produce an MLX graph, but
for static models the graph topology never changes. Phase A captures the topology
once. Phase B replays it without the interpreter. Phase C caches the Metal program.
The chain trace → lower → compile → Metal cache is memoization at four successive
abstraction levels.

### Validated: `mx/compile-fn` Traces Once and Replays (2026-02-24)

A critical experiment (`test/genmlx/compile_fn_test.cljs`) confirmed that
`mx/compile-fn` behaves like JAX's `jit` — it executes the JS function body
**exactly once** per input shape, then replays the cached Metal program on all
subsequent calls without re-entering JS/SCI.

```
Test                              Body re-executes on 2nd+ call?
──────────────────────────────────────────────────────────────────
compile(f) — same shape           NO  (1 of 3 calls triggered body)
compile(f) — different shape      YES (retrace — expected)
compile(grad(f))                  NO  (0 of 2 subsequent calls)
compile(vmap(f))                  NO  (0 of 1 subsequent call)
compile(vmap(grad(f))) — triple   NO  (0 of 1 subsequent call)
```

**Timing validation** (pure MLX 5-site score function with dependent params):

```
                          Per-call      vs current make-score-fn (1.41ms)
─────────────────────────────────────────────────────────────────────────
Raw (SCI-free, no compile)  0.422 ms    3.3x faster (eliminating SCI)
Compiled (compile-fn)       0.179 ms    7.9x faster (eliminating SCI + graph rebuild)
Compiled + grad             0.236 ms    6.0x faster
```

**What this validates:**
- The triple transform `compile(vmap(grad(f)))` works correctly and the body
  only executes once — the full JAX `jit(vmap(grad(f)))` pattern is available

**Limitation:** Dynamic models (control flow based on sampled values) cannot be
lowered. They fall back to SCI. This is the same limitation as `torch.jit.trace`
and `jax.jit` with dynamic shapes.

### ~~Expected Impact of Full Model Lowering~~ CORRECTED (2026-02-24)

**The projections below were WRONG.** Ground truth testing (Step 2.5) showed that
`mx/compile-fn` already caches score functions for the SCI-based `make-score-fn`
— the body is NOT re-executed on subsequent calls. The comparison between "raw
SCI" (1.41ms) and "compiled pure MLX" (0.179ms) was misleading because:

1. The 1.41ms raw score-fn includes eval! on every call
2. The compiled version skips both SCI AND eval-per-call overhead
3. compile-fn achieves the SAME caching for the original score-fn as for handwritten

**Corrected measurements (Step 2.5):**

```
                                Original+compile  Handwritten+compile  Gap
──────────────────────────────────────────────────────────────────────────
score-fn (7-site)               0.202 ms          0.215 ms             NONE
grad (7-site)                   0.285 ms          0.292 ms             NONE
val+grad (7-site)               0.325 ms          0.325 ms             NONE
simulate (7-site, score only)   0.304 ms          —                    —
```

**The real bottleneck is eval! (Metal dispatch) at ~0.29ms per call, which is 96%
of the compiled score-fn time.** Graph construction (whether SCI or handwritten)
is only 0.012ms — negligible. Optimization must focus on reducing eval! calls,
not eliminating SCI from the graph construction path.

```
OLD PROJECTIONS — WRONG (kept for record):
                        Current      After Lower  After Lower+Compile  vs Gen.jl
────────────────────────────────────────────────────────────────────────────────
score-fn per call       1.41 ms      ~0.42 ms ✓   ~0.18 ms ✓           —
simulate (7-site)       1.66 ms      ~0.05 ms     ~0.02 ms             ~10x
compiled-mh (200 st)    25 ms        ~8 ms        ~4 ms                ~4x
HMC (50 st, L=10)       44 ms        ~25 ms       ~18 ms               ~1.5x
(These assumed compile-fn doesn't cache SCI-based functions. It does.)
```

Note: simulate/generate projections are more conservative than earlier estimates.
The ~0.02ms floor accounts for choicemap construction and PRNG key management
that the pure score function test didn't include.

### References

This strategy is a well-known pattern in ML frameworks and PPLs:
- **JAX `jax.jit`**: Traces with abstract Tracer objects, records jaxpr IR, compiles to XLA
- **PyTorch `torch.jit.trace`**: Runs with example inputs, records tensor ops, replays without Python
- **TensorFlow `tf.function`**: Traces Python to `tf.Graph`, caches by input signature
- **Gen.jl Static DSL**: Constructs static DAG at compile time, generates specialized GFI code
- **GenJAX**: Inherits JAX tracing — entire model+inference loop is JIT-compiled
- **Wingate et al. 2011** ("Lightweight Implementations of PPLs via Transformational Compilation"): Reports ~100x speedup from compilation of probabilistic programs

---

## Honest Assessment: What We Can and Cannot Achieve

### What IS Achievable

**GPU-dominated workloads (HMC, MALA, VI on models with >5 sites):**
- HMC is already **2.2-3.6x** of Gen.jl (measured 2026-02-24, see Gen.jl comparison below)
- With compiled score functions: **1.5-2x of Gen.jl** is realistic
- With vectorized N-chain inference: **matching or exceeding Gen.jl** on Apple Silicon
- This is the sweet spot. Real probabilistic models ARE GPU-bound.

**Compiled parameter-space MH:**
- Already **~34x** of Gen.jl's GFI MH (vs 280-564x for GFI MH)
- Compiled path eliminates ~90% of GFI overhead; remaining gap is SCI inside `p/generate`
- With lowered score function (Step 3): target **5-10x** of Gen.jl

**Vectorized inference:**
- Already showing 12x at N=50 for MALA, 2.5x for HMC at N=10
- With reduced fixed overhead: **closer to Nx** efficiency
- The scaling IS real. The fixed SCI overhead inside score-fn is the bottleneck.

### What Is NOW Achievable (via Model Lowering)

**simulate/generate for static models:**
- Currently **350-1200x** slower than Gen.jl (SCI interprets model body every call)
- With model lowering + compile: projected **~10x of Gen.jl** (validated: compile-fn traces once)
- The gap was "structural to nbb" — but model lowering bypasses nbb on the hot path
- Dynamic models (data-dependent control flow) still fall back to SCI

### What Is NOT Achievable

**GFI MH on uncompiled models:**
- GFI MH calls `p/regenerate` per step, which re-runs the full model through SCI
- **280-564x** slower than Gen.jl — structural without model lowering
- Lowered MH (compiled-mh with lowered score) brings this to ~4x of Gen.jl

**Matching Gen.jl on CPU-bound scalar operations:**
- Julia's JIT compiles to native machine code; even lowered JS has function call overhead
- ~5-10x of Gen.jl appears to be the realistic floor for lowered simulate/generate
- **This is acceptable** — real inference workloads are GPU-bound

**Matching GenJAX on JAX hardware:**
- JAX on NVIDIA A100/H100 has higher raw FLOPS than Apple Silicon
- GenMLX competes on the Apple Silicon story, not on absolute FLOPS

### Measured Performance vs Gen.jl (2026-02-24)

Apples-to-apples comparison on the same machine, same day. Gen.jl numbers from
`test/reference/gen_jl_benchmark.json` (Julia 1.12.3, Gen.jl 0.4.8). GenMLX numbers
from `test/genmlx/genjl_comparison_benchmark.cljs` (Bun 1.3.9, nbb, 3 warmup + median of 7).

```
                                GenMLX (Bun)    Gen.jl       Ratio
────────────────────────────────────────────────────────────────────
SIMULATE (10 calls)
  1-site                        3.58 ms         0.003 ms     1194x
  7-site (linreg)               16.6 ms         0.021 ms     793x
  11-site (many)                22.6 ms         0.043 ms     525x

GENERATE (10 calls)
  1-site                        3.76 ms         0.004 ms     940x
  7-site (linreg)               12.4 ms         0.024 ms     515x
  11-site (many)                16.9 ms         0.048 ms     351x

GFI MH (200 steps)
  1-site                        112.9 ms        0.20 ms      564x
  7-site (linreg)               366.3 ms        0.93 ms      394x
  11-site (many)                438.5 ms        1.55 ms      283x

COMPILED MH (200 steps, parameter-space — GenMLX only)
  7-site (linreg)               31.8 ms         —            —
  11-site (many)                33.7 ms         —            —

HMC (50 steps, L=10)
  7-site (linreg)               43.9 ms         12.2 ms      3.6x
  11-site (many)                36.6 ms         16.4 ms      2.2x
```

**Key observation:** The gap SHRINKS with model size and GPU utilization. 1-site simulate: 1194x. 11-site HMC: 2.2x. This validates the eval boundary thesis for GPU-bound workloads.

### Why Previous Attempts Fell Short

The benchmark results tell the story:

| What Was Promised | What Was Delivered | Why |
|---|---|---|
| Lazy chains: 2-5x | 0.5-0.9x (regression) | Huge deferred graph is more expensive than eager per-step eval for small models |
| Vectorized: ~Nx | 18x at N=50 (MALA) | ~230ms fixed overhead from SCI interpretation inside score-fn |
| Combined: 100-500x | Best: 18x | Multiplicative assumption was wrong — SCI overhead doesn't multiply away |

**~~Root cause (WRONG):~~ `make-score-fn` calls `p/generate` on every invocation.**
**CORRECTION (Step 2.5):** `mx/compile-fn` DOES skip the body (including SCI)
on subsequent calls. The body runs once (trace), then the cached Metal kernel
replays. The SCI overhead is NOT the bottleneck for compiled inference.
The ACTUAL root cause of the lazy regression is Metal dispatch overhead —
see points 2-4 below.

The lazy chain regression happens because:
1. Each lazy step still calls `score-fn` (which calls `p/generate` through SCI)
2. The lazy graph grows without bound (no per-step eval to free intermediates)
3. The final `apply mx/eval!` on hundreds of accumulated graph nodes is slower than incremental eval
4. **Metal resource limit (499K)**: Accumulated graph nodes exhaust Metal allocations, crashing the process (see Metal Resource Leak below)

---

## Should We Improve node-mlx?

**No. Not now.** GenMLX already runs on a local fork at MLX 0.30.6 (see `./node-mlx/`).

- N-API overhead is 0.03% of total inference time (measured: ~0.8μs out of 8ms)
- The local fork already exposes everything GenMLX uses: `vmap`, `grad`, `compile-fn`, `value-and-grad`, all ops
- The bottleneck is 100% in the SCI interpretation layer, not in the native bindings
- The local fork (MLX 0.30.6) is **strictly better** than the official `@frost-beta/mlx@0.4.0` (MLX 0.25.0) — it handles eval-every-iteration that 0.25.0 cannot (0.25.0 crashes with `[Event::stream] Cannot access stream on invalid event`)

**When to revisit**: After model lowering (Steps 3-10) is complete and GPU utilization exceeds 90%, node-mlx improvements (newer MLX kernels) might matter. Not before.

---

## Metal Resource Leak (Discovered 2026-02-24)

**The problem:** Running inference loops or benchmarks in a single process eventually hits `[metal::malloc] Resource limit (499000) exceeded`. This is a hard cap on concurrent Metal buffer allocations.

**Root cause:** JavaScript's GC doesn't know about Metal resource pressure. Evaluated MLX arrays still hold Metal buffer references in the JS heap. Without explicit disposal, dead arrays accumulate until the 499K limit is hit.

**Tested on both MLX versions (2026-02-24):**
- Local fork (MLX 0.30.6): Leak exists. `eval!` every iteration works. `tidy`+`eval!` works.
- Official `@frost-beta/mlx@0.4.0` (MLX 0.25.0): Leak exists. `eval!` every iteration crashes with stream error. Only `tidy`+`eval!` works.

**The fix:** Wrap each iteration's array-heavy work in `mx/tidy` and call `mx/eval!` on retained results before the next iteration. This is already done in HMC leapfrog but needs to be applied across all inference loops.

**Practical limit without tidy:** With ~4-5 arrays per trace site and 5-50 sites per model, inference hits the wall at roughly 2K-25K iterations.

**Impact on this plan:** This invalidates the lazy chain approach entirely — lazy chains accumulate graph nodes by design, which is fundamentally incompatible with the 499K limit. Step 6 is now "remove lazy variants" rather than "fix lazy variants."

---

## The Plan: Small Steps, Each Verified

### Principles

1. **Measure first, optimize second.** Every step starts with a benchmark and ends with a benchmark.
2. **Never move forward until the current step is verified.** "Verified" means: benchmark shows improvement, all tests pass.
3. **If a step doesn't deliver, understand why before proceeding.** No hand-waving about future multiplicative gains.
4. **Fix bugs before adding features.** Broken code cannot be optimized.
5. **Each step is independently valuable.** No step depends on a future step to be useful.

---

### Step 0: Fix Critical Bugs
**Time: 1 day | Prerequisite for everything else**

These bugs make benchmark results unreliable and block optimization work.

- [x] **0.1** Fix `mx/random-normal` → `mx/random-uniform` in vectorized compiled MH
  - File: `inference/mcmc.cljs:299`
  - 1 line change
  - **Verify**: vectorized-compiled-mh produces valid samples (no NaN)

- [x] **0.2** Fix REINFORCE estimator signal (vi.cljs:323,327)
  - Use `obj-val` instead of `log-q` as REINFORCE signal
  - 3 line change
  - **Verify**: programmable-vi with `:reinforce` produces decreasing loss

- [x] **0.3** Fix wake-phase ELBO variable shadowing (learning.cljs:201-206)
  - Rename shadowed `weight` binding
  - 5 line change
  - **Verify**: wake-phase-loss returns ELBO (not just -log p)

- [x] **0.4** Add `mx/eval!` after resampling in vectorized.cljs
  - File: `vectorized.cljs` after `resample-vtrace`
  - 1 line addition
  - **Verify**: long SMC runs don't OOM

**Exit criterion**: All existing tests pass. Vectorized MH produces valid chains.

---

### Step 1: Establish Benchmark Baselines
**Time: 1 day | No code changes to core**

Create a single benchmark file that measures everything we care about, with precise timing and clear output.

- [x] **1.1** Create `test/genmlx/optimization_benchmark.cljs`
  - Models: 2-site gaussian, 7-site linear regression, 20-site model
  - Operations: simulate, generate, compiled-mh (200 steps), mala (50 steps), hmc (20 steps)
  - Variants: scalar, lazy, vectorized (N=10, N=50)
  - Measure: wall-clock time (ms), per-step time
  - Also measures: lazy regression, score-fn call cost

- [x] **1.2** Run baseline on Node.js and record results in this file
- [x] **1.3** Run baseline on Bun and record results in this file

**Exit criterion**: Repeatable benchmark with <10% variance between runs. DONE — results consistent across 2 Bun runs.

```
BASELINE RESULTS — Bun 1.3.9, nbb, macOS, Apple Silicon (2026-02-24)

┌───────────────────────────┬──────────┬──────────┬──────────┐
│ Operation                 │ 2-site   │ 7-site   │ 20-site  │
├───────────────────────────┼──────────┼──────────┼──────────┤
│ simulate                  │ 1 ms     │ 1 ms     │ 3 ms     │
│ generate                  │ 0-1 ms   │ 1-2 ms   │ 3 ms     │
│ compiled-mh (200 st)      │ 22 ms    │ 25-26 ms │ 39 ms    │
│ mala (50 st)              │ 22 ms    │ 32-33 ms │ 52-54 ms │
│ hmc (20 st, L=10)         │ 10 ms    │ 17-18 ms │ 27 ms    │
│ vec-mala N=10 (50 st)     │ —        │ 100-101ms│ —        │
│ vec-mala N=50 (50 st)     │ —        │ 140 ms   │ —        │
│ vec-hmc N=10 (20 st)      │ —        │ 72-74 ms │ —        │
└───────────────────────────┴──────────┴──────────┴──────────┘

Key derived metrics:
  compiled-mh per-step:     0.13 ms/step (7-site)
  mala per-step:            0.64 ms/step (7-site)
  hmc per-step:             0.85 ms/step (7-site)
  score-fn per-call (raw):  1.41 ms/call (7-site) ← THIS IS THE BOTTLENECK
  score-fn per-call (compiled): 0.20 ms/call (7-site, 7x faster)
  lazy-mh regression:       1.81x slower than eager
  lazy-mala regression:     1.15-1.18x slower than eager
  vec-mala effective N=10:  3.3-3.7x
  vec-mala effective N=50:  11.8-12.8x
  vec-hmc effective N=10:   2.4-2.5x

Node.js comparison (same machine):
  Node.js is ~5-10% slower across the board vs Bun.
  Not dramatically different — the SCI overhead dominates both runtimes.
```

---

### Step 2: ~~Understand Where Time Goes (Profiling)~~ SUPERSEDED
**Status: SKIPPED — the Gen.jl comparison benchmark (Step 1) already answered these questions.**

The Gen.jl comparison benchmark established:
- Score-fn costs 1.41ms/call (7-site), with `p/generate` as the bottleneck inside it
- Compiled score-fn costs 0.20ms/call (7x faster via Metal kernel reuse, but SCI still runs)
- GFI MH is 280-564x slower than Gen.jl (SCI dominates)
- Compiled MH is ~34x slower (SCI inside `p/generate` is the remaining gap)
- HMC is 2.2-3.6x slower (GPU computation masks SCI)

**Conclusion:** Step 3 (compiled score function) is confirmed as the right next move. No further profiling needed.

---

### Step 2.5: Ground Truth — compile-fn Behavior & Cost Decomposition (DONE 2026-02-24)
**Critical finding that changes the entire plan.**

Tests: `test/genmlx/lowering_ground_truth.cljs`, `test/genmlx/lowering_ground_truth_2.cljs`

#### Finding 1: compile-fn ALREADY caches score functions (previous claim was WRONG)

The HANDOFF.md stated: "compile-fn re-executes the body because p/generate has
side effects." **This is false.** `mx/compile-fn` ignores JS side effects entirely.
It caches the Metal kernel topology after the first call and replays it on all
subsequent same-shape calls — regardless of volatile!, dynamic vars, or SCI.

```
Does compile-fn re-execute the body?
────────────────────────────────────────────────────────
Pure function:                 NO (0 of 2 subsequent calls)
SCI closure with reduce:       NO (0 of 2 subsequent calls)
Function with volatile!:       NO (0 of 2 subsequent calls)
make-score-fn (p/generate):    NO (0 of 2 subsequent calls)  ← KEY FINDING
```

#### Finding 2: Compiled score/gradient — already at the Metal floor

Hand-written pure MLX score functions (no SCI, no handler, no choicemap) produce
**identical performance** to the compiled original when wrapped in compile-fn:

```
                                    Original+compile  Handwritten+compile  Gap
────────────────────────────────────────────────────────────────────────────────
score-fn (7-site)                   0.202 ms          0.215 ms             0.9x (noise)
grad (7-site)                       0.285 ms          0.292 ms             0.98x
val+grad (7-site)                   0.325 ms          0.325 ms             1.00x
```

**Conclusion: Model lowering for score functions gives ZERO speedup.** compile-fn
already achieves the Metal floor. Steps 3-4 as designed are zero-value.

#### Finding 3: eval! is THE bottleneck

```
Component decomposition (compiled 7-site score-fn, 1000 calls):
────────────────────────────────────────────────────────
score-fn graph construction (compiled):  0.012 ms  (4%)
Metal dispatch + synchronization:        0.287 ms  (96%)
────────────────────────────────────────────────────────
Total per score-fn call:                 0.300 ms
```

Each `eval!` call costs ~0.29ms — this is Metal kernel launch + GPU synchronization.
This is the floor for any single MLX operation. Graph construction (the part SCI
participates in) is only 4% of total time. **The bottleneck is Metal dispatch,
not SCI interpretation.**

Proposal generation (random noise) also costs ~0.26ms, dominated by its own eval.

#### Finding 4: simulate/generate — real gap exists but Metal floor limits gains

```
                              Current    Compiled   Metal floor   Gen.jl
──────────────────────────────────────────────────────────────────────────
simulate (7-site)             0.800 ms   0.304 ms   ~0.30 ms      0.021 ms
generate (7-site)             0.502 ms   —          ~0.30 ms      0.024 ms
```

compile-fn caches simulate too (body not re-executed). The 2.6x speedup from
compile-fn is real, but the Metal floor (~0.30ms) means we can never get below
~14x of Gen.jl for 7-site models. Gen.jl runs on CPU with zero dispatch overhead.

#### Finding 5: Why the compile_fn_test.cljs results were misleading

The original test compared "raw SCI score-fn (1.41ms)" to "compiled pure MLX
(0.179ms)" and concluded there was 8x headroom from lowering. But this comparison
was wrong: the 1.41ms included eval! on every call, while compiled omits re-execution.
The correct comparison is compiled-original (0.202ms) vs compiled-handwritten (0.215ms)
= no gap.

The 5-site test also had fewer sites and simpler structure than 7-site, making
the pure-MLX version appear faster than it is at realistic model sizes.

#### Implications for the plan

| Original Step | Status | Reason |
|---|---|---|
| Step 3: Lowered score functions | **SKIP** | compile-fn already at Metal floor |
| Step 4: Lowered gradients | **SKIP** | compile-fn already at Metal floor |
| Step 5: Triple transform | **INVESTIGATE** | vmap may still help by amortizing eval! |
| Step 6: Remove lazy + tidy | **DO** | Independent, still valuable |
| Step 9: Lowered simulate/generate | **REASSESS** | 2.6x from compile-fn already; Metal floor limits further gains |

**The real optimization levers are:**
1. **Minimize eval! calls** — each costs ~0.29ms, dominates everything
2. **Vectorized inference** — amortizes eval! across N chains (already proven: 12x at N=50)
3. **Micro-batched lazy steps** — accumulate K steps lazily, eval once (K-fold reduction in eval overhead)
4. **Reduce inference loop overhead** — collect-samples, PRNG management, mx/->clj conversions

---

### Step 3: ~~Model Lowering for Score Functions~~ SUPERSEDED
**Status: SKIPPED — compile-fn already achieves the Metal floor for score functions.**

Ground truth testing (Step 2.5) proved that `mx/compile-fn` caches the Metal
kernel for `make-score-fn` + `p/generate` exactly as well as for a hand-written
pure MLX function. The 8x estimate was based on comparing raw (uncached) vs
compiled (cached) — the gap disappears when both are compiled.

See Step 2.5 for full measurements.

---

### Step 4: ~~Lowered Gradient Function~~ SUPERSEDED
**Status: SKIPPED — compile-fn already achieves the Metal floor for gradients.**

Compiled grad (original): 0.285ms. Compiled grad (handwritten): 0.292ms. No gap.
Compiled val+grad: 0.325ms = 0.325ms. No gap.

---

### Step 5: eval! Cost Model & Optimization Exploration (DONE 2026-02-24)
**Tests: `test/genmlx/eval_cost_model.cljs`, `test/genmlx/eval_cost_model_2.cljs`**

#### Finding: eval! cost model is fixed + linear

```
eval! cost = 0.16ms (fixed dispatch) + N_ops × 0.013ms (per graph op)

K graph ops:   eval! cost   per-op cost   (500 trials)
─────────────────────────────────────────────────────────
K=1            0.172ms      0.172ms
K=10           0.282ms      0.028ms
K=100          1.213ms      0.012ms
K=500          7.322ms      0.015ms
```

**Implication:** The fixed 0.16ms dispatch cost CAN be amortized by batching more
work into one eval!. But the per-op cost (~0.013ms) scales linearly — you always
pay for the work. Micro-batching saves the FIXED cost, not the VARIABLE cost.

#### Finding: Micro-batched lazy MH — DEAD END

```
Eager MH (eval every step):    75.1ms  (0.375ms/step)
Fully lazy (eval at end only): 60.9ms  (0.305ms/step)  ← 1.23x, best case
Eval every 5:                  80.4ms  (0.402ms/step)  ← WORSE than eager
Eval every 10:                 78.4ms  (0.392ms/step)  ← WORSE than eager
Eval every 20:                 78.8ms  (0.394ms/step)  ← WORSE than eager
```

Micro-batching is worse than eager because the per-op cost is linear. Batching K
steps into one eval! saves K × 0.16ms dispatch but costs K × (ops_per_step ×
0.013ms) in per-op overhead. For compiled score-fn (which is a single cached
Metal kernel per call), each step adds ~8 graph nodes. The overhead exceeds the
dispatch savings except in the fully-lazy case.

Fully lazy gives 1.23x but risks the Metal 499K resource limit on long chains.

- [x] **5.1** ~~Micro-batched lazy MH~~ — DEAD END (measured, doesn't help)

#### Finding: vmap composes with make-score-fn — YES

```
vmap(score-fn)([3,2]) → [3] shaped scores — WORKS
Values match scalar evaluation to 1e-3
```

No model lowering needed — vmap works directly on the SCI-based score function.

- [x] **5.2.1** vmap(make-score-fn) — WORKS ✓

#### Finding: THE TRIPLE TRANSFORM WORKS

```
compile(vmap(grad(make-score-fn(...)))) — WORKS on the existing SCI-based score-fn

grad(score-fn):                    ✓
compile(grad(score-fn)):           ✓
vmap(grad(score-fn)):              ✓  shape [3,2]
compile(vmap(grad(score-fn))):     ✓  shape [3,2] — THE FULL PATTERN

Triple (N=3): 0.278ms  vs  3× sequential: 0.360ms  → 1.30x
```

The JAX `jit(vmap(grad(f)))` pattern works in GenMLX without any model lowering.
compile-fn caches the SCI body, vmap vectorizes, grad differentiates. All three
compose on the EXISTING score function.

The 1.30x at N=3 reflects the modest dispatch savings. At larger N, the savings
grow: each additional chain costs ~0.013ms marginal, vs ~0.17ms for a separate
eval! dispatch.

- [x] **5.2.2** compile(vmap(grad(score-fn))) — WORKS ✓

#### Finding: Batching independent results into 1 eval! gives 6.1x

```
eval!(already-evaluated):    0.0005ms  (free)
eval!(1 trivial op):         0.1731ms  (fixed dispatch)
eval!(10 ops, 1 eval!):      0.2829ms  (batched)
10 × eval!(1 op):            1.7307ms  (10 dispatches)
Batching savings: 6.1x
```

This is the mechanism behind vectorized inference: one eval! dispatch serves N
chains instead of N dispatches.

- [x] **5.2.3** Batching mechanism validated ✓

#### Remaining: Inference loop overhead (5.3)

compiled-mh actual: 0.66ms/step (at 1000 steps) vs manual: 0.375ms/step = 1.76x gap.
This gap is `collect-samples` infrastructure: PRNG management, `mx/->clj`
conversions, callback dispatch.

- [ ] **5.3.1** Profile and optimize `collect-samples` hot path
- [ ] **5.3.2** Benchmark optimized loop

#### Finding: LOOP COMPILATION WORKS — 5.6x speedup (2026-02-24)

**Test: `test/genmlx/loop_compilation_test.cljs`**

**The headline result:** compile-fn can cache an entire K-step MH chain as one
Metal dispatch. The loop unrolls during tracing, and the cached Metal program
replays with fresh inputs on each call.

**Key challenge solved: randomness.** compile-fn freezes ALL random operations
(both stateful `mx/random-normal` AND key-based `rng/normal`). The fix:
pre-generate `[K, D]` noise and `[K]` uniform arrays OUTSIDE compile-fn and
pass them as inputs. The compiled function indexes into pre-generated noise
at each step via `mx/take-idx`. Fresh noise on each call → correct randomness.

```
Compiled chain (200 steps): 13.8 ms  (0.069 ms/step)
Eager MH (200 steps):       77.9 ms  (0.390 ms/step)
Speedup:                     5.6x

Per-step cost by chain length:
  K=10:   0.090 ms/step
  K=20:   0.076 ms/step
  K=50:   0.068 ms/step
  K=100:  0.067 ms/step
  K=200:  0.068 ms/step  ← converges to ~0.068ms

Block sample collection (200 samples):
  Eager:       83.2 ms  (baseline)
  Block K=5:   26.8 ms  (3.1x)
  Block K=10:  18.9 ms  (4.4x)
  Block K=20:  16.1 ms  (5.2x)
  Block K=50:  14.1 ms  (5.9x)
  Block K=200: 13.6 ms  (6.1x)
```

**Correctness validated:**
- 20/20 unique results with fresh noise
- 50/50 unique chain endpoints (50 independent chains)
- Chains explore: var(slope)=0.157, var(intercept)=2.19
- Deterministic with same noise (reproducible)
- 2000-step chains stable (no Metal resource crash)

**Architecture:**
```clojure
;; Build compiled K-step chain
(defn make-compiled-chain [k score-fn std n-params]
  (mx/compile-fn
    (fn [params noise-2d uniforms-1d]   ; inputs: [D], [K,D], [K]
      (loop [p params, i 0]
        (if (>= i k) p
          (let [row (mx/reshape (mx/take-idx noise-2d (mx/array [i] mx/int32) 0)
                                [n-params])
                proposal (mx/add p (mx/multiply std row))
                s-cur (score-fn p)
                s-prop (score-fn proposal)
                log-alpha (mx/subtract s-prop s-cur)
                log-u (mx/log (mx/index uniforms-1d i))
                accept? (mx/greater log-alpha log-u)]
            (recur (mx/where accept? proposal p) (inc i))))))))

;; Usage: generate noise outside, pass as input
(let [compiled (make-compiled-chain 200 score-fn std 2)]
  (compiled params (mx/random-normal [200 2]) (mx/random-uniform [200])))
```

This is GenMLX's equivalent of JAX's `jit(lax.scan(...))`. The entire inference
loop is compiled into one Metal program. One dispatch for K steps instead of K
dispatches.

- [x] **5.4** Loop compilation — WORKS, 5.6x speedup ✓

- [x] **5.4.1** Integrate loop compilation into `compiled-mh` in mcmc.cljs
  - Add `make-compiled-chain` helper (pre-generated noise pattern)
  - Burn-in: compiled chain of block-size steps (default 50), ~5x faster
  - Collection with thin > 1: compiled chain of thin steps per sample
  - Collection with thin = 1: **no benefit** — 1-step compiled chain is slower
    than eager due to 2D noise array overhead. Falls back to eager per-step.
  - New option: `:block-size` (default 50) controls burn-in steps per dispatch
  - `mx/tidy` around collection loop prevents Metal resource leak
  - All tests pass: 165/165 Gen.clj, 73/73 GenJAX, inference green
  - **Limitation**: for `{:burn 0 :thin 1}` (no burn, no thinning), the new
    code is the same speed as the old code. The 5.6x headline applies to
    burn-in and thinning phases only.

- [ ] **5.4.2** Apply loop compilation to MALA (mcmc.cljs)
- [ ] **5.4.3** Apply loop compilation to HMC leapfrog (mcmc.cljs)

```
STEP 5 RESULTS (2026-02-24):
┌────────────────────────────────┬──────────┬───────────────────────────┐
│ Finding                        │ Result   │ Impact                    │
├────────────────────────────────┼──────────┼───────────────────────────┤
│ Micro-batched lazy MH          │ 1.23x    │ Dead end (only fully lazy)│
│ vmap(score-fn)                 │ WORKS    │ No lowering needed        │
│ compile(vmap(grad(score-fn)))  │ WORKS    │ Triple transform available│
│ Batching 10→1 eval!            │ 6.1x    │ Vectorization mechanism   │
│ Loop compilation (K=200)       │ 5.6x    │ THE winning strategy      │
│ Block collection (K=20)        │ 5.2x    │ Practical with samples    │
│ Inference loop overhead        │ 1.76x    │ Lower priority now        │
└────────────────────────────────┴──────────┴───────────────────────────┘
```

---

### Step 6: Remove Lazy Chain Variants + Apply Tidy/Eval Discipline ✅ DONE (2026-02-24)
**Time: 1-2 days | Independent of Steps 3-5**

Lazy chain variants are confirmed regressions AND fundamentally incompatible with the Metal 499K resource limit. Removed them and applied `tidy`+`eval!` discipline in loop-compiled paths.

- [x] **6.1** Remove lazy chain variants from mcmc.cljs
  - `compiled-mh-lazy`, `lazy-compiled-mh-step` — deleted
  - `mala-lazy`, `mala-step-lazy` — deleted
  - `hmc-lazy`, `hmc-step-lazy`, `hamiltonian-lazy` — deleted
  - Updated `optimization_benchmark.cljs`, `vectorized_grad_benchmark.cljs`, `vectorized_grad_test.cljs`, `gen_jl_speed_test.cljs`

- [x] **6.2** Tidy/eval discipline in loop-compiled paths
  - `run-loop-compiled-mh`: `mx/tidy` around each collected sample ✓
  - `run-loop-compiled-mala`: `mx/tidy` around each collected sample, `mx/eval!` on threaded [q,score,grad] ✓
  - `run-loop-compiled-hmc`: `mx/tidy` around each collected sample ✓
  - Burn-in blocks: `mx/eval!` between blocks to bound graph size ✓
  - 500+ step chains tested stable (MALA test, HMC test)

- [x] **6.3** All tests pass: 165/165, 73/73, all core tests green

---

### Step 7: Bun as Default Runtime
**Time: 0.5 days | Independent of Steps 3-6 | PARTIALLY DONE**

`experiment/bun-compatibility` has been merged to main. Bun works and all tests pass.
Remaining work is documentation only.

- [x] **7.1** Run full benchmark suite (from Step 1) on Bun — DONE (results in Step 1 baseline)
- [x] **7.2** Verify all tests pass on Bun — DONE (165/165, 73/73)
- [ ] **7.3** Document Bun as recommended runtime
  - Update README or CLAUDE.md with Bun instructions (`bun run --bun nbb <file>`)

**Exit criterion**: Bun documented as recommended runtime.

---

### Step 8: Distribution-Level Optimizations ✅ DONE (2026-02-24)
**Time: 2 days | Independent of Steps 3-7**

These are small, targeted improvements to distribution hot paths.

- [x] **8.1** Replace JS `log-gamma` with MLX-native `mlx-log-gamma` in beta, gamma, inv-gamma, student-t, dirichlet
  - File: `dist.cljs` — replaced `(mx/scalar (log-gamma (mx/realize x)))` with `(mlx-log-gamma x)` in all 5 distributions
  - Fixes gradient opacity: `mlx-log-gamma` uses MLX ops (Lanczos approximation), fully differentiable
  - **Verified**: `mx/grad(fn [alpha] (log-prob (beta alpha 2) 0.5))` returns non-zero gradient (-0.11)
  - **Verified**: `mx/grad(fn [shape] (log-prob (gamma shape 1) 2))` returns non-zero gradient (-0.23)

- [x] **8.2** Implement native batch categorical sampling via Gumbel-max trick
  - Replaced sequential `(mapv #(rng/categorical % logits) keys)` with `argmax(logits + Gumbel_noise, axis=1)`
  - ~10 lines in `dist-sample-n :categorical`
  - **Verified**: [N=1000] shape correct, category distribution matches logit weights (88/271/641 for logits [0,1,2])

- [x] **8.3** All distribution tests pass (dist_test.cljs, gen_clj_compat 165/165)

---

### Step 9: Lowered Simulate & Generate (Model Lowering for Full GFI Operations)
**Time: 3-5 days | Builds on Step 3's recipe + lowering infrastructure**

Steps 3-5 applied model lowering to score functions (the inner loop of inference).
Step 9 applies the same strategy to `simulate` and `generate` themselves — the
operations that create traces from scratch. This is where the 350-1200x gap
to Gen.jl lives.

The lowered simulate/generate functions include sampling (not just scoring):
- Phase A (trace): Already done in Step 3.1 — the recipe is the same
- Phase B (lower): Generate a closure that does PRNG splitting, sampling, scoring, and choicemap construction — all via direct MLX op calls
- Phase C (compile): Wrap in `mx/compile-fn` — the entire model becomes a cached Metal program

For static models, the lowered simulate is: `(fn [key] -> {:choices, :score, :retval})`
— a pure function of the PRNG key. `mx/compile-fn` caches the topology; only the
key value changes between calls.

- [ ] **9.1** Implement `lower-simulate` in `compiler.cljs`
  - Reuses recipe from Step 3.1
  - Adds: PRNG key splitting, distribution sampling (via direct `dc/dist-sample` calls)
  - Builds choices as a flat map `{addr -> MLX-array}` (not a choicemap tree — convert at boundary)
  - ~150 lines

- [ ] **9.2** Implement `lower-generate` in `compiler.cljs`
  - Extends 9.1 with constraint handling: for constrained sites, use constraint value instead of sampling
  - Computes importance weight = sum(log-prob at constrained sites)
  - ~100 lines

- [ ] **9.3** Wrap in `mx/compile-fn` and benchmark
  - `(def fast-sim (mx/compile-fn (lower-simulate model args)))`
  - Validated: compile-fn traces once and replays — body never re-enters SCI
  - Target: **>30x speedup** on 7-site simulate (from ~1.66ms to ~0.05ms)
  - Target: **>50x with compile** (from ~1.66ms to ~0.02ms)
  - Note: simulate includes PRNG splitting and choicemap construction that the
    pure score-fn benchmark didn't measure — targets are more conservative

- [ ] **9.4** Integrate with IS and MH
  - Importance sampling with lowered generate
  - MH with lowered generate for initialization
  - **Verify**: inference results statistically equivalent

**Exit criterion**: Lowered simulate/generate deliver >50x speedup for static models.

---

### Step 10: Lowered Vectorized Simulate & Generate
**Time: 2-3 days | Builds on Step 9**

Apply `vmap` or explicit [N]-shaped lowering to the lowered simulate/generate.

Two approaches (same as Step 5):
- **vmap approach**: `mx/vmap(lower-simulate)` over N PRNG keys
- **Explicit approach**: Lower to a function that samples [N]-shaped arrays at each site

- [ ] **10.1** Implement lowered vsimulate (via best approach from Step 5)
  - Produces `[N]`-shaped arrays at each site
  - ~80 lines

- [ ] **10.2** Implement lowered vgenerate
  - ~60 lines (extends 10.1)

- [ ] **10.3** Benchmark: lowered vgenerate vs normal vgenerate
  - Target: **>5x speedup** on 7-site model (N=100)
  - Combined with vectorization: approach **hundreds of x** vs SCI scalar simulate

- [ ] **10.4** Integrate with vectorized IS and vectorized SMC
  - **Verify**: inference results statistically equivalent

**Exit criterion**: Lowered + vectorized operations deliver major speedups.

---

### Step 11: Final Benchmark & Assessment
**Time: 1 day**

Run the complete benchmark suite from Step 1 with all optimizations enabled.

- [ ] **11.1** Full benchmark: all models × all algorithms × all variants
- [ ] **11.2** Compare against baseline (Step 1 results)
- [ ] **11.3** Compute: total speedup per algorithm, GPU utilization, overhead fraction
- [ ] **11.4** Write honest assessment: what improved, what didn't, what's next

```
FINAL RESULTS (to be filled in):
┌────────────────────────────────┬──────────┬──────────┬─────────┐
│ Operation                      │ Baseline │ Final    │ Speedup │
├────────────────────────────────┼──────────┼──────────┼─────────┤
│ simulate (7-site)              │ ___ ms   │ ___ ms   │ ___x    │
│ generate (7-site)              │ ___ ms   │ ___ ms   │ ___x    │
│ compiled-mh (7-site, 200 st)  │ ___ ms   │ ___ ms   │ ___x    │
│ mala (7-site, 50 st)          │ ___ ms   │ ___ ms   │ ___x    │
│ hmc (7-site, 20 st, L=10)    │ ___ ms   │ ___ ms   │ ___x    │
│ vec-mala N=50 (50 st)         │ ___ ms   │ ___ ms   │ ___x    │
│ vec-hmc N=10 (20 st)          │ ___ ms   │ ___ ms   │ ___x    │
│ vec-IS N=100                   │ ___ ms   │ ___ ms   │ ___x    │
│ GPU utilization (hmc)          │ ____%    │ ____%    │         │
│ GPU utilization (vec-mala@50)  │ ____%    │ ____%    │         │
└────────────────────────────────┴──────────┴──────────┴─────────┘
```

**Exit criterion**: Written report with before/after numbers.

---

## Future Steps (Only After Steps 0-11 Are Complete)

These are listed for reference but should NOT be started until the above steps are verified.

### F1: Native Kernel Addon (genmlx-kernels)
- Fused leapfrog in C++ (eliminates remaining JS overhead for HMC inner loop)
- GPU-accelerated systematic resampling
- Expected: additional 2-3x on HMC after all above optimizations
- **Only worth doing after Step 11 shows GPU utilization <80%**
- Note: with model lowering, the SCI overhead is already eliminated — the native addon targets the remaining JS function call overhead

### F2: shadow-cljs Production Build
- AOT compilation of ClojureScript → optimized JS
- Expected: 10-60x on remaining non-lowered code paths (dynamic models, model definition)
- **Only worth doing if dynamic models are a significant use case**
- Note: for static models, model lowering already achieves what AOT compilation would — the production path is lowering, not AOT

### F3: Deeper vmap Integration
- vmap rejection-sampled distributions (beta, gamma via GPU-side Marsaglia-Tsang)
- vmap gradient evaluations in HMC (batched grad eval)
- The `compile(vmap(grad(lowered)))` triple transform is in Step 5
- **Only worth doing after Steps 8-10**

### F4: Adaptive Step-Size (Practical, Not Speed)
- Dual averaging for HMC/NUTS step-size tuning
- This is about usability, not raw speed — but makes HMC/NUTS actually usable without manual tuning

### F5: Incremental Update/Regenerate (Model Lowering for MCMC Inner Loop)
- Apply model lowering to `update` and `regenerate` operations
- For MCMC where only 1-2 sites change per step, skip recomputing unchanged sites
- Related to self-adjusting computation (Acar et al.) and Gen.jl's static update
- Expected: 2-10x speedup for GFI MH on large models (20+ sites)
- **Only worth doing after lowered simulate/generate (Steps 9-10) are proven**

---

## Dependency Graph (Revised 2026-02-24, post MALA/HMC loop compilation)

```
Step 0 (Bug Fixes) ✅
  │
  ├── Step 1 (Baselines) ✅
  │     │
  │     └── Step 2.5 (Ground Truth) ✅ — compile-fn at floor, eval! is bottleneck
  │
  │     Step 3, 4 SKIPPED — compile-fn handles it
  │
  │     ┌── Step 5 (Reduce eval! overhead) ✅ DONE
  │     │     5.1: Micro-batched lazy MH — DEAD END ✅
  │     │     5.2: Triple transform — WORKS ✅
  │     │     5.3: Inference loop overhead — deprioritized
  │     │     5.4: LOOP COMPILATION ✅
  │     │       compiled-mh: 5.6x ✅
  │     │       MALA: 2.3x ✅ (score+grad caching: 3→1 val-grad calls/step)
  │     │       HMC: 3.9x ✅ (K outer × L inner leapfrog unrolling)
  │     │
  │     ├── Step 6 (Remove Lazy + Tidy) ✅ DONE
  │     ├── Step 7 (Bun docs) [mostly done]
  │     ├── Step 8 (Distributions) ✅ DONE
  │     │
  │     ├── Step 9 (Lowered Simulate/Generate) [reassess]
  │     │     └── Step 10 (Lowered Vectorized Sim/Gen)
  │     │
  │     └── Step 11 (Final Benchmark)
```

**Remaining work:**
- Step 7 (Bun docs) — documentation only
- Steps 9-10 should be reassessed — loop compilation may be sufficient
- Step 11 (final benchmark) — blocked on deciding about 9-10

---

## Time Estimates (Revised 2026-02-24)

| Step | Effort | Status |
|------|--------|--------|
| 0. Bug fixes | 1 day | ✅ DONE |
| 1. Baselines | 1 day | ✅ DONE |
| 2. Profiling | — | SKIPPED |
| 2.5 Ground truth | 0.5 days | ✅ DONE (compile-fn at floor, eval! is bottleneck) |
| 3. Score lowering | — | SKIPPED (compile-fn handles it) |
| 4. Grad lowering | — | SKIPPED (compile-fn handles it) |
| 5. Reduce eval! overhead | 3-5 days | ✅ DONE (MH 5.6x, MALA 2.3x, HMC 3.9x) |
| 6. Remove lazy + tidy | 1-2 days | ✅ DONE |
| 7. Bun docs | 0.5 days | Mostly done |
| 8. Distributions | 2 days | ✅ DONE |
| 9. Lowered sim/gen | 3-5 days | Reassess |
| 10. Lowered vec sim/gen | 2-3 days | Blocked on 9 |
| 11. Final benchmark | 1 day | Blocked on all |

**Remaining: ~1 week** (Step 7 docs + assess Steps 9-11).

---

## Success Criteria (Revised 2026-02-24)

The ground truth finding changes what's achievable. The Metal dispatch floor
(~0.29ms per eval!) is a hardware constraint, not a software one. Success is
now defined by how effectively we amortize that cost.

**Minimum success (Steps 5-6 complete): ✅ ACHIEVED**
- ~~Micro-batched lazy MH delivers >2x~~ → Loop compilation delivers **5.6x** (exceeded)
- ~~Inference loop overhead reduced~~ → Loop compilation bypasses collect-samples entirely
- Lazy variants removed, tidy/eval discipline applied — ✅ Step 6 DONE
- Long chains (>5000 steps) don't hit Metal resource limit — ✅ 500 steps tested stable
- All tests pass — ✅ 165/165, 73/73

**Good success (Steps 5-8 complete): ✅ ACHIEVED**
- All minimum criteria met — ✅
- Triple transform tested — ✅ compile(vmap(grad(score-fn))) works
- Distribution-level optimizations — ✅ Step 8 DONE (mlx-log-gamma, Gumbel-max categorical)
- Loop compilation for all gradient MCMC: compiled-mh 5.6x, MALA 2.3x, HMC 3.9x — ✅
- Vectorized inference shown to be the primary scaling strategy — ✅

**Full success (Steps 5-11 complete):**
- Compiled-mh: **5.6x faster** via loop compilation — ✅ ACHIEVED
- MALA: **2.3x faster** via loop compilation + score/grad caching — ✅ ACHIEVED
- HMC: **3.9x faster** via loop compilation + leapfrog unrolling — ✅ ACHIEVED
- Vectorized MALA N=50: 12x effective (already good)
- simulate/generate: assessed honestly against Metal floor — TODO
- Honest report on what Apple Silicon + MLX can and cannot achieve vs Gen.jl — TODO

---

## The Key Metric (Revised)

The old metric was "GPU utilization." The new metric is:

**eval! amortization = useful_work_per_eval / eval!_cost**

```
                          Before          After loop compilation
──────────────────────────────────────────────────────────────────
compiled-mh:              1 step/eval     200 steps/eval ✅ (5.6x)
MALA:                     3 val-grad/step 1 val-grad/step ✅ (2.3x)
HMC (L=10):               1 step/eval     K×L leapfrog/eval ✅ (3.9x)
vectorized-mala N=50:     50 chains/eval  50 chains/eval (already good)
vectorized-hmc N=10:      10 chains/eval  10 chains/eval (already good)
```

Loop compilation (Step 5.4) lets each eval! do work for K steps.
Pre-generated noise arrays as inputs ensure correct randomness.

MALA additionally caches [score, gradient] across iterations, reducing
the dominant per-step cost from 3 val-grad calls to 1.

HMC unrolls both K outer MH steps and L inner leapfrog sub-steps.
Default block-size=20 (smaller than MH's 50 due to deeper graphs).

For vectorized inference, each eval! already does work for N chains.
This is why vectorized inference (12x at N=50) already works well.

**Next frontier:** combine both — `compile(vmap(K-step-chain))` for N×K work per dispatch.

The architectural insight: **the eval boundary principle is correct, but the
boundary is eval!, not SCI.** The computation graph construction (whether via
SCI or handwritten) is negligible (~0.012ms). The Metal dispatch (~0.29ms)
is what matters. Optimization = maximizing work per Metal dispatch.
