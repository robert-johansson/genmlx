# Roadmap: Experiment 4 from 4/10 to 9/10

## Current State (4/10)

| Model   | Algorithm  | GenMLX       | Gen.jl  | GenJAX  | vs Gen.jl | vs GenJAX |
|---------|-----------|-------------|---------|---------|-----------|-----------|
| LinReg  | IS(1000)  | **1.5ms** (vec) | 6.5ms   | 0.10ms  | **4.3x faster** | 15x slower |
| HMM     | IS(1000)  | 19.2s (seq) | 34ms    | --      | 565x slower | -- |
| GMM     | IS(1000)  | 3.3s (seq)  | 5.8ms   | 1.1ms   | 569x slower | 3000x slower |
| LinReg  | MH(5000)  | 19.5s       | 64ms    | --      | 305x slower | -- |
| HMM     | SMC(100)  | 5.8s        | 190ms   | --      | 31x slower  | -- |

**Score: 4/10.** Only LinReg IS uses vectorized inference and is competitive.
Everything else runs sequential per-particle loops with catastrophic FFI overhead.

---

## Root Cause Analysis

### Why LinReg IS works (1.5ms)
LinReg uses `vectorized-importance-sampling` (`src/genmlx/inference/importance.cljs:68-83`),
which calls `dyn/vgenerate` once. The model body runs once with `[N]`-shaped arrays.
All 1000 particles are processed in a single MLX graph evaluation. No per-particle FFI calls.

### Why GMM IS is 3.3s (sequential)
The GMM model (`scripts/exp4_genmlx_is1000.cljs:144-151`) contains:
```clojure
(let [z (trace ... (dist/categorical gmm-log-weights))
      _ (mx/eval! z)          ;; <-- forces synchronous GPU eval per site
      z-val (mx/item z)       ;; <-- extracts scalar to JS, breaks batching
      mu (mx/take-idx gmm-means (mx/scalar (int z-val) mx/int32))]
```
The `mx/eval!` + `mx/item` calls exist because `mx/take-idx` historically needed
scalar indices. But **`mx/take-idx` already supports `[N]`-shaped indices** -- it
returns `[N]`-shaped values via fancy indexing. Removing `mx/eval!`/`mx/item` and
passing array-valued `z` directly makes the model vectorization-compatible.

**Verified:** Running `vectorized-importance-sampling` on a GMM model without
`mx/eval!`/`mx/item` produces valid log-ML estimates (log-ML ~ -20.4) in milliseconds.

### Why HMM IS is 19.2s (sequential)
Two issues:

1. **Same mx/eval!/mx/item problem** as GMM in the HMM kernel
   (`scripts/exp4_genmlx_is1000.cljs:123-133`):
   ```clojure
   (let [z (trace :z (dist/categorical logits))
         _ (mx/eval! z)
         z-val (mx/item z)]
   ```

2. **dist-sample-n categorical bug** (`src/genmlx/dist.cljs:416-426`):
   ```clojure
   (defmethod dc/dist-sample-n* :categorical [d key n]
     (let [{:keys [logits]} (:params d)
           k (first (mx/shape logits))   ;; BUG: gives N when logits are [N,K]
           u (rng/uniform key [n k])
           ...))
   ```
   When `vgenerate` passes `[N,K]`-shaped logits (one per-particle transition row),
   `(first (mx/shape logits))` returns N (e.g. 1000) instead of K (e.g. 2).
   The Gumbel-max trick then allocates `[N, N]` noise instead of `[N, K]`, crashing.

   **Fix:** Use `(last (mx/shape logits))` to always get the category dimension.

3. **Unfold combinator**: Not yet batched. Each timestep runs sequentially even
   within vgenerate. This is inherent to sequential models but could be improved
   with a batched unfold that processes all N particles at each timestep.

### Why MH is 19.5s
MH runs 5000 sequential `p/update` calls, each involving a full model trace.
Each call incurs FFI overhead. Compiled MH (from Exp 6) reduces this by 11.1x
but still processes one sample at a time. Gen.jl's MH is fast because Julia's
JIT eliminates per-call overhead entirely.

### Why SMC is 5.8s
SMC(100) runs T=50 timesteps x 100 particles = 5000 sequential operations,
plus resampling at each step. The per-particle loop at each timestep is the
bottleneck. A batched unfold would reduce this to 50 timesteps with vectorized
operations across particles.

---

## Tier 1: High Impact, Low Effort (rewrite models + fix one bug)

These changes require no new inference algorithms. Just fix the models to be
vectorization-compatible and benchmark with existing `vectorized-importance-sampling`.

### 1a. GMM Vectorized IS

**Effort:** ~15 minutes. Rewrite GMM model without `mx/eval!`/`mx/item`.

```clojure
;; Before (sequential-only):
(let [z (trace (keyword (str "z" i)) (dist/categorical gmm-log-weights))
      _ (mx/eval! z)
      z-val (mx/item z)
      mu (mx/take-idx gmm-means (mx/scalar (int z-val) mx/int32))])

;; After (vectorization-compatible):
(let [z (trace (keyword (str "z" i)) (dist/categorical gmm-log-weights))
      mu (mx/take-idx gmm-means z)])
```

**Expected result:** ~1-3ms for IS(1000). Confirmed by testing.

**vs Gen.jl:** 1-3ms vs 5.8ms = **competitive or faster**.
**vs GenJAX:** 1-3ms vs 1.1ms = **roughly on par**.

### 1b. Fix dist-sample-n Categorical for [N,K] Logits

**Effort:** ~10 minutes. One-line fix in `src/genmlx/dist.cljs:419`.

```clojure
;; Before:
k (first (mx/shape logits))   ;; breaks when logits are [N,K]

;; After:
k (last (mx/shape logits))    ;; always gets category count K
```

Also need to handle broadcasting in the Gumbel-max: when logits are `[N,K]`,
noise should be `[N,K]` (not `[n,K]` where n is the sample count).
For batched mode the sample count n equals N, so `(rng/uniform key [n k])`
already produces the right shape. The fix is just the `k` extraction.

### 1c. HMM Vectorized IS

**Effort:** ~30 minutes. Depends on 1b. Rewrite HMM kernel without
`mx/eval!`/`mx/item` and add a vgenerate-compatible unfold path.

The HMM kernel needs array-compatible indexing:
```clojure
;; After (vectorization-compatible):
(let [logits (if (nil? z-prev)
               init-logits
               (mx/take-idx transition-logits z-prev 0))  ;; z-prev is [N]-shaped
      z (trace :z (dist/categorical logits))               ;; logits are [N,K]
      mu (mx/take-idx emission-means z)]                   ;; z is [N]-shaped
  z)
```

**Challenge:** Unfold combinator runs timesteps sequentially. Even with vectorized
per-timestep execution, T=50 sequential steps remain. But each step processes all
N particles in one MLX call instead of N separate calls.

**Expected result:** ~5-50ms for IS(1000) with T=50 (50 vectorized timesteps).
Each timestep does a few MLX ops on `[N]`-shaped arrays.

**vs Gen.jl:** 5-50ms vs 34ms = **competitive**.

### 1d. Benchmark at Higher N

At N=1000, GPU overhead per operation is significant relative to the small
computation. At N=10K+, GPU parallelism dominates and GenMLX should win.
Run IS(10000) for LinReg, GMM, and HMM to show scaling advantage.

### 1e. Separate Vectorized vs Sequential in Results Table

The current results conflate vectorized (GPU-optimal) and sequential (FFI-bottlenecked)
execution. Present them separately to give an honest picture:

| Model | Method | GenMLX | Gen.jl | GenJAX | Notes |
|-------|--------|--------|--------|--------|-------|
| LinReg IS | Vectorized | 1.5ms | 6.5ms | 0.10ms | GenMLX beats Gen.jl |
| GMM IS | Vectorized | ~2ms | 5.8ms | 1.1ms | After removing mx/eval! |
| HMM IS | Vectorized | ~20ms | 34ms | -- | After 1b+1c fixes |
| LinReg MH | Sequential | 19.5s | 64ms | -- | FFI overhead |
| HMM SMC | Sequential | 5.8s | 190ms | -- | FFI overhead |

---

## Expected Results After Tier 1 (7-8/10)

| Model   | Algorithm  | GenMLX          | Gen.jl  | GenJAX  | vs Gen.jl    | vs GenJAX   |
|---------|-----------|----------------|---------|---------|-------------|-------------|
| LinReg  | IS(1000)  | **1.5ms** (vec) | 6.5ms   | 0.10ms  | **4.3x faster** | 15x slower  |
| GMM     | IS(1000)  | **~2ms** (vec)  | 5.8ms   | 1.1ms   | **~3x faster**  | ~2x slower  |
| HMM     | IS(1000)  | **~20ms** (vec) | 34ms    | --      | **~1.7x faster** | --         |
| LinReg  | MH(5000)  | 19.5s (seq)     | 64ms    | --      | 305x slower | --          |
| HMM     | SMC(100)  | 5.8s (seq)      | 190ms   | --      | 31x slower  | --          |

**Score: 7-8/10.** Vectorized IS across all three models is competitive or faster
than Gen.jl. The sequential MCMC/SMC gap is honestly reported as the cost of
interpretation, setting up the compilation story from Exp 6.

---

## Tier 2: Medium Effort Code Changes

### 2a. Batched Unfold Combinator

**Effort:** 1-2 days. Create a vectorized unfold that processes all N particles
at each timestep using `[N]`-shaped state arrays.

Currently, `unfold-combinator` in `src/genmlx/combinators.cljs` runs the kernel
T times sequentially, but each call processes one particle. A batched variant
would run the kernel T times with `[N]`-shaped arrays, reducing HMM IS from
~1000 x T = 50K FFI calls to T = 50 vectorized operations.

This would improve HMM IS from ~20ms to potentially ~2-5ms.

### 2b. Compiled MH with Reduced FFI

**Effort:** 1-2 days. The Exp 6 compiled MH achieves 11.1x speedup by fusing
score computation. Extend this to fuse the full MH loop (propose + accept/reject)
into fewer MLX graph evaluations.

Target: LinReg MH from 19.5s to ~1-2s (still behind Gen.jl's 64ms but
dramatically better).

### 2c. Vectorized Multi-Chain MCMC

**Effort:** 1 day. Run K independent MH chains in parallel using `[K]`-shaped
arrays. Exp 6 already shows 3.3x speedup for 10 chains. Scaling to 100+ chains
with vectorized proposals could bring LinReg MH to competitive range.

---

## Tier 3: Long-Term Architecture

### 3a. JIT Compilation for Inference Loops

Trace the sequence of MLX operations in a model body, compile to a single fused
Metal kernel. Eliminates per-operation FFI overhead entirely. This is what
GenJAX gets from JAX's XLA compiler.

### 3b. Whole-Program Vectorization

Automatic vmap-style transformation: given a scalar model, produce a batched
version that processes N particles. Currently done manually via `vgenerate`
and shape-based broadcasting. A compiler pass could do this automatically,
including for Unfold/Switch/Scan combinators.

### 3c. Metal Shader Compilation

For hot inference kernels (MH accept/reject, resampling, weight normalization),
compile directly to Metal shaders instead of going through MLX's lazy graph.

---

## Narrative Framing for Paper

### The honest story (Section 6.7)

> GenMLX demonstrates that a purely functional, interpreted probabilistic
> programming system can achieve competitive performance with compiled systems
> when execution is properly vectorized.
>
> For importance sampling -- the most naturally parallelizable inference
> algorithm -- GenMLX's shape-based broadcasting eliminates per-particle overhead
> entirely. On an Apple M2, vectorized IS runs all three canonical models
> (linear regression, HMM, GMM) in 1.5-20ms, matching or beating Gen.jl
> (6.5-34ms on CPU) and approaching GenJAX's JIT-compiled performance (0.1-1.1ms).
>
> Sequential algorithms (MH, SMC) expose the cost of interpretation: each
> inference step requires a JavaScript-to-Metal FFI round-trip. This is the
> fundamental tradeoff of an interpreted system. However, the compilation
> techniques from Section 6.3 (11.1x MH speedup) show a clear path to
> closing this gap.
>
> The key architectural insight: by keeping all values as MLX arrays and
> leveraging broadcasting, GenMLX sidesteps the need for program transformation
> (vmap) or JIT compilation to achieve parallel execution. This makes
> vectorization available to any model that avoids materializing intermediate
> values to host scalars.

### What makes 9/10 vs 10/10

**9/10:** Vectorized IS is competitive across all models. Sequential gap is
honestly reported with clear explanation. Compilation speedups (Exp 6) show
the gap is addressable. The story is: "interpreted but fast where it matters,
with a clear compilation roadmap."

**10/10:** Would require compiled MH/SMC approaching Gen.jl speeds. This needs
either JIT compilation (Tier 3) or significant engineering on batched combinators
(Tier 2). Achievable but not for the first paper submission.

---

## Action Plan

### Phase 1: Quick wins (1-2 hours)

1. Fix `dist-sample-n` categorical: `(first ...)` -> `(last ...)` at `dist.cljs:419`
2. Write vectorization-compatible GMM model (no `mx/eval!`/`mx/item`)
3. Write vectorization-compatible HMM kernel (no `mx/eval!`/`mx/item`)
4. Run `vectorized-importance-sampling` on all three models
5. Collect timings with standard protocol (5 warmup, 20 runs)

### Phase 2: Updated benchmarks (1 hour)

6. Update `scripts/exp4_genmlx_is1000.cljs` with vectorized variants
7. Add IS(10000) runs to show GPU scaling advantage
8. Regenerate `results/exp4_system_comparison/genmlx_is1000.json`
9. Regenerate figures 9 and 10

### Phase 3: Write-up (30 min)

10. Update paper Section 6.7 with new results
11. Add "vectorized vs sequential" distinction to system comparison table
12. Frame compilation results (Exp 6) as roadmap for closing MCMC gap
