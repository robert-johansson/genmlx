# MCMC Plan: Closing the Performance Gap with Gen.jl

## Problem Statement

GenMLX's compiled MH is **75x slower** than Gen.jl on linear regression
(4841ms vs 64ms). The bottleneck is not the score function — it's the
per-step host overhead: PRNG splits, materialization, callback dispatch,
and cache cleanup happening in ClojureScript between every Metal dispatch.

The solution is proven: compiled Adam achieved **9.2x speedup** by fusing
gradient + optimizer into a single `mx/compile-fn` call. The same pattern
applies to MCMC chains.

---

## Root Cause Analysis

### Where Time Goes in Compiled MH Today

The current `compiled-mh` in `mcmc.cljs` has three levels of compilation:

```
Level 1: Score function      — mx/compile-fn(raw-score-fn)        ✓ compiled
Level 2: K-step chain block  — mx/compile-fn(K-step loop)         ✓ compiled
Level 3: Outer sample loop   — ClojureScript loop/recur           ✗ interpreted
```

The K-step chain block is compiled (`make-compiled-chain`, lines 165-189):

```clojure
(fn [init-params noise-2d uniforms-1d]
  (loop [p init-params, i 0]
    (if (>= i k-steps)
      p
      (let [row (mx/reshape (mx/take-idx noise-2d ...) [n-params])
            proposal (mx/add p (mx/multiply proposal-std row))
            s-cur (score-fn p)           ;; compiled score
            s-prop (score-fn proposal)   ;; compiled score
            log-alpha (mx/subtract s-prop s-cur)
            log-u (mx/log (mx/index uniforms-1d i))
            accept? (mx/greater log-alpha log-u)]
        (recur (mx/where accept? proposal p) (inc i))))))
```

This is good — K MH steps fuse into one Metal dispatch. But the **outer loop**
(`run-loop-compiled-mh`, lines 233-329) is still per-block:

```clojure
(loop [p init-params, acc (transient []), i 0, rk rk]
  ;; PER-BLOCK HOST OVERHEAD:
  (let [[k1 k2 rk'] (rng/split-n rk 3)           ;; 1. PRNG split (host)
        noise (rng/normal k1 [block-size n-params]) ;; 2. noise generation (host→Metal)
        uniforms (rng/uniform k2 [block-size])      ;; 3. uniform generation (host→Metal)
        p' (chain-fn p noise uniforms)]              ;; 4. compiled chain (Metal)
    (mx/materialize! p')                             ;; 5. materialize (sync)
    (mx/clear-cache!)                                ;; 6. cache cleanup (host)
    (when callback (callback ...))                   ;; 7. callback (host + mx/->clj)
    (recur p' (conj! acc (mx/->clj p')) (inc i) rk')))
```

**Per-block cost breakdown** (for 1000 samples, block-size=25, 40 blocks):

| Operation | Per-Block | 40 Blocks | Notes |
|-----------|-----------|-----------|-------|
| `rng/split-n` | ~0.01ms | 0.4ms | Host-side PRNG |
| `rng/normal` + `rng/uniform` | ~0.1ms | 4ms | Metal dispatch for noise |
| `chain-fn` | ~0.5ms | 20ms | Metal dispatch for K steps |
| `mx/materialize!` | ~0.2ms | 8ms | Metal→host sync barrier |
| `mx/clear-cache!` | ~0.1ms | 4ms | Metal cache flush |
| `mx/->clj` | ~0.1ms | 4ms | Array extraction |
| Callback | ~0.05ms | 2ms | ClojureScript function call |
| **Total** | ~1.1ms | **42ms** | **50% is overhead, not compute** |

Gen.jl does none of this. Its compiled loop runs entirely inside Julia's
compiled code, with no per-step FFI crossings, no sync barriers, no cache
flushes.

### The Proven Pattern: Compiled Adam

`compiled_optimizer.cljs` shows how to eliminate this overhead:

```clojure
;; ONE compiled function wraps: gradient + moment update + bias correction + param update
(let [step-fn
      (fn [params m v t-scalar]
        (let [[loss grad] (vg params)
              new-m (mx/add (mx/multiply beta1-s m) (mx/multiply (mx/subtract one-s beta1-s) grad))
              new-v (mx/add (mx/multiply beta2-s v) (mx/multiply (mx/subtract one-s beta2-s) (mx/multiply grad grad)))
              m-hat (mx/divide new-m (mx/subtract one-s (mx/power beta1-s t-scalar)))
              v-hat (mx/divide new-v (mx/subtract one-s (mx/power beta2-s t-scalar)))
              new-p (mx/subtract params (mx/divide (mx/multiply lr-s m-hat)
                                                    (mx/add (mx/sqrt v-hat) eps-s)))]
          #js [new-p new-m new-v loss]))
      compiled (mx/compile-fn step-fn)]

  ;; Outer loop: ONE compiled call per iteration
  (loop [i 0, params init-params, m zeros, v zeros]
    (let [t-scalar (mx/scalar (double (inc i)))
          result (compiled params m v t-scalar)]   ;; ← ONE Metal dispatch
      ;; Materialize ONLY at log boundaries (not every step!)
      (when (zero? (mod (inc i) log-every))
        (mx/materialize! (aget result 3))))))
```

**Key insights:**
1. All math fused into one `mx/compile-fn`
2. State threaded as MLX arrays (params, m, v) — never extracted to host
3. Iteration counter as MLX scalar (`t-scalar`) — stays in graph
4. Materialization only at log boundaries, not every iteration
5. No per-step PRNG (noise is pre-generated or deterministic)

---

## Strategy: Three-Level Fusion

### Level A: Fully Fused Burn-In

Pre-generate ALL noise for the entire burn-in phase. Run ONE compiled function
that executes B×K MH steps without returning to ClojureScript.

```
Host: generate noise [B*K, D], uniforms [B*K]
Host: call compiled burn-in function
Metal: execute B*K MH steps
Host: receive final params
```

**No per-block overhead.** One Metal dispatch for the entire burn-in.

### Level B: Block-Fused Collection

For sample collection, pre-generate noise for S blocks of K steps each.
Run ONE compiled function that returns S sample points.

```
Host: generate noise [S*K, D], uniforms [S*K]
Host: call compiled collection function
Metal: execute S*K MH steps, save params every K steps → [S, D] tensor
Host: receive [S, D] sample tensor
```

**One Metal dispatch for all samples.** No per-sample extraction.

### Level C: Fully Fused End-to-End

Combine burn-in + collection into a single compiled function. Only
materialize the final `[S, D]` sample tensor.

```
Host: generate noise [(B+S)*K, D], uniforms [(B+S)*K]
Host: call compiled end-to-end function
Metal: burn B*K steps → collect S*K steps → [S, D]
Host: receive [S, D]
```

---

## Detailed Implementation Plan

### Phase 1: Pre-Generated Noise for Entire Chain

**Goal:** Eliminate per-block PRNG overhead by generating all noise upfront.

#### 1.1: Noise Pre-Generation

```clojure
(defn pre-generate-chain-noise [key total-steps n-params]
  "Generate all noise for a complete MH chain.
   Returns {:noise [total-steps, n-params] :uniforms [total-steps]}."
  (let [[k1 k2] (rng/split key)
        noise (rng/normal k1 [total-steps n-params])
        uniforms (rng/uniform k2 [total-steps])]
    (mx/materialize! noise uniforms)
    {:noise noise :uniforms uniforms}))
```

**Key design:** Materialize noise ONCE. Pass as argument to compiled chain.
This is the same pattern used by `fused-mcmc-train` in
`compiled_optimizer.cljs`.

#### 1.2: Modified Chain Builder

Current `make-compiled-chain` takes `[K, D]` noise and `[K]` uniforms for
one block. Extend to take `[T, D]` noise with a start offset:

```clojure
(defn make-compiled-chain-with-offset [score-fn proposal-std n-params k-steps]
  "Build a compiled K-step MH chain that indexes into pre-generated noise
   starting at a given offset."
  (fn [init-params all-noise all-uniforms offset]
    (loop [p init-params, i 0]
      (if (>= i k-steps)
        p
        (let [step-idx (mx/add offset (mx/scalar i mx/int32))
              row (mx/reshape (mx/take-idx all-noise step-idx 0) [n-params])
              proposal (mx/add p (mx/multiply proposal-std row))
              s-cur (score-fn p)
              s-prop (score-fn proposal)
              log-alpha (mx/subtract s-prop s-cur)
              log-u (mx/log (mx/take-idx all-uniforms step-idx 0))
              accept? (mx/greater log-alpha log-u)]
          (recur (mx/where accept? proposal p) (inc i)))))))
```

#### 1.3: Fused Burn-In Function

```clojure
(defn make-fused-burn-in [score-fn proposal-std n-params n-burn-steps]
  "Compile a function that runs n-burn-steps MH iterations in one Metal dispatch.
   Takes pre-generated noise and returns final params."
  (let [chain-fn
        (fn [init-params noise uniforms]
          (loop [p init-params, i 0]
            (if (>= i n-burn-steps)
              p
              (let [row (mx/reshape (mx/take-idx noise (mx/scalar i mx/int32) 0) [n-params])
                    proposal (mx/add p (mx/multiply proposal-std row))
                    s-cur (score-fn p)
                    s-prop (score-fn proposal)
                    log-alpha (mx/subtract s-prop s-cur)
                    log-u (mx/log (mx/take-idx uniforms (mx/scalar i mx/int32) 0))
                    accept? (mx/greater log-alpha log-u)]
                (recur (mx/where accept? proposal p) (inc i))))))]
    (mx/compile-fn chain-fn)))
```

**Total burn-in: ONE Metal dispatch.** No intermediate materialization.

### Phase 2: Trajectory Collection in One Dispatch

**Goal:** Collect S samples in a single compiled function call by saving
params at regular intervals within the compiled loop.

#### 2.1: The Challenge: Accumulating Samples Inside Compiled Loop

MLX's `mx/compile-fn` requires fixed tensor shapes. We can't dynamically
grow a list inside the compiled graph. Solution: **pre-allocate the output
tensor and write samples into it.**

```clojure
(defn make-fused-collection [score-fn proposal-std n-params
                             thin n-samples]
  "Compile a function that runs thin*n-samples MH steps and returns
   [n-samples, n-params] tensor of thinned samples."
  (let [total-steps (* thin n-samples)
        collect-fn
        (fn [init-params noise uniforms]
          ;; Run total-steps MH iterations
          ;; Every `thin` steps, record the current params
          ;; Return stacked [n-samples, n-params] tensor
          (loop [p init-params
                 i 0
                 samples (mx/zeros [n-samples n-params])]
            (if (>= i total-steps)
              samples
              (let [;; MH step
                    row (mx/reshape (mx/take-idx noise (mx/scalar i mx/int32) 0) [n-params])
                    proposal (mx/add p (mx/multiply proposal-std row))
                    s-cur (score-fn p)
                    s-prop (score-fn proposal)
                    log-alpha (mx/subtract s-prop s-cur)
                    log-u (mx/log (mx/take-idx uniforms (mx/scalar i mx/int32) 0))
                    accept? (mx/greater log-alpha log-u)
                    new-p (mx/where accept? proposal p)
                    ;; Record sample every `thin` steps
                    sample-idx (mx/floor-divide (mx/scalar i mx/int32)
                                                (mx/scalar thin mx/int32))
                    is-record-step (mx/equal (mx/remainder (mx/scalar i mx/int32)
                                                           (mx/scalar thin mx/int32))
                                             (mx/scalar 0 mx/int32))
                    ;; Conditional write: update row sample-idx if is-record-step
                    new-samples (write-sample-if samples new-p sample-idx is-record-step n-params)]
                (recur new-p (inc i) new-samples)))))]
    (mx/compile-fn collect-fn)))
```

#### 2.2: In-Graph Sample Recording

The `write-sample-if` function updates one row of the output tensor
conditionally, using `mx/where`:

```clojure
(defn- write-sample-if [samples params sample-idx should-write n-params]
  "Conditionally write params into row sample-idx of samples tensor.
   Uses mx/where to avoid data-dependent branching."
  (let [;; Create one-hot row selector: [n-samples, 1]
        indices (mx/arange 0 (first (mx/shape samples)) 1 mx/int32)
        row-mask (mx/equal indices sample-idx)  ;; [n-samples] boolean
        ;; Combine: should we write AND is this the target row?
        write-mask (mx/logical-and row-mask
                                   (mx/broadcast-to should-write
                                                    (mx/shape row-mask)))
        ;; Expand for broadcasting: [n-samples, 1]
        write-mask-2d (mx/expand-dims write-mask 1)
        ;; New row value broadcast to [n-samples, n-params]
        new-row (mx/broadcast-to (mx/reshape params [1 n-params])
                                 (mx/shape samples))]
    ;; Where write-mask: use new-row, else keep old samples
    (mx/where write-mask-2d new-row samples)))
```

**Alternative (simpler):** If `mx/compile-fn` supports returning JS arrays,
accumulate samples as a vector and `mx/stack` at the end:

```clojure
;; Inside compiled loop, collect samples into #js array:
;; Note: this may not work with mx/compile-fn — needs investigation.
;; Fallback: use the mx/where row-write approach above.
```

#### 2.3: Alternative: Block-Based Collection

If in-graph sample recording is too complex, use block-based collection
with pre-generated noise for all blocks:

```clojure
(defn fused-block-collection [score-fn proposal-std n-params
                              block-size n-blocks]
  "Run n-blocks of block-size MH steps each. Pre-generate all noise.
   Return [n-blocks, n-params] tensor (one sample per block)."
  (let [total-steps (* block-size n-blocks)
        {:keys [noise uniforms]} (pre-generate-chain-noise key total-steps n-params)
        ;; Build compiled chain for one block
        block-chain (make-compiled-chain score-fn proposal-std n-params block-size)
        compiled-block (mx/compile-fn block-chain)]

    ;; Run all blocks WITHOUT per-block overhead
    (loop [p init-params, samples [], i 0]
      (if (>= i n-blocks)
        (mx/stack samples)
        (let [offset (* i block-size)
              block-noise (mx/slice noise offset (+ offset block-size))
              block-uniforms (mx/slice uniforms offset (+ offset block-size))
              p' (compiled-block p block-noise block-uniforms)]
          ;; NO materialize here — stay lazy!
          (recur p' (conj samples p') (inc i)))))))
```

**Key insight:** Don't materialize between blocks. Let MLX's lazy graph
fuse consecutive block calls. Only materialize the final stacked tensor.

**Caution:** MLX's graph size may grow linearly with n-blocks. May need
to materialize every M blocks (e.g., M=10) as a compromise:

```clojure
(when (zero? (mod i materialize-every))
  (mx/materialize! p'))
```

This is still M× fewer materializations than the current approach.

### Phase 3: End-to-End Fused Chain

**Goal:** Single function call for burn-in + collection.

#### 3.1: Combined Compiled Function

```clojure
(defn make-fused-mh-chain [score-fn proposal-std n-params
                           n-burn n-samples thin]
  "Compile a complete MH chain: burn-in + thinned collection.
   Returns [n-samples, n-params] tensor of posterior samples."
  (let [total-steps (+ n-burn (* n-samples thin))
        chain-fn
        (fn [init-params noise uniforms]
          (loop [p init-params, i 0, sample-count 0
                 samples (mx/zeros [n-samples n-params])]
            (if (>= i total-steps)
              #js [p samples]  ;; Return final state + all samples
              (let [;; MH step (same as before)
                    row (mx/reshape (mx/take-idx noise (mx/scalar i mx/int32) 0) [n-params])
                    proposal (mx/add p (mx/multiply proposal-std row))
                    s-cur (score-fn p)
                    s-prop (score-fn proposal)
                    log-alpha (mx/subtract s-prop s-cur)
                    log-u (mx/log (mx/take-idx uniforms (mx/scalar i mx/int32) 0))
                    accept? (mx/greater log-alpha log-u)
                    new-p (mx/where accept? proposal p)
                    ;; After burn-in, record every `thin` steps
                    past-burn? (mx/greater-equal (mx/scalar i mx/int32) (mx/scalar n-burn mx/int32))
                    burn-offset (mx/subtract (mx/scalar i mx/int32) (mx/scalar n-burn mx/int32))
                    is-thin-step (mx/equal (mx/remainder burn-offset (mx/scalar thin mx/int32))
                                           (mx/scalar 0 mx/int32))
                    should-record (mx/logical-and past-burn? is-thin-step)
                    new-samples (write-sample-if samples new-p sample-count should-record n-params)
                    new-count (mx/add sample-count
                                      (mx/astype should-record mx/int32))]
                (recur new-p (inc i) new-count new-samples)))))]
    (mx/compile-fn chain-fn)))
```

#### 3.2: Public API

```clojure
(defn fused-mh [model args observations
                {:keys [burn-in samples thin proposal-std
                        addresses key]
                 :or {burn-in 500 samples 1000 thin 1
                      proposal-std 0.1}}]
  "Fully fused MH chain. One Metal dispatch for entire chain.
   Returns {:samples [S, D] :final-params [D]}."
  (let [;; Build tensor-native score function
        raw-score-fn (u/prepare-mcmc-score model args observations addresses)
        score-fn (first raw-score-fn)  ;; uncompiled — chain builder compiles the whole thing
        n-params (second raw-score-fn)

        ;; Pre-generate ALL noise
        total-steps (+ burn-in (* samples thin))
        {:keys [noise uniforms]} (pre-generate-chain-noise key total-steps n-params)

        ;; Build + compile fused chain
        fused (make-fused-mh-chain score-fn (mx/scalar proposal-std)
                                   n-params burn-in samples thin)

        ;; Initialize from prior
        init-params (initialize-params model args observations addresses key)

        ;; ONE Metal dispatch
        result (fused init-params noise uniforms)]
    (mx/materialize! result)
    {:samples (aget result 1)         ;; [samples, n-params]
     :final-params (aget result 0)})) ;; [n-params]
```

### Phase 4: Fused MALA/HMC

**Goal:** Apply the same fusion pattern to gradient-based MCMC.

#### 4.1: Fused MALA Chain

MALA threads score + gradient through iterations (already done in
`make-compiled-mala-chain`). Extend to full fusion:

```clojure
(defn make-fused-mala-chain [score-fn proposal-std n-params
                             n-burn n-samples thin]
  "Fused MALA: gradient-informed proposals, full chain in one dispatch."
  (let [total-steps (+ n-burn (* n-samples thin))
        val-grad-fn (mx/value-and-grad score-fn)
        chain-fn
        (fn [init-params noise uniforms]
          (let [[init-score init-grad] (val-grad-fn init-params)]
            (loop [p init-params, sp init-score, gp init-grad
                   i 0, sample-count 0
                   samples (mx/zeros [n-samples n-params])]
              (if (>= i total-steps)
                #js [p samples]
                (let [;; MALA proposal: q = p + (std²/2)*grad + std*noise
                      std-sq-half (mx/multiply (mx/scalar 0.5)
                                               (mx/multiply proposal-std proposal-std))
                      drift (mx/multiply std-sq-half gp)
                      row (mx/reshape (mx/take-idx noise (mx/scalar i mx/int32) 0) [n-params])
                      q (mx/add p (mx/add drift (mx/multiply proposal-std row)))

                      ;; Score + gradient at proposal
                      [sq gq] (val-grad-fn q)

                      ;; MALA acceptance (includes proposal asymmetry correction)
                      fwd-log-q (mala-log-proposal p q gp proposal-std)
                      bwd-log-q (mala-log-proposal q p gq proposal-std)
                      log-alpha (mx/add (mx/subtract sq sp)
                                        (mx/subtract bwd-log-q fwd-log-q))
                      log-u (mx/log (mx/take-idx uniforms (mx/scalar i mx/int32) 0))
                      accept? (mx/greater log-alpha log-u)

                      ;; Accept/reject (thread score + grad)
                      new-p (mx/where accept? q p)
                      new-sp (mx/where accept? sq sp)
                      new-gp (mx/where (mx/expand-dims accept? 0) gq gp)

                      ;; Sample recording
                      past-burn? (mx/greater-equal (mx/scalar i mx/int32) (mx/scalar n-burn mx/int32))
                      new-samples (write-sample-if-past-burn
                                    samples new-p sample-count past-burn? thin i n-burn n-params)
                      new-count (update-sample-count sample-count past-burn? thin i n-burn)]
                  (recur new-p new-sp new-gp (inc i) new-count new-samples))))))]
    (mx/compile-fn chain-fn)))
```

**Key advantage over current MALA:** Current `mala-step` computes 3 val-grad
calls per step (lines 87-112 of mcmc.cljs). Fused MALA threads score+grad
through iterations — **1 val-grad per step** (saves 2/3 of gradient compute).

#### 4.2: Fused HMC Chain

HMC is more complex (leapfrog integration), but the pattern is the same.
Pre-generate momentum noise and acceptance uniforms:

```clojure
(defn make-fused-hmc-chain [score-fn n-params step-size n-leapfrog
                            n-burn n-samples thin]
  (let [total-steps (+ n-burn (* n-samples thin))
        grad-fn (mx/grad score-fn)
        chain-fn
        (fn [init-params momentum-noise uniforms]
          (loop [q init-params, i 0, sample-count 0
                 samples (mx/zeros [n-samples n-params])]
            (if (>= i total-steps)
              #js [q samples]
              (let [;; Fresh momentum from pre-generated noise
                    p0 (mx/reshape (mx/take-idx momentum-noise (mx/scalar i mx/int32) 0) [n-params])
                    ;; Leapfrog integration (L steps inside compiled graph)
                    [q-prop p-prop] (leapfrog-steps q p0 grad-fn step-size n-leapfrog)
                    ;; HMC acceptance
                    H-cur (mx/subtract (score-fn q) (mx/multiply (mx/scalar 0.5) (mx/sum (mx/multiply p0 p0))))
                    H-prop (mx/subtract (score-fn q-prop) (mx/multiply (mx/scalar 0.5) (mx/sum (mx/multiply p-prop p-prop))))
                    log-alpha (mx/subtract H-prop H-cur)
                    log-u (mx/log (mx/take-idx uniforms (mx/scalar i mx/int32) 0))
                    accept? (mx/greater log-alpha log-u)
                    new-q (mx/where accept? q-prop q)
                    ;; Recording...
                    ...]
                (recur new-q (inc i) new-count new-samples)))))]
    (mx/compile-fn chain-fn)))
```

Where `leapfrog-steps` is a compiled sub-loop:

```clojure
(defn- leapfrog-steps [q p grad-fn step-size n-steps]
  "L leapfrog integration steps. Pure MLX, no host interaction."
  (let [half-step (mx/multiply (mx/scalar 0.5) step-size)]
    (loop [q q
           p (mx/add p (mx/multiply half-step (grad-fn q)))  ;; half step
           l 0]
      (if (>= l (dec n-steps))
        ;; Final: full q step, half p step
        (let [q' (mx/add q (mx/multiply step-size p))
              p' (mx/add p (mx/multiply half-step (grad-fn q')))]
          [q' p'])
        ;; Interior: full q step, full p step
        (let [q' (mx/add q (mx/multiply step-size p))
              p' (mx/add p (mx/multiply step-size (grad-fn q')))]
          (recur q' p' (inc l)))))))
```

### Phase 5: Vectorized Fused Chains (N Parallel Chains)

**Goal:** Run N independent chains in parallel, all in one Metal dispatch.

#### 5.1: Vectorized Score Function

The existing `make-vectorized-score-fn` in `util.cljs` already handles
`[N, D]`-shaped params → `[N]`-shaped scores via broadcasting:

```clojure
;; params shape: [N, D]
;; score-fn(params) shape: [N]
```

#### 5.2: Vectorized Fused Chain

```clojure
(defn make-fused-vectorized-mh [score-fn proposal-std n-params n-chains
                                 n-burn n-samples thin]
  "N parallel MH chains, fully fused. Returns [N, S, D] sample tensor."
  (let [total-steps (+ n-burn (* n-samples thin))
        chain-fn
        (fn [init-params noise uniforms]
          ;; init-params: [N, D]
          ;; noise: [total-steps, N, D]
          ;; uniforms: [total-steps, N]
          (loop [p init-params, i 0, sample-count 0
                 samples (mx/zeros [n-chains n-samples n-params])]
            (if (>= i total-steps)
              #js [p samples]
              (let [;; All N proposals at once
                    row (mx/take-idx noise (mx/scalar i mx/int32) 0)  ;; [N, D]
                    proposal (mx/add p (mx/multiply proposal-std row))
                    s-cur (score-fn p)           ;; [N]
                    s-prop (score-fn proposal)   ;; [N]
                    log-alpha (mx/subtract s-prop s-cur)  ;; [N]
                    u-row (mx/take-idx uniforms (mx/scalar i mx/int32) 0)  ;; [N]
                    log-u (mx/log u-row)
                    accept? (mx/greater log-alpha log-u)  ;; [N] boolean
                    new-p (mx/where (mx/expand-dims accept? 1) proposal p)  ;; [N, D]
                    ;; Record for all chains simultaneously
                    new-samples (write-vectorized-sample
                                  samples new-p sample-count
                                  i n-burn thin n-chains n-params)]
                (recur new-p (inc i) (inc sample-count) new-samples)))))]
    (mx/compile-fn chain-fn)))
```

**Expected speedup:** For N chains, vectorized is ~N× faster than sequential
(GPU parallelism on the score function). Combined with chain fusion, this
should approach Gen.jl performance.

### Phase 6: Diagnostics Integration

**Goal:** Compute acceptance rate and other diagnostics inside the compiled
graph, without per-step callbacks.

#### 6.1: In-Graph Acceptance Tracking

```clojure
;; Inside fused chain loop, accumulate acceptance count:
(let [accept-count (mx/add accept-count (mx/astype accept? mx/float32))]
  ;; At the end:
  (let [acceptance-rate (mx/divide accept-count (mx/scalar total-steps))]
    #js [final-params samples acceptance-rate]))
```

#### 6.2: In-Graph Score Tracking

```clojure
;; Track running score for convergence diagnostics:
(let [score-sum (mx/add score-sum (score-fn new-p))
      score-sq-sum (mx/add score-sq-sum (mx/multiply (score-fn new-p) (score-fn new-p)))]
  ;; Compute mean/var at the end (no per-step extraction)
  ...)
```

#### 6.3: Post-Chain Diagnostics

After the fused chain returns `[S, D]` samples:

```clojure
(let [result (fused-mh model args obs {:burn-in 500 :samples 1000 :thin 2})
      samples (:samples result)]          ;; [1000, D]
  ;; R-hat requires multiple chains:
  (let [chains (mx/reshape samples [4 250 D])]  ;; split into 4 sub-chains
    (diagnostics/r-hat chains)))
```

---

## Score Function Optimization

### Current Score Function Hierarchy

```
Level 0: GFI score         — p/generate → extract weight      (slowest)
Level 1: Compiled generate — compiled p/generate → weight      (faster)
Level 2: Tensor-native     — flat MLX ops, no handler dispatch (fastest)
```

`util.cljs` already selects the best available:

```clojure
(defn prepare-mcmc-score [model args observations addresses]
  (let [schema (:schema model)]
    (cond
      ;; Tensor-native: schema has compiled sites, all static
      (and schema (:static? schema) (:dep-order schema))
      [:tensor-native (make-tensor-score-fn ...) (count addresses)]

      ;; Compiled generate: model has compiled-generate
      (:compiled-generate schema)
      [:compiled (make-compiled-score-fn ...) (count addresses)]

      ;; Fallback: GFI
      :else
      [:gfi (make-score-fn ...) (count addresses)])))
```

### Optimization: Ensure Tensor-Native Score for Fused Chains

For fused chains to achieve maximum speedup, the score function MUST be
tensor-native (Level 2). Otherwise the compiled chain calls an interpreted
score function, negating most of the fusion benefit.

```clojure
;; In fused-mh, validate score level:
(let [[score-level score-fn n-params] (u/prepare-mcmc-score model args obs addrs)]
  (when (= score-level :gfi)
    (println "Warning: fused-mh using GFI score (slow). Consider static model."))
  ...)
```

### New: Conjugate-Aware Fused Score

For L3-augmented models, the score function should exclude analytically
eliminated addresses (already done in `util.cljs` via
`make-conjugate-aware-score-fn`). Ensure this composes with fusion:

```clojure
;; Residual addresses only (non-eliminated)
(let [residual-addrs (remove (:eliminated-addrs schema) all-addrs)
      score-fn (u/make-conjugate-aware-score-fn model args obs residual-addrs)]
  ;; Fused chain operates on fewer dimensions
  (make-fused-mh-chain score-fn proposal-std (count residual-addrs) ...))
```

---

## MLX Compilation Constraints

### Known Limitations of mx/compile-fn

From the gotchas in MEMORY.md and CLAUDE.md:

1. **`mx/compile-fn` + PRNG:** Compiled functions cache random ops
   (deterministic regardless of key). **Solution:** Pre-generate all noise
   outside compiled function, pass as argument. (Already used by
   `fused-mcmc-train`.)

2. **`mx/compile-fn` + `mx/where`:** Compiled graph traces with fixed
   condition value. **Solution:** Use `mx/where` with boolean mask computed
   from data (not from host-side branching). The acceptance check
   `(mx/greater log-alpha log-u)` produces a data-dependent mask — this
   should work correctly.

3. **Graph size grows with loop iterations.** A 10,000-step loop inside
   `mx/compile-fn` creates a 10,000-node graph. MLX may struggle with very
   large graphs. **Solution:** Use block-based fusion (e.g., blocks of 100
   steps) if single-dispatch fusion hits memory limits.

4. **Nested `mx/compile-fn`:** Calling a compiled function from inside
   another compiled function may cause nested dispatch. **Solution:** Pass
   the raw (uncompiled) score function to the chain builder. Let the chain
   builder compile the whole thing.

### Investigation Gates

Before implementing, verify these with experiments:

**Gate 1:** Can `mx/compile-fn` handle a 1000-step loop with `mx/where`
accept/reject? (Graph size test.)

```clojure
(let [f (mx/compile-fn
          (fn [p noise uniforms]
            (loop [p p, i 0]
              (if (>= i 1000) p
                (let [proposal (mx/add p (mx/take-idx noise (mx/scalar i mx/int32) 0))
                      accept? (mx/greater (mx/take-idx uniforms (mx/scalar i mx/int32) 0)
                                          (mx/scalar 0.5))]
                  (recur (mx/where accept? proposal p) (inc i)))))))]
  (time (mx/materialize! (f (mx/zeros [3]) (rng/normal k [1000 3]) (rng/uniform k2 [1000])))))
```

**Gate 2:** Does `mx/where` with data-dependent boolean produce correct
gradients through the chain? (Needed for fused MALA/HMC.)

**Gate 3:** What's the maximum loop iteration count before graph compilation
becomes prohibitively slow? (Determines block size for block-based fusion.)

**Gate 4:** Can `mx/value-and-grad` wrap a function that contains `mx/where`?
(Needed for MALA gradient threading.)

---

## Testing Plan

### Correctness Tests

```clojure
;; C1: Fused MH produces same posterior mean as handler MH (statistical)
;; C2: Fused burn-in reaches same stationary distribution as block burn-in
;; C3: Acceptance rate matches between fused and handler paths
;; C4: Pre-generated noise is statistically equivalent to per-step generation
;; C5: In-graph sample recording matches per-step sample extraction
;; C6: Fused MALA acceptance rate matches handler MALA
;; C7: Fused HMC energy conservation (H(q,p) stable over leapfrog)
;; C8: Vectorized N-chain results match N independent scalar chains
;; C9: Thin>1 produces correct subsampling
;; C10: Conjugate-aware fused chain matches full-dimension chain on residual
```

### Performance Benchmarks

```clojure
;; B1: Fused MH vs handler MH on linreg (T=50 obs, 1000 samples)
;;     Target: 10-50x speedup
;; B2: Fused MH vs current compiled-mh (same model)
;;     Target: 3-5x speedup (eliminating per-block overhead)
;; B3: Fused MH vs Gen.jl on linreg
;;     Target: within 5x of Gen.jl (down from 75x)
;; B4: Fused MALA vs handler MALA
;;     Target: 5-20x speedup (fewer val-grad calls + fusion)
;; B5: Fused HMC vs handler HMC
;;     Target: 5-15x speedup
;; B6: Vectorized N=8 chains vs 8 sequential chains
;;     Target: ~8x speedup
;; B7: Scaling: fused chain speedup vs chain length
;;     Expected: larger chains → larger speedup (amortize compilation)
;; B8: Graph size: compile time vs loop iterations
;;     Determines maximum block size
```

---

## Implementation Order

### Milestone 1: Investigation Gates (1 day)

Run Gates 1-4 to validate that `mx/compile-fn` can handle long MH loops
with `mx/where` and gradient threading. This determines whether single-
dispatch fusion is feasible or if block-based fusion is needed.

### Milestone 2: Pre-Generated Noise (1 day)

Implement Phase 1: pre-generate all noise upfront, pass to existing compiled
chain. This alone should give 2-3x speedup by eliminating per-block PRNG.

**Estimated scope:** ~60 lines in `mcmc.cljs`, ~40 lines of tests.

### Milestone 3: Fused Burn-In (2 days)

Implement Phase 1.3: single compiled function for entire burn-in. Validate
that post-burn-in params match handler path.

**Estimated scope:** ~100 lines in `mcmc.cljs`, ~80 lines of tests.

### Milestone 4: Fused Collection (3 days)

Implement Phase 2: in-graph sample recording via `mx/where` row-write.
This is the hardest part — needs careful shape management.

**Estimated scope:** ~150 lines in `mcmc.cljs`, ~100 lines of tests.

### Milestone 5: End-to-End Fused MH API (1 day)

Implement Phase 3: `fused-mh` public API combining burn-in + collection.
Benchmark against Gen.jl.

**Estimated scope:** ~60 lines in `mcmc.cljs`, ~40 lines of tests.

### Milestone 6: Fused MALA (2 days)

Implement Phase 4.1: gradient-threaded MALA with 1 val-grad per step.
Requires Gate 4 (mx/value-and-grad + mx/where).

**Estimated scope:** ~120 lines in `mcmc.cljs`, ~80 lines of tests.

### Milestone 7: Fused HMC (2 days)

Implement Phase 4.2: HMC with compiled leapfrog integration.

**Estimated scope:** ~150 lines in `mcmc.cljs`, ~100 lines of tests.

### Milestone 8: Vectorized Chains (2 days)

Implement Phase 5: N parallel chains in one dispatch.

**Estimated scope:** ~100 lines in `mcmc.cljs`, ~80 lines of tests.

### Milestone 9: Diagnostics (1 day)

Implement Phase 6: in-graph acceptance/score tracking + post-chain R-hat.

**Estimated scope:** ~60 lines in `mcmc.cljs` + `diagnostics.cljs`.

---

## Expected Impact

| Configuration | Current | After Fusion | Speedup |
|--------------|---------|-------------|---------|
| MH 1000 samples (linreg) | 4841ms | ~200-500ms | 10-25x |
| MH 1000 samples + vectorized 8 chains | N/A | ~100-200ms | 25-50x vs scalar |
| MALA 1000 samples | ~8000ms | ~400-800ms | 10-20x |
| HMC 1000 samples (L=10) | ~12000ms | ~800-1500ms | 8-15x |
| Gen.jl MH 1000 samples (linreg) | 64ms | — | — |
| **Gap vs Gen.jl** | **75x** | **3-8x** | — |

The remaining gap vs Gen.jl (3-8x) is structural: Julia JIT-compiles to
native code with zero FFI overhead, while GenMLX always has the ClojureScript
→ MLX boundary. This is acceptable — MLX's GPU acceleration on larger models
(where per-step cost is dominated by score function, not overhead) should
close or reverse the gap.

---

## Design Principles

1. **Pre-generate, don't split.** All PRNG noise generated upfront. No
   per-step key splitting inside compiled functions.
2. **Thread state, don't extract.** Params, scores, and gradients stay as
   MLX arrays throughout the chain. Only extract at the very end.
3. **Materialize at boundaries.** Only call `mx/materialize!` at the end of
   the complete chain, not per step or per block.
4. **Compose raw score functions.** Pass uncompiled score functions to chain
   builders. Let `mx/compile-fn` fuse score + chain together.
5. **Block fusion as fallback.** If single-dispatch fusion hits graph size
   limits, use block-based fusion with lazy inter-block connection.
6. **Handler is ground truth.** Fused chains must produce statistically
   identical results to handler-based chains. Validate via KS tests on
   posterior samples.
