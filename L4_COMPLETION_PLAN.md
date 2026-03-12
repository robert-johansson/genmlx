# Level 4 Completion Plan: Single Fused Graph

## Executive Summary

Level 4 makes GenMLX a **compiler from probabilistic programs to optimal GPU
execution plans**. The user writes a model and calls `(fit model data)`. Under
the hood, the system:

1. Analyzes model structure (L3/3.5 — conjugacy, dependency graph, affine analysis)
2. Selects the optimal inference strategy from the analysis metadata
3. Compiles model + inference + gradient + optimizer into a single fused graph
4. Executes the entire optimization loop with minimal host interaction

L0-L3.5 built the individual pieces: compiled gen fns (L1), compiled inference
sweeps (L2), analytical elimination (L3/3.5). L4 **composes them into one graph**.
The challenge is not new math — it's composition, cost control, and the unified
API that connects everything.

### What L4 is NOT

- Not new inference algorithms (we have SMC, MCMC, VI, ADEV, MAP, Gibbs, HMC, NUTS)
- Not new analytical elimination (L3/3.5 handles 5+1 conjugate families)
- Not LLMs as gen fns (that's L5)
- Not fixing tech debt for its own sake (only what blocks fusion)

### The thesis

The existing `train` loop in `learning.cljs` already works:

```clojure
(train {:iterations 1000 :optimizer :adam :lr 0.001}
       loss-grad-fn init-params)
```

But it has three inefficiencies:
1. **Per-iteration host overhead**: `mx/materialize!` at every step breaks the graph
2. **Score function goes through GFI protocol**: ChoiceMap construction, handler dispatch
3. **No automatic method selection**: user must choose IS/MCMC/SMC/VI manually

L4 eliminates all three. The compiled optimization loop fuses gradient + Adam into
`mx/compile-fn`. Tensor-native score bypasses GFI. Method selection reads L3/3.5
metadata and dispatches automatically.

---

## Current State

### What exists (infrastructure inventory)

| Component | Location | Lines | L4 Role |
|-----------|----------|-------|---------|
| **Compiled gen fns** | `compiled.cljs` | 2,545 | Model forward pass in fused graph |
| **Tensor-native score** | `compiled.cljs:2357` | ~80 | Score function bypassing GFI |
| **TensorTrace** | `tensor_trace.cljs` | 114 | Flat tensor traces for gradient |
| **Adam optimizer** | `learning.cljs:70` | 25 | Parameter update step |
| **Training loop** | `learning.cljs:99` | 41 | Host-driven loop (L4 compiles this) |
| **Compiled MH chain** | `mcmc.cljs:166` | 25 | `mx/compile-fn` wrapped chain |
| **Compiled trajectory** | `mcmc.cljs:192` | 28 | Full [K,D] trajectory in one dispatch |
| **Compiled MALA/HMC** | `mcmc.cljs:771,1148` | ~200 | Gradient-based MCMC via compile-fn |
| **Compiled SMC** | `compiled_smc.cljs` | 178 | Bootstrap PF with tensor particles |
| **Differentiable resample** | `differentiable_resample.cljs` | ~120 | Gumbel-top-k + Gumbel-softmax |
| **Gradient through MCMC** | `compiled_gradient.cljs:29` | 60 | Differentiable MH chain |
| **Gradient through SMC** | `compiled_gradient.cljs:105` | 80 | d(log-ML)/d(params) through SMC |
| **Compiled VI/ADVI** | `vi.cljs:123` | 65 | `mx/compile-fn(mx/grad(neg-elbo))` |
| **Conjugacy detection** | `conjugacy.cljs` | 165 | 6 conjugate families |
| **Dependency graph** | `dep_graph.cljs` | 262 | Structure analysis |
| **Affine analysis** | `affine.cljs` | 379 | Linear-Gaussian chain detection |
| **Graph rewriting** | `rewrite.cljs` | 225 | 3 rule types (Kalman, Conjugacy, RB) |
| **Auto-analytical handlers** | `auto_analytical.cljs` | 687 | Address-based dispatch |
| **Score function utilities** | `inference/util.cljs:386` | 65 | `make-tensor-score-fn`, `prepare-mcmc-score` |
| **Param loss function** | `learning.cljs:156` | 25 | `make-param-loss-fn` (GFI-based) |

### What's missing for L4

| Gap | Impact | Addressed by |
|-----|--------|-------------|
| Training loop has per-iteration `mx/materialize!` | Breaks graph fusion | WP-0 |
| Adam step has per-iteration `mx/materialize!` | Breaks graph fusion | WP-0 |
| No compiled loss-grad function (uses GFI protocol) | FFI overhead | WP-1 |
| Compiled inference not composed with auto-handlers | Missed analytical elimination | WP-2 |
| No automatic method selection | User must choose | WP-3 |
| No `fit` API (one-call entry point) | UX gap | WP-4 |
| Three trace types with no unified interface | Composition friction | WP-1 (partial) |

---

## Architecture Overview

### The Compiled Optimization Loop

Current (L2):
```
for iteration in 0..N:
  loss, grad = value-and-grad(score-fn)(params)   ← builds graph
  mx/materialize!(loss, grad)                      ← evaluates graph (breaks it)
  params' = adam-step(params, grad)                 ← builds new graph
  mx/materialize!(params')                          ← evaluates (breaks again)
```

L4 target:
```
compiled-step = mx/compile-fn(fn [params, adam-state, noise]:
  loss, grad = value-and-grad(score-fn)(params)
  params', adam-state' = adam-step(params, grad, adam-state)
  return [params', adam-state', loss])

for iteration in 0..N:
  [params, adam-state, loss] = compiled-step(params, adam-state, noise)
  mx/materialize!(loss)   ← only for logging (optional)
```

The inner loop body (score → gradient → Adam) is a **single `mx/compile-fn` call**.
Host only runs the iteration counter and optional loss logging. Each iteration is
one Metal dispatch instead of ~5 separate dispatches.

### Fused Inference + Optimization

The score function itself can be compiled inference:

```
compiled-inference-step = fn [params, inference-state, noise]:
  ;; 1. Run compiled SMC/MCMC with current params
  score = compiled-smc-sweep(params, noise)   OR
  score = compiled-mh-chain(params, noise)    OR
  score = tensor-score(params)                ← direct (no inference needed)

  ;; 2. Gradient of score w.r.t. params
  grad = mx/grad(score-fn)(params)

  ;; 3. Adam update
  params', adam-state' = adam-step(params, grad, adam-state)

  return [params', adam-state', score]
```

The entire "run inference → compute gradient → update params" cycle is one graph.

### Method Selection

L3/3.5 analysis produces rich metadata per model:

```clojure
(:schema model)
→ {:static? true/false
   :trace-sites [{:addr :mu :dist-type :gaussian :static? true ...} ...]
   :conjugate-pairs [{:prior :mu :obs [:y0 :y1] :family :normal-normal} ...]
   :kalman-chains [{:chain [:x0 :x1 :x2] :obs [:y0 :y1 :y2]} ...]
   :dep-graph {:nodes #{...} :edges #{...}}
   :rewrite-result {:eliminated #{:mu} :handlers {...} :rules-applied [...]}
   :has-loops? false :has-splices? false :dynamic-addresses? false}
```

Method selection reads this metadata and dispatches:

| Model Structure | Selected Method | Why |
|----------------|----------------|-----|
| All conjugate, no residual | Exact (analytical only) | Zero variance |
| Linear-Gaussian temporal | Kalman filter | Exact, O(T) |
| Some conjugate + residual | IS with Rao-Blackwellization | Lower variance |
| Static, few latents (≤10) | Compiled MCMC (MH or HMC) | Well-mixed |
| Static, many latents (>10) | Compiled VI (ADVI) | Scalable gradient |
| Temporal (unfold/scan) | Compiled SMC | Sequential structure |
| Non-static (dynamic addrs) | Handler-based IS | Fallback |

This is a decision tree, not a neural network. Pure function of schema metadata.

---

## Investigation Gates

Five gates. Each validates a design decision before committing to implementation.

### Gate 0: Compiled Adam Speedup

**Experiment:** Compare three optimization loop variants on a 5-parameter
Gaussian model with 20 observations:

(a) Current `train` loop (per-step `mx/materialize!`)
(b) `mx/compile-fn` wrapping gradient + Adam (our L4 target)
(c) Raw `mx/compile-fn` wrapping just gradient (Adam stays host-side)

Run 1000 iterations of each. Measure wall-clock time.

**Success criterion:** (b) is measurably faster than (a). At minimum, (b) ≥ 1.5x (a).

**Decision:** If (b) wins → compiled Adam is the right path. If (b) ≈ (a) → the
graph materialization overhead is negligible and L4-M1 should focus on reducing
FFI calls instead. If (c) ≈ (b) → Adam compilation doesn't help; focus on
gradient compilation.

**Hypothesis:** MLX's lazy evaluation already batches work. The per-step
`mx/materialize!` forces evaluation but MLX may internally fuse anyway.
The gain from `mx/compile-fn` is eliminating the host-side graph construction
overhead (1 FFI call vs ~20 per step).

### Gate 1: Tensor-Score + Auto-Handler Composition

**Experiment:** Build a model with conjugate substructure (Normal-Normal prior
on mean, observed Gaussian). Compare:

(a) `make-tensor-score` alone (ignores conjugacy, all latents sampled)
(b) `make-tensor-score` with auto-handler eliminated addresses excluded
(c) Full handler-based `p/generate` with auto-handlers

Measure: score accuracy (vs exact marginal LL), speed, graph size.

**Success criterion:** (b) is at least as fast as (a) and produces correct
reduced-dimension score. Reduced dimension K' < K.

**Decision:** If (b) works → composed path is viable. If not → need a new
"analytical-aware tensor-score" that inlines marginal LL contributions.

**Context:** `prepare-mcmc-score` in `inference/util.cljs:420` already filters
eliminated addresses. Gate 1 validates this works end-to-end through
compiled gradient → Adam → convergence.

### Gate 2: mx/compile-fn Through Inference + Adam

**Experiment:** Wrap the full cycle (MH chain → score → gradient → Adam step)
in a single `mx/compile-fn`. Test on a 3-parameter model:

(a) Can `mx/compile-fn` trace through the composed function?
(b) Does the compiled function produce correct results?
(c) Memory usage for 100 iterations?

**Success criterion:** (a) compiles without error. (b) matches non-compiled path
within 1e-6. (c) no memory explosion.

**Decision:** If all pass → WP-1 proceeds as designed. If (a) fails → investigate
which operation breaks tracing (likely `mx/where` inside Adam or gradient
computation). May need to separate inference compilation from optimizer compilation.

**Known risk:** `mx/compile-fn` + `mx/where` traces with a fixed condition value.
If the Adam step or accept/reject changes which branch is taken across iterations,
the compiled function may cache stale control flow. The existing compiled MH chain
(`make-compiled-chain` at `mcmc.cljs:166`) already handles this by using `mx/where`
for branchless accept/reject — so this should be fine for the inner loop. But Adam's
bias correction (`(- 1.0 (js/Math.pow beta1 t))`) uses host-side `t` counter which
changes each step — this **cannot** be inside `mx/compile-fn`.

**Likely outcome:** Compile the *body* (gradient + Adam update with fixed t),
increment t on the host. Each iteration is one compiled call, t is a host counter.

### Gate 3: Method Selection Accuracy

**Experiment:** Construct 8 test models spanning the method selection table:

1. All-conjugate (NN + BB)
2. Linear-Gaussian temporal (Kalman-eligible)
3. Mixed conjugate (3/5 sites conjugate)
4. Static, 3 latents (MCMC-eligible)
5. Static, 20 latents (VI-eligible)
6. Temporal unfold (SMC-eligible)
7. Nonlinear temporal (EKF/SMC)
8. Dynamic addresses (handler fallback)

For each, run `select-method` and verify it chooses the correct strategy.
Then run the selected method and verify convergence.

**Success criterion:** All 8 models get the correct method. All converge to
within 2σ of the true posterior (where known).

**Decision:** If correct → method selection works. If wrong on ≤2 → fix
edge cases. If wrong on ≥3 → the decision tree needs more features
(e.g., observation count, dimension ratios).

### Gate 4: End-to-End `fit` Convergence

**Experiment:** Run `(fit model data)` on 4 benchmark models:

1. Linear regression (conjugate → exact)
2. Hierarchical Gaussian (mixed → Rao-Blackwellized IS)
3. Changepoint detection (temporal → compiled SMC)
4. Funnel distribution (non-conjugate → compiled MCMC)

Compare:
- Point estimates vs known ground truth
- Wall-clock time vs manual method selection
- Correctness: posterior mean within 2σ, log-ML within 0.5 nats

**Success criterion:** `fit` produces correct results on all 4 without user
specifying the inference method. Wall-clock ≤ 2x manual (overhead for method
selection + compilation).

**Decision:** If all 4 converge correctly → L4 is feature-complete. If
wall-clock > 2x manual → optimize compilation overhead (e.g., cache compiled
functions, lazy compilation).

---

## Work Packages

### WP-0: Compiled Optimization Step

**Goal:** Wrap gradient + Adam into a compiled function, replacing per-iteration
`mx/materialize!` with a single compiled call.

**Why first:** This is the foundation for everything else. WP-1 composes inference
into this compiled step. WP-2 integrates auto-handlers. WP-3 selects the method.
WP-4 wraps it all in `fit`.

**Architecture:**

The key insight from Gate 2's analysis: Adam's bias correction uses iteration
counter `t` which changes each step. We can't compile the full multi-step loop,
but we **can** compile one step (gradient + update at fixed t):

```clojure
(defn make-compiled-opt-step
  "Build a compiled single optimization step.
   Returns (fn [params adam-m adam-v noise t-scalar]
              -> [new-params new-m new-v loss])

   score-fn: (fn [params-tensor] -> scalar) — differentiable score
   lr, beta1, beta2, epsilon: Adam hyperparameters

   The returned function is wrapped in mx/compile-fn. The iteration
   counter t is passed as an argument (not closed over) so the compiled
   function works correctly across iterations."
  [score-fn {:keys [lr beta1 beta2 epsilon]
             :or {lr 0.001 beta1 0.9 beta2 0.999 epsilon 1e-8}}]
  (let [lr-arr (mx/scalar lr)
        b1 (mx/scalar beta1) b2 (mx/scalar beta2)
        one (mx/scalar 1.0) eps (mx/scalar epsilon)
        neg-one (mx/scalar -1.0)
        ;; Build value-and-grad of negative score (we minimize)
        neg-score (fn [p] (mx/multiply neg-one (score-fn p)))
        vag (mx/value-and-grad neg-score)
        ;; One full step: evaluate gradient + Adam update
        step-fn
        (fn [params m v t-scalar]
          (let [[loss grad] (vag params)
                ;; Adam update (all MLX ops, no host branching)
                new-m (mx/add (mx/multiply b1 m)
                              (mx/multiply (mx/subtract one b1) grad))
                new-v (mx/add (mx/multiply b2 v)
                              (mx/multiply (mx/subtract one b2) (mx/square grad)))
                ;; Bias correction: t is passed as MLX scalar
                m-hat (mx/divide new-m (mx/subtract one (mx/power b1 t-scalar)))
                v-hat (mx/divide new-v (mx/subtract one (mx/power b2 t-scalar)))
                update (mx/divide m-hat (mx/add (mx/sqrt v-hat) eps))
                new-params (mx/subtract params (mx/multiply lr-arr update))]
            [new-params new-m new-v loss]))]
    ;; Compile the step function
    (mx/compile-fn step-fn)))
```

**The iteration counter trick:** Adam uses `(1 - beta1^t)` for bias correction.
By passing `t` as an MLX scalar argument (not a host integer), the compiled
function handles varying `t` correctly — `mx/power` is a traced MLX op that
works with any value. The host loop just increments `t`:

```clojure
(defn compiled-train
  "Training loop using compiled optimization step.

   score-fn: (fn [params-tensor] -> scalar) — differentiable
   init-params: [K] MLX array
   opts: {:iterations :lr :beta1 :beta2 :epsilon :callback :log-every}

   Returns {:params :loss-history}"
  [score-fn init-params
   {:keys [iterations lr callback log-every]
    :or {iterations 1000 lr 0.001 log-every 100}}]
  (let [K (first (mx/shape init-params))
        step (make-compiled-opt-step score-fn {:lr lr})
        init-m (mx/zeros [K])
        init-v (mx/zeros [K])]
    (loop [i 0 params init-params m init-m v init-v
           losses (transient [])]
      (if (>= i iterations)
        {:params params :loss-history (persistent! losses)}
        (let [t-scalar (mx/scalar (inc i))  ;; 1-indexed for Adam
              [new-params new-m new-v loss] (step params m v t-scalar)
              ;; Only materialize loss when logging
              log? (zero? (mod i log-every))
              loss-val (when log?
                         (mx/materialize! loss)
                         (mx/item loss))]
          ;; Periodic cleanup
          (when (zero? (mod i 50))
            (mx/clear-cache!)
            (mx/sweep-dead-arrays!))
          (when (and callback loss-val)
            (callback {:iter i :loss loss-val}))
          (recur (inc i) new-params new-m new-v
                 (if loss-val (conj! losses loss-val) losses)))))))
```

**Why not compile the outer loop too?** ClojureScript's `loop/recur` inside
`mx/compile-fn` works (L2 proved this with compiled MH chains), but:
- The number of iterations is typically large (1000-10000)
- MLX graph would be enormous (1000 unrolled steps)
- Periodic cleanup (`mx/clear-cache!`) is essential for memory
- Logging requires host interaction

The per-step compilation gives 80% of the benefit with none of the risk.

**Files:**
- New: `src/genmlx/inference/compiled_optimizer.cljs`
- Test: `test/genmlx/compiled_optimizer_test.cljs`

**Tests (~25):**
- `make-compiled-opt-step` returns compiled function
- Single step produces correct gradient direction
- Adam bias correction is correct at t=1, t=10, t=100
- 100 iterations converge on a quadratic objective
- 100 iterations converge on a Gaussian score function
- Loss decreases monotonically on convex objective
- Results match `learning/train` within 1e-5 (same objective)
- Memory stays bounded (no leak over 1000 iterations)
- Works with K=1 (scalar), K=5, K=20 parameters
- `compiled-train` produces correct final params
- `compiled-train` respects `log-every` (only materializes when logging)
- Periodic cleanup runs without error

**Gate 0 experiment: benchmark against `learning/train`.**

---

### WP-1: Compiled Loss-Gradient Function

**Goal:** Build a compiled loss-gradient function that composes tensor-native
score with gradient computation, bypassing GFI protocol entirely.

**Why it matters:** The current `make-param-loss-fn` in `learning.cljs:156`
constructs a ChoiceMap per evaluation, runs `p/generate` through the handler,
extracts weight. This is ~20 FFI calls per score evaluation. The tensor-native
path (`make-tensor-score` in `compiled.cljs:2357`) is a single pure function:
`[K]-tensor → scalar`. Wrapping this in `mx/value-and-grad` produces a compiled
loss+gradient function that feeds directly into WP-0's compiled Adam step.

**Architecture:**

```clojure
(defn make-compiled-loss-grad
  "Build a compiled loss+gradient function for parameter learning.

   Tries three paths in priority order:
   1. Tensor-native score (L2) — pure MLX ops, no GFI
   2. Compiled gen fn score (L1) — compiled forward pass, some GFI
   3. Handler-based score (L0) — full GFI, no compilation

   Returns {:loss-grad-fn (fn [params-tensor] -> [loss grad])
            :init-params [K] MLX array
            :n-params int
            :compilation-level :tensor-native | :compiled | :handler
            :latent-index {addr -> int}}"
  [model args observations addresses]
  (let [{:keys [score-fn latent-index tensor-native?]}
        (u/make-tensor-score-fn model args observations addresses)]
    (if tensor-native?
      ;; Path 1: Pure tensor score → compile value-and-grad
      (let [neg-score (fn [p] (mx/negative (score-fn p)))
            compiled-vag (mx/compile-fn (mx/value-and-grad neg-score))]
        {:loss-grad-fn compiled-vag
         :latent-index latent-index
         :n-params (count latent-index)
         :compilation-level :tensor-native})
      ;; Path 2/3: GFI-based score → value-and-grad (not compiled)
      (let [neg-score (fn [p] (mx/negative (score-fn p)))
            vag (mx/value-and-grad neg-score)]
        {:loss-grad-fn vag
         :latent-index latent-index
         :n-params (count latent-index)
         :compilation-level :handler}))))
```

**Composition with inference methods:**

For models where direct scoring isn't sufficient (e.g., marginal likelihood
estimation requires SMC), the loss-gradient function must compose inference:

```clojure
(defn make-inference-loss-grad
  "Build a loss+gradient function that runs inference internally.

   method: :mcmc | :smc | :vi
   The score function is the inference output (final score for MCMC,
   log-ML for SMC, ELBO for VI).

   Returns same shape as make-compiled-loss-grad."
  [model args observations addresses method opts]
  (case method
    :mcmc
    ;; Gradient through compiled MH chain (L2 compiled_gradient.cljs)
    (let [{:keys [score-fn init-params n-params]}
          (u/prepare-mcmc-score model args observations addresses
                                (p/generate model args observations))
          steps (:mcmc-steps opts 20)
          proposal-std (:proposal-std opts 0.1)
          std-arr (mx/scalar proposal-std)
          K n-params]
      {:loss-grad-fn
       (fn [params]
         (let [;; Pre-generate noise for this iteration's chain
               rk (rng/fresh-key)
               [nk uk] (rng/split rk)
               noise (rng/normal nk [steps K])
               uniforms (rng/uniform uk [steps])
               _ (mx/materialize! noise uniforms)
               chain-fn (make-differentiable-chain score-fn std-arr steps K)
               objective (fn [p] (mx/negative (score-fn (chain-fn p noise uniforms))))
               vag (mx/value-and-grad objective)
               [loss grad] (vag params)]
           (mx/materialize! loss grad)
           [loss grad]))
       :init-params init-params
       :n-params n-params
       :compilation-level :mcmc-compiled})

    :smc
    ;; Gradient through compiled SMC (L2 compiled_gradient.cljs)
    ;; Returns d(log-ML)/d(model-params)
    {:loss-grad-fn :smc  ;; placeholder, SMC gradient has different signature
     :compilation-level :smc-compiled}

    :vi
    ;; ELBO gradient via compiled VI (existing compiled-vi)
    ;; This is already done — compiled-vi in vi.cljs
    {:loss-grad-fn :vi
     :compilation-level :vi-compiled}))
```

**Integration with WP-0:**

```clojure
;; The composed path:
(let [{:keys [loss-grad-fn init-params]}
      (make-compiled-loss-grad model args observations addresses)

      ;; Feed into compiled optimizer
      score-fn (fn [p] (mx/negative (first (loss-grad-fn p))))  ;; or extract score-fn directly
      result (compiled-train score-fn init-params
                            {:iterations 1000 :lr 0.001})]
  (:params result))
```

**Handling auto-handler composition:**

The `prepare-mcmc-score` function in `inference/util.cljs:420` already filters
out analytically eliminated addresses. The composition path is:

```
model with schema
  → L3/3.5 analysis detects conjugate pairs
  → get-eliminated-addresses returns #{:mu :sigma}
  → filter-addresses removes them from latent set
  → make-tensor-score builds score over reduced dimensions
  → mx/value-and-grad differentiates the reduced-dimension score
  → compiled Adam optimizes only residual parameters
```

The marginal LL contribution from eliminated sites is baked into the score
function (constant offset). Gradients flow only through residual parameters.

**Files:**
- New: `src/genmlx/inference/compiled_optimizer.cljs` (extend from WP-0)
- Modify: `src/genmlx/inference/util.cljs` (add `make-compiled-loss-grad`)
- Test: `test/genmlx/compiled_optimizer_test.cljs` (extend from WP-0)

**Tests (~30):**
- `make-compiled-loss-grad` returns `:tensor-native` for static model
- Returns `:handler` for dynamic model
- Tensor-native loss matches GFI-based loss within 1e-6
- Gradient direction matches finite differences
- Gradient magnitude within 5x of finite differences
- Convergence: 500 iterations on linear regression → correct slope/intercept
- Convergence: 500 iterations on Gaussian mixture → correct means
- Reduced-dimension score (with auto-handlers) produces correct gradient
- L3 eliminated addresses not in gradient output
- Integration: compiled-loss-grad → compiled-train → correct result
- Fallback: non-static model falls back to handler, still converges
- Memory: 1000 iterations doesn't leak

**Gate 1 experiment: tensor-score + auto-handler composition.**
**Gate 2 experiment: mx/compile-fn through inference + Adam.**

---

### WP-2: Fused Inference + Optimization

**Goal:** Compose compiled inference (MCMC chain, SMC sweep) with compiled
optimization. The "run inference → compute gradient → update params" cycle
uses compiled infrastructure end-to-end.

**Why it matters:** For models where the score function requires inference
(not just a static forward pass), the current path runs inference via GFI
handlers, extracts a scalar result, then differentiates. WP-2 keeps
everything in the compiled graph.

**Architecture:**

Three fused paths, one per inference family:

**Fused MCMC + Optimization:**
```clojure
(defn make-fused-mcmc-opt-step
  "Build a compiled step: run K-step MH chain + gradient + Adam.

   The MH chain uses tensor-native score. Pre-generated noise is passed in.
   Gradient of the final chain score w.r.t. model params flows through
   the entire chain via mx/where's straight-through gradient.

   Returns compiled (fn [model-params adam-m adam-v t-scalar chain-noise uniforms]
                       -> [new-params new-m new-v loss])

   chain-noise: [K-steps, D] standard normal (pre-generated per iteration)
   uniforms: [K-steps] uniform (pre-generated per iteration)"
  [make-score chain-steps D adam-opts]
  (let [std-arr (mx/scalar (:proposal-std adam-opts 0.1))
        lr-arr (mx/scalar (:lr adam-opts 0.001))
        ;; ... Adam constants ...
        step-fn
        (fn [model-params m v t-scalar chain-noise uniforms]
          (let [;; Build score function for current model params
                score-fn (make-score model-params)
                ;; Run differentiable MH chain
                chain-fn (make-differentiable-chain score-fn std-arr chain-steps D)
                ;; Objective: negative final score after chain
                objective (fn [mp]
                            (let [sf (make-score mp)
                                  final (chain-fn (initial-latent mp) chain-noise uniforms)]
                              (mx/negative (sf final))))
                ;; Value + gradient
                [loss grad] ((mx/value-and-grad objective) model-params)
                ;; Adam update
                [new-params new-m new-v] (adam-update model-params grad m v t-scalar
                                                      lr-arr b1 b2 eps)]
            [new-params new-m new-v loss]))]
    ;; NOTE: Cannot mx/compile-fn this if make-score builds closures
    ;; that change structure based on model-params. In practice, the
    ;; score function structure is fixed (same sites, same log-prob ops),
    ;; only the values change. So compilation should work.
    step-fn))
```

**Fused SMC + Optimization:**

```clojure
(defn make-fused-smc-opt-step
  "Build a step: run compiled SMC → gradient of log-ML → Adam.

   Uses differentiable resampling (Gumbel-softmax) so gradient flows
   through the full SMC sweep.

   model-params affect the extend step's distribution arguments.
   Pre-generated noise includes both extend noise [T,N,K] and
   Gumbel noise [T,N] for resampling."
  [kernel init-state observations model-params-to-score
   {:keys [n-particles tau n-steps]}]
  ;; Wraps smc-log-ml-gradient from compiled_gradient.cljs
  ;; with Adam step appended
  ...)
```

**Fused VI (already exists):**

The existing `compiled-vi` in `vi.cljs:123` already does this:
```clojure
grad-neg-elbo (mx/compile-fn (mx/grad neg-elbo-fn))
;; ... Adam loop ...
```

WP-2 standardizes the interface so all three methods share the same
`compiled-train` loop from WP-0:

```clojure
;; Unified interface:
(compiled-train
  score-fn        ;; from WP-1: tensor-native, MCMC, SMC, or VI
  init-params
  {:iterations 1000 :lr 0.001})
```

**Implementation note — when `mx/compile-fn` won't work:**

Some inference methods involve randomness that changes each iteration:
- MCMC: new proposal noise each step
- SMC: new extend noise + resample noise

These can't be compiled *across* iterations. But the *within-iteration*
computation (inference + gradient + Adam) can be compiled if the noise
is passed as an argument:

```clojure
;; Per iteration:
(let [noise (pre-generate-noise key K)  ;; host-side
      _ (mx/materialize! noise)
      [params' m' v' loss] (compiled-step params m v t noise)]
  ;; compiled-step does: inference(noise) → grad → Adam
  ...)
```

This is the same pattern as `make-compiled-chain` in `mcmc.cljs:166` —
noise is generated outside, passed into the compiled function.

**Files:**
- Modify: `src/genmlx/inference/compiled_optimizer.cljs` (add fused paths)
- Modify: `src/genmlx/inference/compiled_gradient.cljs` (standardize interface)
- Test: `test/genmlx/fused_inference_test.cljs`

**Tests (~30):**
- Fused MCMC+Adam: 500 iterations converges on 3-param model
- Fused MCMC+Adam: results match sequential (MCMC → separate Adam)
- Fused SMC+Adam: log-ML gradient is correct (matches finite differences)
- Fused SMC+Adam: 200 iterations converges on temporal model
- Fused VI already works (validate existing compiled-vi still passes)
- All fused paths respect memory bounds (1000 iterations, no leak)
- Noise pre-generation produces correct shapes per method
- Compilation level metadata is correctly reported
- Fused path with auto-handlers: eliminated addresses excluded from gradient

---

### WP-3: Automatic Method Selection

**Goal:** Given a model and data, automatically select the optimal inference
method based on L3/3.5 structural analysis.

**Why it matters:** This is the UX breakthrough. The user writes a model
and calls `fit`. The system figures out the rest.

**Architecture:**

```clojure
(defn select-method
  "Select the optimal inference method for a model given observations.

   Returns {:method keyword
            :reason string
            :opts map-of-method-specific-options
            :eliminated set-of-analytically-eliminated-addresses
            :residual-addrs vector-of-remaining-latent-addresses}

   Method keywords: :exact, :kalman, :is-rb, :mcmc, :hmc, :vi, :smc, :handler-is

   Decision tree (evaluated top-to-bottom, first match wins):

   1. All sites eliminated by L3/3.5 → :exact
   2. Kalman chain covers all latent temporal structure → :kalman
   3. Temporal model (unfold/scan kernel) → :smc
   4. Static model, all residual sites differentiable:
      a. n-residual ≤ 10 → :hmc (or :mcmc if gradient unavailable)
      b. n-residual > 10 → :vi
   5. Non-static (dynamic addresses) → :handler-is
   6. Fallback → :handler-is"
  [model args observations]
  (let [schema (:schema model)
        ;; L3/3.5 analysis results
        rewrite (:rewrite-result schema)
        eliminated (or (:eliminated rewrite) #{})
        all-addrs (mapv :addr (filter :static? (:trace-sites schema)))
        obs-addrs (set (map first (cm/addresses observations)))
        latent-addrs (remove #(or (obs-addrs %) (eliminated %)) all-addrs)
        n-residual (count latent-addrs)
        ;; Model structure flags
        static? (:static? schema)
        has-kalman? (seq (:kalman-chains rewrite))
        has-conjugate? (seq (:conjugate-pairs rewrite))
        temporal? (or (:has-unfold? schema) (:has-scan? schema))
        dynamic? (:dynamic-addresses? schema)]
    (cond
      ;; 1. All eliminated → exact
      (zero? n-residual)
      {:method :exact
       :reason "All latent sites analytically eliminated"
       :opts {}
       :eliminated eliminated
       :residual-addrs []}

      ;; 2. Kalman filter covers temporal structure
      (and has-kalman? temporal? (zero? n-residual))
      {:method :kalman
       :reason "Linear-Gaussian temporal model → Kalman filter"
       :opts {}
       :eliminated eliminated
       :residual-addrs []}

      ;; 3. Temporal model → SMC
      temporal?
      {:method :smc
       :reason (str "Temporal model with " n-residual " residual latents")
       :opts {:particles (if (> n-residual 5) 200 100)
              :resample-method :systematic}
       :eliminated eliminated
       :residual-addrs (vec latent-addrs)}

      ;; 4. Static model
      (and static? (not dynamic?))
      (if (<= n-residual 10)
        {:method :hmc
         :reason (str n-residual " residual latents — gradient-based MCMC")
         :opts {:samples 1000 :burn 200 :step-size 0.01 :n-leapfrog 10}
         :eliminated eliminated
         :residual-addrs (vec latent-addrs)}
        {:method :vi
         :reason (str n-residual " residual latents — variational inference")
         :opts {:iterations 2000 :learning-rate 0.01 :elbo-samples 10}
         :eliminated eliminated
         :residual-addrs (vec latent-addrs)})

      ;; 5. Dynamic → handler fallback
      :else
      {:method :handler-is
       :reason "Dynamic model structure — handler-based IS"
       :opts {:particles 1000}
       :eliminated eliminated
       :residual-addrs (vec latent-addrs)})))
```

**Method-specific option tuning:**

The method selection also sets reasonable defaults based on model analysis:

```clojure
(defn tune-method-opts
  "Tune method-specific options based on model structure.
   Adjusts particles/samples/iterations based on n-residual and data size."
  [method-result model observations]
  (let [{:keys [method opts residual-addrs]} method-result
        n-residual (count residual-addrs)
        n-obs (count (cm/addresses observations))]
    (case method
      :smc (assoc opts
             :particles (max 100 (* 10 n-residual))
             :key (rng/fresh-key))
      :hmc (assoc opts
             :samples (max 500 (* 100 n-residual))
             :burn (max 100 (* 50 n-residual))
             :step-size (/ 0.1 (js/Math.sqrt n-residual)))
      :vi (assoc opts
            :iterations (max 1000 (* 200 n-residual))
            :elbo-samples (max 5 (min 50 (* 2 n-residual))))
      :handler-is (assoc opts
                    :particles (max 1000 (* 100 n-obs)))
      opts)))
```

**Schema extension for method selection:**

The schema needs a few additional fields that L3/3.5 analysis already
computes but doesn't always store:

```clojure
;; Add to schema construction in dynamic.cljs:
{:has-unfold? (boolean (seq (filter #(= (:combinator %) :unfold) (:splice-sites schema))))
 :has-scan?   (boolean (seq (filter #(= (:combinator %) :scan) (:splice-sites schema))))
 :n-trace-sites (count (:trace-sites schema))
 :n-latent-sites nil  ;; filled lazily when observations provided
 :rewrite-result (:rewrite-result schema)}  ;; already stored
```

**Files:**
- New: `src/genmlx/method_selection.cljs`
- Test: `test/genmlx/method_selection_test.cljs`

**Tests (~35):**
- All-conjugate model → `:exact`
- Kalman-eligible model → `:kalman`
- Unfold temporal model → `:smc`
- Static 3-param model → `:hmc`
- Static 20-param model → `:vi`
- Dynamic address model → `:handler-is`
- Mixed model (3/5 conjugate + 2 residual) → `:hmc` with n-residual=2
- Mixed temporal (partial conjugacy + unfold) → `:smc`
- Method option tuning: particle count scales with n-residual
- Method option tuning: MCMC burn-in scales with dimension
- Method option tuning: VI iterations scale with dimension
- Unknown model structure → `:handler-is` (safe fallback)
- Edge case: 0 latent + 0 observed → `:exact`
- Edge case: all observed → `:exact` (trivially)
- `select-method` returns all required fields
- `tune-method-opts` produces valid opts for each method

**Gate 3 experiment: 8 test models with correct method selection.**

---

### WP-4: The `fit` API

**Goal:** Provide the one-call entry point: `(fit model data)` returns posterior
estimates, optionally with parameter optimization.

**Why it matters:** This is the user-facing deliverable. Everything else is
infrastructure. `fit` is the payoff.

**Architecture:**

```clojure
(defn fit
  "Fit a generative model to data. Automatically selects inference method
   and compiles the optimization loop.

   model: generative function (with schema from gen macro)
   args: model arguments
   data: ChoiceMap of observed values

   opts (all optional):
     :method    — override automatic method selection
                  :exact :kalman :smc :mcmc :hmc :vi :handler-is
     :learn     — vector of param names to optimize (enables learning loop)
     :iterations — number of optimization iterations (default: auto)
     :lr         — learning rate (default: 0.001)
     :particles  — number of particles for IS/SMC (default: auto)
     :samples    — number of samples for MCMC (default: auto)
     :callback   — (fn [{:iter :loss :method :elapsed}]) called periodically
     :key        — PRNG key for reproducibility
     :verbose?   — print method selection reasoning (default: false)

   Returns:
     {:method      keyword — which method was used
      :trace       Trace   — best/final trace (MAP estimate)
      :posterior    map     — {:mean :std :samples} per latent address
      :log-ml      number  — log marginal likelihood estimate
      :loss-history [numbers] — optimization loss per iteration (if :learn)
      :params      map     — learned parameter values (if :learn)
      :diagnostics map     — method-specific diagnostics
      :elapsed-ms  number  — wall-clock time}"
  ([model args data] (fit model args data {}))
  ([model args data opts]
   (let [start-time (js/Date.now)
         model (dyn/auto-key model)
         ;; 1. Method selection (or user override)
         selection (if (:method opts)
                     {:method (:method opts)
                      :reason "User-specified"
                      :opts (select-keys opts [:particles :samples :iterations
                                               :step-size :n-leapfrog :tau])
                      :eliminated (get-eliminated-addresses model)
                      :residual-addrs (compute-residual-addrs model data)}
                     (select-method model args data))
         {:keys [method reason]} selection
         _ (when (:verbose? opts)
             (println (str "[fit] Selected method: " (name method) " — " reason)))
         ;; 2. Tune options
         method-opts (merge (tune-method-opts selection model data)
                            (select-keys opts [:lr :iterations :particles :samples :key]))
         ;; 3. Run inference
         result (run-method model args data method method-opts)
         ;; 4. Optional: parameter learning loop
         result (if (:learn opts)
                  (run-learning-loop model args data result
                                    (:learn opts) method-opts opts)
                  result)
         elapsed (- (js/Date.now) start-time)]
     (assoc result
       :method method
       :elapsed-ms elapsed
       :diagnostics (assoc (:diagnostics result) :reason reason)))))
```

**`run-method` dispatcher:**

```clojure
(defn- run-method
  "Execute the selected inference method."
  [model args data method opts]
  (case method
    :exact
    (let [{:keys [trace weight]} (p/generate model args data)]
      {:trace trace :log-ml (mx/item weight)
       :posterior (extract-posterior trace data)})

    :kalman
    ;; Already handled by auto-handlers in p/generate
    (let [{:keys [trace weight]} (p/generate model args data)]
      {:trace trace :log-ml (mx/item weight)
       :posterior (extract-posterior trace data)})

    :smc
    (let [kernel (:kernel model)   ;; unfold kernel
          result (compiled-smc/compiled-smc
                   (select-keys opts [:particles :key :resample-method :tau])
                   kernel (:init-state opts) (extract-obs-seq data))]
      {:trace (best-particle result)
       :log-ml (mx/item (:log-ml result))
       :posterior (smc-posterior result)})

    :hmc
    (let [result (mcmc/hmc opts model args data
                           (:residual-addrs opts))]
      {:trace (last result)
       :posterior (mcmc-posterior result data)
       :samples result})

    :mcmc
    (let [result (mcmc/compiled-mh opts model args data
                                    (:residual-addrs opts))]
      {:trace (last result)
       :posterior (mcmc-posterior result data)
       :samples result})

    :vi
    (let [score-fn (u/make-score-fn model args data (:residual-addrs opts))
          init-params (u/extract-params
                        (:trace (p/generate model args data))
                        (:residual-addrs opts))
          result (vi/compiled-vi opts score-fn init-params)]
      {:trace nil  ;; VI produces distribution, not trace
       :posterior {:mean (:mu result) :sigma (:sigma result)}
       :log-ml (last (:elbo-history result))})

    :handler-is
    (let [traces (importance/importance-resampling
                   opts model args data)]
      {:trace (first traces)
       :posterior (is-posterior traces data)
       :log-ml (importance/log-ml-estimate opts model args data)})))
```

**`run-learning-loop` for parameter optimization:**

```clojure
(defn- run-learning-loop
  "Run parameter learning using compiled optimizer (WP-0/WP-1).
   model-params-to-learn: vector of param keywords
   inference-result: initial inference result from run-method"
  [model args data inference-result param-names method-opts user-opts]
  (let [{:keys [loss-grad-fn init-params n-params compilation-level]}
        (make-compiled-loss-grad model args data param-names)
        ;; Build score function for compiled-train
        score-fn (fn [p] (mx/negative (first (loss-grad-fn p))))
        result (compiled-train score-fn init-params
                 {:iterations (:iterations user-opts 1000)
                  :lr (:lr user-opts 0.001)
                  :log-every (:log-every user-opts 100)
                  :callback (:callback user-opts)})]
    (merge inference-result
           {:params (unpack-learned-params (:params result) param-names)
            :loss-history (:loss-history result)})))
```

**Posterior extraction helpers:**

```clojure
(defn- extract-posterior
  "Extract posterior summary from a single trace."
  [trace data]
  (let [choices (:choices trace)
        obs-addrs (set (map first (cm/addresses data)))]
    (into {}
      (for [[addr sub] (cm/submaps choices)
            :when (not (obs-addrs addr))
            :when (cm/has-value? sub)]
        [addr {:mean (mx/item (cm/get-value sub))}]))))

(defn- mcmc-posterior
  "Extract posterior summary from MCMC samples."
  [traces data]
  (let [obs-addrs (set (map first (cm/addresses data)))
        first-choices (:choices (first traces))
        addrs (for [[addr sub] (cm/submaps first-choices)
                    :when (not (obs-addrs addr))
                    :when (cm/has-value? sub)]
                addr)]
    (into {}
      (for [addr addrs
            :let [vals (mapv #(mx/item (cm/get-value
                              (cm/get-submap (:choices %) addr)))
                            traces)]]
        [addr {:mean (/ (reduce + vals) (count vals))
               :std (js/Math.sqrt (/ (reduce + (map #(* (- % (/ (reduce + vals) (count vals)))
                                                         (- % (/ (reduce + vals) (count vals))))
                                                     vals))
                                     (dec (count vals))))
               :samples vals}]))))
```

**Files:**
- New: `src/genmlx/fit.cljs`
- Test: `test/genmlx/fit_test.cljs`

**Tests (~40):**
- `fit` on all-conjugate model → `:exact`, correct posterior
- `fit` on Kalman model → `:kalman`, correct posterior
- `fit` on unfold model → `:smc`, correct log-ML
- `fit` on 3-param static → `:hmc`, correct posterior
- `fit` on 20-param static → `:vi`, correct posterior mean
- `fit` with `:method` override → uses specified method
- `fit` with `:learn [:theta]` → parameters converge
- `fit` with `:verbose? true` → prints reasoning
- `fit` with `:callback` → called at log points
- `fit` returns all documented fields
- `fit` on dynamic model → `:handler-is`, fallback works
- `fit` with `:key` → reproducible results
- `fit` timing: `:elapsed-ms` is reasonable
- `fit` on linear regression → slope/intercept correct (end-to-end)
- `fit` on hierarchical Gaussian → group means correct (end-to-end)
- `run-method` dispatches correctly for each method
- `extract-posterior` produces correct mean/std
- `mcmc-posterior` handles multiple samples correctly
- Edge case: empty data → simulate-like behavior
- Edge case: all data constrained → score = log-ML

**Gate 4 experiment: 4 benchmark models end-to-end.**

---

### WP-5: Integration, Documentation, and Certification

**Goal:** Wire everything together, ensure no regressions, add schema extensions
needed by method selection, and certify the full L4 test suite.

**Why it matters:** Previous levels created ~50 new test files. WP-5 ensures
L4 doesn't break any of them and that the new L4 API is exercised across the
full model zoo.

**Components:**

**Schema extensions:**
```clojure
;; In dynamic.cljs, extend schema construction:
(defn- enrich-schema-for-l4
  "Add L4-specific metadata to schema for method selection."
  [schema]
  (assoc schema
    :has-unfold? (boolean (some #(= (:combinator-type %) :unfold)
                                (:splice-sites schema)))
    :has-scan? (boolean (some #(= (:combinator-type %) :scan)
                              (:splice-sites schema)))
    :n-static-latents (count (filter :static? (:trace-sites schema)))))
```

**Regression test suite:**
```bash
# L0 certification must still pass
bun run --bun nbb test/genmlx/level0_certification_test.cljs   # 68/68

# L1 tests must still pass
bun run --bun nbb test/genmlx/schema_test.cljs                 # 174/174
bun run --bun nbb test/genmlx/compiled_simulate_test.cljs       # 82/82
bun run --bun nbb test/genmlx/partial_compile_test.cljs         # 92/92
bun run --bun nbb test/genmlx/combinator_compile_test.cljs      # 90/90

# L2+ tests must still pass
bun run --bun nbb test/genmlx/gen_clj_compat_test.cljs          # 162/165
bun run --bun nbb test/genmlx/genjax_compat_test.cljs           # 73/73
```

**L4 certification test:**
```bash
bun run --bun nbb test/genmlx/l4_certification_test.cljs
```

Tests:
1. **Gate-0**: Compiled Adam step ≥ 1.5x faster than `learning/train`
2. **Gate-1**: Tensor-score + auto-handlers produce correct reduced-dim score
3. **Gate-2**: `mx/compile-fn` through gradient + Adam works
4. **Gate-3**: Method selection correct on 8 test models
5. **Gate-4**: `fit` converges on 4 benchmarks
6. **WP-tests**: All WP-specific tests pass

**VISION.md update:**
```
Level 4 (done): Single fused graph — compiled optimization, fused inference, automatic method selection
```

**Files:**
- Modify: `src/genmlx/dynamic.cljs` (schema extensions)
- New: `test/genmlx/l4_certification_test.cljs`
- Modify: `VISION.md` (status update)

**Tests (~20):**
- Schema has `:has-unfold?`, `:has-scan?`, `:n-static-latents` fields
- All L0-L3.5 regression tests still pass (run as part of certification)
- L4 certification: 5 gates verified
- `fit` on real-world-sized model (50+ sites) completes without error
- Memory: `fit` on 50-site model × 1000 iterations stays under 4GB

---

## Dependency Graph

```
WP-0 (Compiled optimizer step)
  │
  ├── WP-1 (Compiled loss-gradient function)
  │     │
  │     ├── WP-2 (Fused inference + optimization)
  │     │     │
  │     │     └── WP-4 (fit API) ← needs WP-0 + WP-1 + WP-2 + WP-3
  │     │
  │     └── WP-4 (partial — direct score path)
  │
  └── WP-4 (partial — compiled-train integration)

WP-3 (Method selection) [independent of WP-0/1/2]
  │
  └── WP-4 (fit API)

WP-5 (Integration + certification) [depends on all]
```

**Critical path:** WP-0 → WP-1 → WP-2 → WP-4 → WP-5

**Parallelizable:** WP-3 can be developed in parallel with WP-0/1/2.

## Recommended Execution Order

1. **WP-0** → Gate 0 (compiled optimization step — foundation)
2. **WP-1** → Gate 1, Gate 2 (compiled loss-gradient — connects score to optimizer)
3. **WP-3** (method selection — independent, can overlap with WP-1)
4. **WP-2** (fused inference — builds on WP-0 + WP-1)
5. **WP-4** → Gate 3, Gate 4 (fit API — capstone, uses everything)
6. **WP-5** (certification — final)

**Agent team parallelism:**
- Team A: WP-0 → WP-1 → WP-2 (compilation path)
- Team B: WP-3 (method selection, independent)
- Merge: WP-4 (fit API, uses both paths)
- Final: WP-5 (certification)

**Stop points:** Each WP delivers standalone value:
- After WP-0: `compiled-train` available (faster training loop)
- After WP-1: `make-compiled-loss-grad` available (composed score + gradient)
- After WP-2: Fused MCMC/SMC+optimization (best performance)
- After WP-3: `select-method` available (automatic method choice)
- After WP-4: `fit` available (one-call API)
- After WP-5: Certified, documented, VISION.md updated

---

## Estimated Scope

| WP | New code | Tests | Difficulty | Gates |
|----|----------|-------|------------|-------|
| WP-0: Compiled optimizer step | ~120 lines | ~25 | Medium | Gate 0 |
| WP-1: Compiled loss-gradient | ~150 lines | ~30 | Medium | Gate 1, 2 |
| WP-2: Fused inference + opt | ~200 lines | ~30 | Hard | — |
| WP-3: Method selection | ~150 lines | ~35 | Medium | Gate 3 |
| WP-4: fit API | ~250 lines | ~40 | Medium | Gate 4 |
| WP-5: Integration + cert | ~80 lines | ~20 | Easy | All |
| **Total** | **~950 lines** | **~180** | | |

---

## mx/compile-fn Applicability

| Component | mx/compile-fn? | Why |
|-----------|---------------|-----|
| Tensor-native score | Yes | Pure: `[K]` → scalar, no data-dependent branches |
| `mx/value-and-grad(score)` | Yes | Gradient of compiled function is compiled |
| Adam step (fixed t) | Yes | All MLX ops, t passed as arg |
| Adam step (varying t) | No — pass t as arg | Host-side t counter, but use `mx/power` with MLX scalar |
| Full opt step (grad + Adam) | **Yes** | Compose value-and-grad + Adam as one compiled fn |
| Compiled MH chain | Yes (existing) | Pre-generated noise, mx/where accept/reject |
| Compiled SMC step | Partial | Extend step yes, resample breaks graph (unless Gumbel-softmax) |
| Method selection | No | Host-side decision tree, runs once |
| `fit` outer loop | No | Host-driven iteration with logging |
| Periodic cleanup | No | Host-side `mx/clear-cache!` |

**Key design constraint:** `mx/compile-fn` caches the graph structure. Functions
where the graph structure changes between calls (e.g., different branch taken)
cannot be compiled. But functions where only *values* change (same ops, different
numbers) compile perfectly. This is why Adam works: same ops every step, just
different parameter values and t.

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Compiled Adam ≈ host Adam (no speedup) | Medium | Low | Gate 0 validates upfront. Still cleaner architecture. |
| mx/compile-fn can't trace through mx/value-and-grad | Low | High | Gate 2 validates. Fallback: compile grad and Adam separately. |
| Method selection picks wrong method | Medium | Medium | Gate 3 validates on 8 models. User can override via `:method`. |
| Memory explosion in fused inference + gradient | Medium | Medium | Same pattern as L2 (periodic cleanup). Gate 2 measures memory. |
| `fit` too slow vs manual method choice | Low | Low | Gate 4 measures. Max 2x overhead acceptable. |
| Schema extensions break existing tests | Low | High | WP-5 runs full regression suite before declaring done. |
| Fused SMC gradient too noisy for learning | Medium | Medium | Gumbel-softmax temperature tuning. Fallback to VI. |

---

## L2 Infrastructure Reuse

| L2 Component | L4 Usage |
|-------------|----------|
| `make-tensor-score` | Core of WP-1's compiled loss-gradient |
| `prepare-mcmc-score` | Auto-handler composition in WP-1 |
| `TensorTrace` / `TensorChoiceMap` | Tensor-backed traces in fused paths |
| `make-compiled-chain` | Inner loop of fused MCMC+Adam (WP-2) |
| `make-differentiable-chain` | Gradient through MCMC in WP-2 |
| `compiled-smc` | Inner loop of fused SMC+Adam (WP-2) |
| `smc-log-ml-gradient` | SMC gradient in WP-2 |
| `compiled-vi` / `compiled-programmable-vi` | VI path in WP-4, already fused |
| Gumbel-softmax resampling | Differentiable SMC in WP-2 |
| `adam-step` / `adam-init` | Compiled inside WP-0's opt step |

## L3/3.5 Infrastructure Reuse

| L3/3.5 Component | L4 Usage |
|------------------|----------|
| `detect-conjugate-pairs` | Method selection (WP-3): is model conjugate? |
| `classify-dependency` | Method selection: affine vs nonlinear |
| `build-dep-graph` | Method selection: structure analysis |
| `apply-rewrite-rules` | Method selection: what was eliminated? |
| `get-eliminated-addresses` | WP-1: reduce score dimension |
| `make-auto-*-handlers` | Transparent — already wired into p/generate |
| `:rewrite-result` on schema | WP-3 reads this directly |

---

## Files to Create/Modify

| File | Action | WP |
|------|--------|-----|
| `src/genmlx/inference/compiled_optimizer.cljs` | New | WP-0, WP-1, WP-2 |
| `src/genmlx/method_selection.cljs` | New | WP-3 |
| `src/genmlx/fit.cljs` | New | WP-4 |
| `src/genmlx/inference/util.cljs` | Modify: add `make-compiled-loss-grad` | WP-1 |
| `src/genmlx/inference/compiled_gradient.cljs` | Modify: standardize interface | WP-2 |
| `src/genmlx/dynamic.cljs` | Modify: schema extensions for L4 | WP-5 |
| `test/genmlx/compiled_optimizer_test.cljs` | New | WP-0, WP-1 |
| `test/genmlx/fused_inference_test.cljs` | New | WP-2 |
| `test/genmlx/method_selection_test.cljs` | New | WP-3 |
| `test/genmlx/fit_test.cljs` | New | WP-4 |
| `test/genmlx/l4_certification_test.cljs` | New | WP-5 |

---

## What "Level 4 Done" Means

1. **`(fit model data)` works.** User calls one function. System selects method,
   compiles the optimization loop, and returns posterior estimates.

2. **Compiled optimization loop.** Gradient computation + Adam update is a single
   `mx/compile-fn` call per iteration. No per-step graph materialization.

3. **Fused inference + optimization.** For models requiring inference (not just
   scoring), the inference sweep is composed with gradient computation and
   parameter update. MCMC chains, SMC sweeps, and VI are all supported.

4. **Automatic method selection.** The decision tree reads L3/3.5 structural
   metadata and selects the optimal inference strategy. User can override.

5. **All L0-L3.5 tests still pass.** Zero regressions.

6. **~180 new tests** across 5 test files, covering all gates and WPs.

This matches VISION.md's promise: *"Model specification, inference algorithm,
and parameter optimization are all expressed as one lazy MLX computation graph.
ClojureScript builds the graph, then sits idle while Metal executes."*

---

## Relationship to Level 5

Level 5 (Cognitive Architecture) is architecturally independent of L4's
compilation work. L5 adds a new distribution type (LLM) and new combinators
(theory search). It uses the GFI contract, not the compilation ladder.

However, L4's `fit` API becomes the entry point for L5 workflows:

```clojure
;; L5 model with LLM distribution
(def hybrid-model
  (gen [description]
    (let [explanation (trace :explanation (llm-dist prompt))]
      (trace :obs (physics-model explanation) measured-data))))

;; L4's fit handles it
(fit hybrid-model [description] {:obs observed-data})
;; Method selection: :handler-is (LLM is non-differentiable, dynamic)
```

L4's method selection correctly falls back to handler-based IS for non-compilable
models. L5 models are inherently non-compilable (API calls, variable-length text),
so they flow through the fallback path. But L4's `fit` provides the unified entry
point that makes this transparent to the user.
