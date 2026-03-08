# GenMLX Vision: GPU-Native Probabilistic Programming

## Current State

GenMLX is ~10,800 lines of ClojureScript implementing the full Generative Function Interface (GFI) on Apple Silicon via MLX. It has 27 distributions, 10 combinators, and comprehensive inference (IS, MCMC, SMC, SMCP3, VI, ADEV, amortized). Shape-based GPU batching gives 10-100x speedups where implemented.

But GPU utilization is uneven. This document maps the path from "selective GPU acceleration" to "everything compiles to a single Metal dispatch."

---

## 1. GPU Gaps in Current GenMLX

### Sequential Bottlenecks (CPU `mapv` loops)

| Algorithm | File | Current | Fix | Est. Speedup |
|-----------|------|---------|-----|--------------|
| `importance-sampling` | inference/importance.cljs | N separate `p/generate` | Use existing `vectorized-importance-sampling` | 10-100x |
| `adev-optimize` | inference/adev.cljs | Sequential `adev-gradient` | Use existing `vadev-gradient` / `compiled-adev-optimize` | 5-10x |
| `smc` init | inference/smc.cljs | `mapv` over particles | `vgenerate` (as `batched-smc-unfold` already does) | 10-50x |
| `smc-step` updates | inference/smc.cljs | `mapv` per particle | Batched update transition (handler supports it) | 10-50x |
| `smc-rejuvenate` | inference/smc.cljs | Nested `mapv` | Use `vectorized-compiled-mh` | 10-50x |
| `smcp3` all steps | inference/smcp3.cljs | `mapv` per particle | Add vectorized variants | 10-50x |
| `systematic-resample` | inference/util.cljs | JS loop + `mx/realize` | GPU cumsum + searchsorted | 2-5x |
| `wake-sleep` | learning.cljs | Nested `mapv` | Batch via `vadev-gradient` pattern | 5-10x |

### All Combinators Are Sequential

Map, Unfold, Switch, Scan in `combinators.cljs` — all use `mapv` internally, calling `p/simulate` per element. Shape-based batching only works at the handler level (vsimulate/vgenerate), not at the combinator level.

### Unused MLX Primitives

| Primitive | Wrapped? | Used? | Potential |
|-----------|----------|-------|-----------|
| `mx/vmap` | Yes | Only in vi.cljs | Native functional vectorization |
| `mx/jvp` / `mx/vjp` | Yes | Never | Forward-mode AD, Fisher information |
| `mx/checkpoint` | No | N/A | Memory-efficient gradients for deep models |
| `asyncEval` | Yes | Never | Pipeline GPU work |
| `einsum` | No | N/A | Cleaner tensor contractions |
| `fast.scaledDotProductAttention` | No | N/A | Fused attention for sequence models |
| `fast.rope` | No | N/A | Rotary position embeddings |
| Metal capture/profiling | No | N/A | GPU shader debugging |

---

## 2. Missing Gen Universe Features

### Inference Algorithms

- **Particle MCMC (PMCMC)** — Use SMC as MH proposal for full trajectories. Standard in Gen.jl. Critical for temporal models (depression unfold).
- **SMC-squared** — Nested SMC for simultaneous parameter + state estimation. Eliminates need for fixed-parameter IS.
- **Rao-Blackwellization** — Analytically marginalize conjugate pairs. Massive variance reduction. E.g., if measurement noise is conjugate, marginalize it instead of sampling.
- **Streaming/Online SMC** — Process new observations incrementally. Natural for clinical monitoring.

### Compilation Strategy

GenJAX's key advantage: lower an entire inference loop to a single XLA dispatch. GenMLX has `mx/compile-fn` for individual functions (gradients, MH scores) but not for whole inference sweeps. The gap is **whole-program compilation**.

### Static Analysis

- **Static DSL** — Gen.jl has static generative functions with compile-time-known trace structure. Enables: pre-allocated traces, compiled update, zero choicemap overhead.
- **Trace type schemas** — Would enable compile-time verification and optimization. (Already in TODO as items 10.14-10.15.)

---

## 3. New Ideas — Where GenMLX Can Lead

GenMLX sits at a unique intersection: purely functional ClojureScript + MLX's lazy computation graph + Apple Silicon GPU. This enables things neither Gen.jl nor GenJAX can easily do.

### A. Compiled Generative Functions ("Static Gen on MLX")

**Insight:** MLX builds a lazy computation graph before evaluating. A gen fn body *is* a graph builder. Trace it once, capture the graph, re-execute without handler overhead.

```clojure
;; Current: every call builds choicemaps, runs handler transitions
(def model (gen [x] (trace :z (dist/gaussian 0 1))))

;; Proposed: compile to a single MLX graph
(def compiled-model (compile-gen model {:trace-schema {:z :float32}}))
;; compiled-model.simulate = ONE MLX dispatch
;; Trace is a flat tensor [z_value, z_score, total_score]
;; No choicemaps, no handler, no volatile!
```

**Why uniquely GenMLX:** MLX's lazy eval means compilation is natural — you're just not breaking the graph. The handler's `volatile!` is the only mutable part; replace it with pure graph accumulation.

**Impact:** 180 patients x 10 timesteps x 20K particles — inner model body becomes a single Metal kernel dispatch instead of 10 handler transitions per execution.

### B. Lazy Inference Graphs

**Insight:** Currently, inference materializes (`mx/eval!`) at every decision point. But MLX's lazy graph can represent the *entire inference sweep* as one graph.

```clojure
;; Current SMC: materialize after each step
(loop [step 0, particles ...]
  (let [new-particles (smc-step ...)   ;; builds graph
        _             (mx/materialize!) ;; BREAKS graph here
        resampled     (resample ...)]   ;; new graph starts
    (recur (inc step) resampled)))

;; Proposed: lazy SMC sweep
(defn lazy-smc-sweep [model obs-seq n-particles]
  (reduce (fn [state obs-t]
            (-> state
                (extend-particles model obs-t)
                (compute-weights)
                (resample-lazy)))
          (init-particles model n-particles)
          obs-seq))
;; Single mx/eval! at the very end
```

**Key challenge:** Resampling requires reading weights, forcing materialization. **Solution:** Pre-generate all random noise (stratified resampling with fixed u0). Then the entire sweep is deterministic given the noise — pure graph, single dispatch.

**Why uniquely GenMLX:** JAX requires explicit `jax.lax.scan` with static shapes. MLX's lazy graph naturally accumulates operations without requiring fixed control flow.

### C. Differentiable Inference Programs

**Insight:** ADEV differentiates through model execution. Extend this to differentiate through *inference itself* — learn model parameters by backpropagating through the inference algorithm.

```clojure
;; Learn theta by differentiating through IS
(defn differentiable-is [theta observations n-particles key]
  (let [traces  (vgenerate (model theta) observations n-particles key)
        log-ml  (log-marginal-likelihood traces)]
    log-ml))

(def grad-fn (mx/grad differentiable-is))
;; Gradient of log-ML w.r.t. model parameters!
```

This is the "learning the model by differentiating through inference" paradigm (Maddison et al., Le et al.). GenMLX already has the pieces (ADEV + vectorized IS + mx/grad). The missing piece is composing them.

**Impact on depression:** Learn population-level parameters (measurement slopes, noise scales, prior variances) by gradient descent on log-ML. Currently hand-calibrated. Differentiable inference finds optimal values automatically.

### D. Combinator-Level GPU Compilation

**Insight:** Combinators are sequential wrappers. But `unfold` has known structure — it's a scan over timesteps. Compile the entire unfolded execution as one graph.

```clojure
;; Current: unfold calls kernel.simulate 10 times sequentially
(unfold kernel 10 init-state)

;; Proposed: compiled-unfold
(compiled-unfold kernel 10 init-state)
;; mx/compile-fn wraps the entire 10-step loop
;; Traces pre-allocated as [10, ...]-shaped tensors
;; Single Metal dispatch for all timesteps
```

For `switch`: pre-compute all branches, then `mx/where` to select (already possible with vectorized-switch, not yet generalized).

For `map` over patients: the flat `[P*K]` trick. Generalize as `compiled-map`.

### E. Amortized Combinator Search (Automatic Theory Discovery)

**Insight from depression reboot:** Mechanism profiling manually searched 8 combinator compositions. Make this search a generative program itself.

```clojure
;; Meta-generative function: samples architecture, then runs it
(def architecture-search
  (gen [patient-data]
    (let [has-self-criticism  (trace :self-crit (dist/bernoulli 0.5))
          has-reward-sens     (trace :reward    (dist/bernoulli 0.5))
          has-barrier-sens    (trace :barrier   (dist/bernoulli 0.5))
          model (compose-architecture
                  has-self-criticism has-reward-sens has-barrier-sens)]
      (splice :inner (model patient-data)))))
```

This is `mix` over an exponential space of compositions. With `switch` + GPU, evaluate all compositions in parallel and let the data choose. The "theory" is whatever the posterior concentrates on.

The depression reboot's mechanism profiling was a manual version. Automating it requires: (a) compiled switch over architectures, (b) GPU-parallel branch evaluation, (c) inference over architecture choices.

### F. Forward-Mode AD for Fisher Information

**Unused capability:** `mx/jvp` (forward-mode AD) gives Jacobian-vector products, which yields the **Fisher information matrix**:

```
Fisher information = E[nabla log p (x) nabla log p]
Forward-mode: one JVP per parameter dimension
```

For models with ~20-30 parameters, forward-mode is competitive with reverse-mode and gives exact Fisher information — enabling:
- Natural gradient optimization
- Laplace approximation (posterior uncertainty)
- Proper model comparison (BIC with exact parameter counts)
- Confidence intervals on mechanism strengths

### G. Checkpoint for Deep Temporal Models

MLX has `checkpoint` (gradient checkpointing) — not yet wrapped. For `unfold` over T timesteps with gradients, checkpointing reduces memory from O(T) to O(sqrt(T)) by recomputing intermediate states during backward pass.

Critical for: daily data (T=63 days), item-level data (many more parameters per timestep), or longer treatment courses.

---

## 4. Priority Roadmap

### Tier 1: Engineering (immediate GPU wins)

| # | Item | Lines | Impact |
|---|------|-------|--------|
| 1a | Vectorize SMC init + step (use vgenerate + batched handler) | ~200 | 10-50x SMC speedup |
| 1b | GPU resampling (cumsum + searchsorted) | ~30 | Remove CPU bottleneck in all particle methods |
| 1c | Population-level flat [P*K] batching | ~165 | 180 patients on GPU simultaneously |
| 1d | Default to vectorized variants for IS, ADEV | ~20 | Users get GPU by default |
| 1e | Structured state resampling (walk map, take-idx each array) | ~15 | Unblock SMC with structured state |

### Tier 2: Compilation (eliminate overhead)

| # | Item | Lines | Impact |
|---|------|-------|--------|
| 2a | Compiled unfold (single Metal dispatch for T steps) | ~100 | 5-10x temporal models |
| 2b | Compiled gen fn (trace-schema → flat tensor) | ~500 | Eliminate handler overhead |
| 2c | Lazy inference graphs (pre-generate noise, single eval) | ~300 | Whole-sweep compilation |

### Tier 3: Research contributions

| # | Item | Lines | Impact |
|---|------|-------|--------|
| 3a | Differentiable inference (grad through vgenerate → log-ML) | ~100 | Automatic parameter learning |
| 3b | PMCMC / SMC-squared | ~200 | Joint parameter + state estimation |
| 3c | Rao-Blackwellization (auto-detect conjugate pairs) | ~300 | Variance reduction |
| 3d | Amortized combinator search | ~200 | Automatic theory discovery |
| 3e | Forward-mode AD / Fisher information | ~100 | Model comparison, natural gradient |
| 3f | Wrap mx/checkpoint for deep temporal gradients | ~30 | Memory-efficient long sequences |

### Tier 4: Ecosystem

| # | Item | Impact |
|---|------|--------|
| 4a | Streaming/online SMC | Real-time clinical inference |
| 4b | Visualization (GenStudio-inspired) | Interactive trace exploration |
| 4c | LLM-as-generative-function | Natural language + probabilistic reasoning |

---

## 5. The Unifying Vision

**GenMLX should be a system where models, inference, and compilation are all expressed in the same purely functional language, and MLX's lazy computation graph optimizes the entire pipeline end-to-end.**

The key insight: MLX's lazy evaluation is not just a performance trick — it's a *semantic match* for probabilistic programming. A generative function builds a computation graph (the model). Inference builds a larger graph around it (the algorithm). Compilation fuses them into minimal Metal dispatches (the execution).

```
Model (gen macro)           → MLX lazy graph (sampling + scoring)
Inference (SMC/IS/MCMC)     → MLX lazy graph (weighting + resampling)
Compilation (compile-gen)   → MLX lazy graph (everything fused)
                                    ↓
                            Single Metal dispatch
```

ClojureScript's purely functional nature makes this possible: no hidden mutation means the graph is always valid. MLX's lazy evaluation makes it efficient: operations accumulate without executing until needed. Apple Silicon's unified memory means no CPU↔GPU transfers.

The result: a PPL where writing `(gen [...] ...)` and running `(smc model obs)` compiles to GPU code as efficient as hand-written Metal shaders, while remaining a 10-line ClojureScript program.

---

## 6. Depression Model as Proof of Concept

The depression reboot (Phase B) demonstrates the vision at small scale:
- **Combinators as mechanisms**: mix (subtypes), switch (modes), unfold (temporal), scan (learning)
- **Bottom-up theory discovery**: 8 compositions searched, profiles emerged that match Blatt's theory
- **GPU acceleration**: VIS runs 20K particles per patient on Metal

The full vision scales this to:
- **Compiled temporal model**: `compiled-unfold` over 10 weeks, single dispatch per patient
- **Population GPU**: flat [P*K] batching, 180 patients × 1000 particles = 180K simultaneous
- **Differentiable inference**: learn measurement slopes, noise scales, prior variances by gradient descent
- **Automatic architecture search**: posterior over combinator compositions, not manual profiling
- **Three-dataset validation**: discover on N=180, replicate on N=90, test on N=90

This progression — from manual mechanism profiling to automatic combinator search on GPU — is the GenMLX roadmap in microcosm.
