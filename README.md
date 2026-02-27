<p align="center">
  <img src="genmlx.png" alt="GenMLX" width="528">
</p>

A probabilistic programming language implemented in ClojureScript on Node.js ([nbb](https://github.com/babashka/nbb)), using Apple's [MLX framework](https://github.com/ml-explore/mlx) for GPU acceleration via [`@frost-beta/mlx`](https://github.com/nicebyte/node-mlx).

GenMLX implements Gen's **Generative Function Interface (GFI)** — the same architecture used by [GenJAX](https://github.com/probcomp/GenJAX) (JAX) and [Gen.jl](https://github.com/probcomp/Gen.jl) (Julia).

## Why GenMLX?

Gen implementations exist for Julia and JAX — but nothing for Apple Silicon's GPU framework. MLX's unified memory model is a natural fit for probabilistic programming: MCMC control flow runs on CPU while all numerics stay on GPU, with zero data transfer cost. ClojureScript on Node.js gives direct access to MLX through a native addon with no FFI overhead, and nbb provides a fast REPL for interactive model development.

- **MLX-native** — unified memory, lazy evaluation, dynamic shapes, `mx/grad` through entire models
- **~14,000 lines of ClojureScript** — protocols, records, persistent data structures, the whole thing is readable in an afternoon
- **GPU end-to-end** — scores and choice values are MLX arrays throughout, extracted with `mx/item` only at inference boundaries

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- [Bun](https://bun.sh/) (recommended) or Node.js 18+
- [nbb](https://github.com/babashka/nbb) (`npm install -g nbb`)

## Quick Start

```bash
npm install
```

### Define a model

```clojure
(ns my-model
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Bayesian linear regression — all values stay as MLX arrays
(def model
  (gen [xs]
    (let [slope     (dyn/trace :slope (dist/gaussian 0 10))
          intercept (dyn/trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (dyn/trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                          intercept) 1)))
      slope)))
```

### Run inference

```clojure
;; Observations — choicemap constructor takes keyword-value pairs
(def xs [1.0 2.0 3.0 4.0 5.0])
(def observations
  (cm/choicemap :y0 (mx/scalar 2.1) :y1 (mx/scalar 3.9) :y2 (mx/scalar 6.2)
                :y3 (mx/scalar 7.8) :y4 (mx/scalar 10.1)))

;; Metropolis-Hastings
(def traces
  (mcmc/mh {:samples 500 :burn 100
            :selection (sel/select :slope :intercept)}
           model [xs] observations))

;; Examine posterior — traces are records, choices are hierarchical maps
(let [slopes (mapv #(mx/item (cm/get-choice (:choices %) [:slope])) traces)]
  (println "Posterior slope mean:" (/ (reduce + slopes) (count slopes))))
;; => ~2.0 (true slope)
```

## Architecture

```
Layer 0: MLX Foundation
  genmlx.mlx          — thin wrapper over @frost-beta/mlx
  genmlx.mlx.random   — functional PRNG key management

Layer 1: Core Data Types
  genmlx.choicemap    — hierarchical address → value trees
  genmlx.trace        — immutable execution records
  genmlx.selection    — composable address selection algebra

Layer 2: GFI Protocols & Execution
  genmlx.protocols    — IGenerativeFunction, IGenerate, IUpdate, IRegenerate,
                        IAssess, IProject, IEdit, IUpdateWithDiffs
  genmlx.handler      — handler-based execution via dynamic vars
  genmlx.edit         — parametric edit interface (constraint, selection, proposal)
  genmlx.diff         — incremental change tracking

Layer 3: DSL
  genmlx.gen          — gen macro
  genmlx.dynamic      — DynamicDSLFunction (full GFI + vsimulate/vgenerate)

Layer 4: Distributions
  genmlx.dist         — 27 distributions, each a GFI participant

Layer 5: Combinators
  genmlx.combinators  — Map, Unfold, Switch, Scan, Mask, Mix, Recurse,
                        Vectorized Switch, Contramap/Dimap

Layer 6: Inference
  genmlx.inference    — IS, MH, MALA, HMC, NUTS, Gibbs, Elliptical Slice,
                        Involutive MCMC, MAP, SMC, SMCP3, VI, Kernel Composition

Layer 7: Vectorized
  genmlx.vectorized   — VectorizedTrace, batched execution, dispatch amortization
```

## Distributions

| Distribution | Constructor | Reparameterized |
|---|---|---|
| Gaussian | `(gaussian mu sigma)` | Yes |
| Uniform | `(uniform lo hi)` | Yes |
| Bernoulli | `(bernoulli p)` | — |
| Beta | `(beta-dist alpha beta)` | — |
| Gamma | `(gamma-dist shape rate)` | — |
| Exponential | `(exponential rate)` | Yes |
| Categorical | `(categorical logits)` | — |
| Poisson | `(poisson rate)` | — |
| Laplace | `(laplace loc scale)` | Yes |
| Student-t | `(student-t df loc scale)` | — |
| Log-Normal | `(log-normal mu sigma)` | Yes |
| Multivariate Normal | `(multivariate-normal mean cov)` | Yes |
| Dirichlet | `(dirichlet alpha)` | — |
| Delta | `(delta v)` | — |
| Cauchy | `(cauchy loc scale)` | Yes |
| Inverse Gamma | `(inv-gamma shape scale)` | — |
| Geometric | `(geometric p)` | — |
| Negative Binomial | `(neg-binomial r p)` | — |
| Binomial | `(binomial n p)` | — |
| Discrete Uniform | `(discrete-uniform lo hi)` | — |
| Truncated Normal | `(truncated-normal mu sigma lo hi)` | Yes |
| Piecewise Uniform | `(piecewise-uniform bounds probs)` | — |
| Beta-Uniform Mixture | `(beta-uniform-mixture theta alpha beta)` | — |
| Wishart | `(wishart df scale)` | — |
| Inverse Wishart | `(inv-wishart df scale)` | — |
| Broadcasted Normal | `(broadcasted-normal mu sigma)` | Yes |
| Mixture | `(mixture components log-weights)` | — |

Aliases: `normal` → `gaussian`, `flip` → `bernoulli`

## Inference Algorithms

- **Importance Sampling** — `importance-sampling`, `importance-resampling`; vectorized variants via `vgenerate`
- **Metropolis-Hastings** — `mh` (via GFI `regenerate`), `mh-custom` (with proposal generative function)
- **MALA** — `mala` (gradient-informed Langevin proposals)
- **HMC** — `hmc` (compiled leapfrog integration, adaptive step-size via dual averaging)
- **NUTS** — `nuts` (adaptive trajectory length, No-U-Turn Sampler)
- **Gibbs Sampling** — `gibbs` (systematic scan with enumerable support)
- **Elliptical Slice Sampling** — `elliptical-slice` (for Gaussian priors)
- **Involutive MCMC** — `involutive-mh` (deterministic involution-based proposals)
- **MAP Optimization** — `map-optimize` (point estimates via gradient ascent)
- **SMC** — `smc` (particle filtering with resampling + rejuvenation), `csmc`
- **SMCP3** — `smcp3` (Sequential Monte Carlo with Probabilistic Program Proposals)
- **Variational Inference** — `vi` (ADVI with mean-field Gaussian guide), `programmable-vi` with pluggable objectives (`elbo`, `iwelbo`, `wake-sleep`) and gradient estimators (`reinforce`, reparameterization)
- **Kernel Composition** — `chain`, `cycle-kernels`, `mix-kernels`, `repeat-kernel`, `seed`
- **Diagnostics** — `ess`, `r-hat`, `summarize`, `sample-quantiles`

## Combinators

- **Map** — apply a generative function independently across sequences
- **Unfold** — sequential state-threading for time-series models
- **Switch** — select between branches for mixture models
- **Scan** — state-threading with accumulation (like `jax.lax.scan`)
- **Mask** — conditionally gate execution on a boolean
- **Mix** — first-class mixture model support
- **Recurse** — fixed-point combinator for recursive generative functions
- **Vectorized Switch** — executes all branches with `[N]`-shaped arrays, selects via `mx/where`
- **Contramap / Dimap** — transform arguments and/or return values of generative functions

## GFI Operations

Every generative function supports the full Gen interface:

```clojure
;; Forward sample
(p/simulate model args)              ;; => Trace

;; Constrained execution
(p/generate model args constraints)  ;; => {:trace Trace :weight MLX-scalar}

;; Update trace with new constraints
(p/update model trace new-constraints) ;; => {:trace Trace :weight MLX-scalar :discard ChoiceMap}

;; Resample selected addresses
(p/regenerate model trace selection) ;; => {:trace Trace :weight MLX-scalar}

;; Score fully-specified choices
(p/assess model args choices)        ;; => {:weight MLX-scalar}

;; Log-probability of selected addresses
(p/project model trace selection)    ;; => MLX-scalar

;; Parametric edit (constraint, selection, or proposal)
(p/edit model trace edit-request)    ;; => {:trace Trace :weight MLX-scalar :discard ChoiceMap}
```

## Vectorized Inference

The key insight: MLX operations broadcast naturally. Sample `[N]` values instead of `[]` at each trace site, and all downstream arithmetic (log-prob, score accumulation, weight computation) just works.

```clojure
;; Run model body ONCE for N particles
(dyn/vsimulate model args n key)         ;; => VectorizedTrace
(dyn/vgenerate model args obs n key)     ;; => VectorizedTrace with weights
```

- `VectorizedTrace` — choices where leaves hold `[N]`-shaped arrays
- 10 distributions have native batch sampling (`dist-sample-n`), others fall back to sequential
- Vectorized importance sampling and SMC initialization built on `vgenerate`

**Benchmarks (N=100, 5-site model — dispatch amortization):**
- vgenerate: 57x
- Vectorized IS: 53x
- Vectorized SMC init: 62x

## Running Tests

```bash
# Individual test files
bun run --bun nbb test/genmlx/dist_test.cljs
bun run --bun nbb test/genmlx/gen_test.cljs
bun run --bun nbb test/genmlx/inference_test.cljs

# All core tests
for f in choicemap_test trace_test selection_test handler_test dist_test gen_test combinators_test inference_test; do
  bun run --bun nbb "test/genmlx/${f}.cljs"
done

# Compatibility suites
bun run --bun nbb test/genmlx/gen_clj_compat_test.cljs     # 165/165 tests (from Gen.clj)
bun run --bun nbb test/genmlx/genjax_compat_test.cljs       # 73/73 tests (GenJAX compat)

# Vectorized tests + benchmarks
bun run --bun nbb test/genmlx/vectorized_test.cljs
bun run --bun nbb test/genmlx/vectorized_benchmark.cljs
```

### Gen.clj Compatibility

165 tests adapted from [Gen.clj](https://github.com/probcomp/Gen.clj)'s test suite verify that GenMLX produces matching results:

- **Distribution logpdf spot checks** — values verified against scipy.stats and Gen.jl (within float32 tolerance)
- **Mathematical properties** — symmetry, normalization, shift invariance
- **GFI semantics** — simulate, generate, update (discard/weight), regenerate
- **Dynamic DSL** — `gen` macro, `trace`, `splice`, nested tracing, score computation
- **Importance sampling** — rejection robustness with branching models
- **End-to-end** — line model with constrained observations

### GenJAX Compatibility

73 tests verify parity with GenJAX's design:

- **Edit interface** — constraint, selection, and proposal edits
- **Diff tracking** — incremental update with change hints
- **SMCP3** — reversible kernel proposals
- **Combinators** — Scan, Mask, Mix, Vectorized Switch
- **Programmable VI** — ELBO, IWELBO, wake-sleep objectives

## MLX Optimization Strategy

- **Loop compilation** — entire MCMC chains compiled into single Metal dispatches (MH 5.6x, MALA 2.3x, HMC 3.9x speedup)
- **`mx/compile-fn`** on score functions — JIT-compiles into cached Metal programs
- **`mx/value-and-grad`** — fused forward+backward in a single GPU dispatch
- **Adaptive step-size** — HMC dual averaging (Hoffman & Gelman 2014) auto-tunes during burn-in
- **`mx/tidy` + `mx/eval!` discipline** — bounds graph size, prevents Metal resource exhaustion
- **`mx/vmap`** in combinators — batch GPU execution across particles/instances
- **Unified memory** — MCMC control flow on CPU, all numerics on GPU, zero transfer cost

## GPU Resource Management

Apple Silicon has a hard kernel-level limit of **499,000 simultaneous Metal buffer objects** per process. This limit is identical across M1–M4 and all Pro/Max/Ultra variants. GenMLX manages this automatically in all built-in inference algorithms, but understanding it helps when writing custom inference loops or running very large models.

### Monitoring

```clojure
;; Quick snapshot of Metal resource usage
(mx/memory-report)
;; => {:active-bytes 124344
;;     :cache-bytes 0
;;     :peak-bytes 126712
;;     :wrappers 42
;;     :resource-limit 499000}

;; Individual queries
(mx/get-active-memory)   ;; bytes of Metal buffers in use
(mx/get-cache-memory)    ;; bytes in recycling cache
(mx/get-peak-memory)     ;; high-water mark since last reset
(mx/get-wrappers-count)  ;; live JS-wrapped MLX array objects
```

### Resource-safe algorithms

All built-in inference algorithms clean up Metal buffers between iterations via `mx/eval!` (detaches computation graphs) and periodic `mx/clear-cache!` (releases cached buffers). Long chains run indefinitely without hitting the resource limit.

| Category | Algorithms | Cleanup strategy |
|----------|-----------|-----------------|
| MCMC (via `collect-samples`) | `mh`, `mh-custom`, `gibbs`, `elliptical-slice`, `involutive-mh`, all kernel combinators | `tidy-step` + `with-resource-guard` + periodic `clear-cache!` |
| Compiled MCMC | `compiled-mh`, `compiled-mala`, `hmc`, `nuts` | `mx/tidy` + `mx/eval!` per step |
| Vectorized MCMC | `vectorized-compiled-mh`, `vectorized-mala` | Internal eval + periodic `clear-cache!` |
| Particle methods | `smc`, `csmc`, `smcp3` | Per-particle eval + periodic `clear-cache!` |
| Importance sampling | `importance-sampling`, vectorized IS | Per-sample eval |
| Optimization | `vi`, `programmable-vi`, `adev-optimize`, `wake-sleep` | `mx/eval!` per iter + periodic `clear-cache!` |

### Custom inference loops

When writing your own loops over MLX operations, follow this pattern:

```clojure
(loop [i 0, state init-state]
  (if (>= i n-iters)
    state
    (let [new-state (my-step state)
          ;; Materialize arrays — detaches computation graph
          _ (mx/eval! (:score new-state) (:weight new-state))
          ;; Periodically release cached Metal buffers
          _ (when (zero? (mod i 50)) (mx/clear-cache!))]
      (recur (inc i) new-state))))
```

For tighter control, set the cache limit lower at program start:

```clojure
(mx/set-cache-limit! (* 128 1024 1024))  ;; 128 MB cache (default is unlimited)
```

## Troubleshooting

**Q: I'm getting `[metal::malloc] Resource limit (499000) exceeded`**

This means too many Metal buffer objects are alive simultaneously. Solutions:

1. **Use vectorized inference** — `vectorized-importance-sampling`, `vsmc` run the model body once for N particles instead of N times
2. **Reduce sample count** — fewer particles/samples means fewer simultaneous buffers
3. **Set a cache limit** — add `(mx/set-cache-limit! (* 128 1024 1024))` at program start to cap the buffer recycling cache
4. **Clear cache between runs** — call `(mx/clear-cache!)` between separate inference calls
5. **Use compiled inference** — `compiled-mh`, `hmc`, `nuts` manage resources automatically via `mx/tidy`

**Q: Inference is slow / memory keeps growing**

- Call `(mx/eval!)` on result arrays inside your loop to materialize the computation graph. Without eval, MLX builds an ever-growing lazy graph.
- Check `(mx/get-wrappers-count)` — if it grows linearly with iterations, arrays aren't being freed. Use `mx/tidy` or `mx/dispose!` to release them.

## License

MIT
