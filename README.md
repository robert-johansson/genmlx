<p align="center">
  <img src="genmlx.png" alt="GenMLX" width="528">
</p>

A probabilistic programming language implemented in ClojureScript on Node.js ([nbb](https://github.com/babashka/nbb)), using Apple's [MLX framework](https://github.com/ml-explore/mlx) for GPU acceleration via [`@frost-beta/mlx`](https://github.com/nicebyte/node-mlx).

GenMLX implements Gen's **Generative Function Interface (GFI)** — the same architecture used by [GenJAX](https://github.com/probcomp/GenJAX) (JAX) and [Gen.jl](https://github.com/probcomp/Gen.jl) (Julia).

## Why GenMLX?

Gen implementations exist for Julia and JAX — but nothing for Apple Silicon's GPU framework. MLX's unified memory model is a natural fit for probabilistic programming: MCMC control flow runs on CPU while all numerics stay on GPU, with zero data transfer cost. ClojureScript on Node.js gives direct access to MLX through a native addon with no FFI overhead, and nbb provides a fast REPL for interactive model development.

- **MLX-native** — unified memory, lazy evaluation, dynamic shapes, `mx/grad` through entire models
- **~10,000 lines of ClojureScript** — protocols, records, persistent data structures, the whole thing is readable in an afternoon
- **GPU end-to-end** — scores and choice values are MLX arrays throughout, extracted with `mx/item` only at inference boundaries

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Node.js 18+
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
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Bayesian linear regression
(def model
  (gen [xs]
    (let [slope     (dyn/trace :slope (dist/gaussian 0 10))
          intercept (dyn/trace :intercept (dist/gaussian 0 10))]
      (mx/eval! slope intercept)
      (let [s (mx/item slope) i (mx/item intercept)]
        (doseq [[j x] (map-indexed vector xs)]
          (dyn/trace (keyword (str "y" j))
                     (dist/gaussian (+ (* s x) i) 1)))
        [s i]))))
```

### Run inference

```clojure
;; Observed data
(def xs [1.0 2.0 3.0 4.0 5.0])
(def observations
  (reduce (fn [cm [j y]]
            (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector [2.1 3.9 6.2 7.8 10.1])))

;; Metropolis-Hastings
(def traces
  (mcmc/mh {:samples 500 :burn 100
            :selection (sel/select :slope :intercept)}
           model [xs] observations))

;; Examine posterior
(let [slopes (mapv (fn [t]
                     (let [v (cm/get-value (cm/get-submap (tr/get-choices t) :slope))]
                       (mx/eval! v) (mx/item v)))
                   traces)]
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
  genmlx.vectorized   — VectorizedTrace, batched execution, 29–122x speedup
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
- **HMC** — `hmc` (compiled leapfrog integration)
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

**Benchmarks (N=100, 5-site model):**
- `dist-sample-n`: 29x speedup
- `vgenerate`: 61–122x speedup
- Vectorized IS: 81x
- Vectorized SMC init: 65x

## Running Tests

```bash
# Individual test files
npx nbb test/genmlx/dist_test.cljs
npx nbb test/genmlx/gen_test.cljs
npx nbb test/genmlx/inference_test.cljs

# All core tests
for f in choicemap_test trace_test selection_test handler_test dist_test gen_test combinators_test inference_test; do
  npx nbb "test/genmlx/${f}.cljs"
done

# Compatibility suites
npx nbb test/genmlx/gen_clj_compat_test.cljs     # 165/165 tests (from Gen.clj)
npx nbb test/genmlx/genjax_compat_test.cljs       # 73/73 tests (GenJAX compat)

# Vectorized tests + benchmarks
npx nbb test/genmlx/vectorized_test.cljs
npx nbb test/genmlx/vectorized_benchmark.cljs
```

600+ test assertions across 29 test files.

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

- **`mx/compile-fn`** on score functions — JIT-compiles into cached Metal programs
- **`mx/value-and-grad`** — fused forward+backward in a single GPU dispatch
- **One `mx/eval!` per leapfrog step** — bounds graph size, avoids Metal OOM
- **`mx/tidy` per step** — automatic disposal of intermediate arrays
- **`mx/vmap`** in combinators — batch GPU execution across particles/instances
- **Unified memory** — MCMC control flow on CPU, all numerics on GPU, zero transfer cost

## License

MIT
