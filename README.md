<p align="center">
  <img src="genmlx.png" alt="GenMLX" width="528">
</p>

A probabilistic programming language implemented in ClojureScript on Node.js ([nbb](https://github.com/babashka/nbb)), using Apple's [MLX framework](https://github.com/ml-explore/mlx) for GPU acceleration via [`@frost-beta/mlx`](https://github.com/nicebyte/node-mlx).

GenMLX implements Gen's **Generative Function Interface (GFI)** — the same architecture used by [GenJAX](https://github.com/probcomp/GenJAX) (JAX) and [Gen.jl](https://github.com/probcomp/Gen.jl) (Julia).

## Why GenMLX?

The JVM-based Gen.clj + MLX approach hit fundamental FFI friction (~1000x overhead per scalar operation). On Node.js, ClojureScript talks to MLX through a native addon with **zero FFI overhead** — the entire ~2000-line JVM interop layer collapses to ~100 lines of JS interop.

**Design principles:**
- MLX as a first-class platform — lean into unified memory, lazy evaluation, dynamic shapes
- Beautiful, idiomatic ClojureScript — protocols, records, persistent data, minimal mutation
- MLX arrays throughout — scores and choice values stay on GPU end-to-end, enabling `mx/grad` through entire models

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
  genmlx.protocols    — IGenerativeFunction, IGenerate, IUpdate, IRegenerate
  genmlx.handler      — handler-based execution via dynamic vars

Layer 3: DSL
  genmlx.gen          — gen macro
  genmlx.dynamic      — DynamicDSLFunction (full GFI implementation)

Layer 4: Distributions
  genmlx.dist         — 13 distributions, each a GFI participant

Layer 5: Combinators
  genmlx.combinators  — Map, Unfold, Switch

Layer 6: Inference
  genmlx.inference    — IS, MH, MALA, HMC, NUTS, SMC, VI
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

## Inference Algorithms

- **Importance Sampling** — `importance-sampling`, `importance-resampling`
- **Metropolis-Hastings** — `mh` (via GFI `regenerate`)
- **MALA** — `mala` (gradient-informed proposals)
- **HMC** — `hmc` (compiled leapfrog integration)
- **NUTS** — `nuts` (adaptive trajectory length)
- **SMC** — `smc` (particle filtering with resampling + rejuvenation)
- **Variational Inference** — `vi` (ADVI with mean-field Gaussian guide)
- **Diagnostics** — `ess`, `r-hat`, `summarize`

## Combinators

- **Map** — apply a generative function independently across sequences
- **Unfold** — sequential application for time-series models
- **Switch** — select between branches for mixture models

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
```

## Running Tests

```bash
# Individual test files
npx nbb -cp src:test test/genmlx/choicemap_test.cljs
npx nbb -cp src:test test/genmlx/dist_test.cljs
npx nbb -cp src:test test/genmlx/gen_test.cljs
npx nbb -cp src:test test/genmlx/inference_test.cljs

# All tests
for f in choicemap_test trace_test selection_test handler_test dist_test gen_test combinators_test inference_test; do
  npx nbb -cp src:test "test/genmlx/${f}.cljs"
done

# Benchmarks
npx nbb -cp src:test test/genmlx/benchmark.cljs
```

## MLX Optimization Strategy

- **`mx/compile-fn`** on score functions — JIT-compiles into cached Metal programs
- **`mx/value-and-grad`** — fused forward+backward in a single GPU dispatch
- **One `mx/eval!` per leapfrog step** — bounds graph size, avoids Metal OOM
- **`mx/tidy` per step** — automatic disposal of intermediate arrays
- **`mx/vmap`** in combinators — batch GPU execution across particles/instances
- **Unified memory** — MCMC control flow on CPU, all numerics on GPU, zero transfer cost

## License

MIT
