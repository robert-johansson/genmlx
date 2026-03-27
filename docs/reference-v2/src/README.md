# GenMLX API Reference

Complete reference for all GenMLX modules. GenMLX is ~25,000 lines of ClojureScript organized into 8 architecture layers, from the MLX foundation through inference and verification.

## Architecture layers

| Layer | Modules | Purpose |
|-------|---------|---------|
| **0** | [mlx](mlx.md), [mlx.random](mlx.md#random-number-generation) | Thin wrapper over `@frost-beta/mlx`: arrays, ops, grad, vmap, PRNG |
| **1** | [choicemap](choicemap.md), [trace](trace.md), [selection](selection.md) | Core data structures: hierarchical choicemaps, immutable traces, address selections |
| **2** | [protocols](gfi.md), handler, [edit](edit.md), diff | GFI protocol definitions, handler-based execution engine, parametric edits |
| **3** | [gen](gfi.md#gen-macro), dynamic | DSL: `gen` macro for defining models, `DynamicGF` with full GFI |
| **4** | [dist](distributions.md), dist/core, dist/macros | 27+ distributions with open multimethods for extensibility |
| **5** | [combinators](combinators.md), [vmap](combinators.md#vmap) | Compositional model building: Map, Unfold, Switch, Scan, Mask, Mix, Recurse, Vmap |
| **6** | [inference/*](inference.md) | IS, MH, MALA, HMC, NUTS, Gibbs, SMC, SMCP3, VI, ADEV, MAP, kernels |
| **7** | vectorized, [gradients](gradients.md), [learning](learning.md), [nn](nn.md), [contracts](contracts.md), [verify](verify.md) | Batched execution, gradient computation, training, neural GFs, verification |

## Quick reference

### Defining models

```clojure
(require '[genmlx.gen :refer [gen]]
         '[genmlx.dynamic :as dyn]
         '[genmlx.dist :as dist])

(def model
  (gen [x]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (trace :y (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1))
      slope)))
```

### GFI operations

```clojure
(require '[genmlx.protocols :as p]
         '[genmlx.choicemap :as cm])

(p/simulate model [x])                          ;; => Trace
(p/generate model [x] (cm/choicemap :y obs))    ;; => {:trace :weight}
(p/update model trace new-constraints)           ;; => {:trace :weight :discard}
(p/regenerate model trace (sel/select :slope))   ;; => {:trace :weight}
(p/assess model [x] choices)                     ;; => {:retval :weight}
```

### Running inference

```clojure
(require '[genmlx.inference.importance :as is]
         '[genmlx.inference.mcmc :as mcmc]
         '[genmlx.inference.smc :as smc])

;; Importance sampling
(is/importance-sampling {:samples 1000} model args obs)

;; MCMC (MH, HMC, NUTS)
(mcmc/mh {:samples 1000 :burn 200} model args obs)
(mcmc/hmc {:samples 500 :step-size 0.01 :leapfrog-steps 10} model args obs)

;; Sequential Monte Carlo
(smc/smc {:particles 500} model args obs-seq)
```

### Vectorized inference

```clojure
(require '[genmlx.dynamic :as dyn])

;; Runs model body ONCE for N particles
(dyn/vsimulate model args 1000 key)     ;; => VectorizedTrace
(dyn/vgenerate model args obs 1000 key) ;; => VectorizedTrace with weights
```
