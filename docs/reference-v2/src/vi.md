# Variational Inference

Source: `src/genmlx/inference/vi.cljs`

Variational inference (VI) turns Bayesian inference into an optimization problem. Instead of sampling from the posterior directly, VI finds a simple distribution (the *guide* or *variational family*) that is closest to the true posterior by maximizing a lower bound on the log marginal likelihood (the ELBO).

GenMLX provides three tiers of VI:

1. **ADVI** -- automatic differentiation VI with a mean-field Gaussian guide. One function call, no guide to write.
2. **Model-aware VI** -- convenience wrappers that extract the log-density from a generative model and observations.
3. **Programmable VI** -- bring your own objective (ELBO, IWELBO, VIMCO, wake-sleep) and gradient estimator (reparameterization or REINFORCE).

All tiers use Adam optimization and support both interpreted and compiled execution paths. For gradient estimation that works directly on generative functions (with automatic reparameterization/REINFORCE selection per trace site), see [ADEV](adev.md).

```clojure
(require '[genmlx.inference.vi :as vi])
```

---

## ADVI

Automatic Differentiation Variational Inference with a mean-field Gaussian guide. The variational parameters are a mean vector and a log-standard-deviation vector, optimized jointly to maximize the ELBO.

### `vi`

```clojure
(vi/vi opts log-density init-params)
```

Run ADVI on a scalar log-density function. The guide is a diagonal Gaussian: each dimension has an independent mean and standard deviation. Optimization uses Adam with reparameterized ELBO gradients.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Options (see below) |
| `log-density` | `(fn [params] -> MLX scalar)` | Unnormalized log-density to approximate |
| `init-params` | MLX array `[d]` | Initial parameter values (sets dimensionality) |

| Option | Default | Description |
|--------|---------|-------------|
| `:iterations` | 1000 | Number of Adam optimization steps |
| `:learning-rate` | 0.01 | Adam learning rate |
| `:elbo-samples` | 10 | Monte Carlo samples per ELBO gradient estimate |
| `:beta1` | 0.9 | Adam first moment decay |
| `:beta2` | 0.999 | Adam second moment decay |
| `:epsilon` | 1e-8 | Adam numerical stability constant |
| `:callback` | `nil` | `(fn [{:iter :elbo :params}])` called periodically |
| `:key` | random | PRNG key for reproducibility |
| `:vectorized-log-density` | `nil` | Pre-vmapped log-density `(fn [[N d]] -> [N])` (computed automatically if not provided) |

**Returns:** `{:mu MLX-array, :sigma MLX-array, :elbo-history [numbers], :sample-fn (fn [n] -> samples)}`

- `:mu` -- posterior mean `[d]`
- `:sigma` -- posterior standard deviation `[d]`
- `:elbo-history` -- ELBO values sampled during optimization
- `:sample-fn` -- draw `n` samples from the fitted guide

```clojure
;; Approximate a 2D Gaussian posterior
(let [log-p (fn [z]
              (let [x (mx/index z 0) y (mx/index z 1)]
                (mx/add (dist/log-prob (dist/gaussian 3 1) x)
                        (dist/log-prob (dist/gaussian -1 2) y))))
      result (vi/vi {:iterations 500 :learning-rate 0.05}
                     log-p (mx/array [0 0]))]
  (println "mu:" (mx/->clj (:mu result)))       ;; ~[3, -1]
  (println "sigma:" (mx/->clj (:sigma result)))  ;; ~[1, 2]
  ((:sample-fn result) 5))                       ;; 5 samples from fitted guide
```

---

## Compiled VI

### `compiled-vi`

```clojure
(vi/compiled-vi opts log-density init-params)
```

Same interface and return type as `vi`, but uses `mx/compile-fn` on the gradient and ELBO functions for faster iteration. The compiled graph avoids repeated MLX tracing overhead. Best for pure tensor log-density functions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Same options as `vi`, plus `:device` |
| `log-density` | `(fn [params] -> MLX scalar)` | Unnormalized log-density (must be pure tensor ops) |
| `init-params` | MLX array `[d]` | Initial parameter values |

| Extra Option | Default | Description |
|--------------|---------|-------------|
| `:device` | `:cpu` | `:cpu` or `:gpu` -- CPU is often faster for small parameter spaces |

**Returns:** Same as `vi` -- `{:mu :sigma :elbo-history :sample-fn}`

```clojure
;; Compiled VI: same API, faster iteration
(vi/compiled-vi {:iterations 1000 :learning-rate 0.01 :device :gpu}
                log-density (mx/array [0 0]))
```

### `compiled-vi-from-model`

```clojure
(vi/compiled-vi-from-model opts model args observations addresses)
```

Convenience wrapper that runs compiled VI on a generative model. Internally delegates to `vi-from-model` because `mx/compile-fn` cannot compile GFI score functions (which use `volatile!` in the handler). Provided for API symmetry.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Same options as `vi` |
| `model` | DynamicGF | Generative function |
| `args` | Vector | Model arguments |
| `observations` | Choicemap | Observed values |
| `addresses` | Vector of keywords | Latent addresses to optimize |

**Returns:** Same as `vi` -- `{:mu :sigma :elbo-history :sample-fn}`

---

## VI from Model

### `vi-from-model`

```clojure
(vi/vi-from-model opts model args observations addresses)
```

Run ADVI directly on a generative model with observations. Constructs the log-density and initial parameters automatically by:

1. Running `generate` to get an initial trace conditioned on observations.
2. Extracting current values at the target `addresses` as initial parameters.
3. Building a score function that evaluates the model's log-joint at given parameter values.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Same options as `vi` |
| `model` | DynamicGF | Generative function |
| `args` | Vector | Model arguments |
| `observations` | Choicemap | Observed values (choicemap) |
| `addresses` | Vector of keywords | Latent addresses to optimize (e.g., `[:slope :intercept]`) |

**Returns:** Same as `vi` -- `{:mu :sigma :elbo-history :sample-fn}`

```clojure
(require '[genmlx.gen :refer [gen]]
         '[genmlx.dist :as dist]
         '[genmlx.choicemap :as cm])

(def model
  (gen [x]
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (trace :y (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1)))))

;; Run VI on the model, observing y=5.0
(let [result (vi/vi-from-model
               {:iterations 500 :learning-rate 0.05}
               model [(mx/scalar 2.0)]
               (cm/choicemap :y (mx/scalar 5.0))
               [:slope :intercept])]
  (println "posterior mean:" (mx/->clj (:mu result)))
  (println "posterior std:" (mx/->clj (:sigma result))))
```

---

## Programmable VI

### `programmable-vi`

```clojure
(vi/programmable-vi opts log-p-fn log-q-fn sample-fn init-params)
```

Fully programmable variational inference. You provide the model density, guide density, guide sampler, and choose the objective and gradient estimator. This is the most flexible VI entry point.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Options (see below) |
| `log-p-fn` | `(fn [z] -> MLX scalar)` | Model log-density |
| `log-q-fn` | `(fn [z params] -> MLX scalar)` | Parameterized guide log-density |
| `sample-fn` | `(fn [params key n] -> MLX [n d])` | Guide sampler drawing `n` samples |
| `init-params` | MLX array | Initial variational parameters |

| Option | Default | Description |
|--------|---------|-------------|
| `:iterations` | 1000 | Optimization steps |
| `:learning-rate` | 0.01 | Adam learning rate |
| `:n-samples` | 10 | MC samples per gradient estimate |
| `:objective` | `:elbo` | `:elbo`, `:iwelbo`, `:vimco`, `:pwake`, `:qwake`, or custom `(fn [log-p log-q samples] -> scalar)` |
| `:estimator` | `:reparam` | `:reparam` (reparameterization trick) or `:reinforce` (score function) |
| `:callback` | `nil` | `(fn [{:iter :loss}])` called each step |
| `:key` | random | PRNG key |

**Returns:** `{:params MLX-array, :loss-history [numbers]}`

```clojure
;; Programmable VI with IWELBO objective
(let [log-p (fn [z] ...)        ;; model log-density
      log-q (fn [z params] ...) ;; guide log-density, parameterized
      sample (fn [params key n] ;; guide sampler
               (let [mu (mx/slice params 0 2)
                     sigma (mx/exp (mx/slice params 2 4))
                     eps (rng/normal key [n 2])]
                 (mx/add mu (mx/multiply sigma eps))))]
  (vi/programmable-vi
    {:iterations 2000 :objective :iwelbo :n-samples 20}
    log-p log-q sample (mx/zeros [4])))
```

### `compiled-programmable-vi`

```clojure
(vi/compiled-programmable-vi opts log-p-fn log-q-fn sample-fn init-params)
```

Same as `programmable-vi` but compiles the gradient and loss functions with `mx/compile-fn`. Sampling is separated from gradient computation to enable compilation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Same as `programmable-vi`, plus `:device` |
| `log-p-fn` | `(fn [z] -> MLX scalar)` | Model log-density (pure tensor ops) |
| `log-q-fn` | `(fn [z params] -> MLX scalar)` | Guide log-density (pure tensor ops) |
| `sample-fn` | `(fn [params key n] -> MLX [n d])` | Guide sampler |
| `init-params` | MLX array | Initial variational parameters |

| Extra Option | Default | Description |
|--------------|---------|-------------|
| `:device` | `:cpu` | `:cpu` or `:gpu` |

**Returns:** `{:params MLX-array, :loss-history [numbers]}`

---

## Objectives

Objective functions take `log-p-fn` and `log-q-fn` and return a function `(fn [samples] -> MLX scalar)` that computes the objective value from a batch of samples `[K d]`. All objectives are designed so that *maximizing* the returned value improves the approximation.

### `elbo-objective`

```clojure
(vi/elbo-objective log-p-fn log-q-fn)
```

Standard Evidence Lower Bound: E\_q[log p(x,z) - log q(z)]. The workhorse of variational inference. Equals the log marginal likelihood minus the KL divergence from guide to posterior.

| Parameter | Type | Description |
|-----------|------|-------------|
| `log-p-fn` | `(fn [z] -> MLX scalar)` | Model log-density |
| `log-q-fn` | `(fn [z] -> MLX scalar)` | Guide log-density |

**Returns:** `(fn [samples] -> MLX scalar)` where `samples` is `[K d]`

### `iwelbo-objective`

```clojure
(vi/iwelbo-objective log-p-fn log-q-fn)
```

Importance-Weighted ELBO (IWAE bound). Uses K samples to compute a tighter bound than the standard ELBO: log(1/K * sum\_k p(z\_k)/q(z\_k)). Approaches the true log marginal likelihood as K grows.

| Parameter | Type | Description |
|-----------|------|-------------|
| `log-p-fn` | `(fn [z] -> MLX scalar)` | Model log-density |
| `log-q-fn` | `(fn [z] -> MLX scalar)` | Guide log-density |

**Returns:** `(fn [samples] -> MLX scalar)` where `samples` is `[K d]`

### `vimco-objective`

```clojure
(vi/vimco-objective log-p-fn log-q-fn)
```

VIMCO (Variational Inference with Multi-sample Objectives). Combines the tighter IWAE bound with leave-one-out control variates for lower-variance gradients than REINFORCE alone. Particularly useful for discrete latent variables.

| Parameter | Type | Description |
|-----------|------|-------------|
| `log-p-fn` | `(fn [z] -> MLX scalar)` | Model log-density |
| `log-q-fn` | `(fn [z] -> MLX scalar)` | Guide log-density |

**Returns:** `(fn [samples] -> MLX scalar)` -- surrogate whose gradient is the VIMCO estimator

### `pwake-objective`

```clojure
(vi/pwake-objective log-p-fn log-q-fn)
```

P-Wake objective: trains the model to match the guide's proposals. Equivalent to minimizing KL(q || p). Maximizes E\_q[log p(z)]; the log-q term is constant w.r.t. model parameters.

| Parameter | Type | Description |
|-----------|------|-------------|
| `log-p-fn` | `(fn [z] -> MLX scalar)` | Model log-density |
| `log-q-fn` | `(fn [z] -> MLX scalar)` | Guide log-density |

**Returns:** `(fn [samples] -> MLX scalar)`

### `qwake-objective`

```clojure
(vi/qwake-objective log-p-fn log-q-fn)
```

Q-Wake objective: trains the guide to approximate the posterior using self-normalized importance weights. Maximizes the importance-weighted E\_p[log q(z)].

| Parameter | Type | Description |
|-----------|------|-------------|
| `log-p-fn` | `(fn [z] -> MLX scalar)` | Model log-density |
| `log-q-fn` | `(fn [z] -> MLX scalar)` | Guide log-density |

**Returns:** `(fn [samples] -> MLX scalar)`

---

## Gradient Estimators

### `reinforce-estimator`

```clojure
(vi/reinforce-estimator objective-fn log-q-fn)
```

REINFORCE (score function) gradient estimator. Wraps an objective function to produce a surrogate loss whose gradient equals the REINFORCE estimator. Uses the mean objective value as a control variate baseline for variance reduction. Use this for non-reparameterizable distributions (discrete, mixture models).

| Parameter | Type | Description |
|-----------|------|-------------|
| `objective-fn` | `(fn [samples] -> MLX scalar)` | Objective function (from `elbo-objective`, etc.) |
| `log-q-fn` | `(fn [z] -> MLX scalar)` | Guide log-density |

**Returns:** `(fn [samples] -> MLX scalar)` -- surrogate whose gradient is the REINFORCE estimator

---

## VIMCO Convenience

### `vimco`

```clojure
(vi/vimco opts log-p-fn log-q-fn sample-fn init-params)
```

Convenience wrapper around `programmable-vi` with the `:vimco` objective pre-selected. Uses leave-one-out control variates for lower-variance gradient estimates.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Same as `programmable-vi` (without `:objective`) |
| `log-p-fn` | `(fn [z] -> MLX scalar)` | Model log-density |
| `log-q-fn` | `(fn [z params] -> MLX scalar)` | Parameterized guide log-density |
| `sample-fn` | `(fn [params key n] -> MLX [n d])` | Guide sampler |
| `init-params` | MLX array | Initial variational parameters |

**Returns:** `{:params MLX-array, :loss-history [numbers]}`

---

## Training Loop Example

A complete VI training loop showing the relationship between the different tiers:

```clojure
(require '[genmlx.inference.vi :as vi]
         '[genmlx.mlx :as mx]
         '[genmlx.mlx.random :as rng]
         '[genmlx.dist :as dist])

;; === Tier 1: ADVI (simplest) ===
;; Fit a mean-field Gaussian to an arbitrary log-density.
(let [target-log-p (fn [z]
                     (let [x (mx/index z 0)]
                       (dc/dist-log-prob (dist/gaussian 5 2) x)))
      result (vi/vi {:iterations 500 :elbo-samples 20}
                     target-log-p (mx/array [0.0]))]
  (println "ADVI mu:" (mx/->clj (:mu result))))

;; === Tier 2: VI from model ===
;; No log-density function needed -- works directly on gen functions.
(let [result (vi/vi-from-model
               {:iterations 300}
               model [x-data]
               observations
               [:slope :intercept])]
  (println "Model VI mu:" (mx/->clj (:mu result))))

;; === Tier 3: Programmable VI ===
;; Custom guide, custom objective, full control.
(let [;; Gaussian guide with learnable mean and log-sigma
      log-q (fn [z params]
              (let [mu    (mx/index params 0)
                    lsig  (mx/index params 1)
                    sigma (mx/exp lsig)]
                (dc/dist-log-prob (dist/gaussian mu sigma) z)))
      sample (fn [params key n]
               (let [mu (mx/index params 0)
                     sigma (mx/exp (mx/index params 1))
                     eps (rng/normal key [n 1])]
                 (mx/add mu (mx/multiply sigma eps))))
      result (vi/programmable-vi
               {:iterations 1000 :objective :iwelbo :n-samples 50}
               target-log-p log-q sample (mx/array [0.0 0.0]))]
  (println "Programmable VI params:" (mx/->clj (:params result)))
  (println "Final loss:" (last (:loss-history result))))
```
