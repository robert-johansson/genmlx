# Fit

The `fit` function is the high-level entry point for probabilistic inference in GenMLX. It automatically selects an inference method based on model structure, runs inference, and optionally performs parameter optimization -- all in a single call.

Under the hood, `fit` composes three systems: the method selection decision tree (`method_selection.cljs`), inference dispatch (importance sampling, MCMC, HMC, VI, SMC, or exact computation), and the compiled optimizer for parameter learning.

```clojure
(require '[genmlx.fit :as fit])
```

Source: `src/genmlx/fit.cljs`, `src/genmlx/method_selection.cljs`

---

## fit

### `fit`

```clojure
(fit/fit model args data)
(fit/fit model args data opts)
```

Fit a generative model to data. Analyzes the model's schema and analytical plan to select the best inference method, runs inference, and returns a structured result with posterior summaries, diagnostics, and timing.

When `:learn` is provided in `opts`, a parameter learning loop runs after initial inference, optimizing the specified parameters via the compiled Adam optimizer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | generative function | Model with schema (from `gen` macro) |
| `args` | vector | Model arguments |
| `data` | ChoiceMap | Observed values to condition on |
| `opts` | map (optional) | Override options (see below) |

Options for `opts`:

| Key | Default | Description |
|-----|---------|-------------|
| `:method` | auto | Override method selection: `:exact`, `:kalman`, `:smc`, `:mcmc`, `:hmc`, `:vi`, `:handler-is` |
| `:learn` | `nil` | Vector of param names to optimize (enables learning loop) |
| `:iterations` | auto | Number of optimization iterations |
| `:lr` | `0.01` | Learning rate for parameter optimization |
| `:particles` | auto | Number of particles for IS/SMC |
| `:samples` | auto | Number of samples for MCMC |
| `:step-size` | auto | Step size for HMC |
| `:n-leapfrog` | auto | Leapfrog steps for HMC |
| `:burn` | auto | Burn-in samples for MCMC |
| `:callback` | `nil` | `(fn [{:iter :loss :method :elapsed}])` called periodically |
| `:key` | `nil` | PRNG key for reproducibility |
| `:verbose?` | `false` | Print method selection reasoning |

**Returns:** A map with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `:method` | keyword | Which inference method was used |
| `:trace` | Trace or nil | Best/final trace (nil for VI and HMC) |
| `:posterior` | map | Per-latent summary: `{addr {:value number}}` or `{addr {:mean :std :samples}}` |
| `:log-ml` | number or nil | Log marginal likelihood estimate (when available) |
| `:loss-history` | vector or nil | Optimization loss per iteration (when `:learn` is set) |
| `:params` | MLX array or nil | Learned parameter values (when `:learn` is set) |
| `:diagnostics` | map | Method-specific diagnostics including `:reason`, `:n-residual`, `:n-latent` |
| `:elapsed-ms` | number | Wall-clock time in milliseconds |

**Example -- automatic method selection:**
```clojure
(def model
  (gen [xs]
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept)
                              (mx/scalar 1.0))))
      slope)))

(def xs [1.0 2.0 3.0])
(def obs (cm/choicemap :y0 (mx/scalar 2.1) :y1 (mx/scalar 3.9) :y2 (mx/scalar 6.2)))

(def result (fit/fit model [xs] obs))

(:method result)      ;; => :hmc (auto-selected for static model with 2 residual dims)
(:posterior result)   ;; => {:slope {:mean ~2.0 :std ~0.3 ...} :intercept {:mean ~0.1 ...}}
(:elapsed-ms result)  ;; => wall-clock time
```

**Example -- with parameter learning:**
```clojure
(def result
  (fit/fit model [xs] obs
    {:learn [:slope :intercept]
     :iterations 200
     :lr 0.01}))

(:loss-history result)  ;; => [5.2 4.8 4.3 ... 1.1]
(:params result)        ;; => MLX array of learned parameter values
```

**Example -- override method:**
```clojure
(def result
  (fit/fit model [xs] obs
    {:method :vi
     :iterations 2000
     :verbose? true}))
;; [fit] Selected method: vi -- User-specified
```

---

## Method Selection

The method selection system is a pure decision tree that reads model schema metadata and the Level 3/3.5 analytical plan to choose the optimal inference algorithm. It runs no MLX operations and performs no inference -- only structural analysis.

### `select-method`

```clojure
(ms/select-method model observations)
```

Select the optimal inference method for a model given observations. Analyzes the model's schema (trace sites, splice sites, static/dynamic classification) and analytical elimination plan to determine which method will be most efficient.

```clojure
(require '[genmlx.method-selection :as ms])
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | generative function | A DynamicGF with `:schema` |
| `observations` | ChoiceMap or nil | Observed data constraints |

**Returns:** A map describing the selected method:

| Key | Type | Description |
|-----|------|-------------|
| `:method` | keyword | Selected method: `:exact`, `:kalman`, `:smc`, `:hmc`, `:vi`, `:handler-is` |
| `:reason` | string | Human-readable explanation of the choice |
| `:opts` | map | Base hyperparameters for the selected method |
| `:eliminated` | set | Addresses eliminated by analytical plan |
| `:residual-addrs` | set | Addresses still requiring inference |
| `:n-residual` | number | Count of residual addresses |
| `:n-latent` | number | Count of all latent (unobserved) addresses |

**Example:**
```clojure
(def selection (ms/select-method model obs))

(:method selection)
;; => :hmc

(:reason selection)
;; => "Static model with 2 residual dims (< 10)"

(:n-residual selection)
;; => 2
```

---

### Decision Tree

The decision tree evaluates conditions in priority order. The first matching rule wins:

| Priority | Condition | Method | Reason |
|----------|-----------|--------|--------|
| 1 | All trace sites eliminated or observed | `:exact` | Analytical plan covers everything |
| 2 | No trace sites at all | `:exact` | Trivial model |
| 3 | Kalman chains in analytical plan | `:kalman` | Linear-Gaussian temporal structure |
| 4 | Splice sites present | `:smc` | Sub-model/temporal structure |
| 5 | Dynamic addresses | `:handler-is` | Shape-based methods not applicable |
| 6 | Static model, residual dims <= 10 | `:hmc` | Low-dimensional continuous posterior |
| 7 | Static model, residual dims > 10 | `:vi` | High-dimensional -- amortized gradient methods |
| 8 | Fallback | `:handler-is` | No specialized method matched |

The threshold between HMC and VI is 10 residual dimensions. Below this, HMC explores the posterior efficiently with exact sampling. Above it, VI's amortized gradient updates scale better.

---

### `tune-method-opts`

```clojure
(ms/tune-method-opts selection)
(ms/tune-method-opts selection user-opts)
```

Tune method-specific hyperparameters based on model structure. Takes the result of `select-method` and adjusts particle counts, step sizes, iteration counts, and other settings based on the number of residual and latent dimensions.

User overrides are merged last, taking priority over tuned values.

| Parameter | Type | Description |
|-----------|------|-------------|
| `selection` | map | Result of `select-method` |
| `user-opts` | map (optional) | User overrides (merged last) |

**Returns:** Map of tuned hyperparameters for the selected method.

**Tuning rules by method:**

**`:smc`** -- particle count scales with residual dimensions:

| Residual dims | Particles |
|---------------|-----------|
| 0 | 50 |
| 1--5 | 100 |
| 6--20 | 500 |
| > 20 | 1000 |

**`:hmc`** -- step size and leapfrog steps adapt to dimensionality:

| Residual dims | Leapfrog steps | Step size | Warmup |
|---------------|----------------|-----------|--------|
| 1--3 | 10 | 0.05 | 200 |
| 4--7 | 15 | 0.01 | 500 |
| > 7 | 20 | 0.005 | 1000 |

**`:vi`** -- iteration count and sample count scale up:

| Residual dims | Iterations | Samples |
|---------------|------------|---------|
| 1--20 | 2000 | 10 |
| 21--50 | 5000 | 20 |
| > 50 | 10000 | 50 |

**`:handler-is`** -- particle count scales with latent dimensions:

| Latent dims | Particles |
|-------------|-----------|
| 1--5 | 1000 |
| 6--20 | 5000 |
| > 20 | 10000 |

**Example:**
```clojure
(def selection (ms/select-method model obs))
(def tuned (ms/tune-method-opts selection))

;; For an HMC selection with 2 residual dims:
tuned
;; => {:n-samples 1000 :n-warmup 200 :step-size 0.05 :n-leapfrog 10}

;; Override learning rate
(ms/tune-method-opts selection {:step-size 0.1})
;; => {:n-samples 1000 :n-warmup 200 :step-size 0.1 :n-leapfrog 10}
```
