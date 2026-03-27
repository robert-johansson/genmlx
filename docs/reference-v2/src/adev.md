# ADEV

Source: `src/genmlx/inference/adev.cljs`

ADEV (Automatic Differentiation of Expected Values) computes unbiased gradients of expected costs through generative functions. Unlike [VI](vi.md), which requires you to write separate log-density and guide functions, ADEV works directly on gen functions: it walks the model's trace sites and automatically selects the right gradient estimator at each one.

- **Reparameterizable distributions** (Gaussian, Uniform, Beta, Gamma, ...): the reparameterization trick. Gradients flow through the sampled value via a deterministic transform of the distribution parameters and a fixed noise source.
- **Non-reparameterizable distributions** (Bernoulli, Categorical, Poisson, ...): the REINFORCE (score function) estimator with `stop-gradient` on the sampled value and a surrogate loss term.

This per-site selection happens transparently. You define a model and a cost function; ADEV handles the rest.

```clojure
(require '[genmlx.inference.adev :as adev])
```

---

## Core

### `has-reparam?`

```clojure
(adev/has-reparam? dist)
```

Check if a distribution supports reparameterized sampling via `dist-reparam`. Distributions with a registered `dist-reparam` multimethod use the reparameterization trick; all others fall back to REINFORCE.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dist` | Distribution | A distribution record |

**Returns:** Boolean

### `adev-execute`

```clojure
(adev/adev-execute gf args key)
```

Execute a generative function under the ADEV handler (sequential, one sample). At each trace site, reparameterizable distributions sample via `dist-reparam` so gradients flow through the value. Non-reparameterizable distributions sample via `dist-sample` with `stop-gradient`, accumulating log-probabilities for the REINFORCE surrogate term.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | DynamicGF | Generative function to execute |
| `args` | Vector | Model arguments |
| `key` | MLX array | PRNG key |

**Returns:** `{:trace Trace, :reinforce-lp MLX-scalar}`

- `:trace` -- standard Trace record with choices, score, retval
- `:reinforce-lp` -- accumulated log-probability from non-reparameterizable sites (zero if all sites are reparameterizable)

### `adev-surrogate`

```clojure
(adev/adev-surrogate gf args cost-fn key)
(adev/adev-surrogate gf args cost-fn key baseline)
```

Build the ADEV surrogate loss for a single sample. The surrogate is constructed so that `mx/grad` of the surrogate gives an unbiased gradient estimate of E[cost]:

```
surrogate = cost + stop_gradient(cost - baseline) * reinforce_lp
```

The first term carries gradients through reparameterizable sites. The second term provides the REINFORCE gradient signal for non-reparameterizable sites. When all sites are reparameterizable, `reinforce_lp` is zero and the surrogate reduces to the cost itself.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | DynamicGF | Generative function |
| `args` | Vector | Model arguments |
| `cost-fn` | `(fn [trace] -> MLX scalar)` | Cost function applied to the trace |
| `key` | MLX array | PRNG key |
| `baseline` | MLX scalar (optional) | Control variate baseline for variance reduction |

**Returns:** MLX scalar -- the surrogate loss

---

## Vectorized Execution

The vectorized path runs the model body **once** for all N particles using shape-based batching. This is the default for `adev-optimize` and is typically 10--100x faster than the sequential path.

### `vadev-execute`

```clojure
(adev/vadev-execute gf args n key)
```

Execute a generative function under the batched ADEV handler. Each trace site samples `[N]`-shaped arrays; all arithmetic broadcasts naturally. Same reparameterization/REINFORCE logic as `adev-execute`, but batched.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | DynamicGF | Generative function |
| `args` | Vector | Model arguments |
| `n` | Integer | Number of particles (batch size) |
| `key` | MLX array | PRNG key |

**Returns:** `{:choices Choicemap, :score MLX [N], :reinforce-lp MLX [N], :retval any}`

- `:choices` -- choicemap with `[N]`-shaped leaf values
- `:score` -- per-particle log-joint scores
- `:reinforce-lp` -- per-particle REINFORCE log-probabilities
- `:retval` -- model return value (with `[N]`-shaped arrays)

### `vadev-surrogate`

```clojure
(adev/vadev-surrogate gf args cost-fn n key)
(adev/vadev-surrogate gf args cost-fn n key baseline)
```

Build the batched ADEV surrogate loss (scalar). Runs the model body once for all N particles and returns the mean surrogate across particles.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | DynamicGF | Generative function |
| `args` | Vector | Model arguments |
| `cost-fn` | `(fn [result] -> MLX [N])` | Cost function; receives the full result map `{:choices :score :reinforce-lp :retval}` and returns per-particle costs |
| `n` | Integer | Batch size |
| `key` | MLX array | PRNG key |
| `baseline` | MLX scalar (optional) | Control variate baseline |

**Returns:** MLX scalar -- mean surrogate loss across particles

---

## Gradients

### `adev-gradient`

```clojure
(adev/adev-gradient opts gf args cost-fn param-names params-array)
```

Compute the ADEV gradient of E[cost] w.r.t. a flat parameter array (sequential path). Uses `adev-surrogate` internally, averaging over `n-samples` independent executions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Options (see below) |
| `gf` | DynamicGF | Generative function |
| `args` | Vector | Model arguments |
| `cost-fn` | `(fn [trace] -> MLX scalar)` | Cost function (receives Trace) |
| `param-names` | Vector of keywords | Parameter names matching `param` calls in the model |
| `params-array` | MLX array `[d]` | Flat parameter vector |

| Option | Default | Description |
|--------|---------|-------------|
| `:n-samples` | 1 | Number of MC samples for gradient estimate |
| `:baseline` | `nil` | Scalar control variate baseline |

**Returns:** `{:loss MLX-scalar, :grad MLX-array [d]}`

### `vadev-gradient`

```clojure
(adev/vadev-gradient opts gf args cost-fn param-names params-array)
```

Compute the ADEV gradient via the vectorized (batched) path. Runs the model body once for all particles. This is the recommended path for models without `splice`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Options (see below) |
| `gf` | DynamicGF | Generative function |
| `args` | Vector | Model arguments |
| `cost-fn` | `(fn [result] -> MLX [N])` | Cost function (receives batched result map) |
| `param-names` | Vector of keywords | Parameter names |
| `params-array` | MLX array `[d]` | Flat parameter vector |

| Option | Default | Description |
|--------|---------|-------------|
| `:n-samples` | 100 | Batch size (all particles run in one model execution) |

**Returns:** `{:loss MLX-scalar, :grad MLX-array [d]}`

---

## Optimization

### `adev-optimize`

```clojure
(adev/adev-optimize opts gf args cost-fn param-names init-params)
```

Optimize E[cost] via ADEV gradient estimation with Adam. Uses the vectorized (batched) path by default for maximum throughput. Set `:sequential true` to use the sequential path (required for models with `splice`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Options (see below) |
| `gf` | DynamicGF | Generative function (model) |
| `args` | Vector | Model arguments |
| `cost-fn` | Function | Cost function -- see note on sequential vs. vectorized below |
| `param-names` | Vector of keywords | Parameter names |
| `init-params` | MLX array `[d]` | Initial flat parameter vector |

| Option | Default | Description |
|--------|---------|-------------|
| `:iterations` | 100 | Number of Adam steps |
| `:lr` | 0.01 | Learning rate |
| `:n-samples` | 1 | Samples per gradient estimate (vectorized: batch size) |
| `:baseline-decay` | `nil` | EMA decay for variance-reduction baseline (e.g., 0.9). `nil` disables baseline. |
| `:callback` | `nil` | `(fn [{:iter :loss :params}])` called each step |
| `:sequential` | `false` | Force sequential path (needed for models with `splice`) |
| `:key` | random | PRNG key (unused in current implementation, kept for API consistency) |

**Cost function signature depends on the execution path:**

- **Vectorized** (default): `(fn [result] -> MLX [N])` where `result` is `{:choices :score :reinforce-lp :retval}` with `[N]`-shaped arrays. Return per-particle costs.
- **Sequential**: `(fn [trace] -> MLX scalar)` where `trace` is a standard Trace record.

**Returns:** `{:params MLX-array [d], :loss-history [numbers]}`

```clojure
(require '[genmlx.gen :refer [gen]]
         '[genmlx.inference.adev :as adev]
         '[genmlx.dist :as dist]
         '[genmlx.mlx :as mx])

;; Model with a learnable parameter
(def model
  (gen [target]
    (let [mu (param :mu 0.0)]
      (trace :x (dist/gaussian mu 1.0)))))

;; Cost: squared distance from target
(defn cost-fn [result]
  ;; Vectorized: result has [N]-shaped arrays
  (let [x (cm/get-value (:choices result) :x)]
    (mx/square (mx/subtract x (mx/scalar 5.0)))))

;; Optimize: find mu that minimizes E[(x - 5)^2]
(let [result (adev/adev-optimize
               {:iterations 200 :lr 0.05 :n-samples 50}
               model [] cost-fn
               [:mu] (mx/array [0.0]))]
  (println "Learned mu:" (mx/->clj (:params result)))  ;; ~[5.0]
  (println "Final loss:" (last (:loss-history result))))
```

### `compiled-adev-optimize`

```clojure
(adev/compiled-adev-optimize opts gf args cost-fn param-names init-params)
```

Compiled vectorized ADEV optimization. Uses `mx/value-and-grad` with `mx/tidy-run` for memory-efficient gradient computation. Each iteration creates a fresh computation graph, evaluates it, and discards intermediates.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Options (see below) |
| `gf` | DynamicGF | Generative function |
| `args` | Vector | Model arguments |
| `cost-fn` | `(fn [result] -> MLX [N])` | Cost function (vectorized path only) |
| `param-names` | Vector of keywords | Parameter names |
| `init-params` | MLX array `[d]` | Initial flat parameter vector |

| Option | Default | Description |
|--------|---------|-------------|
| `:iterations` | 100 | Number of Adam steps |
| `:lr` | 0.01 | Learning rate |
| `:n-samples` | 100 | Batch size per gradient estimate |
| `:baseline-decay` | `nil` | EMA decay for baseline |
| `:callback` | `nil` | `(fn [{:iter :loss :params}])` called each step |

**Returns:** `{:params MLX-array [d], :loss-history [numbers]}`

```clojure
;; Compiled ADEV: same API, tighter memory management
(adev/compiled-adev-optimize
  {:iterations 500 :lr 0.01 :n-samples 100}
  model [] cost-fn
  [:mu] (mx/array [0.0]))
```

---

## Choosing Between VI and ADEV

| | VI (`vi.cljs`) | ADEV (`adev.cljs`) |
|---|---|---|
| **Input** | Separate log-density, guide, sampler | Gen function + cost function |
| **Guide** | You write it (or use mean-field Gaussian) | The model *is* the guide |
| **Estimator selection** | Global (all reparam or all REINFORCE) | Per-site (automatic) |
| **Best for** | Posterior approximation with explicit guide families | Learning model parameters, training proposals |
| **Vectorized** | Manual (`vectorized-log-density`) | Automatic (`vadev-*` path) |
| **Compiled** | `compiled-vi`, `compiled-programmable-vi` | `compiled-adev-optimize` |

Use **VI** when you want to approximate a posterior with a parameterized guide family. Use **ADEV** when you want to optimize expected costs through a generative model's parameters (e.g., learning neural network weights, training amortized proposals, or optimizing model hyperparameters).

---

## Training Loop Example

A complete ADEV training loop with loss monitoring:

```clojure
(require '[genmlx.inference.adev :as adev]
         '[genmlx.gen :refer [gen]]
         '[genmlx.dist :as dist]
         '[genmlx.choicemap :as cm]
         '[genmlx.mlx :as mx])

;; Generative model with two learnable parameters
(def model
  (gen [xs]
    (let [w (param :w 0.0)
          b (param :b 0.0)]
      (doseq [[i x] (map-indexed vector xs)]
        (trace (keyword (str "y" i))
               (dist/gaussian (mx/add (mx/multiply w x) b) 0.5))))))

;; Target: y = 3x + 1
(def xs [(mx/scalar 1) (mx/scalar 2) (mx/scalar 3)])

;; Vectorized cost: negative log-prob of targets under model
(defn cost [result]
  (mx/negative (:score result)))

;; Train with ADEV
(let [result (adev/adev-optimize
               {:iterations 300
                :lr 0.02
                :n-samples 64
                :callback (fn [{:keys [iter loss]}]
                            (when (zero? (mod iter 50))
                              (println (str "iter " iter " loss: " loss))))}
               model xs cost
               [:w :b] (mx/array [0.0 0.0]))]
  (println "Learned w:" (mx/item (mx/index (:params result) 0)))  ;; ~3.0
  (println "Learned b:" (mx/item (mx/index (:params result) 1)))) ;; ~1.0
```
