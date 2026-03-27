# Learning

Parameter stores, optimizers, and training loops for learning model parameters through gradient-based optimization. This module provides the building blocks for maximum likelihood estimation, MAP inference, and amortized inference via wake-sleep learning.

The typical workflow: create a parameter store, build a loss-gradient function from a model and observations, then run a training loop with SGD or Adam. For amortized inference, wake-sleep alternates between fitting a recognition model (guide) to the posterior and fitting it to the prior.

```clojure
(require '[genmlx.learning :as learn])
```

Source: `src/genmlx/learning.cljs`

---

## Parameter Store

A parameter store is a purely functional map holding named MLX arrays and a version counter. Every mutation returns a new store -- no mutable state.

### `make-param-store`

```clojure
(learn/make-param-store)
(learn/make-param-store init-params)
```

Create a functional parameter store. With no arguments, returns an empty store. With a map of `{name -> value}`, initializes all parameters (converting plain numbers to MLX scalars).

| Parameter | Type | Description |
|-----------|------|-------------|
| `init-params` | map (optional) | Map of `{keyword -> MLX-array-or-number}` |

**Returns:** `{:params {keyword -> MLX-array} :version 0}`

**Example:**
```clojure
;; Empty store
(learn/make-param-store)
;; => {:params {} :version 0}

;; Pre-initialized store
(learn/make-param-store {:slope (mx/scalar 0.5) :intercept 1.0})
;; => {:params {:slope #mlx 0.5 :intercept #mlx 1.0} :version 0}
```

---

### `get-param`

```clojure
(learn/get-param store name)
```

Get a parameter value from the store by name.

| Parameter | Type | Description |
|-----------|------|-------------|
| `store` | map | Parameter store |
| `name` | keyword | Parameter name |

**Returns:** MLX array, or `nil` if not found.

**Example:**
```clojure
(def store (learn/make-param-store {:lr (mx/scalar 0.01)}))
(learn/get-param store :lr)
;; => #mlx 0.01
```

---

### `set-param`

```clojure
(learn/set-param store name value)
```

Set a parameter value in the store. Plain numbers are converted to MLX scalars. Increments the version counter.

| Parameter | Type | Description |
|-----------|------|-------------|
| `store` | map | Parameter store |
| `name` | keyword | Parameter name |
| `value` | MLX array or number | New value |

**Returns:** Updated parameter store with incremented `:version`.

**Example:**
```clojure
(def store (learn/make-param-store {:x (mx/scalar 1.0)}))
(def store' (learn/set-param store :x (mx/scalar 2.0)))
(:version store')
;; => 1
```

---

### `update-params`

```clojure
(learn/update-params store updates)
```

Apply a map of updates to the store. Each entry in `updates` calls `set-param`, so the version increments once per parameter updated.

| Parameter | Type | Description |
|-----------|------|-------------|
| `store` | map | Parameter store |
| `updates` | map | Map of `{keyword -> new-value}` |

**Returns:** Updated parameter store.

**Example:**
```clojure
(def store (learn/make-param-store {:a (mx/scalar 1.0) :b (mx/scalar 2.0)}))
(learn/update-params store {:a (mx/scalar 10.0) :b (mx/scalar 20.0)})
```

---

### `param-names`

```clojure
(learn/param-names store)
```

List all parameter names in the store.

| Parameter | Type | Description |
|-----------|------|-------------|
| `store` | map | Parameter store |

**Returns:** Sequence of keywords.

**Example:**
```clojure
(def store (learn/make-param-store {:slope (mx/scalar 0.5) :intercept (mx/scalar 1.0)}))
(learn/param-names store)
;; => (:slope :intercept)
```

---

### `params->array`

```clojure
(learn/params->array store names)
```

Flatten named parameters into a single 1-D MLX array. The order is determined by `names`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `store` | map | Parameter store |
| `names` | vector | Ordered vector of parameter names |

**Returns:** MLX array of shape `[N]` where `N = (count names)`.

**Example:**
```clojure
(def store (learn/make-param-store {:a (mx/scalar 1.0) :b (mx/scalar 2.0)}))
(learn/params->array store [:a :b])
;; => #mlx [1.0, 2.0]
```

---

### `array->params`

```clojure
(learn/array->params arr names)
```

Unflatten a 1-D MLX array back into a map of named parameters. The inverse of `params->array`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `arr` | MLX array | 1-D array of shape `[N]` |
| `names` | vector | Ordered vector of parameter names (length `N`) |

**Returns:** Map of `{keyword -> MLX-scalar}`.

**Example:**
```clojure
(learn/array->params (mx/array [1.0 2.0]) [:a :b])
;; => {:a #mlx 1.0 :b #mlx 2.0}
```

---

## Optimizers

### `sgd-step`

```clojure
(learn/sgd-step params grad lr)
```

One step of stochastic gradient descent: `params - lr * grad`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `params` | MLX array | Current parameter array |
| `grad` | MLX array | Gradient array (same shape as `params`) |
| `lr` | number | Learning rate |

**Returns:** MLX array -- updated parameters.

**Example:**
```clojure
(def p (mx/array [1.0 2.0]))
(def g (mx/array [0.1 -0.2]))
(learn/sgd-step p g 0.01)
;; => #mlx [0.999, 2.002]
```

---

### `adam-init`

```clojure
(learn/adam-init params)
```

Initialize Adam optimizer state. Creates zero-filled first-moment (`m`) and second-moment (`v`) accumulators matching the shape of `params`, and sets the step counter `t` to 0.

| Parameter | Type | Description |
|-----------|------|-------------|
| `params` | MLX array | Parameter array (used only for shape) |

**Returns:** `{:m MLX-zeros :v MLX-zeros :t 0}`

**Example:**
```clojure
(def state (learn/adam-init (mx/array [1.0 2.0 3.0])))
;; state => {:m #mlx [0 0 0], :v #mlx [0 0 0], :t 0}
```

---

### `adam-step`

```clojure
(learn/adam-step params grad state opts)
```

One Adam optimizer step with bias-corrected first and second moment estimates. Calls `mx/materialize!` on the updated parameters and moment accumulators before returning.

| Parameter | Type | Description |
|-----------|------|-------------|
| `params` | MLX array | Current parameter array |
| `grad` | MLX array | Gradient array (same shape as `params`) |
| `state` | map | Adam state from `adam-init` or previous `adam-step` |
| `opts` | map | Hyperparameters (all optional) |

Options for `opts`:

| Key | Default | Description |
|-----|---------|-------------|
| `:lr` | `0.001` | Learning rate |
| `:beta1` | `0.9` | First moment decay |
| `:beta2` | `0.999` | Second moment decay |
| `:epsilon` | `1e-8` | Numerical stability constant |

**Returns:** `[new-params new-state]` -- a two-element vector.

**Example:**
```clojure
(def p (mx/array [1.0 2.0]))
(def g (mx/array [0.1 -0.2]))
(def state (learn/adam-init p))

(let [[p' state'] (learn/adam-step p g state {:lr 0.001})]
  ;; p' is updated parameters, state' has incremented :t
  (:t state'))
;; => 1
```

---

## Training

### `train`

```clojure
(learn/train opts loss-grad-fn init-params)
```

Generic training loop for parameter learning. Runs `iterations` steps of gradient-based optimization, choosing between SGD and Adam. Periodically clears the MLX computation cache to manage memory for large models.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | map | Training options (see below) |
| `loss-grad-fn` | function | `(fn [params key] -> {:loss MLX-scalar :grad MLX-array})` |
| `init-params` | MLX array | Initial parameter values |

Options for `opts`:

| Key | Default | Description |
|-----|---------|-------------|
| `:iterations` | `1000` | Number of training steps |
| `:optimizer` | `:adam` | Optimizer: `:sgd` or `:adam` |
| `:lr` | `0.001` | Learning rate |
| `:callback` | `nil` | `(fn [{:iter :loss :params}])` called each step |
| `:key` | `nil` | PRNG key for reproducibility |

**Returns:** `{:params MLX-array :loss-history [numbers...]}`

**Example:**
```clojure
;; Train a simple quadratic loss
(def result
  (learn/train
    {:iterations 500 :lr 0.01 :optimizer :adam}
    (fn [params _key]
      (let [loss (mx/sum (mx/square params))
            grad-fn (mx/grad (fn [p] (mx/sum (mx/square p))))
            grad (grad-fn params)]
        {:loss loss :grad grad}))
    (mx/array [5.0 3.0 -2.0])))

(mx/item (mx/index (:params result) 0))
;; => ~0.0 (converged to zero)
(count (:loss-history result))
;; => 500
```

---

### `make-param-loss-fn`

```clojure
(learn/make-param-loss-fn model args observations param-names-vec)
```

Create a loss-gradient function for training model parameters via maximum likelihood. The loss is the negative log-joint probability (minimize to maximize likelihood). Uses `mx/grad` to differentiate through the `generate` interface.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | generative function | Model using `param` declarations |
| `args` | vector | Model arguments |
| `observations` | ChoiceMap | Observed data |
| `param-names-vec` | vector | Ordered vector of parameter names (keywords) |

**Returns:** `(fn [params-array key] -> {:loss MLX-scalar :grad MLX-array})` -- suitable for passing to `train`.

**Example:**
```clojure
(def model
  (gen [xs]
    (let [slope (param :slope 0.0)
          intercept (param :intercept 0.0)]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept)
                              (mx/scalar 1.0))))
      slope)))

(def xs [1.0 2.0 3.0])
(def obs (cm/choicemap :y0 (mx/scalar 2.1) :y1 (mx/scalar 3.9) :y2 (mx/scalar 6.2)))

;; Create loss-gradient function
(def loss-grad (learn/make-param-loss-fn model [xs] obs [:slope :intercept]))

;; Train
(def result
  (learn/train
    {:iterations 200 :lr 0.01}
    loss-grad
    (mx/array [0.0 0.0])))

;; Recover learned parameters
(learn/array->params (:params result) [:slope :intercept])
;; => {:slope ~2.0 :intercept ~0.1}
```

---

## Parameterized Execution

### `simulate-with-params`

```clojure
(learn/simulate-with-params model args param-store)
```

Simulate a generative function with a parameter store bound. Parameters declared via `param` inside the model body read their values from this store instead of using defaults.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | generative function | Model with `param` declarations |
| `args` | vector | Model arguments |
| `param-store` | map | Parameter store from `make-param-store` |

**Returns:** `Trace` -- the simulation trace with parameter values from the store.

**Example:**
```clojure
(def store (learn/make-param-store {:slope (mx/scalar 2.0) :intercept (mx/scalar 0.5)}))
(def trace (learn/simulate-with-params model [xs] store))
(:retval trace)
;; => #mlx 2.0 (the slope parameter)
```

---

### `generate-with-params`

```clojure
(learn/generate-with-params model args constraints param-store)
```

Generate from a generative function with a parameter store bound. Combines parameter binding with constrained generation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | generative function | Model with `param` declarations |
| `args` | vector | Model arguments |
| `constraints` | ChoiceMap | Observed values to condition on |
| `param-store` | map | Parameter store from `make-param-store` |

**Returns:** `{:trace Trace :weight MLX-scalar}` -- the generation result.

**Example:**
```clojure
(def store (learn/make-param-store {:slope (mx/scalar 2.0) :intercept (mx/scalar 0.5)}))
(def result (learn/generate-with-params model [xs] obs store))
(mx/item (:weight result))
;; => log-probability of observations under these parameters
```

---

## Wake-Sleep

Wake-sleep learning trains a recognition model (guide) to approximate the posterior by alternating two phases. The wake phase minimizes KL(q||p) using samples from the guide. The sleep phase minimizes KL(p||q) using samples from the model prior.

### `wake-phase-loss`

```clojure
(learn/wake-phase-loss model guide args observations guide-addresses)
```

Build a wake phase loss-gradient function. The wake phase optimizes guide parameters to minimize KL(q||p) -- the negative ELBO. Samples from the guide, scores under the model, and differentiates the gap.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | generative function | Target model |
| `guide` | generative function | Recognition model (guide) |
| `args` | vector | Model arguments |
| `observations` | ChoiceMap | Observed data |
| `guide-addresses` | vector | Addresses in the guide to parameterize |

**Returns:** `(fn [guide-params key] -> {:loss MLX-scalar :grad MLX-array})` -- the wake phase loss function.

---

### `sleep-phase-loss`

```clojure
(learn/sleep-phase-loss model guide args guide-addresses)
```

Build a sleep phase loss-gradient function. The sleep phase samples from the model prior and trains the guide to reconstruct those samples, minimizing KL(p||q).

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | generative function | Target model |
| `guide` | generative function | Recognition model (guide) |
| `args` | vector | Model arguments |
| `guide-addresses` | vector | Addresses in the guide to parameterize |

**Returns:** `(fn [guide-params key] -> {:loss MLX-scalar :grad MLX-array})` -- the sleep phase loss function.

---

### `wake-sleep`

```clojure
(learn/wake-sleep opts model guide args observations guide-addresses init-guide-params)
```

Full wake-sleep learning loop for amortized inference. Alternates wake and sleep phases, using Adam optimization within each phase.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | map | Training options (see below) |
| `model` | generative function | Target model |
| `guide` | generative function | Recognition model |
| `args` | vector | Model arguments |
| `observations` | ChoiceMap | Observed data |
| `guide-addresses` | vector or nil | Guide addresses (nil to auto-discover) |
| `init-guide-params` | MLX array or nil | Initial guide params (nil for zero-init) |

Options for `opts`:

| Key | Default | Description |
|-----|---------|-------------|
| `:iterations` | `1000` | Number of wake-sleep cycles |
| `:wake-steps` | `1` | Wake phase steps per cycle |
| `:sleep-steps` | `1` | Sleep phase steps per cycle |
| `:lr` | `0.001` | Learning rate |
| `:callback` | `nil` | `(fn [{:iter :wake-loss :sleep-loss}])` called each cycle |
| `:key` | `nil` | PRNG key for reproducibility |

**Returns:** `{:params MLX-array :wake-losses [numbers...] :sleep-losses [numbers...]}`

**Example:**
```clojure
(def model
  (gen [x]
    (let [z (trace :z (dist/gaussian 0 1))]
      (trace :x (dist/gaussian z 0.5)))))

(def guide
  (gen [x]
    (trace :z (dist/gaussian 0 1))))

(def obs (cm/choicemap :x (mx/scalar 2.0)))

(def result
  (learn/wake-sleep
    {:iterations 500 :lr 0.001 :key (rng/key 42)}
    model guide [2.0] obs
    nil    ;; auto-discover guide addresses
    nil))  ;; zero-init guide params

(count (:wake-losses result))
;; => 500
```
