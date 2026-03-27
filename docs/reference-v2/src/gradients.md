# Gradients

Per-choice and score gradients from generative function traces. These functions differentiate the log-probability of a trace with respect to individual continuous choices or a flat parameter array. They form the foundation for gradient-based inference algorithms (HMC, MALA), parameter learning, and variational inference.

Both functions use MLX's automatic differentiation through the `generate` interface -- constructing a differentiable score function, then calling `mx/grad` or `mx/value-and-grad` to obtain gradients.

```clojure
(require '[genmlx.gradients :as grad])
```

Source: `src/genmlx/gradients.cljs`

---

## Choice Gradients

### `choice-gradients`

```clojure
(grad/choice-gradients model trace addresses)
```

Compute gradients of the model's log-probability with respect to specified choices. Given a trace and a vector of addresses, returns the partial derivative of the total log-score with respect to each choice value.

The gradient at address `a` is: `d(log p(all choices | args)) / d(choice at a)`.

Internally, the current choice values are packed into a flat MLX array, a score function is constructed via `generate` (constraining all choices), and `mx/grad` differentiates through it.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | generative function | The model that produced the trace |
| `trace` | Trace | A trace from `simulate` or `generate` |
| `addresses` | vector | Vector of choice addresses (keywords) to differentiate |

**Returns:** Map of `{address -> MLX-gradient-array}` -- one gradient per requested address.

**Throws:** If any address in `addresses` is not found in the trace's choices.

**Example:**
```clojure
(require '[genmlx.protocols :as p])
(require '[genmlx.dist :as dist])

(def model
  (gen []
    (let [x (trace :x (dist/gaussian 0 1))
          y (trace :y (dist/gaussian x 1))]
      y)))

;; Simulate to get a trace
(def tr (p/simulate model []))

;; Compute gradients of log p(x, y) w.r.t. both choices
(def grads (grad/choice-gradients model tr [:x :y]))

(get grads :x)
;; => #mlx <gradient of log p(x,y) w.r.t. x>

(get grads :y)
;; => #mlx <gradient of log p(x,y) w.r.t. y>
```

**Usage in inference:** Choice gradients drive gradient-based MCMC methods. HMC and MALA use them to compute the gradient of the log-posterior at each leapfrog step.

```clojure
;; HMC-style gradient computation
(let [grads (grad/choice-gradients model trace latent-addresses)
      grad-vec (mx/array (mapv #(mx/item (get grads %)) latent-addresses))]
  ;; Use grad-vec for leapfrog integration
  ...)
```

---

## Score Gradients

### `score-gradient`

```clojure
(grad/score-gradient model args observations addresses params)
```

Compute the gradient of the model score with respect to a flat parameter array. Unlike `choice-gradients` (which differentiates an existing trace), this function takes a parameter array and observation constraints, builds a score function, and returns both the score value and its gradient.

The score function is: `log p(observations, params | args)` where `params` are placed at `addresses` and `observations` are fixed constraints.

Uses `mx/value-and-grad` for a single forward+backward pass.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | generative function | The model to score |
| `args` | vector | Model arguments |
| `observations` | ChoiceMap | Fixed observed values |
| `addresses` | vector | Vector of addresses for the parameters |
| `params` | MLX array | 1-D array of shape `[N]` where `N = (count addresses)` |

**Returns:** `{:score MLX-scalar :grad MLX-array}` -- the log-score and its gradient with respect to `params`.

**Throws:** If `(count addresses)` does not match the first dimension of `params`.

**Example:**
```clojure
(require '[genmlx.choicemap :as cm])

(def model
  (gen [x]
    (let [mu (trace :mu (dist/gaussian 0 10))
          sigma (trace :sigma (dist/half-normal 1))]
      (trace :y (dist/gaussian mu sigma)))))

(def obs (cm/choicemap :y (mx/scalar 3.0)))

;; Score and gradient for specific parameter values
(def result
  (grad/score-gradient model [0] obs
    [:mu :sigma]
    (mx/array [2.5 1.0])))

(mx/item (:score result))
;; => log p(y=3, mu=2.5, sigma=1 | x=0)

(:grad result)
;; => #mlx [d(score)/d(mu), d(score)/d(sigma)]
```

**Usage in optimization:** `score-gradient` is the building block for parameter optimization. Maximizing the score by following its gradient is equivalent to MAP estimation.

```clojure
;; Simple gradient ascent for MAP
(loop [params (mx/array [0.0 1.0])
       i 0]
  (when (< i 100)
    (let [{:keys [grad]} (grad/score-gradient model args obs [:mu :sigma] params)
          params' (mx/add params (mx/multiply (mx/scalar 0.01) grad))]
      (recur params' (inc i)))))
```
