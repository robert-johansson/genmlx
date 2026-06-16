# Neural Networks

Build neural network layers as functional data and train them with pure-function optimizers over MLX autograd. The `nn` namespace provides constructors for common layers, activation functions, a bridge from a layer to the GFI, and training utilities.

Each layer is a plain ClojureScript map of the form `{:params {...} :forward (fn [x] ...) :type ...}` -- not an MLX `nn.*` module. Parameters are flat maps with deterministic key order, directly usable with MLX autograd.

A `NeuralNetGF` is a deterministic generative function -- it makes no random choices, always scores zero, and its return value is the layer's forward pass. This makes neural networks composable with probabilistic models via `splice`.

```clojure
(require '[genmlx.nn :as nn])
```

Source: `src/genmlx/nn.cljs`

---

## Layers

### `linear`

```clojure
(nn/linear in-dims out-dims & {:keys [bias]})
```

Create a Linear layer that computes `y = x @ W^T + b`. Weights use Kaiming uniform initialization (`uniform(-k, k)` where `k = 1/sqrt(in-dims)`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `in-dims` | integer | Number of input features |
| `out-dims` | integer | Number of output features |
| `bias` | boolean | Whether to include a bias term (default: `true`) |

**Returns:** layer map `{:params {:weight .. :bias ..} :forward (fn [x] ..) :type :linear :rebuild ..}`.

---

### `sequential`

```clojure
(nn/sequential layers)
```

Compose a vector of layers. Layers are applied in order during the forward pass. Parameters are flattened with index-prefixed keys (`:0/weight`, `:0/bias`, `:1/weight`, ...).

| Parameter | Type | Description |
|-----------|------|-------------|
| `layers` | vector | Ordered vector of layer maps |

**Returns:** layer map `{:params {...} :forward (fn [x] ..) :type :sequential :layers [..] :rebuild ..}`.

**Example:**
```clojure
(def net (nn/sequential [(nn/linear 4 16) (nn/relu) (nn/linear 16 1)]))
```

---

### `layer-norm`

```clojure
(nn/layer-norm dims)
```

Create a layer-normalization layer: `y = (x - mean) / sqrt(var + eps) * gamma + beta`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dims` | integer | Normalized dimension size |

**Returns:** layer map with `:params {:gamma .. :beta ..}` and `:type :layer-norm`.

---

### `embedding`

```clojure
(nn/embedding num-embeddings dims)
```

Create an embedding layer that maps integer indices to dense vectors (`indices -> weight[indices]`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `num-embeddings` | integer | Size of the vocabulary |
| `dims` | integer | Embedding dimension |

**Returns:** layer map with `:params {:weight ..}` and `:type :embedding`.

---

### `dropout`

```clojure
(nn/dropout p)
```

Create a no-op dropout layer. Its forward pass is the identity function -- the `p` argument is ignored. Stochasticity during forward passes is handled by the generative function framework, not by dropout masks.

| Parameter | Type | Description |
|-----------|------|-------------|
| `p` | float | Dropout probability (ignored) |

**Returns:** layer map `{:params {} :forward identity :type :dropout}`.

---

## Activations

### `relu`

```clojure
(nn/relu)
```

Create a ReLU activation layer (`max(x, 0)`).

**Returns:** layer map `{:params {} :forward .. :type :relu}`.

---

### `gelu`

```clojure
(nn/gelu)
```

Create a GELU activation layer (tanh approximation).

**Returns:** layer map `{:params {} :forward .. :type :gelu}`.

---

### `tanh-act`

```clojure
(nn/tanh-act)
```

Create a tanh activation layer.

**Returns:** layer map `{:params {} :forward mx/tanh :type :tanh}`.

---

### `sigmoid-act`

```clojure
(nn/sigmoid-act)
```

Create a sigmoid activation layer.

**Returns:** layer map `{:params {} :forward mx/sigmoid :type :sigmoid}`.

---

## nn->gen-fn

### `nn->gen-fn`

```clojure
(nn/nn->gen-fn layer-or-atom)
```

Wrap a layer (a layer map or an atom holding one) as a deterministic generative function. The layer's `:forward` closure becomes the generative function body. The resulting `NeuralNetGF` implements the full GFI: `simulate`, `generate`, `update`, `regenerate`, `assess`, `propose`, and `project`.

Because the function is deterministic, its choicemap is always empty and all weights/scores are zero.

| Parameter | Type | Description |
|-----------|------|-------------|
| `layer-or-atom` | map or atom | A layer map, or an atom containing one |

**Returns:** `NeuralNetGF` -- a generative function wrapping the layer.

**Example:**
```clojure
(def net (nn/sequential [(nn/linear 4 16) (nn/relu) (nn/linear 16 1)]))
(def gen-net (nn/nn->gen-fn net))

;; Use as a generative function
(def trace (p/simulate gen-net [(mx/array [1.0 2.0 3.0 4.0])]))
(:retval trace) ;; => forward pass output

;; Compose with probabilistic models via splice
(def model
  (gen [x]
    (let [pred (splice :nn gen-net [x])
          sigma (trace :sigma (dist/gamma-dist 1 1))]
      (trace :y (dist/gaussian pred sigma)))))
```

---

## Training Utilities

### `value-and-grad`

```clojure
(nn/value-and-grad layer-ref loss-fn)
```

Create a function that computes both a loss value and gradients with respect to the layer's parameters. Bridges to MLX autograd via `mx/value-and-grad`.

The `loss-fn` receives the layer's rebuilt `forward` function as its first argument (so that autograd traces through the tracked parameter arrays), followed by the inputs.

| Parameter | Type | Description |
|-----------|------|-------------|
| `layer-ref` | atom | An atom containing the layer map whose parameters are differentiated |
| `loss-fn` | function | `(fn [forward-fn & inputs] -> MLX scalar loss)` |

**Returns:** `function` -- `(fn [& inputs] -> [loss param-grads])` where `loss` is an MLX scalar and `param-grads` is a flat map from parameter key to gradient array (same keys as `(:params layer)`).

**Example:**
```clojure
(def net (atom (nn/sequential [(nn/linear 4 1)])))
(def loss-and-grad
  (nn/value-and-grad net
    (fn [forward-fn x y]
      (let [pred (forward-fn x)]
        (mx/mean (mx/square (mx/subtract pred y)))))))

(let [[loss grads] (loss-and-grad x-batch y-batch)]
  (println "Loss:" (mx/item loss)))
```

---

### `optimizer`

```clojure
(nn/optimizer type lr)
```

Create a pure-function optimizer step. The returned step function takes the current flat parameter map and the gradient map and returns a new parameter map; `:adam`/`:adamw` close over their own moment state in an internal atom.

```clojure
(nn/optimizer type lr & {:as opts})
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | keyword | Optimizer type: `:sgd`, `:adam`, or `:adamw` |
| `lr` | float | Learning rate |
| `opts` | map | Optional hyperparameters (e.g. `:beta1` `:beta2` `:eps` for Adam, `:weight-decay` for AdamW) |

**Returns:** `function` -- a step function `(fn [params grads] -> new-params)`.

**Example:**
```clojure
(def opt (nn/optimizer :adam 0.001))
(def new-params (opt (:params @net) grads))
```
