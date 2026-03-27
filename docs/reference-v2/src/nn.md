# Neural Networks

Wrap MLX neural network modules as generative functions and train them with native MLX optimizers. The `nn` namespace provides constructors for common layers, activation functions, a bridge from `nn.Module` to the GFI, and training utilities.

A `NeuralNetGF` is a deterministic generative function -- it makes no random choices, always scores zero, and its return value is the module's forward pass. This makes neural networks composable with probabilistic models via `splice`.

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

Create an `nn.Linear` layer that computes `y = x @ W^T + b`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `in-dims` | integer | Number of input features |
| `out-dims` | integer | Number of output features |
| `bias` | boolean | Whether to include a bias term (default: `true`) |

**Returns:** MLX `nn.Linear` module.

---

### `sequential`

```clojure
(nn/sequential layers)
```

Create an `nn.Sequential` module from a vector of layers. Layers are applied in order during the forward pass.

| Parameter | Type | Description |
|-----------|------|-------------|
| `layers` | vector | Ordered vector of MLX `nn.Module` layers |

**Returns:** MLX `nn.Sequential` module.

**Example:**
```clojure
(def net (nn/sequential [(nn/linear 4 16) (nn/relu) (nn/linear 16 1)]))
```

---

### `layer-norm`

```clojure
(nn/layer-norm dims)
```

Create an `nn.LayerNorm` layer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dims` | integer | Normalized dimension size |

**Returns:** MLX `nn.LayerNorm` module.

---

### `embedding`

```clojure
(nn/embedding num-embeddings dims)
```

Create an `nn.Embedding` layer that maps integer indices to dense vectors.

| Parameter | Type | Description |
|-----------|------|-------------|
| `num-embeddings` | integer | Size of the vocabulary |
| `dims` | integer | Embedding dimension |

**Returns:** MLX `nn.Embedding` module.

---

### `dropout`

```clojure
(nn/dropout p)
```

Create an `nn.Dropout` layer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `p` | float | Dropout probability |

**Returns:** MLX `nn.Dropout` module.

---

## Activations

### `relu`

```clojure
(nn/relu)
```

Create an `nn.ReLU` activation module.

**Returns:** MLX `nn.ReLU` module.

---

### `gelu`

```clojure
(nn/gelu)
```

Create an `nn.GELU` activation module.

**Returns:** MLX `nn.GELU` module.

---

### `tanh-act`

```clojure
(nn/tanh-act)
```

Create an `nn.Tanh` activation module.

**Returns:** MLX `nn.Tanh` module.

---

### `sigmoid-act`

```clojure
(nn/sigmoid-act)
```

Create an `nn.Sigmoid` activation module.

**Returns:** MLX `nn.Sigmoid` module.

---

## nn->gen-fn

### `nn->gen-fn`

```clojure
(nn/nn->gen-fn module)
```

Wrap an MLX `nn.Module` as a deterministic generative function. The module's `.forward` method becomes the generative function body. The resulting `NeuralNetGF` implements the full GFI: `simulate`, `generate`, `update`, `regenerate`, `assess`, `propose`, and `project`.

Because the function is deterministic, its choicemap is always empty and all weights/scores are zero.

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | nn.Module | An MLX neural network module |

**Returns:** `NeuralNetGF` -- a generative function wrapping the module.

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
          sigma (trace :sigma (dist/gamma 1 1))]
      (trace :y (dist/gaussian pred sigma)))))
```

---

## Training Utilities

### `value-and-grad`

```clojure
(nn/value-and-grad module loss-fn)
```

Create a function that computes both a loss value and gradients with respect to the module's parameters. Uses MLX's native `nn.valueAndGrad`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | nn.Module | The module whose parameters will be differentiated |
| `loss-fn` | function | `(fn [& inputs] -> MLX scalar loss)` |

**Returns:** `function` -- `(fn [& inputs] -> [loss, grad-tree])` where `loss` is an MLX scalar and `grad-tree` is a tree of parameter gradients.

**Example:**
```clojure
(def net (nn/sequential [(nn/linear 4 1)]))
(def loss-and-grad
  (nn/value-and-grad net
    (fn [x y]
      (let [pred (.forward net x)]
        (mx/mean (mx/square (mx/subtract pred y)))))))

(let [[loss grads] (loss-and-grad x-batch y-batch)]
  (println "Loss:" (mx/item loss)))
```

---

### `optimizer`

```clojure
(nn/optimizer type lr)
```

Create a native MLX optimizer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | keyword | Optimizer type: `:adam`, `:sgd`, or `:adamw` |
| `lr` | float | Learning rate |

**Returns:** MLX optimizer instance.

**Example:**
```clojure
(def opt (nn/optimizer :adam 0.001))
```
