# Your First Model

A probabilistic model is a program that makes random choices. GenMLX lets you write these programs as ordinary ClojureScript functions — with one addition: you **name** each random choice by giving it an address. That's all the `gen` macro adds to a regular function.

## The `gen` macro

```clojure
(def simple-model
  (gen []
    (trace :x (dist/gaussian 0 1))))
```

This defines a generative function called `simple-model`. Inside the body:

- `trace` declares a random choice. It says: "I want a value from this distribution at this address."
- `:x` is the **address** — a keyword that names this particular choice.
- `(dist/gaussian 0 1)` is the **distribution** — a Gaussian with mean 0 and standard deviation 1.

The model doesn't sample, doesn't constrain, doesn't know what will happen to the value. It just declares what it needs. The framework decides the rest.

## Running a model: `simulate`

To run the model forward — sampling every random choice from its distribution — use `simulate`:

```clojure
(let [model (dyn/auto-key simple-model)
      trace (p/simulate model [])]
  (println "value:" (mx/item (cm/get-choice (:choices trace) [:x])))
  (println "score:" (mx/item (:score trace))))
```

`simulate` returns a **trace** — a record of one complete execution. A trace has five fields:

| Field | What it holds |
|-------|---------------|
| `:gen-fn` | The generative function that produced this trace |
| `:args` | The arguments passed to the function |
| `:choices` | A **choice map** — a map from addresses to the values chosen |
| `:retval` | The return value of the function |
| `:score` | The log-probability of all choices: \\(\log p(\text{all choices})\\) |

The score is the sum of log-probabilities of every random choice made during execution. For a single Gaussian choice, it's the log-density of the sampled value under \\(\mathcal{N}(0, 1)\\).

## Inspecting choices

The choice map is a persistent Clojure data structure. You can read individual values:

```clojure
(let [model (dyn/auto-key simple-model)
      trace (p/simulate model [])
      choices (:choices trace)]
  ;; Check if an address exists
  (println "has :x?" (cm/has-value? (cm/get-submap choices :x)))
  ;; Get the value at an address
  (println "x =" (mx/item (cm/get-choice choices [:x])))
  ;; The score
  (println "score =" (mx/item (:score trace))))
```

Values in GenMLX are **MLX arrays** — GPU-resident tensors. Use `mx/item` to extract a plain JavaScript number when you need to print or compare.

## A coin flip model

Let's build something with more structure. A classic Bayesian model: we have a coin with unknown bias, and we flip it three times.

```clojure
(def coin-model
  (gen []
    (let [bias (trace :bias (dist/beta-dist 2 2))]
      (trace :flip1 (dist/bernoulli bias))
      (trace :flip2 (dist/bernoulli bias))
      (trace :flip3 (dist/bernoulli bias))
      bias)))
```

Here:

- `:bias` is sampled from a Beta(2, 2) distribution — a prior that favors values near 0.5 but allows anything between 0 and 1.
- Each flip is a Bernoulli trial with probability equal to the bias.
- The function returns the bias value.

Each `trace` call names its choice. The model has four addresses: `:bias`, `:flip1`, `:flip2`, `:flip3`. Simulate it:

```clojure
(let [model (dyn/auto-key coin-model)
      trace (p/simulate model [])]
  (println "bias:" (mx/item (cm/get-choice (:choices trace) [:bias])))
  (println "flip1:" (mx/item (cm/get-choice (:choices trace) [:flip1])))
  (println "flip2:" (mx/item (cm/get-choice (:choices trace) [:flip2])))
  (println "flip3:" (mx/item (cm/get-choice (:choices trace) [:flip3])))
  (println "score:" (mx/item (:score trace))))
```

The score is the sum: \\(\log p(\text{bias}) + \log p(\text{flip1} \mid \text{bias}) + \log p(\text{flip2} \mid \text{bias}) + \log p(\text{flip3} \mid \text{bias})\\).

## The running example: linear regression

This model will grow throughout the tutorial. We'll keep coming back to it, adding conditioning, inference, composition, and compilation.

A Bayesian linear regression: unknown slope and intercept, with noisy observations at known x-coordinates.

```clojure
(def linear-model
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept) 1)))
      slope)))
```

This model:

1. Samples a slope from \\(\mathcal{N}(0, 10)\\) — a broad prior.
2. Samples an intercept from \\(\mathcal{N}(0, 10)\\).
3. For each x-coordinate, samples an observation \\(y_j \sim \mathcal{N}(\text{slope} \cdot x_j + \text{intercept}, 1)\\).
4. Returns the slope.

Notice the addresses: `:slope` and `:intercept` are fixed keywords, but `:y0`, `:y1`, `:y2` are computed dynamically with `(keyword (str "y" j))`. GenMLX handles both.

```clojure
(let [model (dyn/auto-key linear-model)
      xs [1.0 2.0 3.0]
      trace (p/simulate model [xs])]
  (println "slope:" (mx/item (cm/get-choice (:choices trace) [:slope])))
  (println "intercept:" (mx/item (cm/get-choice (:choices trace) [:intercept])))
  (println "y0:" (mx/item (cm/get-choice (:choices trace) [:y0])))
  (println "y1:" (mx/item (cm/get-choice (:choices trace) [:y1])))
  (println "y2:" (mx/item (cm/get-choice (:choices trace) [:y2])))
  (println "score:" (mx/item (:score trace))))
```

## Every simulation is different

Each call to `simulate` draws fresh random values:

```clojure
(let [model (dyn/auto-key linear-model)
      xs [1.0 2.0 3.0]]
  (dotimes [i 5]
    (let [trace (p/simulate model [xs])
          slope (mx/item (cm/get-choice (:choices trace) [:slope]))]
      (println (str "run " i ": slope=" (.toFixed slope 3))))))
```

You'll see five different slopes. The model defines a probability distribution over possible worlds — each simulation is one draw from that distribution.

## What the score means

The score of a trace is \\(\log p(\boldsymbol{\tau}; x)\\) — the log of the joint probability of all random choices, given the arguments. For our simple Gaussian model, we can verify this manually:

```clojure
(let [model (dyn/auto-key simple-model)
      trace (p/simulate model [])
      x-val (cm/get-choice (:choices trace) [:x])
      ;; Manually compute log p(x) under N(0,1)
      manual-lp (mx/item (dc/dist-log-prob (dist/gaussian 0 1) x-val))
      trace-score (mx/item (:score trace))]
  (println "manual log-prob:" (.toFixed manual-lp 6))
  (println "trace score:    " (.toFixed trace-score 6))
  (println "match?" (< (js/Math.abs (- manual-lp trace-score)) 0.0001)))
```

They match. The score is the sum of log-probabilities across all trace sites.

## MLX arrays

Values in GenMLX live as MLX arrays — lazy GPU tensors. A few operations you'll use often:

```clojure
;; Create scalars and arrays
(def a (mx/scalar 3.0))
(def b (mx/scalar 2.0))
(def arr (mx/array [1.0 2.0 3.0]))

;; Extract to JavaScript
(mx/item a)        ;; => 3.0
(mx/shape arr)     ;; => [3]

;; Arithmetic (returns new MLX arrays)
(mx/add a b)       ;; => 5.0
(mx/multiply a b)  ;; => 6.0
(mx/subtract a b)  ;; => 1.0
(mx/divide a b)    ;; => 1.5
```

MLX operations are **lazy** — they build a computation graph that executes on the GPU when you read a value with `mx/item` or call `mx/eval!`. Inside model bodies, you never need to call `eval!` — the framework handles materialization at inference boundaries.

## Distribution catalog

GenMLX includes 27 built-in distributions. Here are the most common ones:

**Continuous:**

| Distribution | Constructor | Parameters |
|---|---|---|
| Gaussian (Normal) | `(dist/gaussian mean std)` | mean, standard deviation |
| Uniform | `(dist/uniform low high)` | lower bound, upper bound |
| Beta | `(dist/beta-dist alpha beta)` | shape parameters |
| Gamma | `(dist/gamma-dist shape rate)` | shape, rate |
| Exponential | `(dist/exponential rate)` | rate |
| Laplace | `(dist/laplace location scale)` | location, scale |
| Log-Normal | `(dist/log-normal log-mean log-std)` | log-mean, log-std |
| Student-t | `(dist/student-t df loc scale)` | degrees of freedom, location, scale |
| Cauchy | `(dist/cauchy location scale)` | location, scale |

**Discrete:**

| Distribution | Constructor | Parameters |
|---|---|---|
| Bernoulli | `(dist/bernoulli prob)` | success probability |
| Categorical | `(dist/categorical log-weights)` | log-probabilities (unnormalized) |
| Poisson | `(dist/poisson rate)` | rate |
| Geometric | `(dist/geometric prob)` | success probability |
| Binomial | `(dist/binomial n prob)` | trials, probability |
| Discrete Uniform | `(dist/discrete-uniform low high)` | range |

**Multivariate / Special:**

| Distribution | Constructor | Parameters |
|---|---|---|
| Multivariate Normal | `(dist/multivariate-normal mean cov)` | mean vector, covariance matrix |
| Dirichlet | `(dist/dirichlet concentration)` | concentration vector |
| Delta | `(dist/delta value)` | fixed value (score = 0) |

You can also define your own distributions — we'll cover that in [Chapter 10](./ch10-extensions.md).

## What we've learned

- The `gen` macro turns a function into a generative function.
- `trace` declares a random choice at a named address — it says *what* you need, not *how* to get it.
- `simulate` runs the model forward, producing a trace with choices, a return value, and a score.
- The score is \\(\log p(\text{all choices})\\) — the log joint probability.
- Values are MLX arrays (GPU tensors). Use `mx/item` to extract JavaScript numbers.

In the next chapter, we'll condition the model on observed data — and see how the same model code, with no modifications, gives us importance weights for inference.
