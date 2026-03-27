# Importance Sampling & Resampling

Source: `src/genmlx/inference/importance.cljs`

Importance sampling is the simplest Monte Carlo inference method: draw weighted
samples from the model's prior conditioned on observations, then use the weights
to approximate the posterior. GenMLX provides scalar, memory-efficient, and
vectorized variants, plus importance resampling for producing unweighted samples.

**When to use importance sampling:**
- Quick posterior approximation when the prior is a reasonable proposal
- Estimating marginal likelihoods (model evidence)
- As a building block inside more advanced algorithms

**Namespace alias used below:**
```clojure
(require '[genmlx.inference.importance :as is])
```

---

## Importance Sampling

### `importance-sampling`

```clojure
(is/importance-sampling opts model args observations)
```

Draw `N` weighted samples from the model conditioned on observations. Each
sample is an independent `generate` call with the observation constraints.
Returns full traces, log-weights, and a log marginal likelihood estimate.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Options map (see below) |
| `model` | DynamicGF | Generative function to sample from |
| `args` | Vector | Arguments to the model |
| `observations` | ChoiceMap | Observed values to condition on |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | Integer | `100` | Number of importance samples to draw |
| `:key` | MLX array | fresh | PRNG key for reproducibility |

**Returns:** `{:traces [Trace ...] :log-weights [MLX-scalar ...] :log-ml-estimate MLX-scalar}`

- `:traces` -- vector of `N` Trace records, each containing choices, score, and return value
- `:log-weights` -- vector of `N` unnormalized log importance weights
- `:log-ml-estimate` -- log marginal likelihood estimate: `logsumexp(weights) - log(N)`

**Example:**

```clojure
(def model
  (gen [x]
    (let [slope (trace :slope (dist/gaussian 0 10))]
      (trace :y (dist/gaussian (mx/multiply slope x) 1))
      slope)))

(def obs (cm/choicemap :y (mx/scalar 2.5)))

(let [{:keys [traces log-weights log-ml-estimate]}
      (is/importance-sampling {:samples 1000} model [(mx/scalar 1.0)] obs)]
  (println "Particles:" (count traces))
  (println "Log-ML:" (mx/item log-ml-estimate)))
```

**Notes:**
- Automatically strips L3 analytical handlers to ensure particle diversity.
  Analytical handlers produce deterministic posterior means, which would make
  all particles identical.
- Materializes weights every 50 particles and sweeps dead arrays to manage
  Metal buffer pressure.

---

### `tidy-importance-sampling`

```clojure
(is/tidy-importance-sampling opts model args observations)
```

Memory-efficient importance sampling. Each particle runs inside `mx/tidy-run`,
so all trace arrays are disposed immediately after extracting the weight as a
JS number. Only returns log-weights and the log-ML estimate -- no traces.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Options map (see below) |
| `model` | DynamicGF | Generative function to sample from |
| `args` | Vector | Arguments to the model |
| `observations` | ChoiceMap | Observed values to condition on |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | Integer | `100` | Number of importance samples to draw |
| `:key` | MLX array | fresh | PRNG key for reproducibility |

**Returns:** `{:log-weights [JS-number ...] :log-ml-estimate JS-number}`

Note that both log-weights and the log-ML estimate are plain JS numbers (not
MLX arrays), since all GPU arrays are freed during execution.

**When to use instead of `importance-sampling`:**
- The model has many trace sites (large traces that consume Metal buffers)
- You need many particles (thousands+)
- You only need the log marginal likelihood, not the traces themselves

**Example:**

```clojure
(let [{:keys [log-ml-estimate]}
      (is/tidy-importance-sampling {:samples 5000} model args obs)]
  (println "Log-ML estimate:" log-ml-estimate))
```

---

## Importance Resampling

### `importance-resampling`

```clojure
(is/importance-resampling opts model args observations)
```

Importance sampling followed by multinomial resampling: draw `particles`
weighted samples, then resample `samples` unweighted traces proportional to
their importance weights. The result is a set of approximately independent
draws from the posterior.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Options map (see below) |
| `model` | DynamicGF | Generative function to sample from |
| `args` | Vector | Arguments to the model |
| `observations` | ChoiceMap | Observed values to condition on |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | Integer | `100` | Number of resampled traces to return |
| `:particles` | Integer | `1000` | Number of importance samples to draw before resampling |
| `:key` | MLX array | fresh | PRNG key for reproducibility |

**Returns:** Vector of `samples` resampled Trace records (unweighted).

**Example:**

```clojure
(let [traces (is/importance-resampling {:samples 100 :particles 2000}
                                        model args obs)]
  ;; traces is a vector of 100 approximately independent posterior samples
  (println "Posterior mean slope:"
    (/ (reduce + (map #(mx/item (cm/get-choice (:choices %) [:slope])) traces))
       (count traces))))
```

**Notes:**
- For GPU-accelerated resampling on compatible models (no splice, no
  data-dependent branching), use `vectorized-importance-resampling` instead.

---

## Vectorized IS

Vectorized importance sampling runs the model body **once** for all `N`
particles simultaneously using shape-based batching. Instead of `N` sequential
`generate` calls, each trace site samples an `[N]`-shaped array and all
downstream arithmetic broadcasts naturally. This gives 10-100x speedups for
models without splice or data-dependent branching.

### `vectorized-importance-sampling`

```clojure
(is/vectorized-importance-sampling opts model args observations)
```

Vectorized importance sampling. Runs the model body once with a batched
handler for all `N` particles.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Options map (see below) |
| `model` | DynamicGF | Generative function (must support vectorized execution) |
| `args` | Vector | Arguments to the model |
| `observations` | ChoiceMap | Observed values to condition on |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | Integer | `100` | Number of particles |
| `:key` | MLX array | fresh | PRNG key for reproducibility |

**Returns:** `{:vtrace VectorizedTrace :log-ml-estimate MLX-scalar}`

- `:vtrace` -- a `VectorizedTrace` with `[N]`-shaped arrays at each choice site
- `:log-ml-estimate` -- log marginal likelihood estimate

**Example:**

```clojure
(let [{:keys [vtrace log-ml-estimate]}
      (is/vectorized-importance-sampling {:samples 5000} model args obs)]
  (println "ESS:" (vec/vtrace-ess vtrace))
  (println "Log-ML:" (mx/item log-ml-estimate)))
```

**Limitations:** No `splice` calls in the model. No data-dependent branching
(all particles must follow the same control flow).

---

### `vectorized-importance-resampling`

```clojure
(is/vectorized-importance-resampling opts model args observations)
```

Vectorized importance sampling followed by GPU-accelerated systematic
resampling. Runs the model once for all particles, then resamples on the GPU.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Options map (see below) |
| `model` | DynamicGF | Generative function (must support vectorized execution) |
| `args` | Vector | Arguments to the model |
| `observations` | ChoiceMap | Observed values to condition on |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:particles` | Integer | `1000` | Number of particles to draw and resample |
| `:key` | MLX array | fresh | PRNG key for reproducibility |

**Returns:** `{:vtrace VectorizedTrace :log-ml-estimate MLX-scalar}`

The returned `VectorizedTrace` has uniform weights after resampling.

---

## Utilities

### `normalize-log-weights`

```clojure
(u/normalize-log-weights log-weights)
```

Normalize a vector of MLX log-weight scalars via log-softmax.

| Parameter | Type | Description |
|-----------|------|-------------|
| `log-weights` | Vector of MLX scalars | Unnormalized log importance weights |

**Returns:** `{:log-probs MLX-array :probs [JS-number ...]}`

---

### `compute-ess`

```clojure
(u/compute-ess log-weights)
```

Compute the effective sample size from log-weights. ESS measures how many
independent samples the weighted particle set is equivalent to. ESS = N means
all weights are equal (ideal); ESS = 1 means one particle dominates.

| Parameter | Type | Description |
|-----------|------|-------------|
| `log-weights` | Vector of MLX scalars | Unnormalized log importance weights |

**Returns:** JS number (the effective sample size).

---

### `materialize-weights`

```clojure
(u/materialize-weights log-weights)
```

Evaluate a vector of MLX log-weight scalars and stack them into a single
1-D MLX array. Uses `mx/stack` to combine in one MLX call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `log-weights` | Vector of MLX scalars | Log importance weights |

**Returns:** `[N]`-shaped MLX array.

---

### `systematic-resample`

```clojure
(u/systematic-resample log-weights n key)
```

Systematic resampling of particles. Draws `n` indices proportional to the
normalized weights using a single uniform offset, producing lower-variance
index sets than multinomial resampling.

| Parameter | Type | Description |
|-----------|------|-------------|
| `log-weights` | Vector of MLX scalars | Unnormalized log importance weights |
| `n` | Integer | Number of indices to return |
| `key` | MLX array or nil | PRNG key (uses fresh key if nil) |

**Returns:** Vector of `n` integer indices.

---

### `vtrace-log-ml-estimate`

```clojure
(vec/vtrace-log-ml-estimate vtrace)
```

Compute the log marginal likelihood estimate from a `VectorizedTrace`'s
`[N]`-shaped weight array: `logsumexp(weights) - log(N)`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `vtrace` | VectorizedTrace | Vectorized trace with `[N]`-shaped weights |

**Returns:** MLX scalar.

---

### `vtrace-ess`

```clojure
(vec/vtrace-ess vtrace)
```

Effective sample size from a `VectorizedTrace`'s `[N]`-shaped weights.

| Parameter | Type | Description |
|-----------|------|-------------|
| `vtrace` | VectorizedTrace | Vectorized trace with `[N]`-shaped weights |

**Returns:** JS number (the effective sample size).
