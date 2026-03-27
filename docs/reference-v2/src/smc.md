# SMC -- Sequential Monte Carlo

Source: `src/genmlx/inference/smc.cljs`, `src/genmlx/inference/compiled_smc.cljs`

Sequential Monte Carlo (SMC) maintains a weighted particle population that
evolves through a sequence of target distributions. At each timestep, particles
are extended with new observations, reweighted, and optionally resampled and
rejuvenated. SMC is the standard method for filtering in state-space models
and for any inference problem that decomposes into sequential stages.

GenMLX provides four SMC implementations at different points on the
performance/generality tradeoff:

| Variant | Execution | Best for |
|---------|-----------|----------|
| `smc` | Scalar (N sequential generates per step) | General models, splice, branching |
| `vsmc` | Vectorized (1 batched generate per step) | Static models, 10-100x faster |
| `smc-unfold` | Unfold combinator (O(1) per step) | Long sequences, memory-bounded |
| `compiled-smc` | Compiled tensor ops (Level 2) | Compiled kernels, GPU-fused |

**Namespace alias used below:**
```clojure
(require '[genmlx.inference.smc :as smc]
         '[genmlx.inference.compiled-smc :as csmc])
```

---

## SMC

### `smc`

```clojure
(smc/smc opts model args observations-seq)
```

Bootstrap particle filter with adaptive resampling and optional MH
rejuvenation. Each timestep: (1) extend particles with new observations via
`p/update`, (2) reweight, (3) resample if ESS drops below threshold,
(4) optionally rejuvenate with MH moves.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Options map (see below) |
| `model` | DynamicGF | Generative function for the full model |
| `args` | Vector | Arguments to the model |
| `observations-seq` | Seq of ChoiceMaps | Observations for each timestep |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:particles` | Integer | `100` | Number of particles |
| `:ess-threshold` | Number | `0.5` | Resample when ESS/N falls below this ratio |
| `:rejuvenation-steps` | Integer | `0` | Number of MH rejuvenation steps after resampling |
| `:rejuvenation-selection` | Selection | `sel/all` | Addresses to rejuvenate |
| `:resample-method` | Keyword | `:systematic` | One of `:systematic`, `:residual`, `:stratified` |
| `:callback` | Function | `nil` | Called each step with `{:step :ess :resampled?}` |
| `:key` | MLX array | fresh | PRNG key for reproducibility |

**Returns:** `{:traces [Trace ...] :log-weights [MLX-scalar ...] :log-ml-estimate MLX-scalar}`

- `:traces` -- final particle traces after all timesteps
- `:log-weights` -- final unnormalized log importance weights
- `:log-ml-estimate` -- accumulated log marginal likelihood estimate (product of per-step increments)

**Example:**

```clojure
(def hmm-model
  (gen [obs-seq]
    (let [state (trace :z0 (dist/categorical [0.5 0.5]))]
      (doseq [[t obs] (map-indexed vector obs-seq)]
        (let [state (trace (keyword (str "z" (inc t)))
                           (dist/categorical (if (zero? (mx/item state))
                                               [0.9 0.1] [0.2 0.8])))]
          (trace (keyword (str "y" (inc t)))
                 (dist/gaussian (if (zero? (mx/item state))
                                  (mx/scalar 0.0) (mx/scalar 3.0)) 1))))
      state)))

;; Build observation sequence
(def obs-seq
  [(cm/choicemap :y1 (mx/scalar 0.1))
   (cm/choicemap :y2 (mx/scalar 2.8))
   (cm/choicemap :y3 (mx/scalar 3.1))])

(let [{:keys [log-ml-estimate]}
      (smc/smc {:particles 500 :ess-threshold 0.5}
               hmm-model [obs-data] obs-seq)]
  (println "Log-ML:" (mx/item log-ml-estimate)))
```

**Notes:**
- Automatically strips L3 analytical handlers to ensure particle diversity.
- Sweeps dead arrays every step and clears the MLX cache every 5 steps.

---

## Vectorized SMC

### `vsmc`

```clojure
(smc/vsmc opts model args observations-seq)
```

Vectorized Sequential Monte Carlo. Runs the model body **once** per timestep
for all N particles simultaneously via batched handlers. 10-100x faster than
scalar `smc` for models without splice or data-dependent branching.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Options map (see below) |
| `model` | DynamicGF | Generative function (must support vectorized execution) |
| `args` | Vector | Arguments to the model |
| `observations-seq` | Seq of ChoiceMaps | Observations for each timestep |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:particles` | Integer | `100` | Number of particles |
| `:ess-threshold` | Number | `0.5` | Resample when ESS/N falls below this ratio |
| `:rejuvenation-steps` | Integer | `0` | Rounds of vectorized MH rejuvenation |
| `:rejuvenation-selection` | Selection | `sel/all` | Addresses to rejuvenate |
| `:callback` | Function | `nil` | Called each step with `{:step :ess :resampled?}` |
| `:key` | MLX array | fresh | PRNG key for reproducibility |

**Returns:** `{:vtrace VectorizedTrace :log-ml-estimate MLX-scalar}`

- `:vtrace` -- final `VectorizedTrace` with `[N]`-shaped arrays at each choice
- `:log-ml-estimate` -- accumulated log marginal likelihood estimate

**Example:**

```clojure
(let [{:keys [vtrace log-ml-estimate]}
      (smc/vsmc {:particles 1000 :ess-threshold 0.5
                 :rejuvenation-steps 3
                 :callback (fn [{:keys [step ess]}]
                             (println "Step" step "ESS:" ess))}
                model args obs-seq)]
  (println "Log-ML:" (mx/item log-ml-estimate)))
```

**Limitations:** No `splice` calls in the model. No data-dependent branching.

---

### `vsmc-init`

```clojure
(smc/vsmc-init model args observations particles key)
```

Initialize vectorized SMC state. Runs the model once with a batched handler
to produce initial particles. Useful for building custom SMC loops.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | DynamicGF | Generative function |
| `args` | Vector | Arguments to the model |
| `observations` | ChoiceMap | Initial observations |
| `particles` | Integer | Number of particles |
| `key` | MLX array | PRNG key |

**Returns:** `{:vtrace VectorizedTrace :log-ml-estimate MLX-scalar}`

---

## Conditional SMC

### `csmc`

```clojure
(smc/csmc opts model args observations-seq reference-trace)
```

Conditional Sequential Monte Carlo: SMC with a retained reference particle.
The reference particle is never resampled -- its trajectory is preserved across
all timesteps. This is the core kernel for particle Gibbs and particle MCMC
(PMCMC).

The reference particle occupies index 0 in the particle array. During
resampling, its index is forced to survive. During rejuvenation, it is skipped.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Same options as `smc` (see above) |
| `model` | DynamicGF | Generative function |
| `args` | Vector | Arguments to the model |
| `observations-seq` | Seq of ChoiceMaps | Observations for each timestep |
| `reference-trace` | Trace | The retained reference particle from a previous iteration |

**Returns:** `{:traces [Trace ...] :log-weights [MLX-scalar ...] :log-ml-estimate MLX-scalar}`

**Example (particle Gibbs):**

```clojure
;; Initialize with standard SMC
(let [init-result (smc/smc {:particles 100} model args obs-seq)
      ;; Pick a trace proportional to weights
      init-trace (first (:traces init-result))]
  ;; Run conditional SMC iterations
  (loop [ref-trace init-trace, i 0]
    (when (< i 50)
      (let [{:keys [traces log-weights]}
            (smc/csmc {:particles 100} model args obs-seq ref-trace)
            ;; Sample new reference from the particle population
            new-ref (nth traces (rand-int (count traces)))]
        (recur new-ref (inc i))))))
```

---

## SMC Unfold

### `smc-unfold`

```clojure
(smc/smc-unfold opts kernel init-state observations-seq)
```

Always-resample bootstrap particle filter using the Unfold combinator.
Extends traces one step at a time via `unfold-extend`, giving O(1) cost per
step and staying well under Metal buffer limits for long sequences.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Options map (see below) |
| `kernel` | DynamicGF | Kernel taking `[t state & extra]` and returning new state |
| `init-state` | any | Initial state passed to the kernel |
| `observations-seq` | Seq of ChoiceMaps | Kernel-level choice maps, one per timestep |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:particles` | Integer | `100` | Number of particles |
| `:key` | MLX array | fresh | PRNG key |

**Returns:** `{:log-ml MLX-scalar :traces [Trace ...] :final-ess number}`

**Notes:**
- Always resamples (no ESS-based adaptive resampling).
- Uses `mx/tidy-run` at each step to free intermediate arrays.
- Forces GC and clears cache every 2 steps to stay under Metal buffer limits.

---

### `batched-smc-unfold`

```clojure
(smc/batched-smc-unfold opts kernel init-state observations-seq)
```

Batched bootstrap particle filter for sequential models. Runs the kernel
**once** per timestep for all N particles via `vgenerate`. State can be an
MLX array, a map of MLX arrays, or nil.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Options map (see below) |
| `kernel` | DynamicGF | Vectorization-compatible kernel taking `[t state]` |
| `init-state` | any | Initial state (nil, MLX array, or map of arrays) |
| `observations-seq` | Seq of ChoiceMaps | Kernel-level choice maps, one per timestep |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:particles` | Integer | `100` | Number of particles |
| `:key` | MLX array | fresh | PRNG key |
| `:callback` | Function | `nil` | Called each step with `{:step t}` |

**Returns:** `{:log-ml MLX-scalar :final-states [N]-shaped :final-ess number}`

**Notes:**
- Map-valued state is resampled per-key (each value indexed by particle dimension).
- Always resamples at every step using GPU-accelerated systematic resampling.

---

## Compiled SMC

Source: `src/genmlx/inference/compiled_smc.cljs`

Level 2 compiled SMC stores particles as `[N,K]` tensors (N particles, K trace
sites) and broadcasts noise transforms over all particles simultaneously. This
bypasses the GFI handler entirely -- extend steps are compiled from the kernel's
schema into fused MLX operations.

### `compiled-smc`

```clojure
(csmc/compiled-smc opts kernel init-state observations-seq)
```

Compiled bootstrap particle filter for Unfold models. Pre-generates all
randomness as `[T,N,K]` tensors and runs the full sweep with compiled
extend steps.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Options map (see below) |
| `kernel` | DynamicGF | Unfold kernel with compilable schema |
| `init-state` | any | Initial state |
| `observations-seq` | Seq of ChoiceMaps | Observations per timestep |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:particles` | Integer | `100` | Number of particles N |
| `:key` | MLX array | fresh | PRNG key |
| `:callback` | Function | `nil` | Called each step with `{:step :ess :resampled?}` |
| `:resample-method` | Keyword | `:systematic` | One of `:systematic`, `:gumbel-top-k`, `:gumbel-softmax` |
| `:tau` | Number | `1.0` | Temperature for `:gumbel-softmax` resampling |

**Returns:** `{:log-ml MLX-scalar :particles [N,K] :addr-index {addr->int} :final-ess number}`

- `:log-ml` -- accumulated log marginal likelihood
- `:particles` -- final particle states as an `[N,K]` tensor
- `:addr-index` -- mapping from trace site addresses to column indices in the particle tensor
- `:final-ess` -- effective sample size (equals N since always-resample)

**Resample methods:**

| Method | Properties | Use case |
|--------|------------|----------|
| `:systematic` | O(N) via searchsorted, non-differentiable | Default, forward-pass inference |
| `:gumbel-top-k` | All-GPU via Gumbel-max trick, non-differentiable | GPU-only execution, no CPU roundtrips |
| `:gumbel-softmax` | Approximate, fully differentiable | Gradient-through-SMC for learning |

**Example:**

```clojure
(let [{:keys [log-ml particles addr-index]}
      (csmc/compiled-smc {:particles 200 :resample-method :systematic}
                          kernel init-state obs-seq)]
  (println "Log-ML:" (mx/item log-ml))
  (println "Particle shape:" (mx/shape particles)))
```

---

### `generate-smc-noise`

```clojure
(csmc/generate-smc-noise key T N K)
```

Pre-generate all randomness for a T-step, N-particle SMC sweep. Returns
pre-allocated noise tensors that compiled-smc consumes deterministically.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | MLX array | PRNG key |
| `T` | Integer | Number of timesteps |
| `N` | Integer | Number of particles |
| `K` | Integer | Number of trace sites per particle |

**Returns:** `{:extend-noise [T,N,K] :resample-uniforms [T]}`

---

### `systematic-resample-tensor`

```clojure
(csmc/systematic-resample-tensor particles log-weights uniform N)
```

Systematic resampling operating on tensors. Takes `[N,K]`-shaped particles
and `[N]`-shaped log-weights, returns resampled particles and ancestor indices.
Uses O(N) searchsorted for index computation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `particles` | MLX array `[N,K]` | Current particle states |
| `log-weights` | MLX array `[N]` | Unnormalized log-weights |
| `uniform` | MLX scalar | Pre-generated uniform for systematic offset |
| `N` | Integer | Number of particles |

**Returns:** `{:particles [N,K] :ancestors [N] int32}`

---

### `smc-result->traces`

```clojure
(csmc/smc-result->traces result kernel)
```

Convert a compiled SMC result (with `[N,K]` particle tensor) back to a
vector of `TensorTrace` records. Useful for post-processing with GFI
operations.

| Parameter | Type | Description |
|-----------|------|-------------|
| `result` | Map | Return value from `compiled-smc` |
| `kernel` | DynamicGF | The kernel used for the SMC sweep |

**Returns:** Vector of N `TensorTrace` records.

---

## Resampling

GenMLX supports three resampling methods for scalar SMC (`:resample-method`
option in `smc` and `csmc`):

### Systematic resampling (`:systematic`)

The default. Uses a single uniform offset to select particles from the
cumulative weight distribution. O(N), low variance, deterministic given the
uniform.

### Residual resampling (`:residual`)

Deterministically allocates `floor(N * w_i)` copies of each particle, then
multinomially resamples the remainder. Lower variance than systematic for
highly concentrated weight distributions.

### Stratified resampling (`:stratified`)

Draws one independent uniform per stratum `[j/N, (j+1)/N)`. Lower variance
than multinomial resampling due to independent strata. Slightly higher variance
than systematic but more robust to weight ordering.

For compiled SMC, two additional GPU-native methods are available:

### Gumbel-top-k (`:gumbel-top-k`)

Hard resampling via the Gumbel-max trick. Adds independent Gumbel(0,1) noise
to log-weights and takes argmax. Exact categorical sampling, all-GPU (no CPU
roundtrips), but non-differentiable.

### Gumbel-softmax (`:gumbel-softmax`)

Soft resampling via Gumbel-softmax relaxation. Approximate but fully
differentiable through `mx/grad`. Temperature `:tau` controls the
bias-variance tradeoff (lower tau approaches hard resampling). Use for
gradient-through-SMC in learning applications.
