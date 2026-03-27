# MCMC

Markov Chain Monte Carlo methods generate samples from posterior distributions by constructing Markov chains whose stationary distribution matches the target. GenMLX provides a full suite of MCMC algorithms, from basic Metropolis-Hastings through gradient-based methods (MALA, HMC, NUTS) to specialized samplers (Gibbs, elliptical slice, involutive MCMC). All algorithms support optional loop compilation into fused Metal computation graphs for GPU acceleration.

```clojure
(require '[genmlx.inference.mcmc :as mcmc])
```

Source: `src/genmlx/inference/mcmc.cljs`

---

## Metropolis-Hastings

The foundation of MCMC inference. MH proposes new states by regenerating selected addresses from the prior, then accepts or rejects based on the importance weight.

### `mh-step`

```clojure
(mcmc/mh-step current-trace selection)
(mcmc/mh-step current-trace selection key)
```

Single Metropolis-Hastings step. Calls `regenerate` on the given selection, then accepts or rejects the proposal with probability min(1, exp(w)) where w is the regeneration weight.

| Parameter | Type | Description |
|-----------|------|-------------|
| `current-trace` | Trace | Current model trace |
| `selection` | Selection | Addresses to resample |
| `key` | MLX array (optional) | PRNG key for reproducibility |

**Returns:** A `Trace` -- either the new trace (accepted) or the original (rejected).

**Example:**
```clojure
(let [{:keys [trace]} (p/generate model args obs)
      new-trace (mcmc/mh-step trace (sel/select :slope :intercept))]
  (:retval new-trace))
```

---

### `mh`

```clojure
(mcmc/mh opts model args observations)
```

Full Metropolis-Hastings inference chain. Initializes from `generate`, then runs `mh-step` repeatedly with burn-in, thinning, and callback support.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | map | Configuration options |
| `model` | generative function | Target model |
| `args` | vector | Model arguments |
| `observations` | choicemap | Observed values |

**Options map:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | int | -- | Number of post-burn-in samples to collect |
| `:burn` | int | `0` | Burn-in samples to discard |
| `:thin` | int | `1` | Keep every thin-th sample |
| `:selection` | Selection | `sel/all` | Addresses to resample each step |
| `:callback` | fn | `nil` | `(fn [i trace] ...)` called each iteration |
| `:key` | MLX array | random | PRNG key for reproducibility |

**Returns:** Vector of `Trace` records.

**Example:**
```clojure
(def traces
  (mcmc/mh {:samples 1000 :burn 200 :thin 2
             :selection (sel/select :slope :intercept)}
            model [xs] obs))

;; Extract posterior samples
(map #(cm/get-value (:choices %) :slope) traces)
```

---

### `mh-custom-step`

```clojure
(mcmc/mh-custom-step current-trace model proposal-gf key)
(mcmc/mh-custom-step current-trace model proposal-gf backward-gf key)
```

Single MH step with a custom proposal generative function instead of the prior. The proposal takes the current trace's choices and proposes new values. An optional backward proposal handles asymmetric kernels; if omitted, the proposal is assumed symmetric.

| Parameter | Type | Description |
|-----------|------|-------------|
| `current-trace` | Trace | Current model trace |
| `model` | generative function | Target model |
| `proposal-gf` | generative function | Forward proposal: takes `[current-choices]` |
| `backward-gf` | generative function (optional) | Backward proposal for asymmetric kernels |
| `key` | MLX array | PRNG key |

**Returns:** A `Trace` -- either the new trace (accepted) or the original (rejected).

---

### `mh-custom`

```clojure
(mcmc/mh-custom opts model args observations)
```

Full MH chain with a custom proposal generative function. Same chain structure as `mh`, but uses `mh-custom-step` internally.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | map | Configuration options |
| `model` | generative function | Target model |
| `args` | vector | Model arguments |
| `observations` | choicemap | Observed values |

**Options map:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | int | -- | Number of post-burn-in samples |
| `:burn` | int | `0` | Burn-in samples to discard |
| `:thin` | int | `1` | Keep every thin-th sample |
| `:proposal-gf` | generative function | -- | Custom proposal (takes `[current-choices]`) |
| `:backward-gf` | generative function | `nil` | Backward proposal (nil = symmetric) |
| `:callback` | fn | `nil` | `(fn [i trace] ...)` called each iteration |
| `:key` | MLX array | random | PRNG key |

**Returns:** Vector of `Trace` records.

**Example:**
```clojure
;; Custom random-walk proposal with tuned width
(def proposal
  (gen [current-choices]
    (let [old-slope (cm/get-value current-choices :slope)]
      (trace :slope (dist/gaussian old-slope 0.5)))))

(mcmc/mh-custom {:samples 500 :burn 100 :proposal-gf proposal}
                 model [xs] obs)
```

---

## Compiled MH

Compiled MH extracts latent parameters into a flat array, compiles the score function, and iterates in parameter space -- bypassing the GFI regenerate overhead per step.

### `compiled-mh`

```clojure
(mcmc/compiled-mh opts model args observations)
```

MH with random-walk Gaussian proposals in compiled parameter space. When `:compile?` is true (the default), entire K-step chains are compiled into single Metal dispatches. Burn-in runs in blocks; collection uses compiled trajectories that return `[K, D]` tensors per dispatch.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | map | Configuration options |
| `model` | generative function | Target model |
| `args` | vector | Model arguments |
| `observations` | choicemap | Observed values |

**Options map:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | int | -- | Number of post-burn-in samples |
| `:burn` | int | `0` | Burn-in samples to discard |
| `:thin` | int | `1` | Keep every thin-th sample |
| `:addresses` | vector | -- | Vector of address keywords to sample |
| `:proposal-std` | float | `0.1` | Gaussian random walk standard deviation |
| `:compile?` | bool | `true` | Whether to compile the inner loop |
| `:block-size` | int | `50` | Burn-in chain length per Metal dispatch |
| `:block-size-collect` | int | `25` | Collection trajectory length (thin=1 path) |
| `:callback` | fn | `nil` | `(fn [{:iter :value}] ...)` called per sample |
| `:key` | MLX array | random | PRNG key |
| `:device` | keyword | `:cpu` | `:cpu` or `:gpu` |

**Returns:** Vector of parameter samples (JS arrays via `mx/->clj`).

**Example:**
```clojure
(def samples
  (mcmc/compiled-mh {:samples 2000 :burn 500 :addresses [:slope :intercept]
                      :proposal-std 0.1}
                     model [xs] obs))

;; Each sample is a flat JS array [slope, intercept]
(first samples) ;; => [0.47 1.23]
```

---

### `fused-mh`

```clojure
(mcmc/fused-mh opts model args observations)
```

Fully fused MH: burn-in and thinned collection in ONE Metal dispatch. Pre-generates all noise upfront and compiles the entire chain into a single function. Auto-falls back to `compiled-mh` when the chain exceeds Metal graph size limits.

The first call incurs compilation overhead (2-10s depending on chain length). Pass `:chain-fn` from a previous call to skip recompilation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | map | Configuration options |
| `model` | generative function | Target model |
| `args` | vector | Model arguments |
| `observations` | choicemap | Observed values |

**Options map:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | int | `1000` | Number of samples |
| `:burn` | int | `500` | Burn-in samples |
| `:thin` | int | `1` | Thinning interval |
| `:addresses` | vector | -- | Latent addresses to sample |
| `:proposal-std` | float | `0.1` | Random walk standard deviation |
| `:key` | MLX array | random | PRNG key |
| `:device` | keyword | `:cpu` | `:cpu` or `:gpu` |
| `:chain-fn` | compiled fn | `nil` | Reuse compiled chain from a previous call |

**Returns:** `{:samples MLX-array, :final-params MLX-array, :chain-fn compiled-fn, :acceptance-rate float}`

- `:samples` -- `[S, D]` MLX tensor of collected samples
- `:final-params` -- `[D]` MLX array of final parameter state
- `:chain-fn` -- the compiled function (reusable across calls)
- `:acceptance-rate` -- fraction of accepted proposals

**Example:**
```clojure
(let [result (mcmc/fused-mh {:samples 1000 :burn 500
                              :addresses [:slope :intercept]}
                             model [xs] obs)]
  (println "Acceptance rate:" (:acceptance-rate result))
  ;; Reuse compiled chain for another run
  (mcmc/fused-mh {:samples 1000 :burn 0
                   :addresses [:slope :intercept]
                   :chain-fn (:chain-fn result)}
                  model [xs] obs))
```

---

### `vectorized-compiled-mh`

```clojure
(mcmc/vectorized-compiled-mh opts model args observations)
```

Run N independent compiled MH chains in parallel. MLX broadcasting executes the score function ONCE per step for all N chains simultaneously.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | int | -- | Samples to collect |
| `:burn` | int | `0` | Burn-in |
| `:thin` | int | `1` | Thinning |
| `:addresses` | vector | -- | Latent addresses |
| `:proposal-std` | float | `0.1` | Random walk std |
| `:compile?` | bool | `true` | Compile score function |
| `:n-chains` | int | `10` | Number of parallel chains |
| `:callback` | fn | `nil` | `(fn [{:iter :value :n-accepted}] ...)` |
| `:key` | MLX array | random | PRNG key |
| `:device` | keyword | `:gpu` | Default `:gpu` for vectorized ops |

**Returns:** Vector of `[N-chains, D]` JS arrays (one per sample). Metadata contains `{:acceptance-rate mean-rate}`.

**Example:**
```clojure
(def samples
  (mcmc/vectorized-compiled-mh
    {:samples 500 :burn 200 :n-chains 8
     :addresses [:slope :intercept]}
    model [xs] obs))

(:acceptance-rate (meta samples))
```

---

### `vectorized-compiled-trajectory-mh`

```clojure
(mcmc/vectorized-compiled-trajectory-mh opts model args observations)
```

Vectorized compiled trajectory MH: N parallel chains with K steps per Metal dispatch. Combines multi-chain parallelism with loop compilation. Samples are pooled across chains. Uses a single compiled trajectory chain for both burn-in and collection.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | int | -- | Total samples to collect (pooled across chains) |
| `:burn` | int | `0` | Burn-in steps |
| `:addresses` | vector | -- | Latent addresses |
| `:proposal-std` | float | `0.1` | Random walk std |
| `:n-chains` | int | `10` | Number of parallel chains |
| `:block-size` | int | `10` | Steps per Metal dispatch |
| `:callback` | fn | `nil` | Called with `{:total count}` at completion |
| `:key` | MLX array | random | PRNG key |
| `:device` | keyword | `:gpu` | Default `:gpu` |

**Returns:** Vector of `[D]` JS arrays (pooled from all chains).

---

### `fused-vectorized-mh`

```clojure
(mcmc/fused-vectorized-mh opts model args observations)
```

Fully fused vectorized MH: N parallel chains with burn-in and thinned collection in ONE Metal dispatch per call. Same API shape as `fused-mh` but operates on `[N, D]` parameter tensors.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | int | `1000` | Samples to collect |
| `:burn` | int | `500` | Burn-in |
| `:thin` | int | `1` | Thinning |
| `:addresses` | vector | -- | Latent addresses |
| `:proposal-std` | float | `0.1` | Random walk std |
| `:n-chains` | int | `8` | Number of parallel chains |
| `:key` | MLX array | random | PRNG key |
| `:device` | keyword | `:gpu` | Default `:gpu` |
| `:chain-fn` | compiled fn | `nil` | Reuse compiled chain from previous call |

**Returns:** `{:samples [S,N,D], :final-params [N,D], :chain-fn compiled-fn, :acceptance-rate [N]}`

---

## MALA -- Metropolis-Adjusted Langevin Algorithm

MALA uses gradient information to guide proposals toward high-density regions. The proposal follows the score gradient with added noise:

x' = x + (epsilon^2 / 2) * grad(log p(x)) + epsilon * z,  where z ~ N(0, I)

The asymmetric proposal density is corrected via Hastings ratio to maintain detailed balance. Use MALA when the posterior is smooth and continuous -- it converges faster than random-walk MH for well-conditioned targets.

### `mala`

```clojure
(mcmc/mala opts model args observations)
```

MALA inference with optional loop compilation and adaptive step-size tuning.

When `:compile?` is true (the default), entire K-step chains are compiled into single Metal dispatches. Score and gradient are cached across iterations, reducing val-grad calls from 3 to 1 per step.

When `:adapt-step-size` is true, the burn-in phase uses dual averaging (Hoffman and Gelman 2014) to tune the step-size to achieve the target acceptance rate (default 0.574).

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | map | Configuration options |
| `model` | generative function | Target model |
| `args` | vector | Model arguments |
| `observations` | choicemap | Observed values |

**Options map:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | int | -- | Number of samples |
| `:step-size` | float | `0.01` | Langevin step size (epsilon) |
| `:burn` | int | `0` | Burn-in samples |
| `:thin` | int | `1` | Thinning interval |
| `:addresses` | vector | -- | Continuous addresses to update |
| `:compile?` | bool | `true` | Compile inner loop |
| `:block-size` | int | `50` | Steps per compiled burn-in block |
| `:adapt-step-size` | bool | `false` | Enable dual averaging adaptation |
| `:target-accept` | float | `0.574` | Target acceptance rate for adaptation |
| `:callback` | fn | `nil` | `(fn [{:iter :value}] ...)` |
| `:key` | MLX array | random | PRNG key |
| `:device` | keyword | `:cpu` | `:cpu` or `:gpu` |

**Returns:** Vector of parameter samples (JS arrays).

**Example:**
```clojure
(mcmc/mala {:samples 1000 :burn 500 :step-size 0.01
            :addresses [:slope :intercept]
            :adapt-step-size true}
           model [xs] obs)
```

---

### `fused-mala`

```clojure
(mcmc/fused-mala opts model args observations)
```

Fully fused MALA: burn-in and thinned collection in ONE Metal dispatch. Auto-falls back to block-compiled MALA when the chain exceeds Metal graph size limits.

When `:adapt-step-size` is true, runs a short eager warmup phase using dual averaging to tune the step-size, then compiles the fused chain with the adapted value.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | int | `1000` | Number of samples |
| `:burn` | int | `500` | Burn-in |
| `:thin` | int | `1` | Thinning |
| `:addresses` | vector | -- | Latent addresses |
| `:step-size` | float | `0.01` | Langevin step size |
| `:adapt-step-size` | bool | `false` | Dual averaging adaptation |
| `:target-accept` | float | `0.574` | Target acceptance rate |
| `:warmup-steps` | int | `200` | Number of adaptive warmup steps |
| `:key` | MLX array | random | PRNG key |
| `:device` | keyword | `:cpu` | `:cpu` or `:gpu` |
| `:chain-fn` | compiled fn | `nil` | Reuse compiled chain |

**Returns:** `{:samples [S,D], :final-params [D], :chain-fn compiled-fn, :acceptance-rate float}`

---

### `vectorized-mala`

```clojure
(mcmc/vectorized-mala opts model args observations)
```

N independent MALA chains running in parallel. MLX broadcasting computes gradients for all N chains simultaneously via the sum trick.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | int | -- | Samples to collect |
| `:burn` | int | `0` | Burn-in |
| `:thin` | int | `1` | Thinning |
| `:step-size` | float | `0.01` | Langevin step size |
| `:addresses` | vector | -- | Continuous addresses |
| `:compile?` | bool | `true` | Compile score/gradient |
| `:n-chains` | int | `10` | Number of parallel chains |
| `:callback` | fn | `nil` | `(fn [{:iter :value :n-accepted}] ...)` |
| `:key` | MLX array | random | PRNG key |
| `:device` | keyword | `:gpu` | Default `:gpu` for vectorized ops |

**Returns:** Vector of `[N-chains, D]` JS arrays. Metadata contains `{:acceptance-rate mean-rate}`.

---

## HMC -- Hamiltonian Monte Carlo

HMC simulates Hamiltonian dynamics to make distant, low-correlation proposals. By introducing auxiliary momentum variables, HMC explores the target distribution more efficiently than random-walk methods -- especially in high dimensions.

Each HMC step: (1) sample momentum from N(0, M), (2) run L leapfrog integration steps, (3) accept/reject based on Hamiltonian energy conservation. GenMLX uses fused leapfrog (L+1 gradient evaluations instead of 2L) for efficiency.

### `hmc`

```clojure
(mcmc/hmc opts model args observations)
```

Hamiltonian Monte Carlo with leapfrog integration. Supports identity, diagonal, and dense mass matrices. When `:compile?` is true and the mass matrix is nil (identity), uses loop compilation for Metal acceleration.

When `:adapt-step-size` is true, burn-in uses dual averaging (Hoffman and Gelman 2014) to tune the step-size. When `:adapt-metric` is true, a diagonal mass matrix is estimated from warmup samples using Welford's online algorithm.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | map | Configuration options |
| `model` | generative function | Target model |
| `args` | vector | Model arguments |
| `observations` | choicemap | Observed values |

**Options map:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | int | -- | Number of samples |
| `:step-size` | float | `0.01` | Leapfrog step size |
| `:leapfrog-steps` | int | `20` | Leapfrog steps per proposal |
| `:burn` | int | `100` | Burn-in samples |
| `:thin` | int | `1` | Thinning interval |
| `:addresses` | vector | -- | Continuous addresses to sample |
| `:metric` | MLX array | `nil` | Mass matrix (nil=identity, vector=diagonal, matrix=dense) |
| `:compile?` | bool | `true` | Compile inner loop (identity metric only) |
| `:block-size` | int | `20` | Steps per compiled burn-in block |
| `:adapt-step-size` | bool | `false` | Dual averaging step-size adaptation |
| `:target-accept` | float | `0.65` | Target acceptance rate |
| `:adapt-metric` | bool | `false` | Adapt diagonal mass matrix during warmup |
| `:callback` | fn | `nil` | `(fn [{:iter :value}] ...)` |
| `:key` | MLX array | random | PRNG key |
| `:device` | keyword | `:cpu` | `:cpu` or `:gpu` |

**Returns:** Vector of parameter samples (JS arrays via `mx/->clj`).

**Example:**
```clojure
;; Basic HMC
(mcmc/hmc {:samples 1000 :burn 500 :step-size 0.01 :leapfrog-steps 20
            :addresses [:mu :sigma]}
           model args obs)

;; With full adaptation (step-size + mass matrix)
(mcmc/hmc {:samples 2000 :burn 1000 :addresses [:mu :sigma]
            :adapt-step-size true :adapt-metric true}
           model args obs)

;; With a known diagonal mass matrix
(mcmc/hmc {:samples 1000 :addresses [:mu :sigma]
            :metric (mx/array [1.0 0.1])}
           model args obs)
```

---

### `fused-hmc`

```clojure
(mcmc/fused-hmc opts model args observations)
```

Fully fused HMC: burn-in and thinned collection in ONE Metal dispatch. Pre-generates all momentum and acceptance noise upfront. Auto-falls back to block-compiled HMC when the chain exceeds Metal graph size limits (estimated from total-steps * leapfrog-steps). Identity mass matrix only for fused chains.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | int | `1000` | Number of samples |
| `:burn` | int | `500` | Burn-in |
| `:thin` | int | `1` | Thinning |
| `:addresses` | vector | -- | Latent addresses |
| `:step-size` | float | `0.01` | Leapfrog step size |
| `:leapfrog-steps` | int | `20` | Leapfrog steps per proposal |
| `:adapt-step-size` | bool | `false` | Dual averaging adaptation |
| `:target-accept` | float | `0.65` | Target acceptance rate |
| `:warmup-steps` | int | `200` | Adaptive warmup steps |
| `:key` | MLX array | random | PRNG key |
| `:device` | keyword | `:cpu` | `:cpu` or `:gpu` |
| `:chain-fn` | compiled fn | `nil` | Reuse compiled chain |

**Returns:** `{:samples [S,D], :final-params [D], :chain-fn compiled-fn, :acceptance-rate float}`

---

### `vectorized-hmc`

```clojure
(mcmc/vectorized-hmc opts model args observations)
```

N independent HMC chains running in parallel. MLX broadcasting computes leapfrog trajectories for all N chains simultaneously via the sum trick for gradients. Identity mass matrix only.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | int | -- | Samples to collect |
| `:burn` | int | `100` | Burn-in |
| `:thin` | int | `1` | Thinning |
| `:step-size` | float | `0.01` | Leapfrog step size |
| `:leapfrog-steps` | int | `20` | Leapfrog steps per proposal |
| `:addresses` | vector | -- | Continuous addresses |
| `:n-chains` | int | `10` | Number of parallel chains |
| `:callback` | fn | `nil` | `(fn [{:iter :value :n-accepted}] ...)` |
| `:key` | MLX array | random | PRNG key |
| `:device` | keyword | `:gpu` | Default `:gpu` |

**Returns:** Vector of `[N-chains, D]` JS arrays. Metadata contains `{:acceptance-rate mean-rate}`.

---

## NUTS -- No-U-Turn Sampler

NUTS automatically tunes the number of leapfrog steps by building a binary tree of trajectory states until a U-turn is detected. This eliminates the need to hand-tune `:leapfrog-steps` -- the most sensitive HMC parameter. Use NUTS as the default gradient-based sampler when you do not know the optimal trajectory length.

### `nuts`

```clojure
(mcmc/nuts opts model args observations)
```

No-U-Turn Sampler with adaptive step-size and optional mass matrix estimation.

When `:adapt-step-size` is true, burn-in uses dual averaging with `find-reasonable-epsilon` initialization. When `:adapt-metric` is true, a diagonal mass matrix is estimated from warmup samples via Welford's algorithm.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | map | Configuration options |
| `model` | generative function | Target model |
| `args` | vector | Model arguments |
| `observations` | choicemap | Observed values |

**Options map:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | int | -- | Number of samples |
| `:step-size` | float | `0.01` | Initial leapfrog step size |
| `:max-depth` | int | `10` | Maximum binary tree depth |
| `:burn` | int | `0` | Burn-in samples |
| `:thin` | int | `1` | Thinning interval |
| `:addresses` | vector | -- | Continuous addresses to sample |
| `:metric` | MLX array | `nil` | Mass matrix (nil=identity, vector=diagonal, matrix=dense) |
| `:compile?` | bool | `true` | Compile gradient function |
| `:adapt-step-size` | bool | `false` | Dual averaging step-size adaptation |
| `:target-accept` | float | `0.8` | Target acceptance rate |
| `:adapt-metric` | bool | `false` | Adapt diagonal mass matrix during warmup |
| `:callback` | fn | `nil` | `(fn [{:iter :value}] ...)` |
| `:key` | MLX array | random | PRNG key |
| `:device` | keyword | `:cpu` | `:cpu` or `:gpu` |

**Returns:** Vector of parameter samples (JS arrays via `mx/->clj`).

**Example:**
```clojure
;; NUTS with full adaptation -- the recommended default
(mcmc/nuts {:samples 1000 :burn 500
            :addresses [:mu :log-sigma]
            :adapt-step-size true
            :adapt-metric true}
           model args obs)

;; Manual step-size, deeper tree for complex posteriors
(mcmc/nuts {:samples 2000 :step-size 0.005 :max-depth 12
            :addresses [:mu :sigma]}
           model args obs)
```

---

## Gibbs Sampling

Gibbs sampling sweeps through discrete addresses one at a time, sampling each from its full conditional distribution via explicit support enumeration. Use Gibbs for models with discrete latent variables that have known finite support.

### `gibbs-step-with-support`

```clojure
(mcmc/gibbs-step-with-support current-trace addr support-values key)
```

Single Gibbs step: enumerate all support values for the given address, compute the conditional log-probabilities via model `update`, normalize, and sample one.

| Parameter | Type | Description |
|-----------|------|-------------|
| `current-trace` | Trace | Current model trace |
| `addr` | keyword | Address to resample |
| `support-values` | vector | All possible values for the address |
| `key` | MLX array | PRNG key |

**Returns:** A new `Trace` with the resampled address.

**Example:**
```clojure
;; Resample a discrete cluster assignment
(mcmc/gibbs-step-with-support trace :z [0 1 2] key)
```

---

### `gibbs`

```clojure
(mcmc/gibbs opts model args observations schedule)
```

Systematic Gibbs sampling over discrete addresses. Each iteration sweeps through all addresses in the schedule, resampling each from its full conditional.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | map | Configuration options |
| `model` | generative function | Target model |
| `args` | vector | Model arguments |
| `observations` | choicemap | Observed values |
| `schedule` | vector | Sweep specification |

**Options map:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | int | -- | Number of samples |
| `:burn` | int | `0` | Burn-in |
| `:thin` | int | `1` | Thinning |
| `:callback` | fn | `nil` | Iteration callback |
| `:key` | MLX array | random | PRNG key |

**Schedule format:** Vector of `{:addr keyword, :support [values...]}` maps specifying which addresses to sweep and their finite support.

**Returns:** Vector of `Trace` records.

**Example:**
```clojure
;; Mixture model with 3 components
(mcmc/gibbs {:samples 1000 :burn 200}
             model args obs
             [{:addr :z0 :support [0 1 2]}
              {:addr :z1 :support [0 1 2]}
              {:addr :z2 :support [0 1 2]}])
```

---

## Elliptical Slice Sampling

Elliptical slice sampling (Murray, Adams, and MacKay 2010) is designed for models with Gaussian priors. It requires no tuning parameters and always accepts -- making it ideal for Gaussian process models and other settings with multivariate Gaussian priors.

The algorithm proposes along an ellipse defined by the current state and a draw from the prior, then shrinks a bracket until the log-likelihood threshold is exceeded.

### `elliptical-slice-step`

```clojure
(mcmc/elliptical-slice-step current-trace selection prior-std key)
```

Single elliptical slice sampling step.

| Parameter | Type | Description |
|-----------|------|-------------|
| `current-trace` | Trace | Current model trace |
| `selection` | vector of keywords | Addresses with Gaussian prior to resample |
| `prior-std` | float | Standard deviation of the Gaussian prior N(0, prior-std^2 * I) |
| `key` | MLX array | PRNG key |

**Returns:** A new `Trace` (always accepts -- no MH rejection).

---

### `elliptical-slice`

```clojure
(mcmc/elliptical-slice opts model args observations)
```

Full elliptical slice sampling chain.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | map | Configuration options |
| `model` | generative function | Target model |
| `args` | vector | Model arguments |
| `observations` | choicemap | Observed values |

**Options map:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | int | -- | Number of samples |
| `:burn` | int | `0` | Burn-in |
| `:thin` | int | `1` | Thinning |
| `:selection` | vector | -- | Addresses with Gaussian prior |
| `:prior-std` | float | `1.0` | Prior standard deviation |
| `:key` | MLX array | random | PRNG key |

**Returns:** Vector of `Trace` records.

**Example:**
```clojure
;; Gaussian process regression -- sample function values
(mcmc/elliptical-slice {:samples 500 :burn 100
                         :selection [:f0 :f1 :f2 :f3]
                         :prior-std 2.0}
                        gp-model args obs)
```

---

## Involutive MCMC

Involutive MCMC (Cusumano-Towner et al. 2019) generalizes MH by introducing auxiliary random variables and a deterministic involution that maps `(trace, aux) -> (trace', aux')`. The involution must be its own inverse. This framework subsumes many MCMC kernels as special cases, including reversible-jump MCMC for trans-dimensional models.

### `involutive-mh-step`

```clojure
(mcmc/involutive-mh-step current-trace model proposal-gf involution key)
```

Single involutive MH step.

| Parameter | Type | Description |
|-----------|------|-------------|
| `current-trace` | Trace | Current model trace |
| `model` | generative function | Target model |
| `proposal-gf` | generative function | Proposal for auxiliary variables (takes `[current-choices]`) |
| `involution` | function | `(fn [trace-cm aux-cm] -> [new-trace-cm new-aux-cm])` or 3-element vector with log|det J| |
| `key` | MLX array | PRNG key |

The involution may return a 2-element vector `[new-trace-cm new-aux-cm]` or a 3-element vector `[new-trace-cm new-aux-cm log-abs-det-J]` to include a Jacobian correction.

**Returns:** A `Trace` -- either the new trace (accepted) or the original (rejected).

---

### `involutive-mh`

```clojure
(mcmc/involutive-mh opts model args observations)
```

Full involutive MCMC chain.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | map | Configuration options |
| `model` | generative function | Target model |
| `args` | vector | Model arguments |
| `observations` | choicemap | Observed values |

**Options map:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | int | -- | Number of samples |
| `:burn` | int | `0` | Burn-in |
| `:thin` | int | `1` | Thinning |
| `:proposal-gf` | generative function | -- | Proposal for auxiliary variables |
| `:involution` | function | -- | `(fn [trace-cm aux-cm] -> [trace-cm' aux-cm'])` |
| `:callback` | fn | `nil` | Iteration callback |
| `:key` | MLX array | random | PRNG key |

**Returns:** Vector of `Trace` records.

**Example:**
```clojure
;; Birth-death move for variable-dimensional model
(def birth-death-proposal
  (gen [current-choices]
    (trace :move-type (dist/bernoulli 0.5))
    (trace :new-value (dist/gaussian 0 1))))

(defn birth-death-involution [trace-cm aux-cm]
  (let [move (cm/get-value aux-cm :move-type)]
    (if (pos? (mx/item move))
      ;; Birth: add a component
      [(cm/set-choice trace-cm [:new-comp] (cm/get-value aux-cm :new-value))
       (cm/choicemap :move-type (mx/scalar 0)
                     :new-value (cm/get-value trace-cm :new-comp))]
      ;; Death: remove a component
      [(cm/remove-choice trace-cm :new-comp)
       (cm/choicemap :move-type (mx/scalar 1)
                     :new-value (cm/get-value trace-cm :new-comp))])))

(mcmc/involutive-mh {:samples 500 :burn 200
                      :proposal-gf birth-death-proposal
                      :involution birth-death-involution}
                     model args obs)
```

---

## MAP Optimization

MAP finds the maximum a posteriori estimate by gradient ascent on log p(latents | observations). This is point estimation, not sampling -- useful as initialization for MCMC or when a single best estimate is needed.

### `map-optimize`

```clojure
(mcmc/map-optimize opts model args observations)
```

MAP optimization via gradient ascent on the joint log-density.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | map | Configuration options |
| `model` | generative function | Target model |
| `args` | vector | Model arguments |
| `observations` | choicemap | Observed values |

**Options map:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:iterations` | int | `1000` | Number of gradient steps |
| `:optimizer` | keyword | `:adam` | `:sgd` or `:adam` |
| `:lr` | float | `0.01` | Learning rate |
| `:addresses` | vector | -- | Latent addresses to optimize |
| `:compile?` | bool | `true` | Compile val-grad function |
| `:callback` | fn | `nil` | `(fn [{:iter :score :params}] ...)` |
| `:device` | keyword | `:cpu` | `:cpu` or `:gpu` |

**Returns:** `{:trace Trace, :score number, :params [numbers], :score-history [numbers]}`

- `:trace` -- final model trace at the MAP estimate
- `:score` -- log-density at the MAP estimate
- `:params` -- optimized parameter values
- `:score-history` -- log-density at each iteration

**Example:**
```clojure
(let [result (mcmc/map-optimize {:iterations 500 :lr 0.01
                                  :addresses [:slope :intercept]}
                                 model [xs] obs)]
  (println "MAP score:" (:score result))
  (println "MAP params:" (:params result))
  ;; Use MAP trace as MCMC initialization
  (:trace result))
```

---

### `vectorized-map-optimize`

```clojure
(mcmc/vectorized-map-optimize opts model args observations)
```

Vectorized MAP: N random restarts optimized simultaneously. All N parameter vectors are updated in parallel via the sum trick for gradients. After optimization, returns the best restart (highest score).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:iterations` | int | `1000` | Gradient steps |
| `:optimizer` | keyword | `:adam` | `:sgd` or `:adam` |
| `:lr` | float | `0.01` | Learning rate |
| `:addresses` | vector | -- | Latent addresses |
| `:n-restarts` | int | `10` | Number of parallel random restarts |
| `:callback` | fn | `nil` | `(fn [{:iter :best-score :scores}] ...)` |
| `:device` | keyword | `:gpu` | Default `:gpu` for vectorized ops |

**Returns:** `{:trace Trace, :score number, :params [best], :all-params [N,D], :all-scores [N], :score-history [best-per-iter]}`

**Example:**
```clojure
;; Multi-modal posterior -- use 20 random restarts
(let [result (mcmc/vectorized-map-optimize
               {:iterations 1000 :n-restarts 20
                :addresses [:mu :sigma]}
               model args obs)]
  (println "Best score:" (:score result))
  (println "Best params:" (:params result))
  (println "All scores:" (:all-scores result)))
```

---

## Utilities

### Fused chain graph limits

GenMLX automatically estimates whether a chain can be compiled into a single Metal dispatch based on the method type and total step count. When the estimated graph size exceeds safe limits, fused functions (`fused-mh`, `fused-mala`, `fused-hmc`) fall back to block-compiled paths transparently.

Empirical limits (conservative, leaving headroom below measured failure points):

| Method | GFI score | Tensor-native score |
|--------|-----------|---------------------|
| MH | 2,500 steps | 80,000 steps |
| MALA | 1,500 steps | 50,000 steps |
| HMC | 8,000 steps | 200,000 steps |

### Step-size adaptation

All gradient-based methods (MALA, HMC, NUTS) support automatic step-size tuning via dual averaging (Hoffman and Gelman 2014, Algorithm 5). Enable with `:adapt-step-size true`. The algorithm:

1. Initializes epsilon via `find-reasonable-epsilon` (doubling/halving until ~50% acceptance)
2. Runs warmup steps, adjusting epsilon toward the target acceptance rate
3. Uses the averaged step-size for the sampling phase

Recommended target acceptance rates (defaults):
- **MALA:** 0.574
- **HMC:** 0.65
- **NUTS:** 0.8

### Mass matrix adaptation

HMC and NUTS support diagonal mass matrix estimation via Welford's online algorithm. Enable with `:adapt-metric true`. The mass matrix M is set to 1/Var(q) so that parameters with larger posterior variance receive proportionally larger position updates in leapfrog integration. Requires at least 10 warmup samples before the metric estimate is used.

### Device selection

Scalar MCMC methods default to `:cpu` (faster for single-chain, low-dimensional inference). Vectorized methods default to `:gpu` (parallelism benefits from Metal acceleration). Override with `:device :cpu` or `:device :gpu` in the options map.
