# SMCP3 -- SMC with Probabilistic Program Proposals

Source: `src/genmlx/inference/smcp3.cljs`

SMCP3 (Sequential Monte Carlo with Probabilistic Program Proposals) replaces the
prior as the SMC proposal distribution with custom generative functions. Instead
of extending particles by sampling from the model's prior, SMCP3 uses learned or
hand-crafted proposal programs that can be locally optimal, neural, or
problem-specific. This dramatically improves particle efficiency in models where
the prior is a poor proposal.

The key idea: at each timestep, a **forward kernel** (a generative function)
proposes how to extend each particle, and an optional **backward kernel** computes
the reverse probability for proper importance weight correction. When both kernels
are provided, the incremental weight is:

```
w_t = update_weight + backward_score - forward_score
```

Without a backward kernel, SMCP3 falls back to standard constraint-based update
(equivalent to bootstrap SMC).

**When to use SMCP3:**
- The model's prior is a poor match for the posterior (high-dimensional,
  multimodal, or informative observations)
- You have domain knowledge to write a better proposal
- You want to use neural networks as learned proposals
- Standard SMC requires too many particles for acceptable accuracy

**Namespace alias used below:**
```clojure
(require '[genmlx.inference.smcp3 :as smcp3])
```

---

## smcp3

### `smcp3`

```clojure
(smcp3/smcp3 opts model args observations-seq)
```

Full SMCP3 loop: initialize particles (optionally with a custom proposal),
then iterate through the observation sequence using forward/backward kernels
for proposal-based extension.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | Map | Options map (see below) |
| `model` | DynamicGF | Target generative function |
| `args` | Vector | Arguments to the model |
| `observations-seq` | Seq of ChoiceMaps | Observations for each timestep |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:particles` | Integer | `100` | Number of particles |
| `:ess-threshold` | Number | `0.5` | Resample when ESS/N falls below this ratio |
| `:forward-kernel` | DynamicGF or nil | `nil` | Proposal GF for extending particles |
| `:backward-kernel` | DynamicGF or nil | `nil` | Backward kernel for weight correction |
| `:init-proposal` | DynamicGF or nil | `nil` | Proposal GF for initial particles (instead of prior) |
| `:rejuvenation-fn` | Function or nil | `nil` | `(fn [trace key] -> trace)` for MCMC rejuvenation |
| `:callback` | Function or nil | `nil` | Called each step with `{:step :ess :resampled?}` |
| `:key` | MLX array | fresh | PRNG key for reproducibility |

**Returns:** `{:traces [Trace ...] :log-weights [MLX-scalar ...] :log-ml-estimate MLX-scalar}`

- `:traces` -- final particle traces after all timesteps
- `:log-weights` -- final unnormalized log importance weights
- `:log-ml-estimate` -- accumulated log marginal likelihood estimate

**Example -- standard SMC (no custom proposals):**

```clojure
;; Without forward/backward kernels, smcp3 behaves like standard bootstrap SMC
(let [{:keys [log-ml-estimate]}
      (smcp3/smcp3 {:particles 500} model args obs-seq)]
  (println "Log-ML:" (mx/item log-ml-estimate)))
```

**Example -- with custom proposals:**

```clojure
;; Define a proposal that uses the observation to propose latent states
(def forward-proposal
  (gen [current-choices]
    ;; Propose new latent state informed by observation
    (trace :z (dist/gaussian obs-informed-mean obs-informed-std))))

(def backward-proposal
  (gen [current-choices]
    ;; Score the reverse move
    (trace :z (dist/gaussian prior-mean prior-std))))

(let [{:keys [traces log-ml-estimate]}
      (smcp3/smcp3 {:particles 200
                     :forward-kernel forward-proposal
                     :backward-kernel backward-proposal}
                    model args obs-seq)]
  (println "Log-ML:" (mx/item log-ml-estimate)))
```

**Example -- with init proposal and rejuvenation:**

```clojure
(def init-proposal
  (gen []
    ;; Data-informed initialization
    (trace :slope (dist/gaussian data-mean 1.0))
    (trace :intercept (dist/gaussian 0 2.0))))

(defn rejuvenate [trace key]
  (let [{:keys [trace weight]} (p/regenerate (:gen-fn trace) trace (sel/select :slope))]
    (if (> (mx/item weight) (js/Math.log (rand)))
      trace trace)))

(smcp3/smcp3 {:particles 300
               :init-proposal init-proposal
               :rejuvenation-fn rejuvenate
               :callback (fn [{:keys [step ess]}]
                           (println "Step" step "ESS:" ess))}
              model args obs-seq)
```

---

## Configuration and Options

### Kernel design

The **forward kernel** is a generative function that takes `[current-trace-choices]`
and proposes extensions. It should sample at addresses that will be constrained
in the next observation step.

The **backward kernel** is a generative function that scores the reverse proposal.
Together, they ensure proper importance weight computation:

```
incremental_weight = update_weight + backward_score - forward_score
```

If only the forward kernel is provided (backward is nil), SMCP3 falls back to
constraint-based update (`p/update`) and the forward kernel is **not used**.
Both kernels must be provided for proposal-based extension.

### Init proposal

The `:init-proposal` is a generative function taking `[]` that proposes initial
choices for the first timestep. Its choices are merged with the first
observation, and the importance weight accounts for the proposal score.

### ESS-based resampling

Like standard SMC, SMCP3 monitors the effective sample size and resamples
(systematic resampling) when ESS/N drops below `:ess-threshold`. Resampling
resets all weights to zero (uniform).

### Memory management

SMCP3 sweeps dead arrays and clears the MLX cache every 10 steps to manage
Metal buffer pressure during long sequences.

---

## Low-level API

For building custom SMCP3 loops, the init and step functions are available
directly.

### `smcp3-init`

```clojure
(smcp3/smcp3-init model args observations proposal-gf particles key)
```

Initialize SMCP3 particles. If `proposal-gf` is provided, uses it to propose
initial choices (merged with observations) and corrects the importance weight
by subtracting the proposal score. If nil, falls back to standard importance
sampling from the prior.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | DynamicGF | Target model |
| `args` | Vector | Model arguments |
| `observations` | ChoiceMap | First timestep observations |
| `proposal-gf` | DynamicGF or nil | Optional proposal for initial particles |
| `particles` | Integer | Number of particles |
| `key` | MLX array | PRNG key |

**Returns:** `{:traces [Trace ...] :log-weights [MLX-scalar ...] :log-ml-increment MLX-scalar}`

---

### `smcp3-step`

```clojure
(smcp3/smcp3-step traces log-weights model observations
                   forward-kernel backward-kernel
                   particles ess-threshold rejuvenation-fn key)
```

One SMCP3 step: check ESS and resample if needed, extend each particle
using the forward/backward kernels (or fall back to `p/update`), accumulate
weights, and optionally rejuvenate.

| Parameter | Type | Description |
|-----------|------|-------------|
| `traces` | Vector of Traces | Current particle traces |
| `log-weights` | Vector of MLX scalars | Current particle log-weights |
| `model` | DynamicGF | Target model |
| `observations` | ChoiceMap | New observations for this timestep |
| `forward-kernel` | DynamicGF or nil | Proposal GF for extending particles |
| `backward-kernel` | DynamicGF or nil | Backward kernel for weight correction |
| `particles` | Integer | Number of particles |
| `ess-threshold` | Number | Resample when ESS/N falls below this |
| `rejuvenation-fn` | Function or nil | `(fn [trace key] -> trace)` |
| `key` | MLX array or nil | PRNG key |

**Returns:** `{:traces [Trace ...] :log-weights [MLX-scalar ...] :log-ml-increment MLX-scalar :ess number :resampled? boolean}`

**Weight semantics:**

| Kernels provided | Weight computation |
|------------------|--------------------|
| Both forward and backward | `update_weight + backward_score - forward_score` |
| Forward only (no backward) | Falls back to constraint-based `p/update`; forward kernel is not used |
| Neither | Standard `p/update` weight (bootstrap SMC) |
