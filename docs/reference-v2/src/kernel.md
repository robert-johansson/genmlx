# Kernels

```clojure
(require '[genmlx.inference.kernel :as k])
```

Composable MCMC transition kernels. A kernel is a function `(fn [trace key] -> trace)` that transforms a trace via some stochastic transition. Kernels can be composed sequentially, repeated, mixed randomly, or cycled -- and the library automatically propagates reversibility metadata through all compositions.

---

## Creating Kernels

### `mh-kernel`

```clojure
(k/mh-kernel selection)
```

Create a Metropolis-Hastings kernel that proposes new values by regenerating the given selection from the prior. The proposal is accepted or rejected according to the MH acceptance ratio. Marked symmetric by default.

| Parameter | Type | Description |
|-----------|------|-------------|
| `selection` | Selection | Address selection specifying which trace sites to regenerate |

**Returns:** A symmetric kernel `(fn [trace key] -> trace)`.

```clojure
(def k (k/mh-kernel (sel/select :slope :intercept)))
```

### `update-kernel`

```clojure
(k/update-kernel constraints)
```

Create a kernel that deterministically updates the trace with fixed constraints. Always accepts (no MH step). Useful for conditioning or resetting specific addresses.

| Parameter | Type | Description |
|-----------|------|-------------|
| `constraints` | Choicemap | Fixed values to set in the trace |

**Returns:** A kernel `(fn [trace key] -> trace)`.

```clojure
(def k (k/update-kernel (cm/choicemap :mu (mx/scalar 0.0))))
```

---

## DSL Constructors

Higher-level kernel constructors for common MCMC patterns.

### `random-walk`

```clojure
(k/random-walk addr std)
(k/random-walk addr-std-map)
```

Gaussian random-walk MH kernel. Proposes `x' = x + N(0, std)` at the given address. Symmetric by default. The multi-address form chains per-address walks sequentially.

| Parameter | Type | Description |
|-----------|------|-------------|
| `addr` | keyword | Trace address to propose on |
| `std` | number | Standard deviation of the Gaussian perturbation |
| `addr-std-map` | map | Map of `{address std}` pairs for multi-address walks |

**Returns:** A symmetric kernel `(fn [trace key] -> trace)`.

```clojure
;; Single address
(def k (k/random-walk :mu 0.5))

;; Multiple addresses (chains per-address walks)
(def k (k/random-walk {:slope 0.3 :intercept 0.3}))
```

### `prior`

```clojure
(k/prior addr)
(k/prior addr1 addr2 ...)
```

MH kernel that proposes new values by resampling from the prior via `regenerate`. Accepts multiple addresses for joint resampling.

| Parameter | Type | Description |
|-----------|------|-------------|
| `addr` | keyword(s) | One or more trace addresses to resample from the prior |

**Returns:** A symmetric kernel `(fn [trace key] -> trace)`.

```clojure
(def k (k/prior :mu))
(def k (k/prior :x :y :z))  ;; joint resample
```

### `proposal`

```clojure
(k/proposal fwd-gf)
(k/proposal fwd-gf :backward bwd-gf)
```

MH kernel with a custom proposal generative function. The proposal takes the current trace's choicemap as its argument and proposes new choices.

In the symmetric form, the proposal is assumed to be its own reverse -- the acceptance ratio uses only the model's update weight. In the asymmetric form, a separate backward proposal is provided for computing the reverse move probability.

| Parameter | Type | Description |
|-----------|------|-------------|
| `fwd-gf` | gen fn | Forward proposal generative function, called as `(fwd-gf current-choices)` |
| `:backward` | gen fn | (optional) Backward proposal for asymmetric MH |

**Returns:** A kernel `(fn [trace key] -> trace)`. Symmetric if no `:backward` is given; asymmetric with reversal metadata otherwise.

```clojure
;; Symmetric proposal
(def my-proposal
  (gen [current-choices]
    (let [cur-mu (cm/get-value (cm/get-submap current-choices :mu))]
      (trace :mu (dist/gaussian (mx/item cur-mu) 0.5)))))

(def k (k/proposal my-proposal))

;; Asymmetric proposal with forward/backward pair
(def k (k/proposal fwd-proposal :backward bwd-proposal))
```

### `gibbs`

```clojure
(k/gibbs addr1 addr2 ...)
(k/gibbs addr-std-map)
```

Convenience constructor for the Gibbs cycling pattern. With keyword arguments, resamples each address from the prior in sequence (equivalent to chaining `prior` kernels). With a map argument, applies random walks with the specified standard deviations.

| Parameter | Type | Description |
|-----------|------|-------------|
| `addr1 addr2 ...` | keywords | Addresses to resample from prior, applied in sequence |
| `addr-std-map` | map | Map of `{address std}` for random-walk Gibbs |

**Returns:** A composed kernel `(fn [trace key] -> trace)`.

```clojure
;; Prior-based Gibbs: resample each from prior in sequence
(def k (k/gibbs :slope :intercept :noise))

;; Random-walk Gibbs: walk each with specified std
(def k (k/gibbs {:slope 0.3 :intercept 0.3}))
```

---

## Composition

Kernels compose algebraically. All composition operators propagate reversibility metadata: if every input kernel has a declared reversal, the composite kernel will too.

### `chain`

```clojure
(k/chain k1 k2 k3 ...)
```

Compose kernels sequentially. Applies `k1`, then `k2`, then `k3`, etc. Each kernel receives a fresh PRNG key from a split of the input key.

If all input kernels have reversals, the composite does too: `reversal(chain(k1, k2, k3)) = chain(reversal(k3), reversal(k2), reversal(k1))`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `k1, k2, ...` | kernels | Two or more kernels to compose |

**Returns:** A composed kernel `(fn [trace key] -> trace)`.

```clojure
(def k (k/chain (k/random-walk :slope 0.3)
                (k/random-walk :intercept 0.3)
                (k/prior :noise)))
```

### `repeat-kernel`

```clojure
(k/repeat-kernel n kernel)
```

Apply the same kernel `n` times sequentially. Each application receives a fresh PRNG key.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | integer | Number of times to apply the kernel |
| `kernel` | kernel | The kernel to repeat |

**Returns:** A composed kernel `(fn [trace key] -> trace)`.

```clojure
(def k (k/repeat-kernel 5 (k/random-walk :mu 0.5)))
```

### `cycle-kernels`

```clojure
(k/cycle-kernels n kernels)
```

Cycle through a vector of kernels in round-robin order for `n` total applications. For example, `(cycle-kernels 10 [k1 k2 k3])` applies `k1, k2, k3, k1, k2, k3, k1, k2, k3, k1`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | integer | Total number of kernel applications |
| `kernels` | vector | Vector of kernels to cycle through |

**Returns:** A composed kernel `(fn [trace key] -> trace)`.

```clojure
(def k (k/cycle-kernels 12 [(k/random-walk :slope 0.3)
                             (k/random-walk :intercept 0.3)
                             (k/prior :noise)]))
;; Applies: slope, intercept, noise, slope, intercept, noise, ...
```

### `mix-kernels`

```clojure
(k/mix-kernels kernel-weights)
```

Randomly select one kernel per step from a weighted collection. At each application, a kernel is sampled according to the given weights, and that kernel is applied.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel-weights` | vector | Vector of `[kernel weight]` pairs. Weights need not sum to 1 (normalized internally). |

**Returns:** A mixed kernel `(fn [trace key] -> trace)`.

```clojure
(def k (k/mix-kernels [[(k/random-walk :mu 0.5) 0.7]
                        [(k/prior :mu)           0.3]]))
```

---

## Reversibility

Reversibility metadata enables algorithms that require detailed balance verification or involutive MCMC constructions. The library tracks reversals through metadata on kernel functions.

### `with-reversal`

```clojure
(k/with-reversal kernel reverse-kernel)
```

Declare that `reverse-kernel` is the reversal of `kernel`. Sets metadata on both directions so that `(reversal (reversal k))` returns `k`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel` | kernel | The forward kernel |
| `reverse-kernel` | kernel | Its reversal |

**Returns:** The forward kernel with reversal metadata attached.

### `symmetric-kernel`

```clojure
(k/symmetric-kernel kernel)
```

Declare a kernel as symmetric (its own reversal). This means the forward and backward transition probabilities are equal.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel` | kernel | A kernel that is its own reversal |

**Returns:** The kernel with symmetric metadata attached.

### `reversal`

```clojure
(k/reversal kernel)
```

Get the declared reversal of a kernel. Returns the reverse kernel, the kernel itself if symmetric, or `nil` if no reversal has been declared.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel` | kernel | A kernel to query |

**Returns:** The reverse kernel, or `nil`.

### `symmetric?`

```clojure
(k/symmetric? kernel)
```

Check if a kernel is declared symmetric.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel` | kernel | A kernel to query |

**Returns:** `true` if the kernel is symmetric, `false` otherwise.

### `reversed`

```clojure
(k/reversed kernel)
```

Return the reversal of a kernel. Unlike `reversal`, this throws an exception if no reversal has been declared.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel` | kernel | A kernel with a declared reversal |

**Returns:** The reverse kernel. Throws if no reversal exists.

---

## Execution

### `run-kernel`

```clojure
(k/run-kernel opts kernel init-trace)
```

Run a kernel for multiple iterations with burn-in and thinning. Returns a vector of traces with acceptance rate metadata.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | map | Options (see below) |
| `kernel` | kernel | The kernel to run |
| `init-trace` | Trace | Starting trace (typically from `p/generate`) |

**Options map:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `:samples` | integer | required | Number of post-burn-in samples to collect |
| `:burn` | integer | `0` | Number of burn-in iterations to discard |
| `:thin` | integer | `1` | Keep every nth sample after burn-in |
| `:callback` | function | `nil` | Called with `{:iter :trace :accepted?}` for each kept sample |
| `:key` | PRNG key | `nil` | Optional PRNG key for reproducibility |

**Returns:** A vector of `Trace` values. The vector carries `{:acceptance-rate rate}` as metadata, accessible via `(meta traces)`.

```clojure
(let [{:keys [trace]} (p/generate model [xs] observations)
      k      (k/random-walk :mu 0.5)
      traces (k/run-kernel {:samples 1000 :burn 200 :thin 2} k trace)
      rate   (:acceptance-rate (meta traces))]
  (println "Acceptance rate:" rate)
  (println "Collected:" (count traces) "samples"))
```

### `collect-samples`

```clojure
(k/collect-samples opts step-fn extract-fn init-state)
```

Generic sample collection loop with burn-in, thinning, and callback. This is the lower-level primitive underlying `run-kernel` -- use it when your state is not a trace or your step function has custom structure.

| Parameter | Type | Description |
|-----------|------|-------------|
| `opts` | map | Same options as `run-kernel` (`:samples`, `:burn`, `:thin`, `:callback`, `:key`) |
| `step-fn` | function | `(fn [state key] -> {:state new-state :accepted? bool})` |
| `extract-fn` | function | `(fn [state] -> sample)` to extract a sample from the state |
| `init-state` | any | Initial state passed to `step-fn` |

**Returns:** A vector of samples with `{:acceptance-rate rate}` metadata.

### `seed`

```clojure
(k/seed kernel fixed-key)
```

Fix the PRNG key for a kernel. Every call to the resulting kernel uses the same key, making it deterministic. Useful for testing or creating reproducible transitions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel` | kernel | The kernel to seed |
| `fixed-key` | PRNG key | The fixed key to use on every call |

**Returns:** A deterministic kernel `(fn [trace _key] -> trace)`.

```clojure
(def k (k/seed (k/mh-kernel (sel/select :mu)) (rng/fresh-key 42)))
```

---

## Examples

### MCMC with composed kernels

```clojure
(require '[genmlx.inference.kernel :as k]
         '[genmlx.protocols :as p]
         '[genmlx.selection :as sel])

;; Initialize from observations
(let [{:keys [trace]} (p/generate model [xs] observations)]

  ;; Chain: random walk on slope, then intercept, then resample noise from prior
  (let [kernel (k/chain (k/random-walk :slope 0.3)
                        (k/random-walk :intercept 0.3)
                        (k/prior :noise))
        traces (k/run-kernel {:samples 2000 :burn 500 :thin 2} kernel trace)]
    (println "Acceptance rate:" (:acceptance-rate (meta traces)))
    (println "Samples:" (count traces))))
```

### Mixing exploration strategies

```clojure
;; 70% random walk, 30% prior resample -- combines local and global moves
(def kernel
  (k/mix-kernels [[(k/random-walk :mu 0.5) 0.7]
                   [(k/prior :mu)           0.3]]))

(def traces (k/run-kernel {:samples 1000 :burn 200} kernel init-trace))
```

### Cycling through parameters

```clojure
;; Cycle through 3 parameters, 12 total kernel applications per sample
(def kernel
  (k/cycle-kernels 12 [(k/random-walk :x 0.3)
                        (k/random-walk :y 0.3)
                        (k/random-walk :z 0.3)]))

(def traces (k/run-kernel {:samples 500 :burn 100} kernel init-trace))
```
