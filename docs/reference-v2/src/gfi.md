# GFI Protocols

The **Generative Function Interface (GFI)** is the core abstraction of GenMLX. Every generative function -- whether built with `gen`, a combinator, or `nn->gen-fn` -- implements these protocols. The GFI defines six core operations (simulate, generate, assess, update, regenerate, project) plus three extension protocols (edit, batched splice, argument gradients, update with diffs).

Source: `src/genmlx/protocols.cljs`

---

## GFI Protocols

The protocol definitions live in `genmlx.protocols`. In code, they are accessed via the `p/` alias:

```clojure
(require '[genmlx.protocols :as p])
```

### `simulate`

```clojure
(p/simulate gf args)
```

Forward-sample all random choices from their prior distributions. No observations are provided -- every `trace` site in the model body samples freely. The joint log-probability of all sampled choices is accumulated in the trace's `:score`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | Generative function | The model to execute |
| `args` | Vector | Arguments passed to the model body |

**Returns:** A [`Trace`](trace.md) record with `{:gen-fn :args :choices :retval :score}`.

**Example:**
```clojure
(def model
  (gen [x]
    (let [slope (trace :slope (dist/gaussian 0 10))]
      (trace :y (dist/gaussian (mx/multiply slope x) 1))
      slope)))

(let [tr (p/simulate model [(mx/scalar 2.0)])]
  (println "slope:" (mx/item (cm/get-value (:choices tr) :slope)))
  (println "score:" (mx/item (:score tr))))
```

---

### `generate`

```clojure
(p/generate gf args constraints)
```

Execute the model with some choices constrained to observed values. Unconstrained choices are sampled from the prior. Returns a trace and an importance weight.

The weight is \\(\log p(\text{constraints} \mid \text{args})\\) -- specifically, the sum of log-probabilities at constrained addresses. For a fully constrained model, this equals the joint log-probability.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | Generative function | The model to execute |
| `args` | Vector | Arguments passed to the model body |
| `constraints` | [ChoiceMap](choicemap.md) | Observed values at specific addresses |

**Returns:** `{:trace Trace, :weight MLX-scalar}`

**Example:**
```clojure
(let [obs (cm/choicemap {:y (mx/scalar 3.5)})
      {:keys [trace weight]} (p/generate model [(mx/scalar 2.0)] obs)]
  (println "sampled slope:" (mx/item (cm/get-value (:choices trace) :slope)))
  (println "log-weight:" (mx/item weight)))
```

---

### `assess`

```clojure
(p/assess gf args choices)
```

Score fully-specified choices without any sampling. Every address visited by the model body must be present in the choice map. Throws an error if any address is missing.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | Generative function | The model |
| `args` | Vector | Model arguments |
| `choices` | [ChoiceMap](choicemap.md) | Complete choice map (all addresses) |

**Returns:** `{:retval any, :weight MLX-scalar}` where `:weight` is \\(\log p(\text{choices} \mid \text{args})\\).

**Example:**
```clojure
(let [choices (cm/choicemap {:slope (mx/scalar 1.5) :y (mx/scalar 4.0)})
      {:keys [weight]} (p/assess model [(mx/scalar 2.0)] choices)]
  (println "log-joint:" (mx/item weight)))
```

---

### `update`

```clojure
(p/update gf trace constraints)
```

Update an existing trace with new constraints. At each address:
- If a new constraint is provided, use it (the old value goes to the discard).
- If no new constraint and the address existed before, keep the old value.
- If the address is entirely new, sample from the prior.

Returns the updated trace, an incremental weight, and the discarded old values.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | Generative function | The model |
| `trace` | [Trace](trace.md) | Previous execution trace |
| `constraints` | [ChoiceMap](choicemap.md) | New constraints to apply |

**Returns:** `{:trace Trace, :weight MLX-scalar, :discard ChoiceMap}`

**Example:**
```clojure
(let [tr (p/simulate model [(mx/scalar 2.0)])
      new-obs (cm/choicemap {:slope (mx/scalar 0.5)})
      {:keys [trace weight discard]} (p/update model tr new-obs)]
  (println "new slope:" (mx/item (cm/get-value (:choices trace) :slope)))
  (println "incremental weight:" (mx/item weight))
  (println "old slope:" (mx/item (cm/get-value discard :slope))))
```

---

### `regenerate`

```clojure
(p/regenerate gf trace selection)
```

Resample the selected addresses from the prior while keeping unselected choices fixed. The weight is the MH acceptance ratio:

\\[
w = \log p_{\text{new}}(\text{all choices}) - \log p_{\text{old}}(\text{all choices}) - \log q(\text{new selected} \mid \text{new unselected})
\\]

Used as the default proposal in Metropolis-Hastings inference.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | Generative function | The model |
| `trace` | [Trace](trace.md) | Previous execution trace |
| `selection` | [Selection](selection.md) | Addresses to resample |

**Returns:** `{:trace Trace, :weight MLX-scalar}`

**Example:**
```clojure
(let [tr (p/simulate model [(mx/scalar 2.0)])
      sel (sel/select :slope)
      {:keys [trace weight]} (p/regenerate model tr sel)]
  (println "resampled slope:" (mx/item (cm/get-value (:choices trace) :slope)))
  (println "MH weight:" (mx/item weight)))
```

---

### `propose`

```clojure
(p/propose gf args)
```

Forward-sample all choices and return them as a choice map together with their joint log-probability. No observations are involved. This is used for proposal distributions in inference algorithms like SMCP3.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | Generative function | The proposal model |
| `args` | Vector | Arguments passed to the model body |

**Returns:** `{:choices ChoiceMap, :weight MLX-scalar, :retval any}`

**Example:**
```clojure
(let [{:keys [choices weight]} (p/propose model [(mx/scalar 2.0)])]
  (println "proposed slope:" (mx/item (cm/get-value choices :slope)))
  (println "log-joint:" (mx/item weight)))
```

---

### `project`

```clojure
(p/project gf trace selection)
```

Compute the log-probability of the selected choices within a trace, without resampling anything. Replays all values from the existing trace and accumulates log-probabilities only at selected addresses.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | Generative function | The model |
| `trace` | [Trace](trace.md) | Execution trace to evaluate |
| `selection` | [Selection](selection.md) | Addresses to compute log-prob for |

**Returns:** MLX scalar -- the log-probability of the selected choices.

**Example:**
```clojure
(let [tr (p/simulate model [(mx/scalar 2.0)])
      sel (sel/select :slope)
      log-prob (p/project model tr sel)]
  (println "log p(slope):" (mx/item log-prob)))
```

---

### `edit`

```clojure
(p/edit gf trace edit-request)
```

Apply a parametric edit request to a trace. This generalizes `update`, `regenerate`, and proposal-based trace modifications into a single interface. The backward request in the return value enables automatic computation of acceptance weights for reversible kernels -- the foundation of SMCP3.

Source: `src/genmlx/edit.cljs`

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | Generative function | The model |
| `trace` | [Trace](trace.md) | Previous execution trace |
| `edit-request` | `EditRequest` | One of `ConstraintEdit`, `SelectionEdit`, or `ProposalEdit` |

**Returns:** `{:trace Trace, :weight MLX-scalar, :discard ChoiceMap, :backward-request EditRequest}`

There are three edit request types:

| Type | Constructor | Equivalent to |
|------|-------------|---------------|
| `ConstraintEdit` | `(constraint-edit constraints)` | `update` -- change observed values |
| `SelectionEdit` | `(selection-edit selection)` | `regenerate` -- resample selected addresses |
| `ProposalEdit` | `(proposal-edit fwd-gf bwd-gf)` | Reversible kernel for SMCP3-style proposals |

**Example:**
```clojure
(require '[genmlx.edit :as edit])

;; Constraint edit (equivalent to update)
(let [tr (p/simulate model [(mx/scalar 2.0)])
      req (edit/constraint-edit (cm/choicemap {:slope (mx/scalar 1.0)}))
      {:keys [trace weight backward-request]} (p/edit model tr req)]
  (println "new slope:" (mx/item (cm/get-value (:choices trace) :slope)))
  ;; backward-request is a ConstraintEdit with the old discarded values
  (println "backward type:" (type backward-request)))
```

---

### `update-with-diffs`

```clojure
(p/update-with-diffs gf trace constraints argdiffs)
```

Like `update` but with change hints (diffs) describing which arguments changed. This enables combinators to skip unchanged sub-computations during incremental re-execution. When arguments have not changed and constraints are empty, returns the original trace with zero weight.

Source: `src/genmlx/protocols.cljs` (protocol), `src/genmlx/dynamic.cljs` (implementation)

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | Generative function | The model |
| `trace` | [Trace](trace.md) | Previous execution trace |
| `constraints` | [ChoiceMap](choicemap.md) | New constraints to apply |
| `argdiffs` | Diff | Change hints from `genmlx.diff` describing which args changed |

**Returns:** `{:trace Trace, :weight MLX-scalar, :discard ChoiceMap}`

**Example:**
```clojure
(require '[genmlx.diff :as diff])

;; No changes -- fast path returns original trace
(let [tr (p/simulate model [(mx/scalar 2.0)])
      {:keys [trace weight]} (p/update-with-diffs model tr cm/EMPTY (diff/no-change))]
  (println "same trace?:" (= (:choices trace) (:choices tr)))
  (println "weight:" (mx/item weight)))
;; => same trace?: true, weight: 0.0
```

---

### `has-argument-grads`

```clojure
(p/has-argument-grads gf)
```

Returns a vector of booleans indicating which argument positions are differentiable, or `nil` if unknown. Used by gradient-based inference algorithms to determine which arguments can be differentiated through.

DynamicGF returns `nil` (unknown). Custom generative functions and neural network gen-fns may return specific vectors.

Source: `src/genmlx/protocols.cljs`

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | Generative function | The generative function to query |

**Returns:** Vector of booleans (e.g. `[true false true]`) or `nil`.

**Example:**
```clojure
(p/has-argument-grads model)
;; => nil  (DynamicGF does not declare argument differentiability)
```

---

### `batched-splice`

```clojure
(p/batched-splice gf state addr args)
```

Execute a combinator in batched mode within a parent handler. This protocol is used during vectorized execution when a parent gen-fn splices into a sub-generative-function. Combinators that implement this protocol can run natively in batched mode without falling back to per-particle scalar execution.

Source: `src/genmlx/protocols.cljs`

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | Generative function | The combinator being spliced into |
| `state` | Map | Parent handler state (with `:batched? true`, `:batch-size N`) |
| `addr` | Keyword | The splice address in the parent |
| `args` | Vector | Arguments to this combinator |

**Returns:** `[state' retval]` -- updated parent handler state and the return value.

---

## gen Macro

Source: `src/genmlx/gen.cljc`

### `gen`

```clojure
(gen [params ...] body ...)
```

Define a generative function. The `gen` macro transforms a function body into a `DynamicGF` that implements the full GFI. Inside the body, three local bindings are available:

- **`trace`** -- sample from (or constrain to) a distribution at an address
- **`splice`** -- call a sub-generative-function at a namespaced address
- **`param`** -- declare or read a trainable parameter

The return value of the body becomes the trace's `:retval`.

The macro injects a hidden runtime parameter as the first argument and binds `trace`, `splice`, `param` as local names from the runtime object. Because these are local bindings (not namespace-qualified calls), they work naturally with all Clojure constructs: `map`, `for`, higher-order functions, closures.

| Parameter | Type | Description |
|-----------|------|-------------|
| `params` | Vector | Parameter list for the model (like `defn`) |
| `body` | Forms | Model body with `trace`, `splice`, `param` calls |

**Returns:** A `DynamicGF` record implementing the full GFI.

**Example:**
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

### Inside `gen` bodies

#### `trace`

```clojure
(trace addr dist)
```

Sample from (or constrain to) a distribution at the given address. The behavior depends on which handler is active:

- **simulate:** sample from the prior
- **generate:** use the constrained value if present, otherwise sample
- **update:** apply new constraint, keep old value, or sample fresh
- **regenerate:** resample if selected, keep old value otherwise
- **assess:** use the constrained value (error if missing)

| Parameter | Type | Description |
|-----------|------|-------------|
| `addr` | Keyword | Address for this random choice (e.g. `:slope`) |
| `dist` | Distribution | Distribution to sample from |

**Returns:** The sampled (or constrained) value as an MLX array.

#### `splice`

```clojure
(splice addr gf & args)
```

Call a sub-generative-function at the given address. The sub-function's choices are nested under `addr` in the trace's choice map. This is how models compose hierarchically.

| Parameter | Type | Description |
|-----------|------|-------------|
| `addr` | Keyword | Namespace address for the sub-call |
| `gf` | Generative function | Sub-model to invoke |
| `args` | Any | Arguments passed to the sub-model |

**Returns:** The sub-model's return value.

**Example:**
```clojure
(def sub-model
  (gen [mu]
    (trace :x (dist/gaussian mu 1))))

(def parent-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (splice :sub sub-model mu))))

;; Choices: {:mu 1.2, :sub {:x 0.8}}
```

#### `param`

```clojure
(param name default-value)
```

Declare or read a trainable parameter. If a parameter store is bound (during learning), reads from it; otherwise returns the default value as an MLX array. Used for learnable model parameters in variational inference and amortized inference.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | Keyword | Parameter name |
| `default-value` | Number or MLX array | Default value when no param store is bound |

**Returns:** The parameter value as an MLX array.

---

## Dynamic DSL

Source: `src/genmlx/dynamic.cljs`

The `DynamicGF` record is the concrete type returned by the `gen` macro. It holds the body function, the source form, and the extracted schema. All GFI protocol implementations live on this record.

### `make-gen-fn`

```clojure
(dyn/make-gen-fn body-fn source)
```

Create a `DynamicGF` from a body function and its quoted source form. This is the function the `gen` macro expands into. It extracts the schema from the source form and attempts Level 1 compilation for eligible models:
- Static models get full compiled paths (L1-M2).
- Branch models get branch-rewritten paths (L1-M4).
- Other models get compiled prefix paths (L1-M3).

| Parameter | Type | Description |
|-----------|------|-------------|
| `body-fn` | Function | The transformed model body (with runtime parameter) |
| `source` | List | Quoted source form `(params & body)` |

**Returns:** A `DynamicGF` record.

---

### `with-key`

```clojure
(dyn/with-key gf key)
```

Return a copy of the generative function with a specific PRNG key attached as metadata. The key is consumed by GFI methods for reproducible execution. Inference entry points manage keys automatically; use `with-key` when calling GFI methods directly and you need reproducibility.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | Generative function | The gen-fn to attach a key to |
| `key` | MLX array | PRNG key from `rng/key` or `rng/split` |

**Returns:** The same gen-fn with the key stored as metadata.

**Example:**
```clojure
(require '[genmlx.mlx.random :as rng])

(let [k (rng/key 42)
      model-k (dyn/with-key model k)
      tr1 (p/simulate model-k [(mx/scalar 1.0)])
      tr2 (p/simulate model-k [(mx/scalar 1.0)])]
  ;; Same key => same trace
  (= (:choices tr1) (:choices tr2)))
;; => true
```

---

### `auto-key`

```clojure
(dyn/auto-key gf)
```

Mark a generative function to auto-generate fresh PRNG keys for each GFI call. Each call to `simulate`, `generate`, etc. gets a fresh key, so repeated calls produce different results. For REPL use, tests, and interactive exploration. Inference entry points manage keys automatically.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | Generative function | The gen-fn to mark |

**Returns:** The same gen-fn with an auto-key sentinel as metadata.

**Example:**
```clojure
(let [m (dyn/auto-key model)
      tr1 (p/simulate m [(mx/scalar 1.0)])
      tr2 (p/simulate m [(mx/scalar 1.0)])]
  ;; Different keys => different traces
  (not= (:choices tr1) (:choices tr2)))
;; => true
```

---

### `call`

```clojure
(dyn/call gf & args)
```

Call a generative function as a regular function: simulate and return the value. Auto-keys the gen-fn for convenience so you do not need to manage PRNG keys.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | Generative function | The model to call |
| `args` | Any | Arguments passed to the model body |

**Returns:** The model's return value (`:retval` from the trace).

**Example:**
```clojure
(let [slope (dyn/call model (mx/scalar 2.0))]
  (println "slope:" (mx/item slope)))
```

---

### `vsimulate`

```clojure
(dyn/vsimulate gf args n key)
```

Run the model body **once** with a batched handler that samples `[n]`-shaped arrays at each trace site. All downstream arithmetic (log-prob, score accumulation) broadcasts naturally via MLX. Returns a `VectorizedTrace` where every field has an `[n]` batch dimension.

No `splice` is supported in shape-based batched mode. Use `vmap-gf` for models with splice.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | DynamicGF | The model (must be a DynamicGF) |
| `args` | Vector | Model arguments |
| `n` | Integer | Number of particles |
| `key` | MLX array | PRNG key |

**Returns:** `VectorizedTrace`

**Example:**
```clojure
(let [k (rng/key 0)
      vtr (dyn/vsimulate model [(mx/scalar 2.0)] 1000 k)]
  (println "particles:" (:n-particles vtr))
  (println "score shape:" (mx/shape (:score vtr))))
;; => particles: 1000
;; => score shape: [1000]
```

---

### `vgenerate`

```clojure
(dyn/vgenerate gf args constraints n key)
```

Batched `generate`. Constrained sites use the scalar observation (broadcast to all particles via MLX broadcasting); unconstrained sites sample `[n]` independent values. Returns a `VectorizedTrace` with `[n]`-shaped weights.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | DynamicGF | The model |
| `args` | Vector | Model arguments |
| `constraints` | [ChoiceMap](choicemap.md) | Observed values (scalar, broadcast to all particles) |
| `n` | Integer | Number of particles |
| `key` | MLX array | PRNG key |

**Returns:** `VectorizedTrace` with `[n]`-shaped weights.

**Example:**
```clojure
(let [k (rng/key 0)
      obs (cm/choicemap {:y (mx/scalar 3.5)})
      vtr (dyn/vgenerate model [(mx/scalar 2.0)] obs 1000 k)]
  (println "weight shape:" (mx/shape (:weight vtr))))
;; => weight shape: [1000]
```

---

### `vupdate`

```clojure
(dyn/vupdate gf vtrace constraints key)
```

Batched `update` on a `VectorizedTrace`. Runs the model body once with a batched update handler. Returns a map with the new `VectorizedTrace`, weights, and discarded values.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | DynamicGF | The model |
| `vtrace` | VectorizedTrace | Previous batched trace with `[n]`-shaped choices |
| `constraints` | [ChoiceMap](choicemap.md) | New constraints to apply |
| `key` | MLX array | PRNG key |

**Returns:** `{:vtrace VectorizedTrace, :weight MLX-array, :discard ChoiceMap}`

---

### `vregenerate`

```clojure
(dyn/vregenerate gf vtrace selection key)
```

Batched `regenerate` on a `VectorizedTrace`. Resamples selected addresses for all particles simultaneously. Returns a map with the new `VectorizedTrace` and the MH weight.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | DynamicGF | The model |
| `vtrace` | VectorizedTrace | Previous batched trace with `[n]`-shaped choices |
| `selection` | [Selection](selection.md) | Addresses to resample |
| `key` | MLX array | PRNG key |

**Returns:** `{:vtrace VectorizedTrace, :weight MLX-array}`

---

### `loop-obs`

```clojure
(dyn/loop-obs prefix values)
```

Create flat constraints from a prefix string and a sequence of values. A convenience for building observation choicemaps for loop-based models.

| Parameter | Type | Description |
|-----------|------|-------------|
| `prefix` | String | Address prefix (e.g. `"y"`) |
| `values` | Sequence | Values to assign |

**Returns:** ChoiceMap with addresses `:y0`, `:y1`, `:y2`, ...

**Example:**
```clojure
(dyn/loop-obs "y" [1.0 2.0 3.0])
;; => choicemap with {:y0 1.0, :y1 2.0, :y2 3.0}
```

---

### `merge-obs`

```clojure
(dyn/merge-obs & cms)
```

Merge multiple choicemaps into one. A convenience for combining observation sets.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cms` | ChoiceMaps | Any number of choicemaps to merge |

**Returns:** A single merged ChoiceMap.

**Example:**
```clojure
(dyn/merge-obs
  (cm/choicemap {:slope (mx/scalar 1.0)})
  (dyn/loop-obs "y" [2.0 3.0 4.0]))
;; => {:slope 1.0, :y0 2.0, :y1 3.0, :y2 4.0}
```

---

### `param` (standalone)

```clojure
(dyn/param name default-value)
```

Read a trainable parameter outside a gen body. Returns the default value as an MLX array (no param store is available outside gen body execution). Inside gen bodies, use the `param` local binding from the `gen` macro instead.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | Keyword | Parameter name |
| `default-value` | Number or MLX array | Default value |

**Returns:** MLX array.

---

## Handler System

Source: `src/genmlx/handler.cljs`, `src/genmlx/runtime.cljs`

The handler system is the heart of GenMLX. When `gen` body code calls `trace`, it dispatches to whichever handler is active. Each handler is a **pure state transition**:

```clojure
(fn [state addr dist] -> [value state'])
```

State flows through an immutable map with mode-specific keys:

| Mode | State keys |
|------|------------|
| simulate | `:key` `:choices` `:score` `:executor` |
| generate | `:key` `:choices` `:score` `:weight` `:constraints` `:executor` |
| update | `:key` `:choices` `:score` `:weight` `:constraints` `:old-choices` `:discard` `:executor` |
| regenerate | `:key` `:choices` `:score` `:weight` `:old-choices` `:selection` `:executor` |
| project | `:key` `:choices` `:score` `:weight` `:old-choices` `:selection` `:constraints` `:executor` |

Batched variants add `:batch-size` (integer N) and `:batched?` (true). All other keys and semantics are identical -- MLX broadcasting handles the shape difference between scalar and `[N]`-shaped values.

The handler never inspects value shapes. This is what makes batched execution (`[N]`-shaped arrays) work transparently.

### Scalar transitions

### `simulate-transition`

```clojure
(h/simulate-transition state addr dist)
```

Sample from the prior at `addr`, accumulate log-prob into `:score`. Splits the PRNG key, samples a value, computes `log-prob(dist, value)`, and threads the updated state.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | Map | Handler state with `:key`, `:choices`, `:score` |
| `addr` | Keyword | Address for this choice |
| `dist` | Distribution | Distribution to sample from |

**Returns:** `[value state']`

---

### `generate-transition`

```clojure
(h/generate-transition state addr dist)
```

If `addr` is constrained, use the constraint and add its log-prob to both `:score` and `:weight`. Otherwise delegate to `simulate-transition`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | Map | Handler state with `:constraints` |
| `addr` | Keyword | Address for this choice |
| `dist` | Distribution | Distribution to sample from |

**Returns:** `[value state']`

---

### `assess-transition`

```clojure
(h/assess-transition state addr dist)
```

All addresses must be constrained. Throws `ex-info` if `addr` is not found in `:constraints`. Adds the log-prob to both `:score` and `:weight`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | Map | Handler state with `:constraints` |
| `addr` | Keyword | Address for this choice |
| `dist` | Distribution | Distribution to score |

**Returns:** `[value state']`

---

### `update-transition`

```clojure
(h/update-transition state addr dist)
```

Three cases: (1) new constraint provided -- use it, compute weight as difference of new and old log-probs, discard old value; (2) old value exists -- keep it; (3) address is new -- sample fresh from the prior.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | Map | Handler state with `:constraints`, `:old-choices`, `:discard` |
| `addr` | Keyword | Address for this choice |
| `dist` | Distribution | Distribution to evaluate |

**Returns:** `[value state']`

---

### `regenerate-transition`

```clojure
(h/regenerate-transition state addr dist)
```

If `addr` is selected, resample from the prior and compute the weight adjustment (new log-prob minus old log-prob). If not selected, keep the old value. Throws if the address was never sampled before.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | Map | Handler state with `:selection`, `:old-choices` |
| `addr` | Keyword | Address for this choice |
| `dist` | Distribution | Distribution to resample from |

**Returns:** `[value state']`

---

### `project-transition`

```clojure
(h/project-transition state addr dist)
```

Replay the old value (no sampling). If `addr` is selected, add its log-prob to `:weight`. Always add to `:score`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | Map | Handler state with `:selection`, `:old-choices` |
| `addr` | Keyword | Address for this choice |
| `dist` | Distribution | Distribution to evaluate |

**Returns:** `[value state']`

---

### Batched transitions

The batched variants have identical semantics to their scalar counterparts but operate on `[N]`-shaped arrays. They use `dist-sample-n` for batch sampling. MLX broadcasting handles all score and weight accumulation.

### `batched-simulate-transition`

```clojure
(h/batched-simulate-transition state addr dist)
```

Sample `[N]` values via `dist-sample-n`, accumulate `[N]`-shaped score.

**Returns:** `[value state']` where `value` has shape `[N, ...]`.

---

### `batched-generate-transition`

```clojure
(h/batched-generate-transition state addr dist)
```

Constrained sites use scalar observation (broadcasts into `[N]` score/weight). Unconstrained sites delegate to `batched-simulate-transition`.

**Returns:** `[value state']`

---

### `batched-update-transition`

```clojure
(h/batched-update-transition state addr dist)
```

Batched update with `[N]`-shaped old values and scalar or `[N]`-shaped new constraints.

**Returns:** `[value state']`

---

### `batched-regenerate-transition`

```clojure
(h/batched-regenerate-transition state addr dist)
```

Batched regenerate: resample `[N]` values at selected addresses, keep old `[N]`-shaped values at unselected addresses.

**Returns:** `[value state']`

---

### `run-handler`

```clojure
(rt/run-handler transition init-state body-fn)
```

Execute `body-fn` under the given transition function, returning the final state map. This is the single mutable boundary above MLX. It creates a `volatile!` cell, builds closure-based `trace`/`splice`/`param` operations, packages them into a runtime object, and passes it to `body-fn`.

Source: `src/genmlx/runtime.cljs`

| Parameter | Type | Description |
|-----------|------|-------------|
| `transition` | Function | Pure state transition `(fn [state addr dist] -> [value state'])` |
| `init-state` | Map | Initial handler state (immutable) |
| `body-fn` | Function | `(fn [runtime] -> retval)` receiving the runtime object |

**Returns:** Final state map with `:retval` added.

The runtime object `rt` has three fields:
- `(.-trace rt)` -- `(fn [addr dist] -> value)`
- `(.-splice rt)` -- `(fn [addr gf & args] -> retval)`
- `(.-param rt)` -- `(fn [name default] -> value)`

---

### `merge-sub-result`

```clojure
(h/merge-sub-result state addr sub-result)
```

Merge a sub-generative-function's result into the parent handler state. Nests the sub-choices under `addr`, accumulates scores, and propagates weights and discards.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | Map | Parent handler state |
| `addr` | Keyword | The splice address |
| `sub-result` | Map | `{:choices :score :weight :discard :retval}` from the sub-GF |

**Returns:** Updated parent state.

---

### `combinator-batched-fallback`

```clojure
(h/combinator-batched-fallback state addr gf args)
```

Fallback for splicing a non-DynamicGF (e.g. a combinator) in batched mode. Unstacks the `[N]`-particle state, runs the combinator's scalar GFI methods N times, then stacks the results back into `[N]`-shaped arrays.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | Map | Parent batched handler state |
| `addr` | Keyword | The splice address |
| `gf` | Generative function | The combinator to execute |
| `args` | Vector | Arguments to the combinator |

**Returns:** `[state' retval]`

---

### Edit request constructors

Source: `src/genmlx/edit.cljs`

### `constraint-edit`

```clojure
(edit/constraint-edit constraints)
```

Create a `ConstraintEdit` request -- equivalent to `update`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `constraints` | ChoiceMap | New constraints to apply |

**Returns:** `ConstraintEdit` record.

---

### `selection-edit`

```clojure
(edit/selection-edit selection)
```

Create a `SelectionEdit` request -- equivalent to `regenerate`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `selection` | Selection | Addresses to resample |

**Returns:** `SelectionEdit` record.

---

### `proposal-edit`

```clojure
(edit/proposal-edit forward-gf backward-gf)
(edit/proposal-edit forward-gf forward-args backward-gf backward-args)
```

Create a `ProposalEdit` request for SMCP3-style reversible kernels. The forward GF proposes new choices; the backward GF scores the reverse move.

| Parameter | Type | Description |
|-----------|------|-------------|
| `forward-gf` | Generative function | Proposes new choices |
| `forward-args` | Vector or nil | Arguments for forward proposal |
| `backward-gf` | Generative function | Scores the reverse move |
| `backward-args` | Vector or nil | Arguments for backward proposal |

**Returns:** `ProposalEdit` record.

---

### `edit-dispatch`

```clojure
(edit/edit-dispatch gf trace edit-request)
```

Dispatch an edit based on the `EditRequest` type. Called automatically by `p/edit` on `DynamicGF`. Routes to `update` for `ConstraintEdit`, `regenerate` for `SelectionEdit`, and the full forward/backward proposal protocol for `ProposalEdit`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | Generative function | The model |
| `trace` | Trace | Previous execution trace |
| `edit-request` | EditRequest | The edit to apply |

**Returns:** `{:trace Trace, :weight MLX-scalar, :discard ChoiceMap, :backward-request EditRequest}`
