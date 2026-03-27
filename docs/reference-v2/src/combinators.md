# Combinators

Combinators compose generative functions into larger models with structured
address spaces. Each combinator implements the full GFI -- simulate, generate,
update, regenerate -- with efficient incremental computation. Compiled paths
(Level 1+) accelerate execution without changing semantics.

**Source:** `src/genmlx/combinators.cljs`, `src/genmlx/vmap.cljs`

---

## Map

### `map-combinator`

```clojure
(comb/map-combinator kernel)
```

Apply a kernel generative function independently to each element of the input
sequences. Each application is addressed by its integer index. Supports
incremental update: only re-executes changed elements.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel` | Generative function | Applied independently to each element |

**Arguments:** `[n & per-element-args]` where each per-element arg is a vector of
length `n`.

**Returns:** A `MapCombinator` implementing `IGenerativeFunction`, `IGenerate`,
`IUpdate`, `IRegenerate`, `IProject`, `IUpdateWithDiffs`, and `IEdit`.

**Address space:** Integer keys `0, 1, ..., n-1`. A choice at address `:x` in
element `i` lives at `[i :x]` in the combined choicemap.

**Example:**

```clojure
(def kernel (gen [x] (trace :y (dist/gaussian x 1))))
(def mapped (comb/map-combinator kernel))

;; Simulate 3 independent draws
(def tr (p/simulate mapped [3 [(mx/scalar 1) (mx/scalar 2) (mx/scalar 3)]]))

;; Choices at addresses: [0 :y], [1 :y], [2 :y]
(cm/get-choice (:choices tr) [0 :y])  ;; => scalar near 1
(cm/get-choice (:choices tr) [2 :y])  ;; => scalar near 3

;; Generate with constraints on element 1
(def obs (cm/from-map {[1 :y] (mx/scalar 5.0)}))
(def {:keys [trace weight]} (p/generate mapped
                              [3 [(mx/scalar 1) (mx/scalar 2) (mx/scalar 3)]]
                              obs))

;; Update: only element 0 is re-executed
(def new-obs (cm/from-map {[0 :y] (mx/scalar 10.0)}))
(p/update mapped trace new-obs)
```

**GFI behavior:**

- **simulate** calls `(p/simulate kernel elem-args)` for each element. Compiled
  paths (fused or per-element) are used when available.
- **generate** constrains each element independently via `(cm/get-submap constraints i)`.
- **update** reconstructs per-element traces and delegates to `(p/update kernel ...)`.
  With `IUpdateWithDiffs`, unchanged elements are skipped entirely.
- **regenerate** delegates `(sel/get-subselection selection i)` per element.

---

## Unfold

### `unfold-combinator`

```clojure
(comb/unfold-combinator kernel)
```

Sequentially apply a kernel for `n` time steps, threading state through each
step. Models time series, Markov chains, state-space models, and any process
with sequential dependencies.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel` | Generative function | Takes `[t state & extra-args]`, returns new state |

**Arguments:** `[n init-state & extra-args]`

**Returns:** An `UnfoldCombinator` implementing the full GFI. The trace's
`:retval` is a vector of states `[state-0 state-1 ... state-(n-1)]`.

**Address space:** Integer keys `1, 2, ..., n` (but stored 0-indexed internally as
`0, 1, ..., n-1`). Choice `:x` at step `t` is at `[t :x]`.

**Example:**

```clojure
(def step (gen [t state]
            (let [s (trace :state (dist/gaussian state 0.3))]
              (trace :obs (dist/gaussian s 1.0))
              s)))

(def hmm (comb/unfold-combinator step))

;; Run 5-step HMM from initial state 0
(def tr (p/simulate hmm [5 (mx/scalar 0.0)]))

;; Choices at: [0 :state], [0 :obs], [1 :state], [1 :obs], ...
(:retval tr)  ;; => [state0 state1 state2 state3 state4]

;; Generate with observations at each step
(def obs (cm/from-map {[0 :obs] (mx/scalar 0.5)
                       [1 :obs] (mx/scalar 1.2)
                       [2 :obs] (mx/scalar 0.8)
                       [3 :obs] (mx/scalar 1.5)
                       [4 :obs] (mx/scalar 2.0)}))
(def {:keys [trace weight]} (p/generate hmm [5 (mx/scalar 0.0)] obs))

;; Update with prefix-skip: only re-runs from the first changed step
(def new-obs (cm/from-map {[3 :obs] (mx/scalar 99.0)}))
(p/update hmm trace new-obs)  ;; steps 0-2 reused, steps 3-4 re-executed
```

**GFI behavior:**

- **simulate** loops `n` times, passing the previous step's return value as the
  next step's state. Three paths: fused (2 Metal dispatches for all T steps),
  compiled (per-step compiled-simulate), and handler (per-step `p/simulate`).
- **generate** constrains each step via `(cm/get-submap constraints t)`.
- **update** implements prefix-skip: finds the first step with non-empty constraints
  and reuses all prior steps from the old trace without re-execution.
- **batched-splice** (IBatchedSplice) enables efficient vectorized inference by
  running the kernel body once per step with all N particles via the batched handler.

### `unfold-empty-trace`

```clojure
(comb/unfold-empty-trace unfold-gf init-state & extra-args)
```

Create a valid T=0 Unfold trace with no steps executed. Used to initialize
particles for incremental extension with `unfold-extend`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `unfold-gf` | UnfoldCombinator | The unfold generative function |
| `init-state` | any | Initial carry state |
| `extra-args` | any | Additional arguments passed to kernel |

**Returns:** A `Trace` with empty choices, empty retval vector, and zero score.

**Example:**

```clojure
(def hmm (comb/unfold-combinator step))
(def t0 (comb/unfold-empty-trace hmm (mx/scalar 0.0)))

(:choices t0)  ;; => EMPTY
(:retval t0)   ;; => []
(:score t0)    ;; => 0.0
```

### `unfold-extend`

```clojure
(comb/unfold-extend trace step-constraints key)
```

Extend an Unfold trace by one step, returning `{:trace :weight}`. The new step
is generated with the given constraints. Calls `mx/materialize!` on weight and
score to break lazy graph accumulation -- critical for Metal buffer management
in long sequential inference.

| Parameter | Type | Description |
|-----------|------|-------------|
| `trace` | Trace | An existing Unfold trace (from simulate, generate, or prior extend) |
| `step-constraints` | ChoiceMap | Constraints for the new step |
| `key` | PRNG key | Random key for the new step |

**Returns:** `{:trace Trace, :weight scalar}` where the trace has one additional
step appended.

**Example:**

```clojure
;; Incremental SMC: extend one step at a time
(def t0 (comb/unfold-empty-trace hmm (mx/scalar 0.0)))

(def {:keys [trace weight]}
  (comb/unfold-extend t0
                      (cm/from-map {:obs (mx/scalar 0.5)})
                      (rng/fresh-key)))
;; trace now has 1 step, weight is the log-likelihood of the observation

(def {:keys [trace weight]}
  (comb/unfold-extend trace
                      (cm/from-map {:obs (mx/scalar 1.2)})
                      (rng/fresh-key)))
;; trace now has 2 steps
```

---

## Switch

### `switch-combinator`

```clojure
(comb/switch-combinator & branches)
```

Select one of several generative functions based on an integer index argument.
Models mixture components, regime switching, and conditional structure.

| Parameter | Type | Description |
|-----------|------|-------------|
| `branches` | Generative functions (varargs) | One or more branch GFs |

**Arguments:** `[index & branch-args]` where `index` (integer) selects which
branch to execute (0-indexed).

**Returns:** A `SwitchCombinator` implementing the full GFI.

**Address space:** The selected branch's addresses are used directly (no
additional nesting). If branch 0 traces at `:y`, the switch traces at `:y`.

**Example:**

```clojure
(def branch-a (gen [x] (trace :y (dist/gaussian x 1))))
(def branch-b (gen [x] (trace :y (dist/gaussian x 10))))

(def switched (comb/switch-combinator branch-a branch-b))

(p/simulate switched [0 (mx/scalar 5)])  ;; executes branch-a (tight around 5)
(p/simulate switched [1 (mx/scalar 5)])  ;; executes branch-b (wide around 5)

;; Generate with constraints
(def obs (cm/from-map {:y (mx/scalar 7.0)}))
(p/generate switched [0 (mx/scalar 5)] obs)
```

**GFI behavior:**

- **simulate/generate** selects `(nth branches index)` and delegates.
- **update** checks whether the branch index changed. Same branch: updates in
  place. Different branch: generates the new branch from scratch and discards
  old choices.
- **regenerate** delegates to the current branch's regenerate.
- **batched-splice** (IBatchedSplice) runs ALL branches once with the batched
  handler, then combines results per-particle using `mx/where` on the
  `[N]`-shaped index array.

### `vectorized-switch`

```clojure
(comb/vectorized-switch branches index branch-args)
```

Execute all branches with `[N]` independent samples each, then mask-select
results per-particle using an `[N]`-shaped index array. Used in vectorized
inference with stochastic control flow (mixture models, clustering).

| Parameter | Type | Description |
|-----------|------|-------------|
| `branches` | Vector of GFs | The branch generative functions |
| `index` | MLX array `[N]`, int32 | Per-particle branch indices |
| `branch-args` | Vector | Arguments shared across all branches |

**Returns:** `{:choices :score :retval}` with `[N]`-shaped arrays at each address.
Choices and scores from non-selected branches are masked out per particle.

**Example:**

```clojure
(def branches [(gen [x] (trace :y (dist/gaussian x 1)))
               (gen [x] (trace :y (dist/gaussian x 10)))])

;; 4 particles: first two use branch 0, last two use branch 1
(def idx (mx/array [0 0 1 1] mx/int32))
(def result (comb/vectorized-switch branches idx [(mx/scalar 5)]))

(:score result)   ;; [4]-shaped: log-probs from selected branches
(:choices result) ;; :y is [4]-shaped, mixing both branches
```

---

## Scan

### `scan-combinator`

```clojure
(comb/scan-combinator kernel)
```

State-threading sequential combinator, analogous to `jax.lax.scan`. More
general than Unfold: the kernel takes `[carry input]` and returns
`[new-carry output]`, processing external inputs at each step while
accumulating both carry-state and outputs.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel` | Generative function | Takes `[carry input]`, returns `[new-carry output]` |

**Arguments:** `[init-carry inputs]` where `inputs` is a vector of per-step inputs.

**Returns:** A `ScanCombinator` implementing the full GFI. The trace's `:retval`
is `{:carry final-carry, :outputs [output0 output1 ...]}`.

**Address space:** Integer keys `0, 1, ..., n-1` where `n = (count inputs)`.
Choice `:x` at step `t` is at `[t :x]`.

**Example:**

```clojure
(def scan-step
  (gen [carry input]
    (let [new-carry (trace :state (dist/gaussian
                                    (mx/add carry input) 0.1))
          obs (trace :obs (dist/gaussian new-carry 1.0))]
      [new-carry obs])))

(def scanner (comb/scan-combinator scan-step))

(def inputs [(mx/scalar 0.1) (mx/scalar 0.2) (mx/scalar -0.1)])
(def tr (p/simulate scanner [(mx/scalar 0.0) inputs]))

;; retval is {:carry final-state, :outputs [obs0 obs1 obs2]}
(:carry (:retval tr))   ;; final carry state
(:outputs (:retval tr)) ;; vector of observations

;; Choices at: [0 :state], [0 :obs], [1 :state], [1 :obs], ...
```

**GFI behavior:**

- **simulate** loops over inputs, threading carry. Three paths: fused, compiled,
  and handler.
- **generate** constrains each step via `(cm/get-submap constraints t)`.
- **update** implements prefix-skip like Unfold: unchanged prefix steps are reused.
- **batched-splice** (IBatchedSplice) runs one batched kernel call per step.

---

## Mask

### `mask-combinator`

```clojure
(comb/mask-combinator inner)
```

Conditionally execute a generative function based on a boolean flag. When
`active?` is false, the inner GF is not executed and contributes no choices
or score. Used internally by vectorized switch to implement all-branch
execution.

| Parameter | Type | Description |
|-----------|------|-------------|
| `inner` | Generative function | GF to conditionally execute |

**Arguments:** `[active? & inner-args]` where `active?` is a boolean.

**Returns:** A `MaskCombinator` implementing `IGenerativeFunction`, `IGenerate`,
`IUpdate`, `IRegenerate`, `IProject`, `IUpdateWithDiffs`, and `IEdit`.

**Address space:** When active, identical to the inner GF's address space. When
inactive, the choicemap is empty.

**Example:**

```clojure
(def inner-gf (gen [x] (trace :y (dist/gaussian x 1))))
(def masked (comb/mask-combinator inner-gf))

;; Active: executes inner, choices present
(def t1 (p/simulate masked [true (mx/scalar 5)]))
(cm/get-choice (:choices t1) [:y])  ;; => scalar near 5

;; Inactive: no execution, empty choices, zero score
(def t2 (p/simulate masked [false (mx/scalar 5)]))
(:choices t2)  ;; => EMPTY
(:score t2)    ;; => 0.0
(:retval t2)   ;; => nil
```

**GFI behavior:**

- **simulate/generate** check `active?`: if true, delegate to inner; if false,
  return empty trace with zero score.
- **update/regenerate** delegate to inner when active, return unchanged trace
  with zero weight when inactive.

---

## Mix

### `mix-combinator`

```clojure
(comb/mix-combinator components log-weights-fn)
```

Create a first-class mixture model. Samples a component index from a
categorical distribution over the log-weights, then simulates that component.
The component index is stored at address `:component-idx` in the choicemap.

| Parameter | Type | Description |
|-----------|------|-------------|
| `components` | Vector of GFs | Mixture component generative functions |
| `log-weights-fn` | Function or MLX array | `(fn [args] -> MLX array)` of log mixing weights, or a fixed MLX array |

**Returns:** A `MixCombinator` implementing the full GFI.

**Address space:** The selected component's addresses plus `:component-idx`
(an int32 scalar indicating which component was chosen).

**Example:**

```clojure
(def comp-a (gen [x] (trace :y (dist/gaussian x 1))))
(def comp-b (gen [x] (trace :y (dist/gaussian x 10))))

;; Equal mixing weights (log 0.5 each)
(def mixture (comb/mix-combinator
               [comp-a comp-b]
               (mx/array [(Math/log 0.5) (Math/log 0.5)])))

(def tr (p/simulate mixture [(mx/scalar 0)]))

;; Which component was selected?
(mx/item (cm/get-choice (:choices tr) [:component-idx]))  ;; => 0 or 1

;; The component's choices
(cm/get-choice (:choices tr) [:y])  ;; => sampled value

;; Generate with constrained component index
(def obs (cm/from-map {:component-idx (mx/scalar 0 mx/int32)
                       :y (mx/scalar 3.0)}))
(p/generate mixture [(mx/scalar 0)] obs)
```

**GFI behavior:**

- **simulate** samples an index from `categorical(log-weights)`, then simulates
  the selected component.
- **generate** constrains the component index if present in the constraints.
- **update** checks whether the component index changed. Same component: updates
  inner choices. Different component: generates new component from scratch.
- **regenerate** with `:component-idx` selected resamples the index and
  simulates a new component. Without it selected, regenerates within the
  current component.
- **batched-splice** (IBatchedSplice) runs ALL components once with the batched
  handler, samples `[N]`-shaped indices, and combines per-particle using `mx/where`.

---

## Recurse

### `recurse`

```clojure
(comb/recurse maker)
```

Create a recursive generative function. The `maker` receives a reference to
the combinator being defined, enabling self-referential models: random trees,
recursive grammars, linked lists, and other variable-depth structures.

| Parameter | Type | Description |
|-----------|------|-------------|
| `maker` | Function | `(fn [self] -> GF)` where `self` is the `RecurseCombinator` |

**Returns:** A `RecurseCombinator` implementing `IGenerativeFunction`, `IGenerate`,
`IUpdate`, `IRegenerate`, `IProject`, `IUpdateWithDiffs`, and `IEdit`.

**Address space:** Determined by the inner GF returned by `maker`. Since the inner
GF can `splice` back into `self`, addresses nest recursively. Typical pattern: use
`:left`/`:right` or indexed addresses to distinguish recursive branches.

**Example:**

```clojure
;; Recursive random tree
(def tree-model
  (comb/recurse
    (fn [self]
      (gen [depth]
        (let [leaf? (trace :leaf? (dist/bernoulli 0.3))]
          (if (or (pos? (mx/item leaf?)) (>= depth 5))
            (trace :value (dist/gaussian 0 1))
            (let [left  (splice :left self [(inc depth)])
                  right (splice :right self [(inc depth)])]
              (mx/add left right))))))))

(def tr (p/simulate tree-model [0]))

;; Address structure depends on the random tree shape:
;; :leaf? at each node
;; :value at leaves
;; [:left :leaf?], [:left :value], [:right :left :leaf?], etc.
```

**GFI behavior:**

- **simulate/generate** call `(maker this)` to get the inner GF, then delegate.
  The self-reference enables recursion: the inner GF can `splice` back to `this`.
- **update** reconstructs the inner GF via `maker` and delegates.
- **regenerate** delegates to the inner GF's regenerate.
- **update-with-diffs** short-circuits when args and constraints are unchanged.

---

## Vmap

**Source:** `src/genmlx/vmap.cljs`

### `vmap-gf`

```clojure
(vmap/vmap-gf kernel & {:keys [in-axes axis-size]})
```

Create a Vmap combinator that maps a kernel over a batch dimension of the
arguments. Unlike `map-combinator` (which uses integer-indexed sub-traces),
`vmap-gf` stores choices with `[N]`-shaped leaf arrays and supports `splice`
by falling back to per-invocation execution when needed.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel` | Generative function | The model to map |
| `:in-axes` | Vector or nil | `0` or `nil` per argument (0 = batch along axis 0, nil = broadcast). Default: nil (all args batched) |
| `:axis-size` | Integer or nil | Explicit batch size N (required when all in-axes are nil or args are empty) |

**Returns:** A `VmapCombinator` implementing `IGenerativeFunction`, `IGenerate`,
`IUpdate`, `IRegenerate`, `IAssess`, `IPropose`, and `IProject`.

**Address space:** Same address structure as the kernel, but leaf values are
`[N]`-shaped arrays instead of scalars. Address `:y` in the kernel becomes
address `:y` in the vmap trace, holding an `[N]`-shaped array.

**Example:**

```clojure
(def kernel (gen [x] (trace :y (dist/gaussian x 1))))

;; Batch over a [5]-shaped array of means
(def vmapped (vmap/vmap-gf kernel))
(def tr (p/simulate vmapped [(mx/array [1 2 3 4 5])]))
;; (:choices tr) has :y as a [5]-shaped array

;; Broadcast: x is batched, sigma is shared
(def kernel2 (gen [x sigma]
               (trace :y (dist/gaussian x sigma))))
(def vmapped2 (vmap/vmap-gf kernel2 :in-axes [0 nil]))
(def tr2 (p/simulate vmapped2 [(mx/array [1 2 3]) (mx/scalar 0.5)]))
;; :y is [3]-shaped; sigma=0.5 is shared across all 3

;; Generate with [N]-shaped constraints
(def obs (cm/from-map {:y (mx/array [10 20 30])}))
(def {:keys [trace weight]} (vmap/vmap-gf kernel)
  (p/generate vmapped [(mx/array [1 2 3])] obs))
```

**Execution paths:**

- **Fast path:** When the kernel has a `body-fn` and all batched args are MLX
  arrays, the kernel body runs ONCE with the batched handler, producing
  `[N]`-shaped arrays at each trace site.
- **Slow path:** When args include sequences or the kernel lacks `body-fn`,
  falls back to N independent `p/simulate`/`p/generate` calls, then stacks
  results.

### `repeat-gf`

```clojure
(vmap/repeat-gf kernel n)
```

Create a Vmap combinator that runs the kernel `n` times with no batched
arguments. Shorthand for `(vmap-gf kernel :axis-size n)`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel` | Generative function | The model to repeat |
| `n` | Integer | Number of independent repetitions |

**Returns:** A `VmapCombinator` with `:axis-size n` and no in-axes.

**Example:**

```clojure
(def coin (gen [] (trace :flip (dist/bernoulli 0.5))))

;; Flip 100 independent coins
(def repeated (vmap/repeat-gf coin 100))
(def tr (p/simulate repeated []))

;; :flip is a [100]-shaped array of 0s and 1s
(mx/sum (cm/get-choice (:choices tr) [:flip]))  ;; approximately 50
```

---

## Contramap / Dimap

Functional transforms on generative functions that modify inputs, outputs, or
both. These do not change the address space -- they only transform the data
flowing in and out.

### `contramap-gf`

```clojure
(comb/contramap-gf gf f)
```

Transform arguments before passing to the inner generative function.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | Generative function | The inner GF to wrap |
| `f` | Function | `(fn [args] -> transformed-args)` |

**Returns:** A `ContramapGF` implementing `IGenerativeFunction`, `IGenerate`,
`IUpdate`, `IRegenerate`, `IUpdateWithDiffs`, and `IEdit`.

**Example:**

```clojure
(def model (gen [x y] (trace :z (dist/gaussian (mx/add x y) 1))))

;; Swap arguments
(def swapped (comb/contramap-gf model (fn [[a b]] [b a])))
(p/simulate swapped [(mx/scalar 1) (mx/scalar 2)])
;; model receives args [2, 1]

;; Extract from a map
(def from-map (comb/contramap-gf model
                (fn [[m]] [(:x m) (:y m)])))
```

### `map-retval`

```clojure
(comb/map-retval gf g)
```

Transform the return value of a generative function. The address space and
score are unchanged.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | Generative function | The inner GF to wrap |
| `g` | Function | `(fn [retval] -> transformed-retval)` |

**Returns:** A `MapRetvalGF` implementing `IGenerativeFunction`, `IGenerate`,
`IUpdate`, `IRegenerate`, `IUpdateWithDiffs`, and `IEdit`.

**Example:**

```clojure
(def model (gen [x] (trace :y (dist/gaussian x 1))))

;; Square the return value
(def squared (comb/map-retval model (fn [v] (mx/multiply v v))))
(def tr (p/simulate squared [(mx/scalar 3)]))
;; (:retval tr) is the squared value of :y
```

### `dimap`

```clojure
(comb/dimap gf f g)
```

Transform both arguments and return value of a generative function. Equivalent
to `(-> gf (contramap-gf f) (map-retval g))`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | Generative function | The inner GF to wrap |
| `f` | Function | `(fn [args] -> transformed-args)` for input |
| `g` | Function | `(fn [retval] -> transformed-retval)` for output |

**Returns:** A composed `MapRetvalGF` wrapping a `ContramapGF`.

**Example:**

```clojure
(def model (gen [x] (trace :y (dist/gaussian x 1))))

;; Double the input, negate the output
(def transformed (comb/dimap model
                   (fn [[x]] [(mx/multiply x (mx/scalar 2))])
                   (fn [v] (mx/negative v))))

(def tr (p/simulate transformed [(mx/scalar 3)]))
;; model receives x=6, retval is negated
```
