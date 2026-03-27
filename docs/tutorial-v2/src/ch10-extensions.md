# Extensions and Verification

GenMLX is designed to be extended. This final chapter covers defining custom distributions, verifying GFI correctness with contracts and static validation, and where to go from here.

## Custom distributions with `defdist`

The `defdist` macro defines a new distribution type with sampling and log-probability:

```clojure
(defdist my-uniform-int
  "Uniform integer distribution on [lo, hi]."
  [lo hi]
  (sample [key]
    (let [range-size (- hi lo -1)
          u (rng/uniform key [] mx/float32)
          idx (mx/astype (mx/floor (mx/multiply u (mx/scalar range-size)))
                         mx/int32)]
      (mx/add idx (mx/scalar lo mx/int32))))
  (log-prob [value]
    (let [range-size (- hi lo -1)]
      (mx/negative (mx/log (mx/scalar range-size))))))
```

This creates a constructor function `my-uniform-int` and registers multimethod implementations for `dist-sample*` and `dist-log-prob`. The new distribution works immediately with all GFI operations:

```clojure
(let [model (dyn/auto-key (gen []
              (trace :roll (my-uniform-int 1 6))))
      trace (p/simulate model [])]
  (println "roll:" (mx/item (cm/get-choice (:choices trace) [:roll]))))
```

Optional clauses: `(reparam [key] ...)` for reparameterized sampling (gradient flow), and `(support [] ...)` for enumerable distributions.

## Quick custom distributions with `map->dist`

For one-off distributions, `map->dist` creates a `Distribution` from a plain map without the macro:

```clojure
(def spike (dc/map->dist {:type :my-spike
                           :sample (fn [key] (mx/scalar 42.0))
                           :log-prob (fn [value] (mx/scalar 0.0))}))
(dc/dist-sample spike (rng/fresh-key))  ;; => 42.0
```

## GFI contracts

GenMLX includes 11 executable contracts that verify the measure-theoretic invariants of GFI implementations. These are property-based tests that any valid generative function should satisfy:

```clojure
(let [model (dyn/auto-key (gen []
              (let [mu (trace :mu (dist/gaussian 0 1))]
                (trace :x (dist/gaussian mu 1))
                mu)))
      result (contracts/verify-gfi-contracts model [] :n-trials 5)]
  (println "pass:" (:total-pass result) "fail:" (:total-fail result)))
```

The contracts include:

| Contract | What it checks |
|---|---|
| `generate-weight-equals-score` | Fully constrained generate: weight = score |
| `update-empty-identity` | Update with same choices: weight \\(\approx 0\\) |
| `update-weight-correctness` | Weight = new\_score - old\_score |
| `update-round-trip` | Update then reverse recovers original |
| `regenerate-empty-identity` | Empty selection changes nothing |
| `project-all-equals-score` | Project everything = total score |
| `score-decomposition` | Per-address projections sum to total |
| `simulate-has-valid-score` | Simulated traces have finite scores |
| `assess-agrees-with-score` | Assess on trace choices matches score |

Use contracts to verify custom generative functions and combinators.

## Static validation

`validate-gen-fn` catches structural errors before running inference:

```clojure
(let [model (dyn/auto-key (gen []
              (trace :x (dist/gaussian 0 1))))
      result (verify/validate-gen-fn model [] {:n-trials 3})]
  (println "valid?" (:valid? result)))
```

It checks for: duplicate addresses, non-finite scores, execution errors, and materialization warnings. Running validation with multiple trials catches branch-dependent issues in models with stochastic control flow.

## The compilation ladder

To recap the performance story from [Chapter 8](./ch08-vectorization.md):

| Level | What it does | How it works |
|---|---|---|
| 0 | GPU batching | \\([N]\\)-shaped arrays through unchanged handlers |
| 1 | Compiled models | Macro-time schema \\(\to\\) single Metal dispatch |
| 2 | Compiled inference | Full SMC/MCMC sweep in one graph |
| 3 | Analytical elimination | Conjugacy detection \\(\to\\) exact marginal |
| 4 | Fused graph | Model + inference + gradient + optimizer fused |

Each level subsumes the previous. The handler path remains ground truth.

## The functional-probabilistic correspondence

Throughout this tutorial, we've seen functional programming concepts map onto probabilistic programming:

| Functional Programming | Probabilistic Programming |
|---|---|
| Algebraic effect (declare what) | Random choice (declare what) |
| Effect handler (determine how) | GFI operation (determine how) |
| Pure function | Generative function |
| Immutable record | Trace |
| Persistent map | Choice map |
| Boolean algebra | Selection |
| Function composition | Model composition (splice) |
| Higher-order function | Combinator (Map, Unfold, Switch) |
| Lazy evaluation | MLX computation graph |
| Macro / metaprogramming | Schema extraction |

This correspondence is the organizing principle behind GenMLX's architecture. Your model is pure — the framework manages state.

## Where to go next

- **Source code**: Browse the complete implementation at [github.com/robert-johansson/genmlx](https://github.com/robert-johansson/genmlx)
- **The paper**: "GenMLX: A Functional Probabilistic Programming Language on GPU via Algebraic Effects and Lazy Evaluation" describes the architecture and compilation ladder in detail
- **Gen.jl**: The original Julia implementation of the GFI — [gen.dev](https://www.gen.dev/)
- **Level 5**: The frontier — LLMs as generative functions, theory search, composing language with continuous models through the same GFI

Thank you for working through this tutorial. You now understand the core of GenMLX: the `gen` macro, the handler loop, the GFI operations, composition via splice and combinators, inference as composable kernels, vectorized GPU execution, gradients and learning, and the compilation ladder. Everything else is built from these pieces.
