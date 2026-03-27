# GenMLX Tutorial

**Probabilistic programming with a functional core.**

GenMLX is a probabilistic programming language in ClojureScript that runs on Apple's MLX GPU framework. It implements the *Generative Function Interface* — a small set of operations that separate how you write models from how you run inference.

The architecture is organized around one principle: **your model is pure; the framework manages state.** When you write a model, you declare random choices — you say *what* you need (a value from this distribution at this address). The handler determines *how* to provide it: sampling from the prior, constraining to observed data, updating from a previous execution, or regenerating for MCMC. The same model code works with all of these — without modification, without knowing which one is running. You never touch mutable state. The framework handles it.

By the end of this tutorial, you will:

- Write probabilistic models using `gen` and `trace`
- Condition on data and run inference (IS, MCMC, SMC, VI)
- Understand how the handler architecture works and why it's pure
- Compose models with splice and combinators
- Use GPU-accelerated vectorized inference
- Train models with gradients and neural networks

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- [Bun](https://bun.sh/) (or Node.js 18+)
- The GenMLX repository cloned and set up:

```bash
git clone https://github.com/robert-johansson/genmlx
cd genmlx
git submodule update --init --recursive node-mlx
cd node-mlx && npm install && npm run build && cd ..
bun install
```

## Running examples

Save any code snippet to a file and run it:

```bash
bun run --bun nbb example.cljs
```

## A 30-second demo

```clojure
(ns demo
  (:require [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm])
  (:require-macros [genmlx.gen :refer [gen]]))

(def coin-model
  (gen []
    (let [bias (trace :bias (dist/beta-dist 2 2))]
      (trace :flip (dist/bernoulli bias)))))

(let [model (dyn/auto-key coin-model)
      trace (p/simulate model [])]
  (println "bias:" (mx/item (cm/get-choice (:choices trace) [:bias])))
  (println "flip:" (mx/item (cm/get-choice (:choices trace) [:flip])))
  (println "score:" (mx/item (:score trace))))
```

That's it. A model, a simulation, a trace. Let's build from here.
