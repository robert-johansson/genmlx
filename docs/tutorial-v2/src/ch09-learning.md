# Gradients, Learning, and Neural Models

GenMLX supports differentiable probabilistic programming: gradients flow through models, enabling gradient-based inference and parameter learning. This chapter covers choice gradients, parameter stores, optimizers, and the bridge to neural networks.

## Choice gradients

Given a trace, `choice-gradients` computes the gradient of the log-joint probability with respect to specific random choices:

```clojure
(let [model (dyn/auto-key grad-model)
      trace (p/simulate model [0])
      grads (grad/choice-gradients model trace [:mu])]
  (println ":mu gradient:" (mx/item (:mu grads))))
```

This returns a map from addresses to MLX gradient arrays. The gradient \\(\partial \log p / \partial \mu\\) tells you which direction to move \\(\mu\\) to increase the log-probability. This is the foundation for HMC, MALA, and variational inference.

## Parameter stores

Models can have **trainable parameters** — values that persist across inference runs and are optimized by gradient descent. A parameter store is a map:

```clojure
(def ps (learn/make-param-store {:theta (mx/scalar 1.0)
                                  :sigma (mx/scalar 0.5)}))
;; ps = {:params {:theta #mlx[1.0], :sigma #mlx[0.5]}, :version 0}
```

Inside a `gen` body, `param` reads from the store:

```clojure
(def parameterized-model
  (gen []
    (let [mu (param :mu 0.0)]  ;; reads :mu from store, or 0.0 if absent
      (trace :x (dist/gaussian mu 1))
      mu)))
```

Use `simulate-with-params` or `generate-with-params` to run with a specific parameter store:

```clojure
(let [ps (learn/make-param-store {:mu (mx/scalar 5.0)})
      trace (learn/simulate-with-params parameterized-model [] ps)]
  ;; :x is sampled from N(5.0, 1) because :mu was set to 5.0
  )
```

## Optimizers

GenMLX provides SGD and Adam on flat MLX arrays:

```clojure
;; SGD: params' = params - lr * grad
(let [params (mx/scalar 5.0)
      grad (mx/scalar 1.0)
      updated (learn/sgd-step params grad 0.1)]
  (println "updated:" (mx/item updated)))  ;; 4.9

;; Adam: maintains momentum and variance estimates
(let [params (mx/array [5.0 3.0])
      state (learn/adam-init params)
      grad (mx/array [1.0 0.5])
      [new-params new-state] (learn/adam-step params grad state {:lr 0.01})]
  (println "new params:" (mx/->clj new-params)))
```

## Training loop

`learn/train` runs a generic optimization loop:

```clojure
;; Minimize (theta - 3)^2
(let [loss-grad-fn (fn [params _key]
                     (let [theta (mx/item params)
                           loss (mx/scalar (* (- theta 3.0) (- theta 3.0)))
                           grad (mx/scalar (* 2.0 (- theta 3.0)))]
                       {:loss loss :grad grad}))
      result (learn/train {:iterations 50 :optimizer :sgd :lr 0.1}
                           loss-grad-fn (mx/scalar 0.0))]
  (println "final theta:" (mx/item (:params result)))
  (println "final loss:" (last (:loss-history result))))
```

The loss-grad function receives `(params, key)` and returns `{:loss MLX-scalar, :grad MLX-array}`. The training loop handles optimizer state management, loss history tracking, and optional callbacks.

For models with multiple parameters, represent parameters as a flat MLX array and use `learn/params->array` / `learn/array->params` to convert between named parameters and flat arrays.

## Wake-sleep learning

For amortized inference, `learn/wake-sleep` alternates between two phases:

- **Wake phase**: fix the model, improve the proposal (guide) to match the posterior.
- **Sleep phase**: fix the proposal, improve the model to match the guide's samples.

This trains a neural proposal to produce good importance sampling proposals without running MCMC at test time.

## Neural networks as generative functions

GenMLX bridges MLX's neural network module with the GFI via `nn->gen-fn`. A neural network wrapped as a generative function has no random choices (it's deterministic) but implements the GFI protocols, so it can be used with `splice`, combinators, and all inference algorithms.

This enables VAE-style architectures: a neural encoder maps data to approximate posterior parameters, and a generative decoder maps latent variables to observations. Both are generative functions composed via `splice`.

## What we've learned

- `choice-gradients` computes \\(\partial \log p / \partial \text{choice}\\) for gradient-based inference.
- Parameter stores hold trainable values; `param` reads them inside models.
- SGD and Adam operate on flat MLX arrays.
- `learn/train` runs optimization loops with loss-grad functions.
- Wake-sleep learning trains amortized proposals.
- Neural networks bridge to the GFI via `nn->gen-fn`.

In the final chapter, we'll cover extending GenMLX with custom distributions, verifying GFI correctness, and where to go next.
