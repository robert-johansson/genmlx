# The Inference Toolkit

In Chapter 2 we built importance sampling by hand. In Chapter 5 we built Metropolis-Hastings by hand. GenMLX provides a library of inference algorithms and composable building blocks so you don't have to. This chapter covers the toolkit: kernel constructors, kernel composition, built-in MCMC and SMC, and diagnostics.

## Importance sampling

The built-in IS function from Chapter 2, for reference:

```clojure
(let [{:keys [traces log-weights log-ml-estimate]}
      (importance/importance-sampling {:samples 50} simple-model [] obs)]
  (println "traces:" (count traces))
  (println "log-ML:" (mx/item log-ml-estimate)))
```

## Kernel constructors

GenMLX provides ready-made inference kernels — functions of type `(trace, key) -> trace`:

```clojure
;; Prior-proposal MH: resample :mu from the prior
(def k1 (kern/mh-kernel (sel/select :mu)))

;; Random-walk MH: propose mu' = mu + N(0, 0.5)
(def k2 (kern/random-walk :mu 0.5))

;; Prior resample (shorthand for mh-kernel with select)
(def k3 (kern/prior :mu))

;; Gibbs: cycle through addresses with prior proposals
(def k4 (kern/gibbs :mu))
```

All kernels have the same type. This uniform interface enables composition.

## Kernel composition

Kernels compose algebraically:

```clojure
;; Sequential: apply k1 then k2
(def composed (kern/chain (kern/prior :mu) (kern/random-walk :mu 0.5)))

;; Repeat: apply the same kernel 10 times
(def repeated (kern/repeat-kernel 10 (kern/prior :mu)))

;; Cycle: round-robin through a list
(def cycled (kern/cycle-kernels 6 [(kern/prior :mu) (kern/random-walk :mu 0.5)]))
```

Reversals propagate automatically: the reversal of `(chain k1 k2 k3)` is `(chain (rev k3) (rev k2) (rev k1))`. This is essential for reversible-jump MCMC.

## Running a kernel

`run-kernel` executes a kernel for a specified number of iterations with burn-in, thinning, and optional callbacks:

```clojure
(let [model (dyn/auto-key simple-model)
      init-trace (:trace (p/generate model [] obs))
      kernel (kern/random-walk :mu 1.0)
      samples (kern/run-kernel {:samples 50 :burn 20} kernel init-trace)]
  (println "samples:" (count samples))
  (println "acceptance rate:" (:acceptance-rate (meta samples))))
```

The return value is a vector of traces with an acceptance rate in the metadata.

## Built-in MH

For convenience, `mcmc/mh` initializes from observations and runs a chain:

```clojure
(let [samples (mcmc/mh {:samples 50 :burn 20 :selection (sel/select :mu)}
                        simple-model [] obs)]
  (println "samples:" (count samples))
  (println "acceptance:" (:acceptance-rate (meta samples))))
```

## Hamiltonian Monte Carlo

HMC uses gradients of the log-posterior for more efficient proposals. Specify the step size, number of leapfrog steps, and which addresses to update:

```clojure
(let [samples (mcmc/hmc {:samples 30 :burn 10
                          :step-size 0.1 :leapfrog-steps 5
                          :addresses [:mu]}
                         simple-model [] obs)]
  (println "HMC samples:" (count samples)))
```

HMC also supports adaptive step-size tuning (dual averaging) and diagonal mass matrix estimation during burn-in via `:adapt-step-size true` and `:adapt-metric true`.

GenMLX also provides MALA (`mcmc/mala`), NUTS (`mcmc/nuts`), elliptical slice sampling (`mcmc/elliptical-slice`), and MAP optimization (`mcmc/map-optimize`).

## Sequential Monte Carlo

SMC maintains a population of weighted particles, extending them through a sequence of observations:

```clojure
(let [result (smc/smc {:particles 20 :ess-threshold 0.5}
                       model args observations)]
  (println "traces:" (count (:traces result)))
  (println "log-ML:" (mx/item (:log-ml-estimate result))))
```

SMC supports systematic, stratified, and residual resampling, rejuvenation via MCMC kernels, and conditional SMC (cSMC) for particle MCMC.

## Diagnostics

After collecting samples, assess convergence and summarize:

```clojure
(let [traces (kern/run-kernel {:samples 100 :burn 50} kernel init-trace)
      mu-arrays (mapv #(cm/get-choice (:choices %) [:mu]) traces)]
  (println "mean:" (mx/item (diag/sample-mean mu-arrays)))
  (println "std:" (mx/item (diag/sample-std mu-arrays)))
  (let [qs (diag/sample-quantiles mu-arrays)]
    (println "2.5%:" (:q025 qs))
    (println "median:" (:median qs))
    (println "97.5%:" (:q975 qs))))
```

For multiple chains, `diag/r-hat` computes the Gelman-Rubin convergence diagnostic. Values near 1.0 indicate convergence.

## When to use what

| Situation | Recommended method |
|---|---|
| Quick posterior estimate | Importance sampling |
| Few continuous parameters | Random-walk MH or HMC |
| Many continuous parameters | HMC or NUTS with adaptation |
| Sequential/temporal data | SMC |
| Discrete latent variables | Gibbs with enumeration |
| Model comparison | Log marginal likelihood from IS or SMC |
| Fast prototyping | `kern/run-kernel` with `kern/prior` |

## What we've learned

- Inference kernels are functions `(trace, key) -> trace` with a uniform interface.
- `chain`, `repeat-kernel`, `cycle-kernels`, and `mix-kernels` compose kernels algebraically.
- `run-kernel` collects samples with burn-in, thinning, and acceptance tracking.
- Built-in algorithms: MH, HMC, NUTS, MALA, SMC, elliptical slice, Gibbs, MAP.
- Diagnostics: mean, std, quantiles, R-hat.

In the next chapter, we'll make everything faster — vectorized inference via shape polymorphism and the compilation ladder.
