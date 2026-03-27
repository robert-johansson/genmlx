# Diagnostics

```clojure
(require '[genmlx.inference.diagnostics :as diag])
```

Convergence diagnostics and summary statistics for MCMC samples. All functions accept vectors of MLX arrays (the format returned by kernel execution and MCMC samplers).

---

## Convergence

### `ess`

```clojure
(diag/ess samples)
```

Compute the effective sample size from a single chain of parameter samples. Uses the autocorrelation-based estimator with Geyer's initial positive sequence truncation -- autocorrelation pairs are summed until the first negative pair, preventing overestimation of ESS from noisy autocorrelation tails.

ESS measures how many independent samples the chain is equivalent to. A well-tuned MCMC chain typically achieves ESS between 10% and 50% of the nominal sample count. ESS equal to the sample count indicates uncorrelated (IID) draws.

| Parameter | Type | Description |
|-----------|------|-------------|
| `samples` | vector of MLX arrays | Scalar parameter samples from a single chain |

**Returns:** A number (the effective sample size).

```clojure
(let [traces (k/run-kernel {:samples 2000 :burn 500} kernel init-trace)
      mu-samples (mapv #(cm/get-value (cm/get-submap (:choices %) :mu)) traces)
      effective (diag/ess mu-samples)]
  (println "ESS:" effective "out of" (count mu-samples) "samples")
  (println "Efficiency:" (/ effective (count mu-samples))))
```

### `r-hat`

```clojure
(diag/r-hat chains)
```

Compute the Gelman-Rubin R-hat statistic from multiple independent chains. R-hat compares the between-chain variance to the within-chain variance. Values close to 1.0 indicate that chains have converged to the same distribution; values above 1.1 suggest the chains have not yet mixed.

Run at least 4 independent chains with different initial states for a reliable diagnostic. Each chain should have the same number of samples.

| Parameter | Type | Description |
|-----------|------|-------------|
| `chains` | vector of vectors | Each inner vector is a chain of MLX array samples. All chains must have the same length. |

**Returns:** A number (R-hat). Values below 1.05 indicate good convergence; below 1.1 is acceptable.

```clojure
;; Run 4 independent chains
(let [run-chain (fn [seed]
                  (let [{:keys [trace]} (p/generate model args obs
                                          :key (rng/fresh-key seed))
                        kernel (k/random-walk :mu 0.5)]
                    (k/run-kernel {:samples 1000 :burn 500
                                   :key (rng/fresh-key seed)}
                                  kernel trace)))
      chains (mapv run-chain [42 137 271 999])
      ;; Extract :mu from each chain as MLX arrays
      mu-chains (mapv (fn [traces]
                        (mapv #(cm/get-value (cm/get-submap (:choices %) :mu))
                              traces))
                      chains)
      r (diag/r-hat mu-chains)]
  (println "R-hat:" r)
  (when (< r 1.1) (println "Chains have converged")))
```

---

## Summary Statistics

### `sample-mean`

```clojure
(diag/sample-mean samples)
```

Compute the mean of parameter samples by stacking them into a single array and reducing along the sample axis.

| Parameter | Type | Description |
|-----------|------|-------------|
| `samples` | vector of MLX arrays | Scalar parameter samples |

**Returns:** An MLX array (the mean). Call `mx/item` to extract as a JS number.

```clojure
(let [mu (diag/sample-mean mu-samples)]
  (println "Mean:" (mx/item mu)))
```

### `sample-std`

```clojure
(diag/sample-std samples)
```

Compute the standard deviation of parameter samples.

| Parameter | Type | Description |
|-----------|------|-------------|
| `samples` | vector of MLX arrays | Scalar parameter samples |

**Returns:** An MLX array (the standard deviation). Call `mx/item` to extract as a JS number.

```clojure
(let [sd (diag/sample-std mu-samples)]
  (println "Std:" (mx/item sd)))
```

### `sample-quantiles`

```clojure
(diag/sample-quantiles samples)
```

Compute the median and 95% credible interval from parameter samples. Extracts values to the host, sorts them, and selects the appropriate order statistics.

| Parameter | Type | Description |
|-----------|------|-------------|
| `samples` | vector of MLX arrays | Scalar parameter samples |

**Returns:** A map `{:median :q025 :q975}` where all values are JS numbers.

```clojure
(let [{:keys [median q025 q975]} (diag/sample-quantiles mu-samples)]
  (println "Median:" median)
  (println "95% CI: [" q025 "," q975 "]"))
```

---

## Reporting

### `summarize`

```clojure
(diag/summarize samples)
(diag/summarize samples :name "mu")
```

Compute a full diagnostic summary for a parameter: mean, standard deviation, median, 95% credible interval, and effective sample size. Combines `sample-mean`, `sample-std`, `sample-quantiles`, and `ess` into a single call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `samples` | vector of MLX arrays | Scalar parameter samples |
| `:name` | string | (optional) Parameter name for the summary. Default: `"param"`. |

**Returns:** A map with keys:

| Key | Type | Description |
|-----|------|-------------|
| `:name` | string | The parameter name |
| `:mean` | number | Posterior mean |
| `:std` | number | Posterior standard deviation |
| `:median` | number | Posterior median |
| `:q025` | number | 2.5th percentile (lower bound of 95% CI) |
| `:q975` | number | 97.5th percentile (upper bound of 95% CI) |
| `:ess` | number | Effective sample size |

```clojure
(let [summary (diag/summarize mu-samples :name "mu")]
  (println (:name summary)
           "mean=" (:mean summary)
           "std=" (:std summary)
           "95% CI=[" (:q025 summary) "," (:q975 summary) "]"
           "ESS=" (:ess summary)))
;; mu mean=1.02 std=0.71 95% CI=[ -0.35 , 2.38 ] ESS=487
```

---

## Examples

### Full MCMC diagnostic workflow

```clojure
(require '[genmlx.inference.kernel :as k]
         '[genmlx.inference.diagnostics :as diag]
         '[genmlx.protocols :as p]
         '[genmlx.choicemap :as cm])

;; 1. Run multiple chains for R-hat
(let [run-chain
      (fn [seed]
        (let [{:keys [trace]} (p/generate model args obs
                                :key (rng/fresh-key seed))
              kernel (k/gibbs {:slope 0.3 :intercept 0.3})]
          (k/run-kernel {:samples 1000 :burn 500
                         :key (rng/fresh-key seed)}
                        kernel trace)))
      chains (mapv run-chain [1 2 3 4])

      ;; 2. Extract parameter samples from each chain
      extract (fn [traces addr]
                (mapv #(cm/get-value (cm/get-submap (:choices %) addr))
                      traces))

      slope-chains     (mapv #(extract % :slope) chains)
      intercept-chains (mapv #(extract % :intercept) chains)]

  ;; 3. Check convergence with R-hat
  (println "R-hat slope:" (diag/r-hat slope-chains))
  (println "R-hat intercept:" (diag/r-hat intercept-chains))

  ;; 4. Summarize a single chain
  (let [slope-samples (first slope-chains)
        intercept-samples (first intercept-chains)]
    (println (diag/summarize slope-samples :name "slope"))
    (println (diag/summarize intercept-samples :name "intercept"))))
```

### Monitoring ESS during sampling

```clojure
;; Use the run-kernel callback to monitor diagnostics during sampling
(let [collected (atom [])
      kernel (k/random-walk :mu 0.5)
      traces (k/run-kernel
               {:samples 2000 :burn 500
                :callback (fn [{:keys [iter trace]}]
                            (swap! collected conj
                              (cm/get-value (cm/get-submap (:choices trace) :mu)))
                            (when (zero? (mod (inc iter) 500))
                              (println "Iter" (inc iter)
                                       "ESS:" (diag/ess @collected))))}
               kernel init-trace)]
  (println "Final ESS:" (diag/ess @collected)))
```
