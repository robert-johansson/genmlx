(ns genmlx.inference.importance
  "Importance sampling inference."
  (:require [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.util :as u]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vec]))

(defn importance-sampling
  "Importance sampling. Generate traces constrained by observations,
   return weighted samples.

   opts: {:samples N :key prng-key}
   model: generative function
   args: model arguments
   observations: choice map of observed values

   Returns {:traces [Trace ...] :log-weights [MLX-scalar ...]
            :log-ml-estimate MLX-scalar}"
  [{:keys [samples key] :or {samples 100}} model args observations]
  (let [keys (rng/split-n (rng/ensure-key key) samples)
        results (mapv (fn [ki]
                        (let [r (dyn/with-key ki
                                  #(p/generate model args observations))]
                          (mx/eval! (:weight r) (:score (:trace r)))
                          r))
                      keys)
        traces     (mapv :trace results)
        log-weights (mapv :weight results)
        ;; log marginal likelihood estimate = logsumexp(weights) - log(N)
        weights-arr (u/materialize-weights log-weights)
        log-ml (mx/subtract (mx/logsumexp weights-arr)
                             (mx/scalar (js/Math.log samples)))]
    {:traces traces
     :log-weights log-weights
     :log-ml-estimate log-ml}))

(defn importance-resampling
  "Importance resampling. Generate traces via importance sampling,
   then resample proportional to weights.

   opts: {:samples N :particles M :key prng-key}
   Returns vector of N resampled traces."
  [{:keys [samples particles key] :or {samples 100 particles 1000}}
   model args observations]
  (let [{:keys [traces log-weights]}
        (importance-sampling {:samples particles :key key} model args observations)
        ;; Normalize weights via log-softmax
        {:keys [probs]} (u/normalize-log-weights log-weights)
        key (or key (rng/fresh-key))]
    ;; Resample
    (let [keys (rng/split-n key samples)]
      (mapv (fn [ki]
              (let [u (mx/realize (rng/uniform ki []))
                    result (reduce (fn [cumsum [i p]]
                                     (let [cumsum' (+ cumsum p)]
                                       (if (>= cumsum' u)
                                         (reduced (nth traces i))
                                         cumsum')))
                                   0.0
                                   (map-indexed vector probs))]
                (if (number? result) (last traces) result)))
            keys))))

(defn vectorized-importance-sampling
  "Vectorized importance sampling. Runs model ONCE with batched handler
   instead of N sequential generate calls.

   opts: {:samples N :key prng-key}
   model: DynamicGF
   args: model arguments
   observations: choice map of observed values

   Returns {:vtrace VectorizedTrace :log-ml-estimate MLX-scalar}"
  [{:keys [samples key] :or {samples 100}} model args observations]
  (let [key (rng/ensure-key key)
        vtrace (dyn/vgenerate model args observations samples key)]
    {:vtrace vtrace
     :log-ml-estimate (vec/vtrace-log-ml-estimate vtrace)}))
