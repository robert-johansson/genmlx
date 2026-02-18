(ns genmlx.inference.importance
  "Importance sampling inference."
  (:require [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.mlx :as mx]))

(defn importance-sampling
  "Importance sampling. Generate traces constrained by observations,
   return weighted samples.

   opts: {:samples N}
   model: generative function
   args: model arguments
   observations: choice map of observed values

   Returns {:traces [Trace ...] :log-weights [MLX-scalar ...]
            :log-ml-estimate MLX-scalar}"
  [{:keys [samples] :or {samples 100}} model args observations]
  (let [results (mapv (fn [_]
                        (p/generate model args observations))
                      (range samples))
        traces     (mapv :trace results)
        log-weights (mapv :weight results)
        ;; log marginal likelihood estimate = logsumexp(weights) - log(N)
        weights-arr (mx/array (mapv (fn [w] (mx/eval! w) (mx/item w)) log-weights))
        log-ml (mx/subtract (mx/logsumexp weights-arr)
                             (mx/scalar (js/Math.log samples)))]
    {:traces traces
     :log-weights log-weights
     :log-ml-estimate log-ml}))

(defn importance-resampling
  "Importance resampling. Generate traces via importance sampling,
   then resample proportional to weights.

   opts: {:samples N :particles M}
   Returns vector of N resampled traces."
  [{:keys [samples particles] :or {samples 100 particles 1000}}
   model args observations]
  (let [{:keys [traces log-weights]}
        (importance-sampling {:samples particles} model args observations)
        ;; Normalize weights via log-softmax
        weights-arr (mx/array (mapv (fn [w] (mx/eval! w) (mx/item w)) log-weights))
        log-probs (mx/subtract weights-arr (mx/logsumexp weights-arr))
        _ (mx/eval! log-probs)
        probs (mx/->clj (mx/exp log-probs))]
    ;; Resample
    (mapv (fn [_]
            (let [u (js/Math.random)]
              (loop [i 0 cumsum 0.0]
                (if (>= i (count traces))
                  (last traces)
                  (let [cumsum' (+ cumsum (nth probs i))]
                    (if (>= cumsum' u)
                      (nth traces i)
                      (recur (inc i) cumsum')))))))
          (range samples))))
