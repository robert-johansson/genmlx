(ns genmlx.inference.importance
  "Importance sampling inference."
  (:require [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.util :as u]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vec]))

(defn- strip-analytical
  "Remove L3 analytical handlers so generate samples from the prior,
   not the deterministic posterior mean. The analytical path is correct
   for single-trace operations (MCMC, marginal LL) but returns identical
   particles in multi-particle IS, breaking the method."
  [model]
  (if-let [schema (:schema model)]
    (assoc model :schema (dissoc schema :auto-handlers :conjugate-pairs))
    model))

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
  (let [model (-> model dyn/auto-key strip-analytical)
        keys (rng/split-n (rng/ensure-key key) samples)
        results (into []
                      (map-indexed
                       (fn [i ki]
                         (let [r (p/generate (dyn/with-key model ki) args observations)]
                           (mx/materialize! (:weight r) (:score (:trace r)))
                           (when (zero? (mod (inc i) 50)) (mx/sweep-dead-arrays!))
                           r)))
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

(defn tidy-importance-sampling
  "Memory-efficient importance sampling. Each particle runs inside mx/tidy-run
   so all trace arrays are disposed immediately after extracting the weight.
   Only returns log-weights (as JS numbers) and log-ML estimate.

   Use this instead of importance-sampling when:
   - The model has many trace sites (large traces)
   - You need many particles
   - You only need log-ML, not the traces themselves

   opts: {:samples N :key prng-key}
   Returns {:log-weights [JS-number ...] :log-ml-estimate JS-number}"
  [{:keys [samples key] :or {samples 100}} model args observations]
  (let [model (-> model dyn/auto-key strip-analytical)
        keys (rng/split-n (rng/ensure-key key) samples)
        ws (mapv (fn [ki]
                   (mx/tidy-run
                     (fn []
                       (let [{:keys [weight]} (p/generate (dyn/with-key model ki) args observations)]
                         (mx/materialize! weight)
                         (mx/item weight)))
                     (fn [_] [])))  ;; nothing to preserve — weight extracted as JS number
                 keys)
        max-w (apply max ws)
        lse (+ max-w (js/Math.log (reduce + (map #(js/Math.exp (- % max-w)) ws))))
        log-ml (- lse (js/Math.log samples))]
    {:log-weights ws
     :log-ml-estimate log-ml}))

(defn vectorized-importance-sampling
  "Vectorized importance sampling. Runs model ONCE with batched handler
   instead of N sequential generate calls. ~10-100x faster for models
   without splice or data-dependent branching.

   opts: {:samples N :key prng-key}
   model: DynamicGF
   args: model arguments
   observations: choice map of observed values

   Returns {:vtrace VectorizedTrace :log-ml-estimate MLX-scalar}"
  [{:keys [samples key] :or {samples 100}} model args observations]
  (let [model (dyn/auto-key model)
        key (rng/ensure-key key)
        vtrace (dyn/vgenerate model args observations samples key)]
    {:vtrace vtrace
     :log-ml-estimate (vec/vtrace-log-ml-estimate vtrace)}))

(defn vectorized-importance-resampling
  "Vectorized importance resampling. Runs model ONCE with batched handler,
   then resamples on GPU. ~10-100x faster than importance-resampling for
   models without splice or data-dependent branching.

   opts: {:samples N :particles M :key prng-key}
   Returns {:vtrace VectorizedTrace (resampled, uniform weights)
            :log-ml-estimate MLX-scalar}"
  [{:keys [particles key] :or {particles 1000}} model args observations]
  (let [{:keys [vtrace log-ml-estimate]}
        (vectorized-importance-sampling {:samples particles :key key}
                                        model args observations)
        resampled (vec/resample-vtrace vtrace (or key (rng/fresh-key)))]
    {:vtrace resampled
     :log-ml-estimate log-ml-estimate}))

(defn importance-resampling
  "Importance resampling. Generate traces via importance sampling,
   then resample proportional to weights.

   opts: {:samples N :particles M :key prng-key}
   Returns vector of N resampled traces.

   For GPU-accelerated resampling on compatible models (no splice,
   no data-dependent branching), use vectorized-importance-resampling."
  [{:keys [samples particles key] :or {samples 100 particles 1000}}
   model args observations]
  (let [model (dyn/auto-key model)
        {:keys [traces log-weights]}
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
