(ns genmlx.inference.importance
  "Importance sampling inference."
  (:require [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.util :as u]
            [genmlx.inference.smc :as smc]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vec]))

(def ^:private strip-analytical
  "Remove the L3 analytical path so generate samples from the prior, not
   the deterministic posterior mean — identical particles break
   multi-particle IS (genmlx-540f). Delegates to the canonical strip in
   genmlx.dynamic: the old local copy here removed only 2 of the 5
   analytical schema keys (genmlx-jr90 copy drift)."
  dyn/strip-analytical-path)

(defn- log-ml-from-weights
  "Marginal-likelihood estimate from JS-number log-weights:
   logsumexp(ws) - log(n), via the numerically stable max-shift."
  [ws n]
  (let [max-w (apply max ws)
        lse (+ max-w (js/Math.log (reduce + (map #(js/Math.exp (- % max-w)) ws))))]
    (- lse (js/Math.log n))))

(defn importance-sampling
  "Importance sampling. Generate traces constrained by observations,
   return weighted samples.

   opts: {:samples N :key prng-key :gc-every K}
   model: generative function
   args: model arguments
   observations: choice map of observed values
   :gc-every — sweep dead native arrays every K particles (default 50); pass a
   smaller K for heavy nested-combinator/plate models that allocate hundreds of
   leaves per sample.

   Returns {:traces [Trace ...] :log-weights [MLX-scalar ...]
            :log-ml-estimate MLX-scalar}"
  [{:keys [samples key gc-every] :or {samples 100 gc-every 50}} model args observations]
  (let [model (-> model dyn/auto-key strip-analytical)
        keys (rng/split-n (rng/ensure-key key) samples)
        ;; Deep-materialize EACH FULL trace (all choice leaves + retval + score),
        ;; not just weight+score: an un-materialized leaf MxArray pins its whole
        ;; per-sample computation subgraph (hundreds of native buffers for a
        ;; nested-combinator plate) alive, which the dead-buffer sweep cannot free
        ;; while the retained trace still references it. u/materialize-state
        ;; evaluates every leaf, detaching it so the per-sample intermediates
        ;; become dead + sweepable (genmlx-py4a). mh already does this via
        ;; collect-samples/tidy-step.
        results (into []
                      (map-indexed
                       (fn [i ki]
                         (let [r (p/generate (dyn/with-key model ki) args observations)]
                           (mx/materialize! (:weight r))
                           (u/materialize-state (:trace r))
                           (when (zero? (mod (inc i) gc-every)) (mx/sweep-dead-arrays!))
                           r)))
                      keys)
        traces     (mapv :trace results)
        log-weights (mapv :weight results)
        ;; log marginal likelihood estimate = logsumexp(weights) - log(N)
        weights-arr (u/materialize-weights log-weights)
        log-ml (smc/log-ml-increment weights-arr samples)]
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
        log-ml (log-ml-from-weights ws samples)]
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

   opts: {:particles M :key prng-key}
   Returns {:vtrace VectorizedTrace (resampled, uniform weights)
            :log-ml-estimate MLX-scalar}"
  [{:keys [particles key] :or {particles 1000}} model args observations]
  (let [;; Split once: reusing the caller's key for both vgenerate and the
        ;; resample uniforms correlates resampling with particle generation
        ;; (genmlx-njaq).
        [k-gen k-res] (rng/split (rng/ensure-key key))
        {:keys [vtrace log-ml-estimate]}
        (vectorized-importance-sampling {:samples particles :key k-gen}
                                        model args observations)
        resampled (vec/resample-vtrace vtrace k-res)]
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
        ;; Split once: split-n of the SAME key for per-particle generate and
        ;; again for the resample uniforms makes the first `samples` resample
        ;; keys identical to the generate keys (split prefix semantics) —
        ;; resampling correlated with particle generation (genmlx-njaq).
        [k-gen k-res] (rng/split (rng/ensure-key key))
        {:keys [traces log-weights]}
        (importance-sampling {:samples particles :key k-gen} model args observations)
        ;; Normalize weights via log-softmax
        {:keys [probs]} (u/normalize-log-weights log-weights)
        keys (rng/split-n k-res samples)]
    ;; Resample
    (mapv (fn [ki]
            (let [u (mx/realize (rng/uniform ki []))
                  idx (->> (reductions + probs)
                           (keep-indexed (fn [i c] (when (>= c u) i)))
                           first)]
              (nth traces (or idx (dec (count traces))))))
          keys)))
