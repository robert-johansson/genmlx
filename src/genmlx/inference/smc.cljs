(ns genmlx.inference.smc
  "Sequential Monte Carlo (particle filtering) inference."
  (:require [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]))

(defn- systematic-resample
  "Systematic resampling of particles. Returns vector of indices."
  [log-weights n]
  (let [weights-arr (mx/array (mapv (fn [w] (mx/eval! w) (mx/item w)) log-weights))
        log-probs (mx/subtract weights-arr (mx/logsumexp weights-arr))
        _ (mx/eval! log-probs)
        probs (mx/->clj (mx/exp log-probs))
        u (/ (js/Math.random) n)]
    (loop [i 0, cumsum 0.0, j 0, indices (transient [])]
      (if (>= j n)
        (persistent! indices)
        (let [threshold (+ u (/ j n))
              cumsum' (+ cumsum (nth probs i))]
          (if (>= cumsum' threshold)
            (recur i cumsum (inc j) (conj! indices i))
            (recur (inc i) cumsum' j indices)))))))

(defn- compute-ess
  "Compute effective sample size from log-weights."
  [log-weights]
  (let [weights-arr (mx/array (mapv (fn [w] (mx/eval! w) (mx/item w)) log-weights))
        log-probs (mx/subtract weights-arr (mx/logsumexp weights-arr))
        _ (mx/eval! log-probs)
        probs (mx/->clj (mx/exp log-probs))]
    (/ 1.0 (reduce + (map #(* % %) probs)))))

(defn smc
  "Sequential Monte Carlo (particle filtering).

   opts: {:particles N :ess-threshold ratio :rejuvenation-steps K
          :rejuvenation-selection sel :callback fn}

   observations-seq: sequence of choice maps, one per timestep
   model: generative function
   args: model arguments (used for each step)

   Each timestep:
   1. Extend particles with new observations
   2. Reweight
   3. Optionally resample (when ESS < threshold)
   4. Optionally rejuvenate (MH steps)

   Returns {:traces [Trace ...] :log-weights [MLX-scalar ...]
            :log-ml-estimate MLX-scalar}"
  [{:keys [particles ess-threshold rejuvenation-steps rejuvenation-selection callback]
    :or {particles 100 ess-threshold 0.5 rejuvenation-steps 0
         rejuvenation-selection sel/all}}
   model args observations-seq]
  (let [obs-vec (vec observations-seq)
        n-steps (count obs-vec)]
    ;; Initialize with first observation
    (loop [t 0
           traces nil
           log-weights nil
           log-ml (mx/scalar 0.0)]
      (if (>= t n-steps)
        {:traces traces
         :log-weights log-weights
         :log-ml-estimate log-ml}
        (let [obs-t (nth obs-vec t)]
          (if (zero? t)
            ;; First step: generate from prior with constraints
            (let [results (mapv (fn [_] (p/generate model args obs-t))
                                (range particles))
                  new-traces (mapv :trace results)
                  new-weights (mapv :weight results)
                  ;; log ml increment = logsumexp(w) - log(N)
                  w-arr (mx/array (mapv (fn [w] (mx/eval! w) (mx/item w)) new-weights))
                  ml-inc (mx/subtract (mx/logsumexp w-arr)
                                       (mx/scalar (js/Math.log particles)))]
              (when callback
                (callback {:step t :ess (compute-ess new-weights)}))
              (recur (inc t) new-traces new-weights (mx/add log-ml ml-inc)))
            ;; Subsequent steps: update particles with new observations
            (let [;; Check ESS and resample if needed
                  ess (compute-ess log-weights)
                  resample? (< ess (* ess-threshold particles))
                  [traces' weights'] (if resample?
                                       (let [indices (systematic-resample log-weights particles)]
                                         [(mapv #(nth traces %) indices)
                                          (vec (repeat particles (mx/scalar 0.0)))])
                                       [traces log-weights])
                  ;; Update each particle with new observations
                  results (mapv (fn [trace]
                                  (p/update (tr/get-gen-fn trace) trace obs-t))
                                traces')
                  new-traces (mapv :trace results)
                  update-weights (mapv :weight results)
                  new-weights (mapv (fn [w uw] (mx/add w uw))
                                    weights' update-weights)
                  ;; Rejuvenation (MH steps)
                  final-traces (if (pos? rejuvenation-steps)
                                 (mapv (fn [trace]
                                         (reduce (fn [t _]
                                                   (let [gf (tr/get-gen-fn t)
                                                         result (p/regenerate gf t rejuvenation-selection)
                                                         w (do (mx/eval! (:weight result))
                                                               (mx/item (:weight result)))
                                                         accept? (or (>= w 0)
                                                                     (< (js/Math.log (js/Math.random)) w))]
                                                     (if accept? (:trace result) t)))
                                                 trace (range rejuvenation-steps)))
                                       new-traces)
                                 new-traces)
                  ;; log ml increment
                  w-arr (mx/array (mapv (fn [w] (mx/eval! w) (mx/item w)) new-weights))
                  ml-inc (mx/subtract (mx/logsumexp w-arr)
                                       (mx/scalar (js/Math.log particles)))]
              (when callback
                (callback {:step t :ess (compute-ess new-weights) :resampled? resample?}))
              (recur (inc t) final-traces new-weights (mx/add log-ml ml-inc)))))))))
