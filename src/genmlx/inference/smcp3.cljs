(ns genmlx.inference.smcp3
  "SMCP3 — Sequential Monte Carlo with Probabilistic Program Proposals.
   The most powerful inference algorithm in GenJAX. Uses custom generative
   functions as proposals in SMC, with automatic incremental weight computation
   via the edit interface.

   SMCP3 enables locally-optimal proposals, neural proposals, and
   problem-specific sequential inference strategies."
  (:require [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.edit :as edit]
            [genmlx.inference.util :as u]))

;; ---------------------------------------------------------------------------
;; Resampling
;; ---------------------------------------------------------------------------

(defn- systematic-resample
  "Systematic resampling. Returns vector of indices."
  [log-weights n key]
  (let [{:keys [probs]} (u/normalize-log-weights log-weights)
        u0 (if key
             (/ (mx/realize (rng/uniform key [])) n)
             (/ (js/Math.random) n))]
    (loop [i 0 cumsum 0.0 j 0 indices (transient [])]
      (if (>= j n)
        (persistent! indices)
        (let [threshold (+ u0 (/ j n))
              cumsum' (+ cumsum (nth probs i))]
          (if (>= cumsum' threshold)
            (recur i cumsum (inc j) (conj! indices i))
            (recur (inc i) cumsum' j indices)))))))

(defn- compute-ess
  "Effective sample size from log-weights vector."
  [log-weights]
  (let [{:keys [probs]} (u/normalize-log-weights log-weights)]
    (/ 1.0 (reduce + (map #(* % %) probs)))))

;; ---------------------------------------------------------------------------
;; SMCP3 init step
;; ---------------------------------------------------------------------------

(defn smcp3-init
  "Initialize SMCP3 particles using a proposal generative function.
   model: target model GF
   args: model arguments
   observations: first timestep observations
   proposal-gf: proposal GF that takes [] and returns initial choices
   particles: number of particles
   key: PRNG key

   Returns {:traces :log-weights :log-ml-increment}"
  [model args observations proposal-gf particles key]
  (let [keys (rng/split-n (rng/ensure-key key) particles)
        results (mapv
                  (fn [ki]
                    (if proposal-gf
                      ;; Use proposal to generate initial choices
                      (let [proposal-result (p/propose proposal-gf [])
                            proposal-choices (:choices proposal-result)
                            proposal-score (:weight proposal-result)
                            ;; Merge proposal choices with observations
                            merged (cm/merge-cm proposal-choices observations)
                            ;; Generate model trace with merged constraints
                            {:keys [trace weight]} (p/generate model args merged)
                            ;; Importance weight = model-weight - proposal-score
                            iw (mx/subtract weight proposal-score)]
                        {:trace trace :weight iw})
                      ;; No proposal: standard importance sampling
                      (p/generate model args observations)))
                  keys)
        traces (mapv :trace results)
        log-weights (mapv :weight results)
        w-arr (u/materialize-weights log-weights)
        ml-inc (mx/subtract (mx/logsumexp w-arr)
                             (mx/scalar (js/Math.log particles)))]
    {:traces traces :log-weights log-weights :log-ml-increment ml-inc}))

;; ---------------------------------------------------------------------------
;; SMCP3 step
;; ---------------------------------------------------------------------------

(defn smcp3-step
  "One SMCP3 step: extend traces with new observations using proposal kernels.

   traces: current particle traces
   log-weights: current particle log-weights
   model: target model GF
   observations: new observations for this timestep
   forward-kernel: GF that takes [current-trace-choices] and proposes extensions
   backward-kernel: (optional) GF for computing backward proposal score.
                    If nil, uses the edit interface's automatic backward computation.
   particles: number of particles
   ess-threshold: resample when ESS < threshold * N
   rejuvenation-fn: (optional) fn [trace key] -> trace for MCMC rejuvenation
   key: PRNG key

   Returns {:traces :log-weights :log-ml-increment :ess :resampled?}"
  [traces log-weights model observations
   forward-kernel backward-kernel
   particles ess-threshold rejuvenation-fn key]
  (let [;; Check ESS and resample if needed
        ess (compute-ess log-weights)
        resample? (< ess (* ess-threshold particles))
        [resample-key step-key rejuv-key] (rng/split-n-or-nils key 3)
        [traces' weights'] (if resample?
                             (let [indices (systematic-resample log-weights particles resample-key)]
                               [(mapv #(nth traces %) indices)
                                (vec (repeat particles (mx/scalar 0.0)))])
                             [traces log-weights])
        ;; Apply forward kernel to each particle via edit
        step-keys (rng/split-n-or-nils step-key particles)
        results (mapv
                  (fn [trace ki]
                    (if forward-kernel
                      ;; Use proposal edit
                      (let [edit-req (if backward-kernel
                                      (edit/proposal-edit forward-kernel backward-kernel)
                                      ;; Without backward kernel, fall back to constraint update
                                      (edit/constraint-edit observations))
                            result (edit/edit-dispatch (:gen-fn trace) trace edit-req)]
                        {:trace (:trace result) :weight (:weight result)})
                      ;; Standard update
                      (let [result (p/update (:gen-fn trace) trace observations)]
                        {:trace (:trace result) :weight (:weight result)})))
                  traces' step-keys)
        new-traces (mapv :trace results)
        update-weights (mapv :weight results)
        new-weights (mapv mx/add weights' update-weights)
        ;; Rejuvenation
        final-traces (if rejuvenation-fn
                       (let [rkeys (rng/split-n-or-nils rejuv-key particles)]
                         (mapv (fn [t rk] (rejuvenation-fn t rk)) new-traces rkeys))
                       new-traces)
        ;; log-ML increment
        w-arr (u/materialize-weights new-weights)
        ml-inc (mx/subtract (mx/logsumexp w-arr)
                             (mx/scalar (js/Math.log particles)))]
    {:traces final-traces :log-weights new-weights :log-ml-increment ml-inc
     :ess ess :resampled? resample?}))

;; ---------------------------------------------------------------------------
;; SMCP3 main loop
;; ---------------------------------------------------------------------------

(defn smcp3
  "SMCP3: Sequential Monte Carlo with Probabilistic Program Proposals.

   opts:
     :particles         - number of particles (default 100)
     :ess-threshold     - resample when ESS/N < threshold (default 0.5)
     :forward-kernel    - GF for proposing extensions (or nil for standard SMC)
     :backward-kernel   - GF for backward proposals (or nil)
     :init-proposal     - GF for initial particle proposals (or nil)
     :rejuvenation-fn   - fn [trace key] -> trace for MCMC rejuvenation
     :callback          - fn called at each step
     :key               - PRNG key

   observations-seq: sequence of choice maps, one per timestep
   model: generative function
   args: model arguments

   Returns {:traces :log-weights :log-ml-estimate}"
  [{:keys [particles ess-threshold forward-kernel backward-kernel
           init-proposal rejuvenation-fn callback key]
    :or {particles 100 ess-threshold 0.5}}
   model args observations-seq]
  (let [obs-vec (vec observations-seq)
        n-steps (count obs-vec)]
    (loop [t 0
           traces nil
           log-weights nil
           log-ml (mx/scalar 0.0)
           rk key]
      (if (>= t n-steps)
        {:traces traces
         :log-weights log-weights
         :log-ml-estimate log-ml}
        (let [obs-t (nth obs-vec t)
              [step-key next-key] (rng/split-or-nils rk)]
          (if (zero? t)
            ;; Init step
            (let [{:keys [traces log-weights log-ml-increment]}
                  (smcp3-init model args obs-t init-proposal particles step-key)]
              (when callback
                (callback {:step t :ess (compute-ess log-weights)}))
              (recur (inc t) traces log-weights
                     (mx/add log-ml log-ml-increment) next-key))
            ;; Subsequent steps
            (let [{:keys [traces log-weights log-ml-increment ess resampled?]}
                  (smcp3-step traces log-weights model obs-t
                              forward-kernel backward-kernel
                              particles ess-threshold rejuvenation-fn step-key)]
              (when callback
                (callback {:step t :ess ess :resampled? resampled?}))
              (recur (inc t) traces log-weights
                     (mx/add log-ml log-ml-increment) next-key))))))))
