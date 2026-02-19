(ns genmlx.inference.smc
  "Sequential Monte Carlo (particle filtering) inference."
  (:require [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.util :as u]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vec]))

(defn- systematic-resample
  "Systematic resampling of particles. Returns vector of indices.
   Optional `key` uses functional PRNG; nil falls back to js/Math.random."
  [log-weights n key]
  (let [{:keys [probs]} (u/normalize-log-weights log-weights)
        u (if key
            (/ (mx/realize (rng/uniform key [])) n)
            (/ (js/Math.random) n))]
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
  (let [{:keys [probs]} (u/normalize-log-weights log-weights)]
    (/ 1.0 (reduce + (map #(* % %) probs)))))

(defn- smc-init-step
  "First timestep: generate particles from prior with constraints.
   Returns {:traces :log-weights :log-ml-increment}."
  [model args obs particles]
  (let [results    (mapv (fn [_] (p/generate model args obs)) (range particles))
        traces     (mapv :trace results)
        log-weights (mapv :weight results)
        w-arr      (u/materialize-weights log-weights)
        ml-inc     (mx/subtract (mx/logsumexp w-arr)
                                (mx/scalar (js/Math.log particles)))]
    {:traces traces :log-weights log-weights :log-ml-increment ml-inc}))

(defn- smc-rejuvenate
  "Apply rejuvenation-steps MH moves to each trace.
   Returns vector of (possibly updated) traces."
  [traces rejuvenation-steps rejuvenation-selection key]
  (if (pos? rejuvenation-steps)
    (let [keys (if key (rng/split-n key (count traces)) (repeat (count traces) nil))]
      (mapv (fn [trace ki]
              (let [trace-keys (if ki (rng/split-n ki rejuvenation-steps)
                                      (repeat rejuvenation-steps nil))]
                (reduce (fn [t rk]
                          (let [gf     (:gen-fn t)
                                result (p/regenerate gf t rejuvenation-selection)
                                w      (mx/realize (:weight result))]
                            (if (u/accept-mh? w rk) (:trace result) t)))
                        trace trace-keys)))
            traces keys))
    traces))

(defn- smc-step
  "Subsequent timestep: resample (if ESS low), update particles, rejuvenate.
   Returns {:traces :log-weights :log-ml-increment}."
  [traces log-weights model obs particles ess-threshold
   rejuvenation-steps rejuvenation-selection key]
  (let [;; Check ESS and resample if needed
        ess        (compute-ess log-weights)
        resample?  (< ess (* ess-threshold particles))
        [resample-key step-key rejuv-key]
        (rng/split-n-or-nils key 3)
        [traces' weights'] (if resample?
                             (let [indices (systematic-resample log-weights particles resample-key)]
                               [(mapv #(nth traces %) indices)
                                (vec (repeat particles (mx/scalar 0.0)))])
                             [traces log-weights])
        ;; Update each particle with new observations
        results       (mapv (fn [trace]
                              (p/update (:gen-fn trace) trace obs))
                            traces')
        new-traces    (mapv :trace results)
        update-weights (mapv :weight results)
        new-weights   (mapv mx/add weights' update-weights)
        ;; Rejuvenation
        final-traces  (smc-rejuvenate new-traces rejuvenation-steps
                                       rejuvenation-selection rejuv-key)
        ;; log ml increment
        w-arr         (u/materialize-weights new-weights)
        ml-inc        (mx/subtract (mx/logsumexp w-arr)
                                    (mx/scalar (js/Math.log particles)))]
    {:traces final-traces :log-weights new-weights :log-ml-increment ml-inc
     :ess ess :resampled? resample?}))

(defn smc
  "Sequential Monte Carlo (particle filtering).

   opts: {:particles N :ess-threshold ratio :rejuvenation-steps K
          :rejuvenation-selection sel :callback fn :key prng-key}

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
  [{:keys [particles ess-threshold rejuvenation-steps rejuvenation-selection callback key]
    :or {particles 100 ess-threshold 0.5 rejuvenation-steps 0
         rejuvenation-selection sel/all}}
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
            (let [{:keys [traces log-weights log-ml-increment]}
                  (smc-init-step model args obs-t particles)]
              (when callback
                (callback {:step t :ess (compute-ess log-weights)}))
              (recur (inc t) traces log-weights
                     (mx/add log-ml log-ml-increment) next-key))
            (let [{:keys [traces log-weights log-ml-increment ess resampled?]}
                  (smc-step traces log-weights model obs-t particles ess-threshold
                            rejuvenation-steps rejuvenation-selection step-key)]
              (when callback
                (callback {:step t :ess ess :resampled? resampled?}))
              (recur (inc t) traces log-weights
                     (mx/add log-ml log-ml-increment) next-key))))))))

;; ---------------------------------------------------------------------------
;; Conditional SMC (cSMC) for particle MCMC / PMCMC
;; ---------------------------------------------------------------------------

(defn csmc
  "Conditional Sequential Monte Carlo: SMC with a retained reference particle.
   The reference particle is never resampled — its trajectory is preserved.
   This is the core kernel for particle Gibbs and particle MCMC.

   opts: {:particles N :ess-threshold ratio :rejuvenation-steps K
          :rejuvenation-selection sel :callback fn :key prng-key}
   model: generative function
   args: model arguments
   observations-seq: sequence of choice maps, one per timestep
   reference-trace: the retained reference particle (from previous PMCMC iteration)

   Returns {:traces :log-weights :log-ml-estimate}"
  [{:keys [particles ess-threshold rejuvenation-steps rejuvenation-selection callback key]
    :or {particles 100 ess-threshold 0.5 rejuvenation-steps 0
         rejuvenation-selection sel/all}}
   model args observations-seq reference-trace]
  (let [obs-vec (vec observations-seq)
        n-steps (count obs-vec)
        ref-idx 0]  ;; reference particle is always at index 0
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
            ;; Init step: reference trace at index 0, rest from prior
            (let [other-results (mapv (fn [_] (p/generate model args obs-t))
                                      (range (dec particles)))
                  ;; Score reference trace
                  ref-result (p/generate model args obs-t)
                  traces (into [(:trace ref-result)] (mapv :trace other-results))
                  log-weights (into [(:weight ref-result)] (mapv :weight other-results))
                  w-arr (u/materialize-weights log-weights)
                  ml-inc (mx/subtract (mx/logsumexp w-arr)
                                       (mx/scalar (js/Math.log particles)))]
              (when callback
                (callback {:step t :ess (compute-ess log-weights)}))
              (recur (inc t) traces log-weights
                     (mx/add log-ml ml-inc) next-key))
            ;; Subsequent steps with conditional resampling
            (let [ess (compute-ess log-weights)
                  resample? (< ess (* ess-threshold particles))
                  [resample-key step-rk rejuv-key] (rng/split-n-or-nils step-key 3)
                  ;; Conditional resampling: reference particle always survives
                  [traces' weights'] (if resample?
                                       (let [indices (systematic-resample
                                                       log-weights particles resample-key)
                                             ;; Force reference particle at index 0
                                             indices' (assoc indices ref-idx ref-idx)]
                                         [(mapv #(nth traces %) indices')
                                          (vec (repeat particles (mx/scalar 0.0)))])
                                       [traces log-weights])
                  ;; Update all particles
                  results (mapv (fn [trace]
                                  (p/update (:gen-fn trace) trace obs-t))
                                traces')
                  new-traces (mapv :trace results)
                  update-weights (mapv :weight results)
                  new-weights (mapv mx/add weights' update-weights)
                  ;; Rejuvenate all except reference
                  final-traces (if (pos? rejuvenation-steps)
                                 (let [keys (if rejuv-key
                                              (rng/split-n rejuv-key particles)
                                              (repeat particles nil))]
                                   (mapv (fn [i trace ki]
                                           (if (= i ref-idx)
                                             trace  ;; Don't rejuvenate reference
                                             (reduce (fn [t rk]
                                                       (let [gf (:gen-fn t)
                                                             result (p/regenerate gf t rejuvenation-selection)
                                                             w (mx/realize (:weight result))]
                                                         (if (u/accept-mh? w rk) (:trace result) t)))
                                                     trace
                                                     (if ki (rng/split-n ki rejuvenation-steps)
                                                            (repeat rejuvenation-steps nil)))))
                                         (range particles) new-traces keys))
                                 new-traces)
                  w-arr (u/materialize-weights new-weights)
                  ml-inc (mx/subtract (mx/logsumexp w-arr)
                                       (mx/scalar (js/Math.log particles)))]
              (when callback
                (callback {:step t :ess ess :resampled? resample?}))
              (recur (inc t) final-traces new-weights
                     (mx/add log-ml ml-inc) next-key))))))))

;; ---------------------------------------------------------------------------
;; Vectorized SMC (single-step, batched init)
;; ---------------------------------------------------------------------------

(defn vsmc-init
  "Vectorized SMC initialization. Runs model ONCE with batched handler
   instead of N sequential generate calls.

   model: DynamicGF
   args: model arguments
   observations: choice map of observed values
   particles: number of particles
   key: PRNG key

   Returns {:vtrace VectorizedTrace :log-ml-estimate MLX-scalar}"
  [model args observations particles key]
  (let [key (rng/ensure-key key)
        vtrace (dyn/vgenerate model args observations particles key)
        log-ml (vec/vtrace-log-ml-estimate vtrace)]
    {:vtrace vtrace :log-ml-estimate log-ml}))
