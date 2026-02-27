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


(defn- residual-resample
  "Residual resampling: deterministically allocate floor(N * w_i) copies,
   then multinomially resample the remainder. Lower variance than systematic."
  [log-weights n key]
  (let [{:keys [probs]} (u/normalize-log-weights log-weights)
        ;; Deterministic part: floor(N * w_i) copies of each particle
        scaled    (mapv #(* n %) probs)
        floors    (mapv #(js/Math.floor %) scaled)
        n-det     (reduce + floors)
        ;; Build deterministic indices
        det-indices (into []
                      (mapcat (fn [i cnt] (repeat cnt i))
                              (range) floors))
        ;; Stochastic part: resample remainder from residuals
        n-resid   (- n n-det)]
    (if (zero? n-resid)
      det-indices
      (let [residuals  (mapv #(- %1 %2) scaled floors)
            resid-sum  (reduce + residuals)
            resid-probs (mapv #(/ % resid-sum) residuals)
            ;; Use systematic resampling on the residuals
            u (/ (mx/realize (rng/uniform (rng/ensure-key key) [])) n-resid)
            resid-indices
              (loop [i 0, cumsum 0.0, j 0, acc (transient [])]
                (if (>= j n-resid)
                  (persistent! acc)
                  (let [threshold (+ u (/ j n-resid))
                        cumsum' (+ cumsum (nth resid-probs i))]
                    (if (>= cumsum' threshold)
                      (recur i cumsum (inc j) (conj! acc i))
                      (recur (inc i) cumsum' j acc)))))]
        (into det-indices resid-indices)))))

(defn- stratified-resample
  "Stratified resampling: draw one uniform per stratum [j/N, (j+1)/N).
   Lower variance than systematic (independent strata)."
  [log-weights n key]
  (let [{:keys [probs]} (u/normalize-log-weights log-weights)
        ;; Generate N stratified uniforms
        keys (rng/split-n (rng/ensure-key key) n)
        uniforms (mapv (fn [j]
                         (let [u (mx/realize (rng/uniform (nth keys j) []))]
                           (/ (+ j u) n)))
                       (range n))]
    (loop [i 0, cumsum 0.0, j 0, indices (transient [])]
      (if (>= j n)
        (persistent! indices)
        (let [threshold (nth uniforms j)
              cumsum' (+ cumsum (nth probs i))]
          (if (>= cumsum' threshold)
            (recur i cumsum (inc j) (conj! indices i))
            (recur (inc i) cumsum' j indices)))))))

(defn- dispatch-resample
  "Dispatch to the appropriate resampling method.
   method: :systematic (default), :residual, or :stratified."
  [method log-weights n key]
  (case (or method :systematic)
    :systematic (u/systematic-resample log-weights n key)
    :residual   (residual-resample log-weights n key)
    :stratified (stratified-resample log-weights n key)))


(defn- smc-init-step
  "First timestep: generate particles from prior with constraints.
   Returns {:traces :log-weights :log-ml-increment}."
  [model args obs particles]
  (let [results    (mapv (fn [_]
                          (let [r (p/generate model args obs)]
                            (mx/eval! (:weight r) (:score (:trace r)))
                            r))
                        (range particles))
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
   rejuvenation-steps rejuvenation-selection resample-method key]
  (let [;; Check ESS and resample if needed
        ess        (u/compute-ess log-weights)
        resample?  (< ess (* ess-threshold particles))
        [resample-key step-key rejuv-key]
        (rng/split-n-or-nils key 3)
        [traces' weights'] (if resample?
                             (let [indices (dispatch-resample resample-method
                                            log-weights particles resample-key)]
                               [(mapv #(nth traces %) indices)
                                (vec (repeat particles (mx/scalar 0.0)))])
                             [traces log-weights])
        ;; Update each particle with new observations
        results       (mapv (fn [trace]
                              (let [r (p/update (:gen-fn trace) trace obs)]
                                (mx/eval! (:weight r) (:score (:trace r)))
                                r))
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
          :rejuvenation-selection sel :resample-method method
          :callback fn :key prng-key}

   resample-method: :systematic (default), :residual, or :stratified

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
  [{:keys [particles ess-threshold rejuvenation-steps rejuvenation-selection
           resample-method callback key]
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
              [step-key next-key] (rng/split-or-nils rk)
              _ (when (and (pos? t) (zero? (mod t 10))) (mx/clear-cache!))]
          (if (zero? t)
            (let [{:keys [traces log-weights log-ml-increment]}
                  (smc-init-step model args obs-t particles)]
              (when callback
                (callback {:step t :ess (u/compute-ess log-weights)}))
              (recur (inc t) traces log-weights
                     (mx/add log-ml log-ml-increment) next-key))
            (let [{:keys [traces log-weights log-ml-increment ess resampled?]}
                  (smc-step traces log-weights model obs-t particles ess-threshold
                            rejuvenation-steps rejuvenation-selection
                            resample-method step-key)]
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
          :rejuvenation-selection sel :resample-method method
          :callback fn :key prng-key}

   resample-method: :systematic (default), :residual, or :stratified

   model: generative function
   args: model arguments
   observations-seq: sequence of choice maps, one per timestep
   reference-trace: the retained reference particle (from previous PMCMC iteration)

   Returns {:traces :log-weights :log-ml-estimate}"
  [{:keys [particles ess-threshold rejuvenation-steps rejuvenation-selection
           resample-method callback key]
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
              [step-key next-key] (rng/split-or-nils rk)
              _ (when (and (pos? t) (zero? (mod t 10))) (mx/clear-cache!))]
          (if (zero? t)
            ;; Init step: reference trace at index 0, rest from prior
            (let [other-results (mapv (fn [_]
                                        (let [r (p/generate model args obs-t)]
                                          (mx/eval! (:weight r) (:score (:trace r)))
                                          r))
                                      (range (dec particles)))
                  ;; Use reference trace at index 0 (the core of cSMC)
                  ref-result (let [r (p/generate model args (:choices reference-trace))]
                               (mx/eval! (:weight r) (:score (:trace r)))
                               r)
                  traces (into [(:trace ref-result)] (mapv :trace other-results))
                  log-weights (into [(:weight ref-result)] (mapv :weight other-results))
                  w-arr (u/materialize-weights log-weights)
                  ml-inc (mx/subtract (mx/logsumexp w-arr)
                                       (mx/scalar (js/Math.log particles)))]
              (when callback
                (callback {:step t :ess (u/compute-ess log-weights)}))
              (recur (inc t) traces log-weights
                     (mx/add log-ml ml-inc) next-key))
            ;; Subsequent steps with conditional resampling
            (let [ess (u/compute-ess log-weights)
                  resample? (< ess (* ess-threshold particles))
                  [resample-key step-rk rejuv-key] (rng/split-n-or-nils step-key 3)
                  ;; Conditional resampling: reference particle always survives
                  [traces' weights'] (if resample?
                                       (let [indices (dispatch-resample resample-method
                                                       log-weights particles resample-key)
                                             ;; Force reference particle at index 0
                                             indices' (assoc indices ref-idx ref-idx)]
                                         [(mapv #(nth traces %) indices')
                                          (vec (repeat particles (mx/scalar 0.0)))])
                                       [traces log-weights])
                  ;; Update all particles
                  results (mapv (fn [trace]
                                  (let [r (p/update (:gen-fn trace) trace obs-t)]
                                    (mx/eval! (:weight r) (:score (:trace r)))
                                    r))
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
;; Vectorized SMC — multi-step batched particle filtering
;; ---------------------------------------------------------------------------

(defn- vsmc-rejuvenate
  "K rounds of vectorized MH rejuvenation on a VectorizedTrace.
   Each round: vregenerate -> sample accept mask -> merge per-particle."
  [vtrace steps selection key]
  (if (zero? steps)
    vtrace
    (loop [k 0, vtrace vtrace, rk key]
      (if (>= k steps)
        vtrace
        (let [[step-key next-key] (rng/split-or-nils rk)
              [regen-key accept-key] (rng/split-or-nils step-key)
              n (:n-particles vtrace)
              {proposed :vtrace mh-weight :weight}
                (dyn/vregenerate (:gen-fn vtrace) vtrace selection regen-key)
              _ (mx/eval! mh-weight)
              u (rng/uniform (rng/ensure-key accept-key) [n])
              accept-mask (mx/less (mx/log u) mh-weight)
              _ (mx/eval! accept-mask)
              vtrace (vec/merge-vtraces-by-mask vtrace proposed accept-mask)]
          (recur (inc k) vtrace next-key))))))

(defn vsmc
  "Vectorized Sequential Monte Carlo (particle filtering).
   Runs model body ONCE per timestep for all N particles via batched handlers.

   opts: {:particles N :ess-threshold ratio :rejuvenation-steps K
          :rejuvenation-selection sel :callback fn :key prng-key}

   observations-seq: sequence of choice maps, one per timestep
   model: DynamicGF
   args: model arguments

   Returns {:vtrace VectorizedTrace :log-ml-estimate MLX-scalar}"
  [{:keys [particles ess-threshold rejuvenation-steps rejuvenation-selection callback key]
    :or {particles 100 ess-threshold 0.5 rejuvenation-steps 0
         rejuvenation-selection sel/all}}
   model args observations-seq]
  (let [obs-vec (vec observations-seq)
        n-steps (count obs-vec)
        [init-key next-key] (rng/split-or-nils key)
        ;; Step 0: batched init
        vtrace (dyn/vgenerate model args (first obs-vec) particles
                              (rng/ensure-key init-key))
        log-ml (vec/vtrace-log-ml-estimate vtrace)]
    (when callback (callback {:step 0 :ess (vec/vtrace-ess vtrace)}))
    (loop [t 1, vtrace vtrace, log-ml log-ml, rk next-key]
      (if (>= t n-steps)
        {:vtrace vtrace :log-ml-estimate log-ml}
        (let [[step-key next-key] (rng/split-or-nils rk)
              [resample-key update-key rejuv-key] (rng/split-n-or-nils step-key 3)
              ;; 1. ESS check + conditional resample
              ess (vec/vtrace-ess vtrace)
              resample? (< ess (* ess-threshold particles))
              vtrace (if resample?
                       (vec/resample-vtrace vtrace resample-key)
                       vtrace)
              prev-weights (:weight vtrace)
              ;; 2. Batched update
              {updated-vtrace :vtrace update-weight :weight}
                (dyn/vupdate (:gen-fn vtrace) vtrace (nth obs-vec t) update-key)
              ;; 3. Accumulate weights
              cumul-weights (mx/add prev-weights update-weight)
              _ (mx/eval! cumul-weights)
              vtrace (assoc updated-vtrace :weight cumul-weights)
              ;; 4. Log-ML increment
              log-ml-inc (vec/vtrace-log-ml-estimate vtrace)
              ;; 5. Rejuvenation
              vtrace (vsmc-rejuvenate vtrace rejuvenation-steps
                                       rejuvenation-selection rejuv-key)]
          (when callback
            (callback {:step t :ess ess :resampled? resample?}))
          (recur (inc t) vtrace (mx/add log-ml log-ml-inc) next-key))))))

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
