(ns genmlx.inference.smc
  "Sequential Monte Carlo (particle filtering) inference."
  (:require [genmlx.protocols :as p]
            [genmlx.selection :as sel]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.trace :as tr]
            [genmlx.inference.util :as u]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vz]
            [genmlx.combinators :as comb]))


(defn log-ml-increment
  "Marginal-likelihood increment for one SMC step: logsumexp(weights) - log N.
   `w-arr` is a materialized [N]-shaped MLX weight array; `n` is the particle count.

   This is the increment for a step whose INCOMING weights are uniform — the
   init step, or any step immediately after a resample (which resets weights
   to 0, whose logsumexp is log N). When weights carry over from a skipped
   resample, the increment must instead subtract logsumexp of the carried
   weights (see `log-ml-increment-from`), otherwise the previously counted
   mass is counted again every step."
  [w-arr n]
  (mx/subtract (mx/logsumexp w-arr)
               (mx/scalar (js/Math.log n))))

(defn log-ml-increment-from
  "Marginal-likelihood increment with adaptive resampling:
   logsumexp(W_t) - logsumexp(W'_{t-1}), where `prev-arr` holds the
   post-resample weights the step started from (all-zero after a resample,
   so its logsumexp is log N and this reduces to `log-ml-increment`).
   Telescoping across steps gives total = logsumexp(W_T) - log N when no
   resample ever fires."
  [w-arr prev-arr]
  (mx/subtract (mx/logsumexp w-arr)
               (mx/logsumexp prev-arr)))

(defn break-particle-graph!
  "Break the lazy graph for a per-particle generate/update result, preventing
   N-deep accumulation. Materializes the result's weight and score, then sweeps
   dead arrays every 50 particles. Returns the result unchanged."
  [result i]
  (mx/materialize! (:weight result) (:score (:trace result)))
  (when (zero? (mod (inc i) 50)) (mx/sweep-dead-arrays!))
  result)

(defn rejuvenate-trace
  "Apply `steps` MH rejuvenation moves to a single trace over `selection`.
   Each step regenerates the selection and accepts via the MH ratio.
   `step-keys` is a per-step sequence of PRNG keys (or nils)."
  [trace selection step-keys]
  (reduce (fn [t rk]
            (let [{:keys [trace weight]} (p/regenerate (:gen-fn t) t selection)]
              ;; MH ratios are only valid for joint-scored regenerate results
              ;; (genmlx-lbae) — particles enter with the stripped model, so
              ;; this only fires if the entry strip regresses (genmlx-540f).
              (tr/assert-joint! trace :rejuvenate-trace)
              (if (u/accept-mh? (mx/realize weight) rk) trace t)))
          trace step-keys))

(defn break-trace-graph!
  "Break the lazy graph riding a rejuvenated trace: materialize its score
   (forcing the regenerated choice values it references), then sweep dead
   arrays every 50 particles. The rejuvenation counterpart of
   break-particle-graph! — without it a full rejuvenation pass holds all
   N x steps regenerate graphs live at once, and on multi-stage runs that
   peak blows the Metal buffer-count ceiling (genmlx-q6lh). Returns the
   trace unchanged."
  [trace i]
  (mx/materialize! (:score trace))
  (when (zero? (mod (inc i) 50)) (mx/sweep-dead-arrays!))
  trace)

(def strip-analytical
  "Strip analytical handlers from a model for particle-based inference.
   The analytical path returns deterministic posterior means, eliminating
   particle diversity. Particle methods need stochastic prior sampling.
   Public so every particle method (smc, csmc, smcp3) shares ONE stripping
   — mixed stripped/un-stripped scoring puts particles on different weight
   scales. Delegates to the canonical strip in genmlx.dynamic
   (genmlx-jr90: one strip, no copy drift)."
  dyn/strip-analytical-path)

(defn- obs-selection
  "Selection covering exactly the addresses present in an observation
   choicemap (hierarchical when the choicemap nests). Used to put the cSMC
   reference particle's weight on the same obs-only scale as the other
   particles: project of the obs sites = the generate weight the particle
   would have received had only the observations been constrained."
  [obs]
  (letfn [(paths->sel [paths]
            (sel/->Hierarchical
              (into {}
                    (map (fn [[head ps]]
                           (let [tails (map rest ps)]
                             [head (if (some empty? tails)
                                     sel/all
                                     (paths->sel (map vec tails)))])))
                    (group-by first paths))))]
    (paths->sel (cm/addresses obs))))

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
                  (if (>= i (count resid-probs))
                    ;; Floating-point exhaustion: residual probs sum to <1.0;
                    ;; fill remaining slots with the last particle (mirrors
                    ;; u/systematic-resample — genmlx-vqh9).
                    (persistent! (reduce conj! acc (repeat (- n-resid j) (dec (count resid-probs)))))
                    (let [threshold (+ u (/ j n-resid))
                          cumsum' (+ cumsum (nth resid-probs i))]
                      (if (>= cumsum' threshold)
                        (recur i cumsum (inc j) (conj! acc i))
                        (recur (inc i) cumsum' j acc))))))]
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
        (if (>= i (count probs))
          ;; Floating-point exhaustion: probs sum to <1.0 due to rounding; fill
          ;; remaining strata with the last particle (mirrors
          ;; u/systematic-resample — genmlx-vqh9). Avoids an out-of-bounds nth.
          (persistent! (reduce conj! indices (repeat (- n j) (dec (count probs)))))
          (let [threshold (nth uniforms j)
                cumsum' (+ cumsum (nth probs i))]
            (if (>= cumsum' threshold)
              (recur i cumsum (inc j) (conj! indices i))
              (recur (inc i) cumsum' j indices))))))))

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
   Per-particle keys derive from `key` (nil → fresh entropy) — unkeyed
   init made a seeded SMC run irreproducible (genmlx-njaq).
   Returns {:traces :log-weights :log-ml-increment}."
  [model args obs particles key]
  (let [pkeys      (rng/split-n-or-nils key particles)
        results    (mapv (fn [i ki]
                           (break-particle-graph!
                            (p/generate (if ki (dyn/with-key model ki) model)
                                        args obs)
                            i))
                         (range particles) pkeys)
        traces     (mapv :trace results)
        log-weights (mapv :weight results)
        w-arr      (u/materialize-weights log-weights)
        ml-inc     (log-ml-increment w-arr particles)]
    {:traces traces :log-weights log-weights :log-ml-increment ml-inc}))

(defn- smc-rejuvenate
  "Apply rejuvenation-steps MH moves to each trace, breaking each
   particle's regenerate graph as it completes (genmlx-q6lh).
   Returns vector of (possibly updated) traces."
  [traces rejuvenation-steps rejuvenation-selection key]
  (if (pos? rejuvenation-steps)
    (let [keys (if key (rng/split-n key (count traces)) (repeat (count traces) nil))]
      (mapv (fn [i trace ki]
              (let [trace-keys (if ki (rng/split-n ki rejuvenation-steps)
                                      (repeat rejuvenation-steps nil))]
                (break-trace-graph!
                 (rejuvenate-trace trace rejuvenation-selection trace-keys)
                 i)))
            (range) traces keys))
    traces))

(defn- smc-step
  "Subsequent timestep: resample (if ESS low), update particles, rejuvenate.
   cfg holds the fixed per-filter config (:model :particles :ess-threshold
   :rejuvenation-steps :rejuvenation-selection :resample-method); the remaining
   positional args are the per-step loop state. step-spec (nilable) carries
   per-step model arguments {:args :argdiffs} — when present the particle
   advance is update-with-args under those args (genmlx-s8e8), otherwise
   plain update at the trace's own args.
   Returns {:traces :log-weights :log-ml-increment}."
  [{:keys [model particles ess-threshold rejuvenation-steps
           rejuvenation-selection resample-method]}
   traces log-weights obs key step-spec]
  (let [;; Check ESS and resample if needed
        ess        (u/compute-ess log-weights)
        resample?  (< ess (* ess-threshold particles))
        [resample-key update-key rejuv-key]
        (rng/split-n-or-nils key 3)
        [traces' weights'] (if resample?
                             (let [indices (dispatch-resample resample-method
                                            log-weights particles resample-key)]
                               [(mapv #(nth traces %) indices)
                                (vec (repeat particles (mx/scalar 0.0)))])
                             [traces log-weights])
        ;; Update each particle with new observations. Per-particle keys —
        ;; the middle split used to be reserved and never threaded, so
        ;; updates ran on fresh entropy (genmlx-njaq).
        update-keys   (rng/split-n-or-nils update-key particles)
        results       (mapv (fn [i ki trace]
                              (break-particle-graph!
                               (let [gf (if ki
                                          (dyn/with-key (:gen-fn trace) ki)
                                          (:gen-fn trace))]
                                 (if step-spec
                                   (p/update-with-args gf trace (:args step-spec)
                                                       (:argdiffs step-spec :unknown)
                                                       obs)
                                   (p/update gf trace obs)))
                               i))
                            (range) update-keys traces')
        new-traces    (mapv :trace results)
        update-weights (mapv :weight results)
        new-weights   (mapv mx/add weights' update-weights)
        ;; Rejuvenation
        final-traces  (smc-rejuvenate new-traces rejuvenation-steps
                                       rejuvenation-selection rejuv-key)
        ;; log-ML increment: measured against the post-resample weights the
        ;; step started from. After a resample those are uniform (lse = log N);
        ;; when the resample is skipped they carry the already-counted mass,
        ;; which must be subtracted or it is double-counted every step.
        w-arr         (u/materialize-weights new-weights)
        prev-arr      (u/materialize-weights weights')
        ml-inc        (log-ml-increment-from w-arr prev-arr)]
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
    :or {particles 100 ess-threshold 0.5 rejuvenation-steps 0}}
   model args observations-seq]
  (let [model (-> model dyn/auto-key strip-analytical)
        obs-vec (vec observations-seq)
        ;; Default: rejuvenate only the latents. sel/all also resampled the
        ;; observed addresses — targeting the prior (genmlx-7ca0).
        rejuvenation-selection
        (or rejuvenation-selection
            (sel/complement-sel (sel/from-paths (mapcat cm/addresses obs-vec))))
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
              _ (when (pos? t) (mx/sweep-dead-arrays!))
              _ (when (and (pos? t) (zero? (mod t 5))) (mx/clear-cache!))]
          (if (zero? t)
            (let [{:keys [traces log-weights log-ml-increment]}
                  (smc-init-step model args obs-t particles step-key)]
              (when callback
                (callback {:step t :ess (u/compute-ess log-weights)}))
              (recur (inc t) traces log-weights
                     (mx/add log-ml log-ml-increment) next-key))
            (let [{:keys [traces log-weights log-ml-increment ess resampled?]}
                  (smc-step {:model model :particles particles
                             :ess-threshold ess-threshold
                             :rejuvenation-steps rejuvenation-steps
                             :rejuvenation-selection rejuvenation-selection
                             :resample-method resample-method}
                            traces log-weights obs-t step-key nil)]
              (when callback
                (callback {:step t :ess ess :resampled? resampled?}))
              (recur (inc t) traces log-weights
                     (mx/add log-ml log-ml-increment) next-key))))))))

(defn smc-args
  "SMC with changing arguments (genmlx-s8e8): a generic particle filter
   over a sequence of {:args :constraints :argdiffs?} steps — standard
   SMC for growing data without combinator-specific machinery.

   Step 0 initializes particles by constrained generate under its args;
   each later step advances every particle via update-with-args, so the
   per-particle increment is the thesis update weight
   nonfresh(t'; args_t) - score(t; args_(t-1)) — exactly the incremental
   importance weight when the new data site is constrained.

   opts: {:particles N :ess-threshold ratio :resample-method method
          :rejuvenation-steps K :rejuvenation-selection sel :key prng-key}

   Returns {:log-ml MLX-scalar :traces [Trace ...] :log-weights [...]
            :final-ess number}; :final-ess is the post-update ESS at the
   final step (the honest particle-collapse diagnostic). nil when steps
   is empty."
  [{:keys [particles ess-threshold rejuvenation-steps rejuvenation-selection
           resample-method key]
    :or {particles 100 ess-threshold 0.5 rejuvenation-steps 0}}
   model steps]
  (let [model (-> model dyn/auto-key strip-analytical)
        steps-vec (vec steps)
        n-steps (count steps-vec)
        ;; Default: rejuvenate only the latents — never the observed
        ;; addresses (genmlx-7ca0), collected across ALL steps.
        rejuvenation-selection
        (or rejuvenation-selection
            (sel/complement-sel
              (sel/from-paths
                (mapcat (comp cm/addresses :constraints) steps-vec))))]
    (when (pos? n-steps)
      (loop [t 0
             traces nil
             log-weights nil
             log-ml (mx/scalar 0.0)
             rk key]
        (if (>= t n-steps)
          {:log-ml log-ml :traces traces :log-weights log-weights
           :final-ess (u/compute-ess log-weights)}
          (let [{args-t :args obs-t :constraints :as step} (nth steps-vec t)
                obs-t (or obs-t cm/EMPTY)
                [step-key next-key] (rng/split-or-nils rk)
                _ (when (pos? t) (mx/sweep-dead-arrays!))
                _ (when (and (pos? t) (zero? (mod t 5))) (mx/clear-cache!))
                {:keys [traces log-weights log-ml-increment]}
                (if (zero? t)
                  (smc-init-step model args-t obs-t particles step-key)
                  (smc-step {:model model :particles particles
                             :ess-threshold ess-threshold
                             :rejuvenation-steps rejuvenation-steps
                             :rejuvenation-selection rejuvenation-selection
                             :resample-method resample-method}
                            traces log-weights obs-t step-key
                            {:args args-t :argdiffs (:argdiffs step :unknown)}))]
            (recur (inc t) traces log-weights
                   (mx/add log-ml log-ml-increment) next-key)))))))

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
    :or {particles 100 ess-threshold 0.5 rejuvenation-steps 0}}
   model args observations-seq reference-trace]
  (let [model (dyn/auto-key model)
        particle-model (strip-analytical model)
        obs-vec (vec observations-seq)
        ;; Default: rejuvenate only the latents (genmlx-7ca0, see smc).
        rejuvenation-selection
        (or rejuvenation-selection
            (sel/complement-sel (sel/from-paths (mapcat cm/addresses obs-vec))))
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
              _ (when (pos? t) (mx/sweep-dead-arrays!))
              _ (when (and (pos? t) (zero? (mod t 5))) (mx/clear-cache!))]
          (if (zero? t)
            ;; Init step: reference trace at index 0, rest from prior. Thread
            ;; per-particle keys from the user key so a seeded cSMC is
            ;; bit-for-bit reproducible — generating from particle-model's
            ;; auto-key alone leaked fresh entropy, making csmc nondeterministic
            ;; under an explicit :key (genmlx-g5ys; the same fix smc-init-step
            ;; got under genmlx-njaq).
            (let [pkeys (rng/split-n-or-nils step-key (dec particles))
                  other-results (mapv (fn [i ki]
                                        (break-particle-graph!
                                         (p/generate (if ki (dyn/with-key particle-model ki)
                                                         particle-model)
                                                     args obs-t)
                                         i))
                                      (range (dec particles)) pkeys)
                  ;; Reference trace at index 0 (the core of cSMC). Its
                  ;; trajectory is reproduced by constraining ALL its choices,
                  ;; but that puts the generate weight on the full-joint scale
                  ;; (prior densities of the retained latents included) while
                  ;; the other particles carry obs-only weights. Re-score it
                  ;; via project over the step's observation addresses — the
                  ;; weight it would have received had only the observations
                  ;; been constrained — so resampling and the final weighted
                  ;; draw (pmcmc) compare like with like. Same stripped model
                  ;; as the other particles: one score semantics for all.
                  ref-gen (p/generate particle-model args (:choices reference-trace))
                  ref-result (break-particle-graph!
                              {:trace (:trace ref-gen)
                               :weight (p/project particle-model (:trace ref-gen)
                                                  (obs-selection obs-t))}
                              0)
                  traces (into [(:trace ref-result)] (mapv :trace other-results))
                  log-weights (into [(:weight ref-result)] (mapv :weight other-results))
                  w-arr (u/materialize-weights log-weights)
                  ml-inc (log-ml-increment w-arr particles)]
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
                  ;; Update all particles. The reference particle keeps its
                  ;; retained trajectory (the update constrains values it
                  ;; already holds) but its incremental weight is re-scored
                  ;; via project over the step's observation addresses —
                  ;; log p(y_t | x_ref) — the same obs-only scale as the
                  ;; freshly updated particles.
                  obs-sel (obs-selection obs-t)
                  results (mapv (fn [i trace]
                                  (let [r (p/update (:gen-fn trace) trace obs-t)
                                        r (if (= i ref-idx)
                                            (assoc r :weight
                                                   (p/project particle-model (:trace r)
                                                              obs-sel))
                                            r)]
                                    (break-particle-graph! r i)))
                                (range particles) traces')
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
                                             (break-trace-graph!
                                              (rejuvenate-trace
                                               trace rejuvenation-selection
                                               (if ki (rng/split-n ki rejuvenation-steps)
                                                      (repeat rejuvenation-steps nil)))
                                              i)))
                                         (range particles) new-traces keys))
                                 new-traces)
                  ;; Same adaptive-resampling increment as smc-step: subtract
                  ;; the carried mass when the resample was skipped.
                  w-arr (u/materialize-weights new-weights)
                  prev-arr (u/materialize-weights weights')
                  ml-inc (log-ml-increment-from w-arr prev-arr)]
              (when callback
                (callback {:step t :ess ess :resampled? resample?}))
              (recur (inc t) final-traces new-weights
                     (mx/add log-ml ml-inc) next-key))))))))

;; ---------------------------------------------------------------------------
;; Unfold-based SMC — incremental particle filter via unfold-extend
;; ---------------------------------------------------------------------------

(defn smc-unfold
  "Always-resample bootstrap particle filter using Unfold combinator.
   Extends traces one step at a time via unfold-extend, giving O(1) per step
   and staying well under Metal buffer limits.

   kernel: a generative function taking [t state & extra] → new-state
   init-state: initial state passed to the kernel
   observations-seq: sequence of kernel-level choice maps, one per timestep

   opts: {:particles N :key prng-key}

   Returns {:log-ml MLX-scalar :traces [Trace ...] :final-ess number}.
   :final-ess is the PRE-resample ESS at the final timestep — the honest
   particle-collapse diagnostic (post-resample ESS is N by construction).
   nil when observations-seq is empty."
  [{:keys [particles key] :or {particles 100}}
   kernel init-state observations-seq]
  (let [unfold-gf (comb/unfold-combinator kernel)
        obs-vec (vec observations-seq)
        n-steps (count obs-vec)
        init-traces (vec (repeat particles (comb/unfold-empty-trace unfold-gf init-state)))]
    (loop [t 0
           traces init-traces
           log-ml (mx/scalar 0.0)
           final-ess nil
           rk (rng/ensure-key key)]
      (if (>= t n-steps)
        {:log-ml log-ml :traces traces :final-ess final-ess}
        (let [[step-key next-rk] (rng/split rk)
              [extend-key resample-key] (rng/split step-key)
              ;; Extend all particles and resample inside tidy to free intermediates
              step-result
              (mx/tidy-run
                (fn []
                  (let [particle-keys (rng/split-n extend-key particles)
                        results (mapv (fn [tr pk]
                                        (comb/unfold-extend tr (nth obs-vec t) pk))
                                      traces particle-keys)
                        step-weights (mapv :weight results)
                        new-traces (mapv :trace results)
                        w-arr (u/materialize-weights step-weights)
                        ml-inc (log-ml-increment w-arr particles)
                        ;; Materialize before resampling uses ml-inc as JS number
                        _ (mx/materialize! ml-inc)
                        ;; Pre-resample ESS, only needed at the final step
                        ;; (a JS number — safe to escape the tidy scope)
                        ess (when (= t (dec n-steps))
                              (u/ess-from-log-weight-array w-arr))
                        indices (u/systematic-resample step-weights particles resample-key)
                        resampled (mapv #(nth new-traces %) indices)]
                    {:traces resampled :ml-inc ml-inc :ess ess}))
                (fn [result]
                  ;; Preserve resampled trace arrays and ml-inc
                  (into (vec (mapcat u/collect-trace-arrays (:traces result)))
                        [(:ml-inc result)])))
              new-log-ml (mx/add log-ml (:ml-inc step-result))
              ;; Periodic cleanup (every 2 steps to stay under Metal buffer limits)
              _ (when (zero? (mod (inc t) 2))
                  (mx/force-gc!)
                  (mx/clear-cache!))]
          (recur (inc t) (:traces step-result) new-log-ml
                 (or (:ess step-result) final-ess) next-rk))))))

;; ---------------------------------------------------------------------------
;; Batched Unfold SMC — one vgenerate call per timestep for all particles
;; ---------------------------------------------------------------------------

(defn- resample-state
  "Resample state by indices. Handles nil, MLX array, and map-valued state."
  [state indices]
  (cond
    (nil? state) nil
    (map? state) (update-vals state #(mx/take-idx % indices))
    :else (mx/take-idx state indices)))

(defn- materialize-state! [state]
  (cond
    (nil? state) nil
    (map? state) (mx/materialize! (vals state))
    :else (mx/materialize! state)))

(defn batched-smc-unfold
  "Batched bootstrap particle filter for sequential models.
   Runs kernel ONCE per timestep for all N particles via vgenerate.

   kernel: vectorization-compatible gen fn taking [t state]
   init-state: initial state (nil for HMM, or map/array)
   observations-seq: sequence of kernel-level choice maps

   State can be an MLX array, a map of MLX arrays, or nil.
   Map-valued state is resampled per-key (each value indexed by particle).

   Returns {:log-ml MLX-scalar :final-states [N]-shaped :final-ess number}.
   :final-ess is the PRE-resample ESS at the final timestep (post-resample
   ESS is N by construction). nil when observations-seq is empty."
  [{:keys [particles key callback] :or {particles 100}}
   kernel init-state observations-seq]
  (let [obs-vec (vec observations-seq)
        n-steps (count obs-vec)
        kernel (dyn/auto-key kernel)]
    (loop [t 0
           state init-state
           log-ml (mx/scalar 0.0)
           final-ess nil
           rk (rng/ensure-key key)]
      (if (>= t n-steps)
        {:log-ml log-ml :final-states state :final-ess final-ess}
        (let [[step-key next-rk] (rng/split rk)
              [vgen-key resample-key] (rng/split step-key)
              ;; 1. Run kernel ONCE for all N particles
              vtrace (dyn/vgenerate kernel [t state] (nth obs-vec t)
                                    particles vgen-key)
              step-weights (:weight vtrace)
              new-state (:retval vtrace)
              ;; 2. Log-ML increment
              ;; Break lazy graph — weights needed for resampling below
              _ (mx/materialize! step-weights)
              ml-inc (log-ml-increment step-weights particles)
              ;; Materialize ml-inc before accumulation in next iteration
              _ (mx/materialize! ml-inc)
              ;; Pre-resample ESS, only computed at the final step
              ess (when (= t (dec n-steps))
                    (u/ess-from-log-weight-array step-weights))
              ;; 3. Resample (handles array, map, or nil state)
              indices (vz/systematic-resample-indices step-weights
                                                       particles resample-key)
              resampled-state (resample-state new-state indices)
              ;; Break lazy graph — resampled state carried to next timestep
              _ (materialize-state! resampled-state)
              ;; 4. Periodic cleanup
              _ (when (zero? (mod (inc t) 5)) (mx/sweep-dead-arrays!) (mx/clear-cache!))
              _ (when callback (callback {:step t}))]
          (recur (inc t) resampled-state
                 (mx/add log-ml ml-inc) (or ess final-ess) next-rk))))))

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
              ;; Break lazy graph — weight needed for acceptance comparison
              _ (mx/materialize! mh-weight)
              u (rng/uniform (rng/ensure-key accept-key) [n])
              accept-mask (mx/less (mx/log u) mh-weight)
              ;; Materialize mask before per-particle merge
              _ (mx/materialize! accept-mask)
              vtrace (vz/merge-vtraces-by-mask vtrace proposed accept-mask)]
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
    :or {particles 100 ess-threshold 0.5 rejuvenation-steps 0}}
   model args observations-seq]
  (let [model (dyn/auto-key model)
        obs-vec (vec observations-seq)
        ;; Default: rejuvenate only the latents (genmlx-7ca0, see smc).
        rejuvenation-selection
        (or rejuvenation-selection
            (sel/complement-sel (sel/from-paths (mapcat cm/addresses obs-vec))))
        n-steps (count obs-vec)
        [init-key next-key] (rng/split-or-nils key)
        ;; Step 0: batched init
        vtrace (dyn/vgenerate model args (first obs-vec) particles
                              (rng/ensure-key init-key))
        log-ml (vz/vtrace-log-ml-estimate vtrace)]
    (when callback (callback {:step 0 :ess (vz/vtrace-ess vtrace)}))
    (loop [t 1, vtrace vtrace, log-ml log-ml, rk next-key]
      (if (>= t n-steps)
        {:vtrace vtrace :log-ml-estimate log-ml}
        (let [[step-key next-key] (rng/split-or-nils rk)
              [resample-key update-key rejuv-key] (rng/split-n-or-nils step-key 3)
              ;; 1. ESS check + conditional resample
              ess (vz/vtrace-ess vtrace)
              resample? (< ess (* ess-threshold particles))
              vtrace (if resample?
                       (vz/resample-vtrace vtrace resample-key)
                       vtrace)
              prev-weights (:weight vtrace)
              ;; 2. Batched update
              {updated-vtrace :vtrace update-weight :weight}
                (dyn/vupdate (:gen-fn vtrace) vtrace (nth obs-vec t) update-key)
              ;; 3. Accumulate weights
              cumul-weights (mx/add prev-weights update-weight)
              ;; Break lazy graph — weights carried across timesteps
              _ (mx/materialize! cumul-weights)
              vtrace (assoc updated-vtrace :weight cumul-weights)
              ;; 4. Log-ML increment, measured against the post-resample
              ;; weights the step started from (uniform after a resample,
              ;; lse = log N; the carried mass otherwise — subtracting it
              ;; prevents double-counting when the resample is skipped).
              log-ml-inc (mx/subtract (mx/logsumexp cumul-weights)
                                      (mx/logsumexp prev-weights))
              ;; 5. Rejuvenation
              vtrace (vsmc-rejuvenate vtrace rejuvenation-steps
                                       rejuvenation-selection rejuv-key)]
          (when callback
            (callback {:step t :ess ess :resampled? resample?}))
          (when (zero? (mod t 5)) (mx/sweep-dead-arrays!) (mx/clear-cache!))
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
  (let [model (dyn/auto-key model)
        key (rng/ensure-key key)
        vtrace (dyn/vgenerate model args observations particles key)
        log-ml (vz/vtrace-log-ml-estimate vtrace)]
    {:vtrace vtrace :log-ml-estimate log-ml}))
