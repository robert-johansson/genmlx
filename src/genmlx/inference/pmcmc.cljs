(ns genmlx.inference.pmcmc
  "Particle MCMC: joint parameter and state inference.

   PMMH (Particle Marginal Metropolis-Hastings) samples from p(θ | y)
   by using importance sampling to estimate p(θ, y) within an MH loop.
   The IS estimate is unbiased, so the chain targets the correct
   posterior (Andrieu, Doucet & Holenstein, 2010).

   Particle Gibbs alternates conditional SMC for state trajectory
   refresh with random-walk MH for parameter updates."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.util :as u]
            [genmlx.inference.smc :as smc]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- build-constraints
  "Merge parameter values into an observation choicemap."
  [param-addrs param-vals observations]
  (reduce (fn [cm [addr val]]
            (cm/set-choice cm [addr] val))
          observations
          (map vector param-addrs param-vals)))

(defn- random-walk-propose
  "θ' = θ + N(0, σ). Returns vector of proposed MLX scalars."
  [param-vals proposal-std key]
  (let [stds (if (number? proposal-std)
               (repeat (count param-vals) proposal-std)
               proposal-std)
        keys (rng/split-n key (count param-vals))]
    (mapv (fn [val std ki]
            (mx/add val (mx/multiply (rng/normal ki []) (mx/scalar std))))
          param-vals stds keys)))

(defn- extract-param-vals
  "Extract parameter values from a trace as a vector of MLX scalars."
  [trace param-addrs]
  (mapv #(cm/get-choice (:choices trace) [%]) param-addrs))

(defn- estimate-log-joint
  "Vectorized IS estimate of log p(θ, y).
   Constrains params and observations, samples latent variables.
   Returns JS number."
  [model args constraints n-particles key]
  (let [vtrace (dyn/vgenerate model args constraints n-particles key)
        w (:weight vtrace)
        _ (mx/materialize! w)
        log-ml (mx/subtract (mx/logsumexp w)
                             (mx/scalar (js/Math.log n-particles)))]
    (mx/materialize! log-ml)
    (mx/item log-ml)))

(defn- sample-weighted-trace
  "Categorical sample from weighted particles. Returns one trace."
  [traces log-weights key]
  (let [w-arr (u/materialize-weights log-weights)
        log-probs (mx/subtract w-arr (mx/logsumexp w-arr))
        _ (mx/materialize! log-probs)
        idx (int (mx/realize (rng/categorical key log-probs)))]
    (nth traces idx)))

(defn- mh-param-sweep
  "Random-walk MH on each parameter address in sequence.
   Later params see earlier accepted updates (Gibbs-within-Metropolis)."
  [trace param-addrs proposal-std key]
  (let [stds (if (number? proposal-std)
               (repeat (count param-addrs) proposal-std)
               proposal-std)
        keys (rng/split-n key (count param-addrs))]
    (reduce
      (fn [trace [addr std ki]]
        (let [[noise-key accept-key] (rng/split ki)
              cur-val (cm/get-choice (:choices trace) [addr])
              proposed (mx/add cur-val (mx/multiply (rng/normal noise-key [])
                                                    (mx/scalar std)))
              result (p/update (:gen-fn trace) trace (cm/choicemap addr proposed))
              w (mx/realize (:weight result))]
          (if (u/accept-mh? w accept-key)
            (:trace result)
            trace)))
      trace
      (map vector param-addrs stds keys))))

;; ---------------------------------------------------------------------------
;; PMMH (Particle Marginal Metropolis-Hastings)
;; ---------------------------------------------------------------------------

(defn pmmh
  "Particle Marginal Metropolis-Hastings.

   Samples from p(θ | y) by using IS to estimate p(θ, y) in an MH loop.
   Parameters are trace addresses (with priors). Latent variables are
   marginalized by IS. Random-walk proposals on parameters.

   opts:
     :n-particles    — IS particles per estimate (default 100)
     :n-samples      — posterior samples to collect
     :burn           — burn-in iterations (default n-samples)
     :param-addrs    — vector of parameter address keywords
     :observations   — choicemap of observed data
     :proposal-std   — random-walk std (scalar or per-param vector, default 0.1)
     :init-params    — initial param values as MLX scalars (optional)
     :key            — PRNG key
     :extract-fn     — (fn [param-vals] -> sample) custom extraction
     :callback       — (fn [{:keys [iter params log-ml accepted?]}])

   Returns {:samples [...] :acceptance-rate float :log-mls [...]}."
  [{:keys [n-particles n-samples burn param-addrs observations proposal-std
           init-params key extract-fn callback]
    :or {n-particles 100 proposal-std 0.1}}
   model args]
  (let [burn (or burn n-samples)
        model (dyn/auto-key model)
        extract (or extract-fn (fn [vals] (mapv mx/item vals)))
        ;; Initialize
        [init-key loop-key] (rng/split (rng/ensure-key key))
        init-vals (or init-params
                      (let [trace (:trace (p/generate model args observations))]
                        (extract-param-vals trace param-addrs)))
        _ (doseq [v init-vals] (mx/materialize! v))
        [est-key loop-key] (rng/split loop-key)
        init-log-ml (estimate-log-joint model args
                      (build-constraints param-addrs init-vals observations)
                      n-particles est-key)
        total (+ burn n-samples)]
    (loop [i 0, params init-vals, log-ml init-log-ml
           samples (transient []), log-mls (transient [])
           n-accepted 0, rk loop-key]
      (if (>= i total)
        {:samples (persistent! samples)
         :log-mls (persistent! log-mls)
         :acceptance-rate (/ n-accepted total)}
        (let [[step-key next-key] (rng/split rk)
              [propose-key est-key accept-key] (rng/split-n step-key 3)
              ;; Propose θ' via random walk
              proposed (random-walk-propose params proposal-std propose-key)
              _ (doseq [v proposed] (mx/materialize! v))
              ;; Estimate log p(θ', y)
              constraints (build-constraints param-addrs proposed observations)
              proposed-log-ml (estimate-log-joint model args constraints
                                                  n-particles est-key)
              ;; MH accept/reject
              log-alpha (- proposed-log-ml log-ml)
              accepted? (u/accept-mh? log-alpha accept-key)
              params' (if accepted? proposed params)
              log-ml' (if accepted? proposed-log-ml log-ml)
              past-burn? (>= i burn)]
          (when callback
            (callback {:iter i :params (extract params')
                       :log-ml log-ml' :accepted? accepted?}))
          (when (zero? (mod (inc i) 10)) (mx/sweep-dead-arrays!))
          (recur (inc i) params' log-ml'
                 (if past-burn? (conj! samples (extract params')) samples)
                 (if past-burn? (conj! log-mls log-ml') log-mls)
                 (if accepted? (inc n-accepted) n-accepted)
                 next-key))))))

;; ---------------------------------------------------------------------------
;; Particle Gibbs
;; ---------------------------------------------------------------------------

(defn particle-gibbs
  "Particle Gibbs: conditional SMC for states + MH for parameters.

   Each iteration:
   1. Run csmc conditioned on current trace → sample new trajectory
   2. Random-walk MH sweep on parameter addresses

   For flat models: pass a single observations choicemap (wrapped in a
   vector internally; csmc does conditional IS in one step).
   For sequential models: pass a vector of per-step choicemaps.

   opts:
     :n-particles         — csmc particles (default 50)
     :n-samples           — posterior samples to collect
     :burn                — burn-in iterations (default n-samples)
     :param-addrs         — parameter address keywords (for MH)
     :observations        — choicemap or vector of per-step choicemaps
     :proposal-std        — random-walk std for param MH (default 0.1)
     :rejuvenation-steps  — MH rejuvenation within csmc (default 0)
     :key                 — PRNG key
     :extract-fn          — (fn [trace] -> sample)
     :callback            — per-iteration callback

   Returns {:samples [...] :acceptance-rate float}."
  [{:keys [n-particles n-samples burn param-addrs observations proposal-std
           rejuvenation-steps key extract-fn callback]
    :or {n-particles 50 proposal-std 0.1 rejuvenation-steps 0}}
   model args]
  (let [burn (or burn n-samples)
        model (dyn/auto-key model)
        obs-seq (if (vector? observations) observations [observations])
        extract (or extract-fn
                    (fn [trace]
                      (mapv #(mx/item (cm/get-choice (:choices trace) [%]))
                            param-addrs)))
        ;; Initialize: generate with first obs, update through rest
        [init-key loop-key] (rng/split (rng/ensure-key key))
        init-trace (reduce
                     (fn [trace obs-t]
                       (:trace (p/update (:gen-fn trace) trace obs-t)))
                     (:trace (p/generate model args (first obs-seq)))
                     (rest obs-seq))
        total (+ burn n-samples)]
    (loop [i 0, trace init-trace
           samples (transient []), n-accepted 0, rk loop-key]
      (if (>= i total)
        {:samples (persistent! samples)
         :acceptance-rate (/ n-accepted total)}
        (let [[csmc-key param-key next-key] (rng/split-n rk 3)
              ;; 1. Conditional SMC: refresh state trajectory
              [sample-key param-key'] (rng/split param-key)
              csmc-result (smc/csmc {:particles n-particles
                                     :key csmc-key
                                     :rejuvenation-steps rejuvenation-steps}
                                    model args obs-seq trace)
              new-trace (sample-weighted-trace (:traces csmc-result)
                                              (:log-weights csmc-result)
                                              sample-key)
              ;; 2. MH sweep on parameters
              updated-trace (mh-param-sweep new-trace param-addrs
                                            proposal-std param-key')
              ;; Track acceptance (did params change?)
              accepted? (not (identical? new-trace updated-trace))
              past-burn? (>= i burn)]
          (when callback
            (callback {:iter i :params (extract updated-trace) :accepted? accepted?}))
          (when (zero? (mod (inc i) 5))
            (mx/sweep-dead-arrays!)
            (mx/clear-cache!))
          (recur (inc i) updated-trace
                 (if past-burn? (conj! samples (extract updated-trace)) samples)
                 (if accepted? (inc n-accepted) n-accepted)
                 next-key))))))
