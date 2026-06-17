(ns genmlx.fit
  "The `fit` API — one-call entry point for probabilistic inference.
   Composes method selection (WP-3), inference dispatch, and optional
   parameter learning (WP-0/1/2) into a single function.

   (fit model args data)          ;; auto-select method
   (fit model args data opts)     ;; with overrides

   Returns {:method :trace :posterior :log-ml :loss-history :params
            :diagnostics :elapsed-ms}"
  (:require [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.inference.importance :as importance]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.compiled-optimizer :as co]
            [genmlx.method-selection :as ms]))

;; ---------------------------------------------------------------------------
;; Posterior extraction helpers
;; ---------------------------------------------------------------------------

(defn- observation-addr-set
  "Set of top-level observation addresses from a constraints ChoiceMap."
  [data]
  (if (instance? cm/Node data)
    (set (keys (:m data)))
    #{}))

(defn- extract-posterior
  "Extract posterior summary from a single trace.
   Returns {addr {:value number}} for each latent (non-observed) address."
  [trace data]
  (let [choices (:choices trace)
        obs-addrs (observation-addr-set data)]
    (into {}
          (for [[addr sub] (cm/-submaps choices)
                :when (not (obs-addrs addr))
                :when (cm/has-value? sub)]
            [addr {:value (mx/item (cm/get-value sub))}]))))

(defn- mcmc-posterior
  "Extract posterior summary from a vector of MCMC samples.
   samples: vector of clj vectors (each [D]-length, from HMC/compiled-mh).
   addrs: vector of latent address keywords (index-aligned with D).
   Returns {addr {:mean :std :samples}}."
  [samples addrs]
  (let [n (count samples)]
    (when (pos? n)
      (into {}
            (map-indexed
             (fn [i addr]
               (let [vals (mapv #(nth % i) samples)
                     mean (/ (reduce + vals) n)
                     variance (if (> n 1)
                                (/ (transduce (map #(let [d (- % mean)] (* d d))) + vals)
                                   (dec n))
                                0.0)
                     std (js/Math.sqrt variance)]
                 [addr {:mean mean :std std :samples vals}]))
             addrs)))))

;; ---------------------------------------------------------------------------
;; Method dispatcher
;; ---------------------------------------------------------------------------

(defn- opt
  "Coalesce an option across alias keys, falling back to default.
   (opt opts 100 :samples :n-samples) => first truthy of those keys, else 100."
  [opts default & ks]
  (or (some opts ks) default))

(defn- run-method
  "Execute the selected inference method. Returns a partial result map
   (without :method and :elapsed-ms — those are added by `fit`)."
  [model args data method opts]
  (case method
    ;; --- Exact (all-conjugate / trivial) and Kalman (auto-handlers): both
    ;;     get the answer directly from p/generate's weight ---
    (:exact :kalman)
    (let [{:keys [trace weight]} (p/generate model args data)]
      (mx/materialize! weight)
      {:trace trace
       :log-ml (mx/item weight)
       :posterior (extract-posterior trace data)})

    ;; --- HMC ---
    :hmc
    (let [residual (or (:residual-addrs opts) [])
          addrs (vec residual)
          hmc-opts {:samples (opt opts 100 :samples :n-samples)
                    :step-size (opt opts 0.01 :step-size)
                    :leapfrog-steps (opt opts 10 :n-leapfrog)
                    :burn (opt opts 50 :burn :n-warmup)
                    :addresses addrs
                    :key (:key opts)}
          samples (mcmc/hmc hmc-opts model args data)]
      {:trace nil
       :posterior (mcmc-posterior samples addrs)
       :log-ml nil
       :samples samples})

    ;; --- MH (generic MCMC) ---
    :mcmc
    (let [mh-opts {:samples (opt opts 200 :samples)
                   :burn (opt opts 100 :burn)
                   :key (:key opts)}
          traces (mcmc/mh mh-opts model args data)]
      {:trace (last traces)
       :posterior (extract-posterior (last traces) data)
       :log-ml nil
       :samples traces})

    ;; --- :vi — currently MAP point estimation via co/learn, NOT
    ;; variational inference (true VI is avoided here: vmap is not safe in
    ;; all processes). The optimized joint log-density is reported as
    ;; :log-joint; it is NOT a marginal likelihood, so :log-ml stays nil
    ;; (genmlx-7sqe). ---
    :vi
    (let [residual (or (:residual-addrs opts) [])
          addrs (vec residual)
          vi-opts (merge {:iterations (opt opts 500 :iterations :n-iters)
                          :lr (opt opts 0.01 :lr :learning-rate)}
                         (select-keys opts [:key]))
          result (co/learn model args data addrs vi-opts)]
      {:trace nil
       :posterior {:params (:params result)
                   :latent-index (:latent-index result)}
       :log-ml nil
       :log-joint (when (seq (:loss-history result))
                    (- (last (:loss-history result))))
       :loss-history (:loss-history result)})

    ;; --- SMC (no temporal structure → fall back to IS) and handler-based
    ;;     importance sampling (safest fallback): identical IS path ---
    (:smc :handler-is)
    (let [is-opts {:samples (opt opts 200 :particles :n-particles)
                   :key (:key opts)}
          {:keys [traces log-ml-estimate]} (importance/importance-sampling
                                             is-opts model args data)
          best-trace (first traces)]
      (mx/materialize! log-ml-estimate)
      {:trace best-trace
       :posterior (when best-trace (extract-posterior best-trace data))
       :log-ml (mx/item log-ml-estimate)})

    ;; --- Unknown method ---
    (throw (ex-info (str "Unknown inference method: " method)
                    {:method method}))))

;; ---------------------------------------------------------------------------
;; Learning loop (parameter optimization via WP-0/1/2)
;; ---------------------------------------------------------------------------

(defn- run-learning-loop
  "Run parameter learning using compiled optimizer (WP-0/WP-1).
   param-names: vector of param keywords to optimize.
   inference-result: initial inference result from run-method."
  [model args data inference-result param-names method-opts user-opts]
  (let [learn-opts {:iterations (opt user-opts 200 :iterations)
                    :lr (opt user-opts 0.01 :lr)
                    :log-every (opt user-opts 50 :log-every)
                    :callback (:callback user-opts)}
        result (co/learn model args data param-names learn-opts)]
    (merge inference-result
           {:params (:params result)
            :loss-history (:loss-history result)
            :diagnostics {:compilation-level (:compilation-level result)
                          :n-params (:n-params result)}})))

;; ---------------------------------------------------------------------------
;; fit — the public entry point
;; ---------------------------------------------------------------------------

(defn fit
  "Fit a generative model to data. Automatically selects inference method
   and optionally runs parameter optimization.

   model: generative function (with schema from gen macro)
   args:  model arguments (vector)
   data:  ChoiceMap of observed values

   opts (all optional):
     :method     — override automatic method selection
                   :exact :kalman :smc :mcmc :hmc :vi :handler-is
     :learn      — vector of param names to optimize (enables learning loop)
     :iterations — number of optimization iterations (default: auto)
     :lr         — learning rate (default: 0.01)
     :particles  — number of particles for IS/SMC (default: auto)
     :samples    — number of samples for MCMC (default: auto)
     :callback   — (fn [{:iter :loss :method :elapsed}]) called periodically
     :key        — PRNG key for reproducibility
     :verbose?   — print method selection reasoning (default: false)

   Returns:
     {:method       keyword — which method was used
      :trace        Trace   — best/final trace (or nil for VI/HMC)
      :posterior    map     — per-latent summary or distribution params
      :log-ml      number  — log marginal likelihood estimate (when available;
                              nil for :vi, which is currently MAP point
                              estimation and reports :log-joint instead)
      :loss-history [nums] — optimization loss per iteration (if :learn)
      :params      MLX     — learned parameter values (if :learn)
      :diagnostics map     — method-specific diagnostics
      :elapsed-ms  number  — wall-clock time}"
  ([model args data] (fit model args data {}))
  ([model args data opts]
   (let [start-time (js/Date.now)
         model (dyn/auto-key model)
         ;; 1. Method selection — always run to get residual/eliminated info
         auto-selection (ms/select-method model data)
         selection (if (:method opts)
                     (assoc auto-selection
                            :method (:method opts)
                            :reason "User-specified"
                            :opts (merge (:opts auto-selection)
                                         (select-keys opts [:particles :samples :iterations
                                                            :step-size :n-leapfrog])))
                     auto-selection)
         {:keys [method reason]} selection]
     (when (:verbose? opts)
       (println (str "[fit] Selected method: " (name method) " — " reason)))
     (let [;; 2. Tune options (safe — user may override with methods the tuner
           ;;    doesn't know, e.g. :mcmc. Fall back to user opts in that case.)
           tuned (try (ms/tune-method-opts selection)
                      (catch :default _e (:opts selection)))
           method-opts (merge tuned
                              (select-keys opts [:lr :iterations :particles :samples
                                                 :key :step-size :n-leapfrog :burn])
                              ;; Pass residual-addrs through for HMC/VI
                              {:residual-addrs (:residual-addrs selection)})
           ;; 3. Run inference
           result (run-method model args data method method-opts)
           ;; 4. Optional: parameter learning loop
           result (if (:learn opts)
                    (run-learning-loop model args data result
                                       (:learn opts) method-opts opts)
                    result)
           elapsed (- (js/Date.now) start-time)]
       (assoc result
              :method method
              :elapsed-ms elapsed
              :diagnostics (merge (:diagnostics result)
                                {:reason reason
                                 :n-residual (:n-residual selection)
                                 :n-latent (:n-latent selection)}))))))
