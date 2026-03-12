(ns genmlx.method-selection
  "Automatic inference method selection from model schema metadata.
   Pure decision tree — no MLX ops, no inference execution.
   Reads L3/3.5 analytical plan and schema fields to pick the
   optimal method and tune its hyperparameters."
  (:require [clojure.set]
            [clojure.string :as str]
            [genmlx.inference.util :as u]))

;; ---------------------------------------------------------------------------
;; Helpers — extract structural properties from schema
;; ---------------------------------------------------------------------------

(defn- count-trace-sites
  "Number of trace sites in the schema."
  [schema]
  (count (:trace-sites schema)))

(defn- count-splice-sites
  "Number of splice sites in the schema."
  [schema]
  (count (:splice-sites schema)))

(defn- has-splice?
  "Does the model have any splice sites (sub-model calls)?"
  [schema]
  (pos? (count-splice-sites schema)))

(defn- all-trace-addrs
  "Set of all trace-site addresses."
  [schema]
  (set (map :addr (:trace-sites schema))))

(defn- eliminated-addrs
  "Set of addresses eliminated by L3/3.5 analytical plan.
   Delegates to inference.util/get-eliminated-addresses (model-level).
   Returns #{} when no analytical plan exists."
  [model]
  (or (u/get-eliminated-addresses model) #{}))

(defn- observation-addrs
  "Extract observation addresses from a constraints choice map.
   Accepts either a ChoiceMap Node (with :m internal map) or nil.
   Returns a set of keyword addresses."
  [observations]
  (cond
    (nil? observations) #{}
    ;; ChoiceMap Node — record field :m holds the internal map
    (map? (:m observations))
    (set (keys (:m observations)))
    :else #{}))

(defn- residual-addrs
  "Trace-site addresses NOT eliminated by analytical plan.
   These are the addresses that still require MCMC/VI/etc."
  [model observations]
  (let [schema (:schema model)
        all (all-trace-addrs schema)
        elim (eliminated-addrs model)
        obs (observation-addrs observations)]
    (clojure.set/difference all elim obs)))

(defn- n-residual
  "Number of residual (non-eliminated, non-observed) trace sites."
  [model observations]
  (count (residual-addrs model observations)))

(defn- has-kalman-chains?
  "Does the analytical plan include Kalman chains?"
  [schema]
  (boolean (seq (get-in schema [:analytical-plan :kalman-chains]))))

(defn- has-temporal-splice?
  "Does the model have splice sites that likely reference temporal
   combinators (Unfold or Scan)?  Checks if the gf-form symbol name
   contains 'unfold' or 'scan' (case-insensitive)."
  [schema]
  (boolean
   (some (fn [s]
           (let [gf-name (str/lower-case (str (:gf-form s)))]
             (or (str/includes? gf-name "unfold")
                 (str/includes? gf-name "scan"))))
         (:splice-sites schema))))

(defn- latent-addrs
  "Trace-site addresses that are NOT observed (latent variables)."
  [schema observations]
  (let [all-addrs (all-trace-addrs schema)
        obs (observation-addrs observations)]
    (clojure.set/difference all-addrs obs)))

(defn- n-latent
  "Number of latent (unobserved) trace sites."
  [schema observations]
  (count (latent-addrs schema observations)))

;; ---------------------------------------------------------------------------
;; Decision tree — select-method
;; ---------------------------------------------------------------------------

(def ^:private hmc-threshold
  "Maximum number of residual dimensions for HMC.
   Above this, VI is preferred (gradient-based but amortized)."
  10)

(defn select-method
  "Select the optimal inference method for a model given observations.

   Arguments:
     model        — a DynamicGF (gen function) with :schema
     observations — ChoiceMap of observed data, or nil

   Returns a map:
     :method         — keyword (:exact :kalman :smc :hmc :vi :handler-is)
     :reason         — human-readable string explaining the choice
     :opts           — base method options (tunable via tune-method-opts)
     :eliminated     — set of eliminated addresses
     :residual-addrs — set of residual (non-eliminated, non-observed) addresses
     :n-residual     — count of residual addresses
     :n-latent       — count of latent (unobserved) trace sites"
  [model observations]
  (let [schema (:schema model)
        elim (eliminated-addrs model)
        resid (residual-addrs model observations)
        n-res (count resid)
        n-lat (n-latent schema observations)
        base {:eliminated elim
              :residual-addrs resid
              :n-residual n-res
              :n-latent n-lat}]
    (merge base
           (cond
        ;; 1. All trace sites eliminated or observed → exact
             (and (pos? (count-trace-sites schema))
                  (zero? n-res))
             {:method :exact
              :reason "All trace sites eliminated by analytical plan"
              :opts {}}

        ;; 2. No trace sites at all (empty model) → exact
             (zero? (count-trace-sites schema))
             {:method :exact
              :reason "No trace sites — trivial model"
              :opts {}}

        ;; 3. Kalman chains cover temporal structure → kalman
             (has-kalman-chains? schema)
             {:method :kalman
              :reason "Kalman chains found in analytical plan"
              :opts {:kalman-chains (get-in schema [:analytical-plan :kalman-chains])}}

        ;; 4. Has splice sites → likely temporal/hierarchical → smc
             (has-splice? schema)
             {:method :smc
              :reason (if (has-temporal-splice? schema)
                        "Temporal combinator detected in splice sites"
                        "Splice sites present — SMC for sub-model structure")
              :opts {:n-particles 100
                     :ess-threshold 0.5}}

        ;; 5. Dynamic addresses → handler-based IS (safest fallback)
             (:dynamic-addresses? schema)
             {:method :handler-is
              :reason "Dynamic addresses — shape-based methods not applicable"
              :opts {:n-particles 1000}}

        ;; 6. Static model, few residual dims → HMC
             (and (:static? schema) (<= n-res hmc-threshold))
             {:method :hmc
              :reason (str "Static model with " n-res " residual dims (≤ " hmc-threshold ")")
              :opts {:n-samples 1000
                     :n-warmup 500
                     :step-size 0.01
                     :n-leapfrog 10}}

        ;; 7. Static model, many residual dims → VI
             (and (:static? schema) (> n-res hmc-threshold))
             {:method :vi
              :reason (str "Static model with " n-res " residual dims (> " hmc-threshold ") — VI preferred")
              :opts {:n-iters 2000
                     :n-samples 10
                     :learning-rate 0.01}}

        ;; 8. Fallback → handler-based IS
             :else
             {:method :handler-is
              :reason "Fallback — no specialized method matched"
              :opts {:n-particles 1000}}))))

;; ---------------------------------------------------------------------------
;; Hyperparameter tuning — tune-method-opts
;; ---------------------------------------------------------------------------

(defn tune-method-opts
  "Tune method-specific options based on model structure and data size.

   Arguments:
     selection — result of select-method
     opts      — optional user overrides (merged last, takes priority)

   Returns: updated opts map with tuned hyperparameters."
  ([selection] (tune-method-opts selection {}))
  ([selection user-opts]
   (let [{:keys [method n-residual n-latent]} selection
         base-opts (:opts selection)]
     (merge
      (case method
        :exact
        {}

        :kalman
        base-opts

        :smc
        (let [particles (cond
                          (zero? n-residual) 50
                          (<= n-residual 5) 100
                          (<= n-residual 20) 500
                          :else 1000)]
          (assoc base-opts :n-particles particles))

        :hmc
        (let [n-leapfrog (cond
                           (<= n-residual 3) 10
                           (<= n-residual 7) 15
                           :else 20)
              step-size (cond
                          (<= n-residual 3) 0.05
                          (<= n-residual 7) 0.01
                          :else 0.005)
              n-warmup (cond
                         (<= n-residual 3) 200
                         (<= n-residual 7) 500
                         :else 1000)]
          (assoc base-opts
                 :n-leapfrog n-leapfrog
                 :step-size step-size
                 :n-warmup n-warmup))

        :vi
        (let [n-iters (cond
                        (<= n-residual 20) 2000
                        (<= n-residual 50) 5000
                        :else 10000)
              n-samples (cond
                          (<= n-residual 20) 10
                          (<= n-residual 50) 20
                          :else 50)]
          (assoc base-opts
                 :n-iters n-iters
                 :n-samples n-samples))

        :handler-is
        (let [particles (cond
                          (<= n-latent 5) 1000
                          (<= n-latent 20) 5000
                          :else 10000)]
          (assoc base-opts :n-particles particles))

         ;; Unknown method — throw
        (throw (ex-info "Unknown method" {:method method})))
      user-opts))))
