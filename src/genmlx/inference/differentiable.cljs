(ns genmlx.inference.differentiable
  "Differentiable inference: gradient of log-ML w.r.t. model parameters.
   Composes vectorized importance sampling with mx/value-and-grad to enable
   automatic parameter learning (empirical Bayes, hyperparameter optimization).

   Key idea: wrap vgenerate in a differentiable loss function, then use
   mx/value-and-grad to get ∂(log-ML) / ∂θ. All operations are MLX ops,
   so autograd traces through the entire importance sampling pipeline."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vec]
            [genmlx.learning :as learn]))

;; ---------------------------------------------------------------------------
;; Core: gradient of log-ML w.r.t. model parameters
;; ---------------------------------------------------------------------------

(defn- build-param-store
  "Build a param-store map from a flat params array and param names.
   Each param is a slice of the array (differentiable via mx/index)."
  [params-array param-names]
  {:params (into {}
             (map-indexed (fn [i nm] [nm (mx/index params-array i)])
                          param-names))})

(defn log-ml-gradient
  "Compute log-ML and its gradient w.r.t. model parameters via vectorized IS.

   model: DynamicGF using (param :name default) for learnable parameters
   args: model arguments
   observations: ChoiceMap of observed values
   param-names: vector of parameter name keywords
   params-array: flat [D]-shaped MLX array of parameter values
   opts:
     :n-particles  - number of IS particles (default 1000)
     :key          - PRNG key (default: fresh)

   Returns {:log-ml MLX-scalar, :grad [D]-shaped MLX array}.

   The gradient is ∇_θ log p(observations; θ), estimated via the
   log-mean-exp of importance weights from N particles."
  [{:keys [n-particles key] :or {n-particles 1000}}
   model args observations param-names params-array]
  (let [model (dyn/auto-key model)
        key (rng/ensure-key key)
        loss-fn (fn [p]
                  (let [store (build-param-store p param-names)
                        gf (vary-meta model assoc :genmlx.dynamic/param-store store)
                        vtrace (dyn/vgenerate gf args observations n-particles key)]
                    ;; Negative log-ML (minimize)
                    (mx/negative (vec/vtrace-log-ml-estimate vtrace))))
        vg (mx/value-and-grad loss-fn)
        [neg-log-ml grad] (vg params-array)]
    {:log-ml (mx/negative neg-log-ml)
     :grad grad}))

;; ---------------------------------------------------------------------------
;; Optimization loop
;; ---------------------------------------------------------------------------

(defn optimize-params
  "Optimize model parameters by maximizing log p(observations; θ) via
   gradient ascent on the log-ML estimated by vectorized importance sampling.

   opts:
     :iterations   - number of optimization steps (default 200)
     :lr           - learning rate (default 0.01)
     :n-particles  - IS particles per gradient estimate (default 1000)
     :callback     - (fn [{:iter :log-ml :params}]) called each step
     :key          - PRNG key

   model: DynamicGF with (param ...) sites
   args: model arguments
   observations: ChoiceMap of observed values
   param-names: vector of parameter name keywords
   init-params: flat [D]-shaped MLX initial parameter array

   Returns {:params final-params, :log-ml-history [numbers...]}"
  [{:keys [iterations lr n-particles callback key]
    :or {iterations 200 lr 0.01 n-particles 1000}}
   model args observations param-names init-params]
  (let [model (dyn/auto-key model)
        opt-state (learn/adam-init init-params)]
    (loop [i 0
           params init-params
           opt-st opt-state
           history (transient [])]
      (if (>= i iterations)
        {:params params :log-ml-history (persistent! history)}
        (let [step-key (rng/fresh-key)
              {:keys [log-ml grad]}
              (log-ml-gradient {:n-particles n-particles :key step-key}
                               model args observations param-names params)
              ;; Negate grad: log-ml-gradient returns ∂(-log-ML)/∂θ,
              ;; but Adam minimizes, so we pass grad directly (it's already
              ;; the gradient of the negative log-ML)
              _ (mx/materialize! log-ml grad)
              _ (when (zero? (mod i 50)) (mx/clear-cache!))
              log-ml-val (mx/item log-ml)
              [new-params new-opt-st] (learn/adam-step params grad opt-st {:lr lr})]
          (when callback
            (callback {:iter i :log-ml log-ml-val :params new-params}))
          (recur (inc i) new-params new-opt-st
                 (conj! history log-ml-val)))))))
