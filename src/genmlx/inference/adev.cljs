(ns genmlx.inference.adev
  "ADEV (Automatic Differentiation of Expected Values) gradient estimation.
   Computes unbiased gradients ∇_θ E_{p(·;θ)}[cost(trace)] by automatically
   choosing the right estimator at each trace site:
   - Reparameterizable distributions: reparameterization trick (gradient flows through sample)
   - Non-reparameterizable: score function estimator (REINFORCE surrogate)"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist.core :as dc]
            [genmlx.handler :as h]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.learning :as learn]))

;; ---------------------------------------------------------------------------
;; Reparam detection
;; ---------------------------------------------------------------------------

(defn has-reparam?
  "Check if a distribution type has a registered dist-reparam method."
  [dist]
  (contains? (methods dc/dist-reparam) (:type dist)))

;; ---------------------------------------------------------------------------
;; ADEV handler transition
;; ---------------------------------------------------------------------------

(defn- adev-transition
  "Pure state transition for ADEV execution.
   Reparameterizable sites: sample via dist-reparam, gradients flow through value.
   Non-reparameterizable: sample via dist-sample, stop-gradient the value,
   accumulate log-prob into :reinforce-lp for the REINFORCE surrogate term."
  [state addr dist]
  (let [[k1 k2] (rng/split (:key state))
        reparam? (has-reparam? dist)
        value (if reparam?
                (dc/dist-reparam dist k2)
                (mx/stop-gradient (dc/dist-sample dist k2)))
        lp (dc/dist-log-prob dist value)]
    [value (-> state
             (assoc :key k1)
             (update :choices #(cm/set-choice % [addr] value))
             (update :score #(mx/add % lp))
             (cond-> (not reparam?)
               (update :reinforce-lp #(mx/add % lp))))]))

(defn- adev-handler
  "Handler wrapper for ADEV execution."
  [addr dist]
  (let [[value state'] (adev-transition @h/*state* addr dist)]
    (vreset! h/*state* state')
    value))

;; ---------------------------------------------------------------------------
;; ADEV execution
;; ---------------------------------------------------------------------------

(defn adev-execute
  "Execute a generative function under the ADEV handler.
   Returns {:trace Trace, :reinforce-lp MLX-scalar}."
  [gf args key]
  (let [key (rng/ensure-key key)
        result (h/run-handler adev-handler
                 {:choices cm/EMPTY
                  :score (mx/scalar 0.0)
                  :reinforce-lp (mx/scalar 0.0)
                  :key key
                  :executor nil}
                 #(apply (:body-fn gf) args))]
    {:trace (tr/make-trace
              {:gen-fn gf :args args
               :choices (:choices result)
               :retval (:retval result)
               :score (:score result)})
     :reinforce-lp (:reinforce-lp result)}))

;; ---------------------------------------------------------------------------
;; Surrogate loss
;; ---------------------------------------------------------------------------

(defn adev-surrogate
  "Build the ADEV surrogate loss for a single sample.
   cost-fn: (fn [trace] -> MLX-scalar) — the cost to minimize.
   The surrogate is: cost + stop_gradient(cost) * reinforce-lp
   Taking mx/grad of this gives an unbiased gradient estimate."
  [gf args cost-fn key]
  (let [{:keys [trace reinforce-lp]} (adev-execute gf args key)
        cost (cost-fn trace)]
    (mx/add cost (mx/multiply (mx/stop-gradient cost) reinforce-lp))))

;; ---------------------------------------------------------------------------
;; Gradient estimation with param-store integration
;; ---------------------------------------------------------------------------

(defn adev-gradient
  "Compute ADEV gradient of E[cost] w.r.t. a flat parameter array.
   opts: {:n-samples N} — number of samples for Monte Carlo estimate (default 1).
   param-names: vector of parameter name keywords.
   params-array: flat 1-D MLX array of parameter values.
   Returns {:loss MLX-scalar, :grad MLX-array}."
  [{:keys [n-samples] :or {n-samples 1}} gf args cost-fn param-names params-array]
  (let [loss-fn (fn [p]
                  (let [store {:params (into {}
                                        (map-indexed
                                          (fn [i nm] [nm (mx/index p i)])
                                          param-names))}
                        keys (rng/split-n (rng/fresh-key) n-samples)
                        surrogates (mapv (fn [k]
                                          (binding [h/*param-store* store]
                                            (adev-surrogate gf args cost-fn k)))
                                        keys)]
                    ;; Average over samples
                    (mx/divide (reduce mx/add surrogates)
                               (mx/scalar (double n-samples)))))
        vg (mx/value-and-grad loss-fn)
        [loss grad] (vg params-array)]
    {:loss loss :grad grad}))

;; ---------------------------------------------------------------------------
;; Optimization loop
;; ---------------------------------------------------------------------------

(defn adev-optimize
  "Optimize E[cost] via ADEV gradient estimation with Adam.
   opts:
     :iterations  - number of steps (default 100)
     :lr          - learning rate (default 0.01)
     :n-samples   - samples per gradient estimate (default 1)
     :callback    - (fn [{:iter :loss :params}]) called each step
     :key         - PRNG key (unused, kept for API consistency)
   gf: DynamicGF model
   args: model arguments
   cost-fn: (fn [trace] -> MLX-scalar)
   param-names: vector of parameter name keywords
   init-params: initial flat MLX parameter array
   Returns {:params final-params, :loss-history [numbers...]}."
  [{:keys [iterations lr n-samples callback key]
    :or {iterations 100 lr 0.01 n-samples 1}}
   gf args cost-fn param-names init-params]
  (let [opt-state (learn/adam-init init-params)]
    (loop [i 0
           params init-params
           opt-st opt-state
           losses (transient [])]
      (if (>= i iterations)
        {:params params :loss-history (persistent! losses)}
        (let [{:keys [loss grad]} (adev-gradient {:n-samples n-samples}
                                                  gf args cost-fn
                                                  param-names params)
              _ (mx/eval! loss grad)
              loss-val (mx/item loss)
              [new-params new-opt-st] (learn/adam-step params grad opt-st {:lr lr})]
          (when callback
            (callback {:iter i :loss loss-val :params new-params}))
          (recur (inc i) new-params new-opt-st
                 (conj! losses loss-val)))))))
