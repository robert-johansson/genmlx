(ns genmlx.inference.differentiable
  "Differentiable inference: gradient of log-ML w.r.t. model parameters.
   Composes vectorized IS with mx/value-and-grad for automatic parameter
   learning (empirical Bayes, hyperparameter optimization).

   Two core primitives:
   - make-is-loss-fn: fixed-key loss function (params → neg-log-ML)
   - make-is-loss-grad-fn: variable-key loss+gradient (params, key → {:loss :grad})

   Higher-level functions compose on these:
   - log-ml-gradient: one-shot gradient estimate
   - optimize-params: full optimization via learning/train"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vec]
            [genmlx.learning :as learn]))

;; ---------------------------------------------------------------------------
;; Core: IS-based loss functions for parameter learning
;; ---------------------------------------------------------------------------

(defn make-is-loss-fn
  "Build a differentiable loss function: params → neg-log-ML via IS.
   Key is frozen in closure (same particles every call).
   For use with mx/grad, mx/value-and-grad, or mx/compile-fn."
  [model args observations param-names n-particles key]
  (let [model (dyn/auto-key model)]
    (fn [p]
      (let [store {:params (learn/array->params p param-names)}
            gf (vary-meta model assoc :genmlx.dynamic/param-store store)
            vtrace (dyn/vgenerate gf args observations n-particles key)]
        (mx/negative (vec/vtrace-log-ml-estimate vtrace))))))

(defn make-is-loss-grad-fn
  "Build (fn [params key] -> {:loss :grad}) for IS-based log-ML gradient.
   Creates value-and-grad once; key is a regular argument.
   Gradient is w.r.t. params only (argnums=[0])."
  [model args observations param-names n-particles]
  (let [model (dyn/auto-key model)
        raw-fn (fn [p key]
                 (let [store {:params (learn/array->params p param-names)}
                       gf (vary-meta model assoc :genmlx.dynamic/param-store store)
                       vtrace (dyn/vgenerate gf args observations n-particles key)]
                   (mx/negative (vec/vtrace-log-ml-estimate vtrace))))
        vg (mx/value-and-grad raw-fn [0])]
    (fn [params key]
      (let [[loss grad] (vg params key)]
        {:loss loss :grad grad}))))

;; ---------------------------------------------------------------------------
;; One-shot gradient estimate
;; ---------------------------------------------------------------------------

(defn log-ml-gradient
  "Estimate log p(y;θ) and ∇_θ log p(y;θ) via vectorized IS + autodiff.
   Returns {:log-ml MLX-scalar, :grad [D]-shaped MLX array}.

   opts:
     :n-particles  - IS particles (default 1000)
     :key          - PRNG key (default: fresh)"
  [{:keys [n-particles key] :or {n-particles 1000}}
   model args observations param-names params-array]
  (let [key (rng/ensure-key key)
        loss-grad-fn (make-is-loss-grad-fn model args observations param-names n-particles)
        {:keys [loss grad]} (loss-grad-fn params-array key)]
    {:log-ml (mx/negative loss) :grad grad}))

;; ---------------------------------------------------------------------------
;; Parameter optimization via learning/train
;; ---------------------------------------------------------------------------

(defn optimize-params
  "Maximize log p(y;θ) via Adam on IS-estimated log-ML gradient.
   Delegates to learning/train.

   opts:
     :iterations   - optimization steps (default 200)
     :lr           - learning rate (default 0.01)
     :n-particles  - IS particles per step (default 1000)
     :callback     - (fn [{:iter :log-ml :params}]) per step
     :key          - PRNG key

   Returns {:params final-params, :log-ml-history [numbers...]}."
  [{:keys [iterations lr n-particles callback key]
    :or {iterations 200 lr 0.01 n-particles 1000}}
   model args observations param-names init-params]
  (let [loss-grad-fn (make-is-loss-grad-fn model args observations param-names n-particles)
        result (learn/train
                 {:iterations iterations :lr lr :key key
                  :callback (when callback
                              (fn [{:keys [iter loss params]}]
                                (callback {:iter iter :log-ml (- loss) :params params})))}
                 loss-grad-fn init-params)]
    {:params (:params result)
     :log-ml-history (mapv - (:loss-history result))}))
