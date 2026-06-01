(ns genmlx.compiled-gen
  "Compile IS-based log-ML estimators for fast repeated evaluation.
   Wraps differentiable inference loss functions in mx/compile-fn
   for single-Metal-kernel dispatch.

   Three variants:
   - compile-log-ml: frozen key, returns (fn [params] -> neg-log-ml)
   - compile-log-ml-gradient: frozen key, returns (fn [params] -> [loss, grad])
   - compile-log-ml-gradient-keyed: variable key, returns (fn [params key] -> [loss, grad])

   All compose on diff/make-is-loss-fn — no schema introspection, no
   multimethod pre-resolution, no parallel handler path. Just mx/compile-fn
   on top of the standard vgenerate pipeline."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.differentiable :as diff]))

;; ---------------------------------------------------------------------------
;; Warm-up helper
;; ---------------------------------------------------------------------------

(defn- warm-up
  "Trigger Metal kernel compilation with dummy inputs."
  [compiled-fn d & extra-args]
  (let [dummy (mx/zeros [d])
        materialize-result (fn [r] (mx/materialize! (if (sequential? r) (first r) r)))]
    (dotimes [_ 2]
      (materialize-result (apply compiled-fn dummy extra-args)))))

;; ---------------------------------------------------------------------------
;; Compiled log-ML (fixed key — deterministic IS)
;; ---------------------------------------------------------------------------

(defn compile-log-ml
  "Compile the log-ML estimator as a single Metal kernel dispatch.
   Key is frozen (same particles every call). For Fisher information.

   opts:
     :n-particles  - IS particles (default 1000)
     :key          - PRNG key (frozen into compilation)"
  [{:keys [n-particles key] :or {n-particles 1000}}
   model args observations param-names]
  (let [key (rng/ensure-key key)
        loss-fn (diff/make-is-loss-fn model args observations param-names n-particles key)
        compiled (mx/compile-fn loss-fn)]
    (warm-up compiled (count param-names))
    compiled))

;; ---------------------------------------------------------------------------
;; Compiled gradient (fixed key)
;; ---------------------------------------------------------------------------

(defn compile-log-ml-gradient
  "Compile the log-ML gradient as a single Metal kernel dispatch.
   Key is frozen. Returns (fn [params] -> [neg-log-ml, grad]).

   opts:
     :n-particles  - IS particles (default 1000)
     :key          - PRNG key (frozen)"
  [{:keys [n-particles key] :or {n-particles 1000}}
   model args observations param-names]
  (let [key (rng/ensure-key key)
        loss-fn (diff/make-is-loss-fn model args observations param-names n-particles key)
        compiled (mx/compile-fn (mx/value-and-grad loss-fn))]
    (warm-up compiled (count param-names))
    compiled))

;; ---------------------------------------------------------------------------
;; Compiled gradient (variable key — for optimization loops)
;; ---------------------------------------------------------------------------

(defn- compile-log-ml-gradient-keyed
  "Compile the log-ML gradient with key as input.
   Returns (fn [params key] -> [neg-log-ml, grad]).
   Gradient is w.r.t. params only.

   opts:
     :n-particles  - IS particles (default 1000)"
  [{:keys [n-particles] :or {n-particles 1000}}
   model args observations param-names]
  (let [loss-grad-fn (diff/make-is-loss-grad-fn model args observations param-names n-particles)
        compiled (mx/compile-fn
                   (fn [p key]
                     (let [{:keys [loss grad]} (loss-grad-fn p key)]
                       [loss grad])))]
    (warm-up compiled (count param-names) (rng/fresh-key 0))
    compiled))
