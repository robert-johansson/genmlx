(ns genmlx.compiled-gen
  "Build IS-based log-ML estimator functions for repeated evaluation.

   Two variants:
   - compile-log-ml: frozen key, returns (fn [params] -> neg-log-ml)
   - compile-log-ml-gradient: frozen key, returns (fn [params] -> [loss, grad])

   All compose on diff/make-is-loss-fn — no schema introspection, no
   multimethod pre-resolution, no parallel handler path. mx/compile-fn is a
   documented identity (see mlx.cljs); the value here is the closed-over
   loss graph construction, not Metal kernel caching."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.differentiable :as diff]))

;; ---------------------------------------------------------------------------
;; Compiled log-ML (fixed key — deterministic IS)
;; ---------------------------------------------------------------------------

(defn compile-log-ml
  "Build the deterministic log-ML estimator (fn [params] -> neg-log-ml).
   Key is frozen (same particles every call). For Fisher information.

   opts:
     :n-particles  - IS particles (default 1000)
     :key          - PRNG key (frozen into the estimator)"
  [{:keys [n-particles key] :or {n-particles 1000}}
   model args observations param-names]
  (let [key (rng/ensure-key key)
        loss-fn (diff/make-is-loss-fn model args observations param-names n-particles key)]
    (mx/compile-fn loss-fn)))

;; ---------------------------------------------------------------------------
;; Compiled gradient (fixed key)
;; ---------------------------------------------------------------------------

(defn compile-log-ml-gradient
  "Build the log-ML gradient estimator (fn [params] -> [neg-log-ml, grad]).
   Key is frozen.

   opts:
     :n-particles  - IS particles (default 1000)
     :key          - PRNG key (frozen)"
  [{:keys [n-particles key] :or {n-particles 1000}}
   model args observations param-names]
  (let [key (rng/ensure-key key)
        loss-fn (diff/make-is-loss-fn model args observations param-names n-particles key)]
    (mx/compile-fn (mx/value-and-grad loss-fn))))

