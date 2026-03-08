(ns genmlx.compiled-gen
  "Compile gen functions for fast repeated inference.

   Wraps the full vgenerate + gradient pipeline in mx/compile-fn so that
   all MLX operations (sampling, scoring, logsumexp, backward pass) fuse
   into a single Metal kernel dispatch. ClojureScript code (handler dispatch,
   volatile!, multimethods) runs normally — only the MLX ops are compiled.

   Requires: model has static trace structure (same addresses every execution).
   First call triggers compilation (slow). Subsequent same-shape calls are fast."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vec]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- build-param-store
  [params-array param-names]
  {:params (into {}
             (map-indexed (fn [i nm] [nm (mx/index params-array i)])
                          param-names))})

(defn- make-loss-fn
  "Build the raw loss function: params-array → neg-log-ML scalar.
   key, model, args, observations, n-particles are captured in closure (frozen)."
  [model args observations param-names n-particles key]
  (fn [p]
    (let [store (build-param-store p param-names)
          gf (vary-meta model assoc :genmlx.dynamic/param-store store)
          vtrace (dyn/vgenerate gf args observations n-particles key)]
      (mx/negative (vec/vtrace-log-ml-estimate vtrace)))))

;; ---------------------------------------------------------------------------
;; Compiled log-ML
;; ---------------------------------------------------------------------------

(defn compile-log-ml
  "Compile the log-ML estimator for a parametric model.
   Returns a function (fn [params-array] -> neg-log-ml) that runs as a
   single Metal kernel dispatch.

   key is frozen into the compiled function (same particles every call).
   Use for Fisher information where deterministic evaluation is required.

   opts:
     :n-particles  - IS particles (default 1000)
     :key          - PRNG key (frozen into compilation)"
  [{:keys [n-particles key] :or {n-particles 1000}}
   model args observations param-names]
  (let [model (dyn/auto-key model)
        key (rng/ensure-key key)
        loss-fn (make-loss-fn model args observations param-names n-particles key)
        compiled (mx/compile-fn loss-fn)]
    ;; Warm up: trigger compilation with dummy params
    (let [D (count param-names)
          dummy (mx/zeros [D])]
      (compiled dummy)
      (mx/materialize! (compiled dummy)))
    compiled))

;; ---------------------------------------------------------------------------
;; Compiled gradient
;; ---------------------------------------------------------------------------

(defn compile-log-ml-gradient
  "Compile the log-ML gradient estimator.
   Returns a function (fn [params-array] -> [neg-log-ml, grad])
   where both forward and backward passes are a single Metal dispatch.

   key is frozen (deterministic IS, same particles every call).
   Ideal for Fisher information and fixed-key optimization.

   opts:
     :n-particles  - IS particles (default 1000)
     :key          - PRNG key (frozen)"
  [{:keys [n-particles key] :or {n-particles 1000}}
   model args observations param-names]
  (let [model (dyn/auto-key model)
        key (rng/ensure-key key)
        loss-fn (make-loss-fn model args observations param-names n-particles key)
        vg (mx/value-and-grad loss-fn)
        compiled-vg (mx/compile-fn vg)]
    ;; Warm up
    (let [D (count param-names)
          dummy (mx/zeros [D])]
      (compiled-vg dummy)
      (mx/materialize! (first (compiled-vg dummy))))
    compiled-vg))

;; ---------------------------------------------------------------------------
;; Compiled gradient with varying key (for optimization loops)
;; ---------------------------------------------------------------------------

(defn compile-log-ml-gradient-keyed
  "Like compile-log-ml-gradient but key is an INPUT, not frozen.
   Returns (fn [params-array key-array] -> [neg-log-ml, grad]).
   Gradient is w.r.t. params only (argnums=[0]).

   For optimization loops where each step uses a fresh key."
  [{:keys [n-particles] :or {n-particles 1000}}
   model args observations param-names]
  (let [model (dyn/auto-key model)
        loss-fn (fn [p key]
                  (let [store (build-param-store p param-names)
                        gf (vary-meta model assoc
                             :genmlx.dynamic/param-store store
                             :genmlx.dynamic/key key)
                        vtrace (dyn/vgenerate gf args observations n-particles key)]
                    (mx/negative (vec/vtrace-log-ml-estimate vtrace))))
        vg (mx/value-and-grad loss-fn [0])
        compiled-vg (mx/compile-fn vg)]
    ;; Warm up
    (let [D (count param-names)
          dummy-p (mx/zeros [D])
          dummy-k (rng/fresh-key 0)]
      (compiled-vg dummy-p dummy-k)
      (mx/materialize! (first (compiled-vg dummy-p dummy-k))))
    compiled-vg))
