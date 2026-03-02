(ns genmlx.kernel-reversal-test
  "Tests for kernel reversal declarations and auto-reversal of composites."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.selection :as sel]
            [genmlx.inference.kernel :as kern])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn- assert-true [desc pred]
  (if pred
    (println (str "  PASS: " desc))
    (println (str "  FAIL: " desc))))

;; ---------------------------------------------------------------------------
;; Basic reversal declarations
;; ---------------------------------------------------------------------------

(println "\n== Kernel reversal declarations ==")

(println "\n-- with-reversal --")
(let [k-fwd (fn [trace key] trace)
      k-bwd (fn [trace key] trace)
      k (kern/with-reversal k-fwd k-bwd)]
  (assert-true "reversal of forward is backward"
    (some? (kern/reversal k)))
  (assert-true "reversal is a function"
    (fn? (kern/reversal k)))
  (assert-true "not marked symmetric"
    (not (kern/symmetric? k))))

(println "\n-- symmetric-kernel --")
(let [k-raw (fn [trace key] trace)
      k (kern/symmetric-kernel k-raw)]
  (assert-true "symmetric? returns true"
    (kern/symmetric? k))
  (assert-true "reversal exists"
    (some? (kern/reversal k)))
  (assert-true "reversal is a function"
    (fn? (kern/reversal k))))

(println "\n-- reversed throws on undecorated --")
(let [k (fn [trace key] trace)
      threw? (try
               (kern/reversed k)
               false
               (catch :default _ true))]
  (assert-true "throws on kernel without reversal" threw?))

(println "\n-- reversed returns reversal --")
(let [k-fwd (fn [trace key] trace)
      k-bwd (fn [trace key] trace)
      k (kern/with-reversal k-fwd k-bwd)]
  (assert-true "reversed returns the backward kernel"
    (fn? (kern/reversed k))))

;; ---------------------------------------------------------------------------
;; Built-in kernels are symmetric
;; ---------------------------------------------------------------------------

(println "\n-- built-in kernels are symmetric --")

(let [k (kern/mh-kernel (sel/select :x))]
  (assert-true "mh-kernel is symmetric" (kern/symmetric? k)))

(let [k (kern/prior :x)]
  (assert-true "prior is symmetric" (kern/symmetric? k)))

(let [k (kern/random-walk :x 0.5)]
  (assert-true "random-walk (single addr) is symmetric" (kern/symmetric? k)))

;; ---------------------------------------------------------------------------
;; Composite reversal: chain
;; ---------------------------------------------------------------------------

(println "\n-- composite reversal: chain --")
(let [k1 (kern/symmetric-kernel (fn [t k] t))
      k2 (kern/symmetric-kernel (fn [t k] t))
      k3 (kern/symmetric-kernel (fn [t k] t))
      composed (kern/chain k1 k2 k3)]
  (assert-true "chain of symmetric kernels has reversal"
    (some? (kern/reversal composed))))

(let [k1 (fn [t k] t)  ;; no reversal
      k2 (kern/symmetric-kernel (fn [t k] t))
      composed (kern/chain k1 k2)]
  (assert-true "chain with un-decorated kernel has no reversal"
    (nil? (kern/reversal composed))))

;; ---------------------------------------------------------------------------
;; Composite reversal: repeat-kernel
;; ---------------------------------------------------------------------------

(println "\n-- composite reversal: repeat-kernel --")
(let [k (kern/symmetric-kernel (fn [t key] t))
      repeated (kern/repeat-kernel 5 k)]
  (assert-true "repeat of symmetric kernel has reversal"
    (some? (kern/reversal repeated))))

(let [k (fn [t key] t)  ;; no reversal
      repeated (kern/repeat-kernel 5 k)]
  (assert-true "repeat of undecorated kernel has no reversal"
    (nil? (kern/reversal repeated))))

;; ---------------------------------------------------------------------------
;; Composite reversal: cycle-kernels
;; ---------------------------------------------------------------------------

(println "\n-- composite reversal: cycle-kernels --")
(let [k1 (kern/symmetric-kernel (fn [t k] t))
      k2 (kern/symmetric-kernel (fn [t k] t))
      cycled (kern/cycle-kernels 6 [k1 k2])]
  (assert-true "cycle of symmetric kernels has reversal"
    (some? (kern/reversal cycled))))

;; ---------------------------------------------------------------------------
;; Composite reversal: mix-kernels
;; ---------------------------------------------------------------------------

(println "\n-- composite reversal: mix-kernels --")
(let [k1 (kern/symmetric-kernel (fn [t k] t))
      k2 (kern/symmetric-kernel (fn [t k] t))
      mixed (kern/mix-kernels [[k1 0.5] [k2 0.5]])]
  (assert-true "mix of symmetric kernels has reversal"
    (some? (kern/reversal mixed))))

;; ---------------------------------------------------------------------------
;; Composite reversal: seed
;; ---------------------------------------------------------------------------

(println "\n-- composite reversal: seed --")
(let [k (kern/symmetric-kernel (fn [t key] t))
      seeded (kern/seed k (rng/fresh-key))]
  (assert-true "seed of symmetric kernel has reversal"
    (some? (kern/reversal seeded))))

;; ---------------------------------------------------------------------------
;; Integration: MCMC with reversed kernel produces valid traces
;; ---------------------------------------------------------------------------

(println "\n-- integration: reversed kernel produces valid traces --")
(let [model (gen []
              (let [x (trace :x (dist/gaussian 0 1))]
                x))
      model (dyn/auto-key model)
      trace (:trace (p/generate model [] (cm/choicemap :x (mx/scalar 1.0))))
      k (kern/mh-kernel (sel/select :x))
      rev-k (kern/reversed k)
      key (rng/fresh-key)
      result-trace (rev-k trace key)]
  (assert-true "reversed kernel returns a trace with gen-fn"
    (some? (:gen-fn result-trace)))
  (assert-true "reversed kernel returns a trace with choices"
    (some? (:choices result-trace))))

(println "\n== All kernel reversal tests complete ==")
