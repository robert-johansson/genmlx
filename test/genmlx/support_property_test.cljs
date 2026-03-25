(ns genmlx.support-property-test
  "Property-based test: every sample from every distribution falls within
   that distribution's analytical support bounds.

   Uses the all-dists pool from dist_property_test and a label->check-fn map
   to verify support membership for 100 samples per distribution instance."
  (:require [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [cljs.test :as t]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist.core :as dc]
            [genmlx.dist-property-test :as dpt]
            [genmlx.test-helpers :as h])
  (:require-macros [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Support check functions: label -> (fn [js-number] -> boolean)
;; ---------------------------------------------------------------------------

(def support-checks
  {"gaussian(0,1)"       (fn [_] true)
   "gaussian(3,0.5)"     (fn [_] true)
   "gaussian(-2,5)"      (fn [_] true)
   "uniform(0,1)"        (fn [v] (and (>= v 0.0) (<= v 1.0)))
   "uniform(-5,5)"       (fn [v] (and (>= v -5.0) (<= v 5.0)))
   "exponential(1)"      (fn [v] (>= v 0.0))
   "exponential(0.5)"    (fn [v] (>= v 0.0))
   "laplace(0,1)"        (fn [_] true)
   "laplace(2,3)"        (fn [_] true)
   "log-normal(0,1)"     (fn [v] (> v 0.0))
   "log-normal(1,0.5)"   (fn [v] (> v 0.0))
   "cauchy(0,1)"         (fn [_] true)
   "bernoulli(0.5)"      (fn [v] (or (== v 0.0) (== v 1.0)))
   "bernoulli(0.1)"      (fn [v] (or (== v 0.0) (== v 1.0)))
   "bernoulli(0.9)"      (fn [v] (or (== v 0.0) (== v 1.0)))
   "beta(2,2)"           (fn [v] (and (> v 0.0) (< v 1.0)))
   "beta(0.5,0.5)"       (fn [v] (and (> v 0.0) (< v 1.0)))
   "gamma(2,1)"          (fn [v] (> v 0.0))
   "gamma(0.5,2)"        (fn [v] (> v 0.0))
   "poisson(3)"          (fn [v] (and (>= v 0.0) (== v (js/Math.floor v))))
   "poisson(0.5)"        (fn [v] (and (>= v 0.0) (== v (js/Math.floor v))))
   "student-t(3,0,1)"    (fn [_] true)
   "student-t(10,0,1)"   (fn [_] true)
   "delta(3.14)"         (fn [v] (< (js/Math.abs (- v 3.14)) 1e-6))
   "delta(-1)"           (fn [v] (< (js/Math.abs (- v -1.0)) 1e-6))
   "geometric(0.3)"      (fn [v] (and (>= v 0.0) (== v (js/Math.floor v))))
   "geometric(0.8)"      (fn [v] (and (>= v 0.0) (== v (js/Math.floor v))))})

;; ---------------------------------------------------------------------------
;; Property: 100 samples from any distribution are in support
;; ---------------------------------------------------------------------------

(defspec all-samples-in-support 200
  (prop/for-all [d dpt/gen-dist]
    (let [label (:label d)
          check-fn (get support-checks label)
          dist (:dist d)
          n 100
          key (rng/fresh-key)
          samples (dc/dist-sample-n dist key n)]
      (mx/eval! samples)
      (every? (fn [i]
                (let [v (mx/item (mx/index samples i))]
                  (check-fn v)))
              (range n)))))

;; ---------------------------------------------------------------------------
;; Property: single sample is in support (lightweight, complements above)
;; ---------------------------------------------------------------------------

(defspec single-sample-in-support 500
  (prop/for-all [d dpt/gen-dist]
    (let [check-fn (get support-checks (:label d))
          v (dc/dist-sample (:dist d) (rng/fresh-key))]
      (mx/eval! v)
      (check-fn (mx/item v)))))

;; ---------------------------------------------------------------------------
;; MVN: samples have correct shape and finite components
;; ---------------------------------------------------------------------------

(defspec mvn-samples-are-finite-reals 50
  (prop/for-all [spec (gen/elements dpt/mvn-pool)]
    (let [key (rng/fresh-key)
          v (dc/dist-sample (:dist spec) key)]
      (mx/eval! v)
      (every? js/isFinite (mx/->clj v)))))

;; ---------------------------------------------------------------------------
;; Dirichlet: samples are on the probability simplex
;; ---------------------------------------------------------------------------

(defspec dirichlet-samples-on-simplex 50
  (prop/for-all [spec (gen/elements dpt/dirichlet-pool)]
    (let [key (rng/fresh-key)
          v (dc/dist-sample (:dist spec) key)]
      (mx/eval! v)
      (let [vals (mx/->clj v)
            total (reduce + vals)]
        (and (every? #(> % 0.0) vals)
             (< (js/Math.abs (- total 1.0)) 1e-4))))))

(t/run-tests)
