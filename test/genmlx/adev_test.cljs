(ns genmlx.adev-test
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as h]
            [genmlx.inference.adev :as adev])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(println "\n=== ADEV Gradient Estimation Tests ===\n")

;; ---------------------------------------------------------------------------
;; Test 1: has-reparam? detection
;; ---------------------------------------------------------------------------

(println "-- has-reparam? detection --")
(assert-true "gaussian is reparameterizable"
  (adev/has-reparam? (dist/gaussian 0 1)))
(assert-true "uniform is reparameterizable"
  (adev/has-reparam? (dist/uniform 0 1)))
(assert-true "exponential is reparameterizable"
  (adev/has-reparam? (dist/exponential 1)))
(assert-true "laplace is reparameterizable"
  (adev/has-reparam? (dist/laplace 0 1)))
(assert-true "bernoulli is NOT reparameterizable"
  (not (adev/has-reparam? (dist/bernoulli 0.5))))
(assert-true "categorical is NOT reparameterizable"
  (not (adev/has-reparam? (dist/categorical (mx/array [-1 -1])))))
(assert-true "beta is NOT reparameterizable"
  (not (adev/has-reparam? (dist/beta-dist 2 2))))

;; ---------------------------------------------------------------------------
;; Test 2: Pure reparam model — ADEV execution
;; ---------------------------------------------------------------------------

(println "\n-- Pure reparam model (ADEV execute) --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))]
                x))
      key (rng/fresh-key)
      {:keys [trace reinforce-lp]} (adev/adev-execute model [] key)]
  (assert-true "trace has choices" (cm/has-value? (cm/get-submap (:choices trace) :x)))
  (assert-true "trace has score" (number? (mx/item (:score trace))))
  ;; Pure reparam model: no REINFORCE terms, reinforce-lp should be 0
  (assert-close "reinforce-lp is 0 for pure reparam" 0.0 (mx/item reinforce-lp) 1e-6))

;; ---------------------------------------------------------------------------
;; Test 3: Mixed model — gaussian (reparam) + bernoulli (REINFORCE)
;; ---------------------------------------------------------------------------

(println "\n-- Mixed model (reparam + REINFORCE) --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))
                    b (dyn/trace :b (dist/bernoulli 0.5))]
                (mx/add x b)))
      key (rng/fresh-key)
      {:keys [trace reinforce-lp]} (adev/adev-execute model [] key)]
  (assert-true "trace has gaussian choice"
    (cm/has-value? (cm/get-submap (:choices trace) :x)))
  (assert-true "trace has bernoulli choice"
    (cm/has-value? (cm/get-submap (:choices trace) :b)))
  ;; reinforce-lp should be non-zero (it's the log-prob of the bernoulli site)
  (assert-true "reinforce-lp is finite" (js/isFinite (mx/item reinforce-lp))))

;; ---------------------------------------------------------------------------
;; Test 4: ADEV surrogate produces finite scalar
;; ---------------------------------------------------------------------------

(println "\n-- ADEV surrogate loss --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))
                    b (dyn/trace :b (dist/bernoulli 0.5))]
                (mx/add x b)))
      cost-fn (fn [trace] (mx/square (:retval trace)))
      key (rng/fresh-key)
      surrogate (adev/adev-surrogate model [] cost-fn key)]
  (mx/eval! surrogate)
  (assert-true "surrogate is finite" (js/isFinite (mx/item surrogate)))
  (assert-true "surrogate is non-negative (squared cost)" (>= (mx/item surrogate) 0.0)))

;; ---------------------------------------------------------------------------
;; Test 5: ADEV gradient with param store
;; ---------------------------------------------------------------------------

(println "\n-- ADEV gradient with params --")
(let [;; Model: sample x ~ gaussian(mu, 1), cost = (x - 3)^2
      ;; Optimal mu = 3
      model (gen []
              (let [mu (dyn/param :mu 0.0)
                    x (dyn/trace :x (dist/gaussian mu (mx/scalar 1.0)))]
                x))
      cost-fn (fn [trace]
                (mx/square (mx/subtract (:retval trace) (mx/scalar 3.0))))
      param-names [:mu]
      params (mx/array [0.0])
      {:keys [loss grad]} (adev/adev-gradient {:n-samples 10}
                                               model [] cost-fn
                                               param-names params)]
  (mx/eval! loss grad)
  (assert-true "loss is finite" (js/isFinite (mx/item loss)))
  (assert-true "grad is finite" (js/isFinite (mx/item (mx/index grad 0))))
  ;; At mu=0, cost=(x-3)^2, gradient should be negative (need to increase mu)
  (assert-true "grad is negative (should increase mu)" (< (mx/item (mx/index grad 0)) 0)))

;; ---------------------------------------------------------------------------
;; Test 6: ADEV optimization convergence
;; ---------------------------------------------------------------------------

(println "\n-- ADEV optimization convergence --")
(let [;; Model: x ~ gaussian(mu, 1), cost = (x - 5)^2
      ;; Optimal mu = 5
      model (gen []
              (let [mu (dyn/param :mu 0.0)
                    x (dyn/trace :x (dist/gaussian mu (mx/scalar 1.0)))]
                x))
      cost-fn (fn [trace]
                (mx/square (mx/subtract (:retval trace) (mx/scalar 5.0))))
      param-names [:mu]
      init-params (mx/array [0.0])
      {:keys [params loss-history]} (adev/adev-optimize
                                      {:iterations 200 :lr 0.1 :n-samples 10}
                                      model [] cost-fn param-names init-params)
      final-mu (mx/item (mx/index params 0))
      first-loss (first loss-history)
      last-loss (last loss-history)]
  (assert-close "mu converges near 5.0" 5.0 final-mu 1.5)
  (assert-true "loss decreases" (< last-loss first-loss)))

;; ---------------------------------------------------------------------------
;; Test 7: Gradient correctness via finite differences (single gaussian site)
;; ---------------------------------------------------------------------------

(println "\n-- Gradient correctness (finite difference check) --")
(let [;; Model: x ~ gaussian(mu, 1), cost = x^2
      ;; E[cost] = mu^2 + 1, so ∇_mu E[cost] = 2*mu
      model (gen []
              (let [mu (dyn/param :mu 0.0)
                    x (dyn/trace :x (dist/gaussian mu (mx/scalar 1.0)))]
                x))
      cost-fn (fn [trace] (mx/square (:retval trace)))
      param-names [:mu]
      mu-val 2.0
      params (mx/array [mu-val])
      ;; ADEV gradient estimate (many samples for accuracy)
      {:keys [grad]} (adev/adev-gradient {:n-samples 500}
                                          model [] cost-fn
                                          param-names params)
      adev-grad (mx/item (mx/index grad 0))
      ;; Analytical gradient: 2*mu = 4.0
      analytical-grad (* 2.0 mu-val)]
  (assert-close "ADEV grad ≈ analytical 2*mu"
    analytical-grad adev-grad 0.5))

(println "\n=== ADEV Tests Complete ===")
