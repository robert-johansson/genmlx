(ns genmlx.differentiable-test
  "Tests for Tier 3a: Differentiable inference — gradient of log-ML w.r.t. model params."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.differentiable :as diff])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [label pred]
  (if pred
    (println (str "  PASS: " label))
    (println (str "  FAIL: " label))))

(defn assert-close [label expected actual tol]
  (let [ok (< (js/Math.abs (- expected actual)) tol)]
    (if ok
      (println (str "  PASS: " label " (expected " (.toFixed expected 4) ", got " (.toFixed actual 4) ")"))
      (println (str "  FAIL: " label " (expected " (.toFixed expected 4) ", got " (.toFixed actual 4) ")")))))

;; ---------------------------------------------------------------------------
;; Test 1: Gradient computation — non-zero gradients
;; ---------------------------------------------------------------------------

(println "\n=== Test 1: Gradient computation ===")

;; Simple model: Gaussian with learnable mean
;; p(y | mu) = N(y; mu, 1)
;; Observations: y=3.0
;; At mu=0, gradient of log p(y|mu) should push mu toward 3.0 (positive gradient)

(def model-1
  (gen []
    (let [mu (param :mu 0.0)]
      (trace :y (dist/gaussian mu 1.0))
      mu)))

(def obs-1 (cm/choicemap :y (mx/scalar 3.0)))

(let [{:keys [log-ml grad]}
      (diff/log-ml-gradient {:n-particles 1000 :key (rng/fresh-key 42)}
                            model-1 [] obs-1 [:mu] (mx/array [0.0]))]
  (mx/materialize! log-ml grad)
  (let [log-ml-val (mx/item log-ml)
        grad-val (mx/item (mx/index grad 0))]
    (println (str "  log-ML: " (.toFixed log-ml-val 4)))
    (println (str "  grad: " (.toFixed grad-val 4)))
    ;; Log-ML should be finite
    (assert-true "log-ML is finite" (js/isFinite log-ml-val))
    ;; Gradient should be negative (it's ∂(-log-ML)/∂μ, so negative means
    ;; increasing μ increases log-ML — correct direction toward μ=3)
    (assert-true "grad pushes mu toward data (grad < 0 for neg-log-ML)" (< grad-val 0))))

;; ---------------------------------------------------------------------------
;; Test 2: Optimization converges for single Gaussian
;; ---------------------------------------------------------------------------

(println "\n=== Test 2: Optimize Gaussian mean ===")

;; True mu=3.0, observe y=3.0. Starting from mu=0, should converge to ~3.0.

(let [result (diff/optimize-params
               {:iterations 100 :lr 0.05 :n-particles 500
                :callback (fn [{:keys [iter log-ml params]}]
                            (when (zero? (mod iter 25))
                              (println (str "  iter " iter
                                           ": log-ml=" (.toFixed log-ml 3)
                                           " mu=" (.toFixed (mx/item (mx/index params 0)) 3)))))}
               model-1 [] obs-1 [:mu] (mx/array [0.0]))
      final-mu (mx/item (mx/index (:params result) 0))]
  (println (str "  Final mu: " (.toFixed final-mu 4)))
  (assert-close "mu converges near 3.0" 3.0 final-mu 0.5))

;; ---------------------------------------------------------------------------
;; Test 3: Two-parameter model (mean + scale)
;; ---------------------------------------------------------------------------

(println "\n=== Test 3: Two-parameter Gaussian (mean + log-scale) ===")

;; Observe y=5.0 from N(mu, exp(log_sigma)).
;; True: mu=5.0, sigma=1.0 (log_sigma=0).
;; Start from mu=0, log_sigma=1 (sigma=e≈2.7).

(def model-2
  (gen []
    (let [mu (param :mu 0.0)
          log-sigma (param :log-sigma 1.0)
          sigma (mx/exp log-sigma)]
      (trace :y (dist/gaussian mu sigma))
      mu)))

(def obs-2 (cm/choicemap :y (mx/scalar 5.0)))

(let [result (diff/optimize-params
               {:iterations 150 :lr 0.05 :n-particles 500
                :callback (fn [{:keys [iter log-ml params]}]
                            (when (zero? (mod iter 50))
                              (println (str "  iter " iter
                                           ": log-ml=" (.toFixed log-ml 3)
                                           " mu=" (.toFixed (mx/item (mx/index params 0)) 3)
                                           " log-s=" (.toFixed (mx/item (mx/index params 1)) 3)))))}
               model-2 [] obs-2 [:mu :log-sigma] (mx/array [0.0 1.0]))
      final-mu (mx/item (mx/index (:params result) 0))
      final-log-sigma (mx/item (mx/index (:params result) 1))]
  (println (str "  Final mu: " (.toFixed final-mu 4)
               ", log-sigma: " (.toFixed final-log-sigma 4)
               " (sigma=" (.toFixed (js/Math.exp final-log-sigma) 4) ")"))
  (assert-close "mu converges near 5.0" 5.0 final-mu 1.0)
  ;; log-sigma should decrease from 1.0 toward 0.0 (tighter fit)
  (assert-true "log-sigma decreases" (< final-log-sigma 1.0)))

;; ---------------------------------------------------------------------------
;; Test 4: Multiple observations (linear regression hyperparams)
;; ---------------------------------------------------------------------------

(println "\n=== Test 4: Linear regression with learnable prior scale ===")

;; Model: slope ~ N(0, exp(log_prior_scale)), y_i ~ N(slope * x_i, 1)
;; True slope = 2.0, observe at x=1..5.
;; Learn log_prior_scale to maximize marginal likelihood.

(def xs-4 (mapv #(mx/scalar (double %)) (range 1 6)))

(def model-4
  (gen [xs]
    (let [log-ps (param :log-prior-scale 0.0)
          prior-scale (mx/exp log-ps)
          slope (trace :slope (dist/gaussian 0 prior-scale))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/multiply slope x) 1.0)))
      slope)))

(def obs-4
  (apply cm/choicemap
    (mapcat (fn [j] [(keyword (str "y" j)) (mx/scalar (* 2.0 (inc j)))])
            (range 5))))

(let [{:keys [log-ml grad]}
      (diff/log-ml-gradient {:n-particles 1000 :key (rng/fresh-key 123)}
                            model-4 [xs-4] obs-4
                            [:log-prior-scale] (mx/array [0.0]))]
  (mx/materialize! log-ml grad)
  (println (str "  log-ML: " (.toFixed (mx/item log-ml) 4)))
  (println (str "  grad: " (.toFixed (mx/item (mx/index grad 0)) 6)))
  (assert-true "log-ML is finite" (js/isFinite (mx/item log-ml)))
  (assert-true "gradient is non-zero" (not= 0.0 (mx/item (mx/index grad 0)))))

(println "\nDone.")
