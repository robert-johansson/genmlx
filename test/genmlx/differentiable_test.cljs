(ns genmlx.differentiable-test
  "Tests for Tier 3a: Differentiable inference — gradient of log-ML w.r.t. model params."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.differentiable :as diff])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Models and observations
;; ---------------------------------------------------------------------------

(def model-1
  (gen []
    (let [mu (param :mu 0.0)]
      (trace :y (dist/gaussian mu 1.0))
      mu)))

(def obs-1 (cm/choicemap :y (mx/scalar 3.0)))

(def model-2
  (gen []
    (let [mu (param :mu 0.0)
          log-sigma (param :log-sigma 1.0)
          sigma (mx/exp log-sigma)]
      (trace :y (dist/gaussian mu sigma))
      mu)))

(def obs-2 (cm/choicemap :y (mx/scalar 5.0)))

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

;; ---------------------------------------------------------------------------
;; Test 1: Gradient computation — non-zero gradients
;; ---------------------------------------------------------------------------

(deftest gradient-computation
  (testing "gradient of single Gaussian model"
    (let [{:keys [log-ml grad]}
          (diff/log-ml-gradient {:n-particles 1000 :key (rng/fresh-key 42)}
                                model-1 [] obs-1 [:mu] (mx/array [0.0]))]
      (mx/materialize! log-ml grad)
      (let [log-ml-val (mx/item log-ml)
            grad-val (mx/item (mx/index grad 0))]
        (is (js/isFinite log-ml-val) "log-ML is finite")
        (is (< grad-val 0) "grad pushes mu toward data (grad < 0 for neg-log-ML)")))))

;; ---------------------------------------------------------------------------
;; Test 2: Optimization converges for single Gaussian
;; ---------------------------------------------------------------------------

(deftest optimize-gaussian-mean
  (testing "single Gaussian mean converges to data"
    (let [result (diff/optimize-params
                   {:iterations 100 :lr 0.05 :n-particles 500}
                   model-1 [] obs-1 [:mu] (mx/array [0.0]))
          final-mu (mx/item (mx/index (:params result) 0))]
      (is (h/close? 3.0 final-mu 0.5) "mu converges near 3.0"))))

;; ---------------------------------------------------------------------------
;; Test 3: Two-parameter model (mean + scale)
;; ---------------------------------------------------------------------------

(deftest two-parameter-gaussian
  (testing "mean + log-scale optimization"
    (let [result (diff/optimize-params
                   {:iterations 150 :lr 0.05 :n-particles 500}
                   model-2 [] obs-2 [:mu :log-sigma] (mx/array [0.0 1.0]))
          final-mu (mx/item (mx/index (:params result) 0))
          final-log-sigma (mx/item (mx/index (:params result) 1))]
      (is (h/close? 5.0 final-mu 1.0) "mu converges near 5.0")
      (is (< final-log-sigma 1.0) "log-sigma decreases"))))

;; ---------------------------------------------------------------------------
;; Test 4: Multiple observations (linear regression hyperparams)
;; ---------------------------------------------------------------------------

(deftest linear-regression-gradient
  (testing "gradient for linear regression prior scale"
    (let [{:keys [log-ml grad]}
          (diff/log-ml-gradient {:n-particles 1000 :key (rng/fresh-key 123)}
                                model-4 [xs-4] obs-4
                                [:log-prior-scale] (mx/array [0.0]))]
      (mx/materialize! log-ml grad)
      (is (js/isFinite (mx/item log-ml)) "log-ML is finite")
      (is (not= 0.0 (mx/item (mx/index grad 0))) "gradient is non-zero"))))

(cljs.test/run-tests)
