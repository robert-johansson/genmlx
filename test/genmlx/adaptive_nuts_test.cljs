(ns genmlx.adaptive-nuts-test
  "Tests for NUTS dual-averaging step-size + mass matrix adaptation,
   and HMC adapt-metric."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.gen :refer [gen]]
            [genmlx.inference.mcmc :as mcmc]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- mean [xs] (/ (reduce + xs) (count xs)))

;; ---------------------------------------------------------------------------
;; Model: Gaussian with known posterior
;; Prior: mu ~ N(0, 10), Likelihood: y_i ~ N(mu, 1)
;; Posterior: mu | y ~ N(~5.097, ~0.577)
;; ---------------------------------------------------------------------------

(def model
  (gen [n]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (dotimes [i n]
        (trace (keyword (str "y" i)) (dist/gaussian mu 1)))
      mu)))

(def obs (cm/choicemap :y0 5.0 :y1 5.5 :y2 4.8))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest nuts-adapted-step-size-test
  (testing "NUTS adapted step-size is valid"
    (let [samples (mcmc/nuts
                    {:samples 50 :burn 100 :step-size 0.01
                     :addresses [:mu] :adapt-step-size true :compile? false
                     :device :cpu}
                    model [3] obs)]
      (is (= 50 (count samples)) "returns 50 samples")
      (is (every? #(not (js/isNaN (first %))) samples) "samples are finite"))))

(deftest nuts-posterior-accuracy-test
  (testing "NUTS posterior accuracy"
    (let [samples (mcmc/nuts
                    {:samples 300 :burn 200 :step-size 0.01
                     :addresses [:mu] :adapt-step-size true :compile? false
                     :device :cpu}
                    model [3] obs)
          vals (mapv first samples)
          m (mean vals)
          expected 5.097]
      (is (< (js/Math.abs (- m expected)) 0.5) "posterior mean ~ 5.1"))))

(deftest nuts-bad-initial-recovery-test
  (testing "NUTS from bad initial step-size adapts"
    (let [samples (mcmc/nuts
                    {:samples 100 :burn 150 :step-size 1.0
                     :addresses [:mu] :adapt-step-size true :compile? false
                     :device :cpu}
                    model [3] obs)
          vals (mapv first samples)
          n-unique (count (distinct (map #(.toFixed % 2) vals)))
          m (mean vals)]
      (is (> n-unique 10) "adapted from bad init has diversity")
      (is (< (js/Math.abs (- m 5.097)) 0.5) "posterior mean correct"))))

(deftest nuts-adapt-metric-test
  (testing "NUTS adapt-metric"
    (let [samples (mcmc/nuts
                    {:samples 200 :burn 200 :step-size 0.01
                     :addresses [:mu] :adapt-step-size true :adapt-metric true
                     :compile? false :device :cpu}
                    model [3] obs)
          vals (mapv first samples)
          m (mean vals)]
      (is (= 200 (count samples)) "adapt-metric returns 200 samples")
      (is (every? #(not (js/isNaN (first %))) samples) "adapt-metric samples finite")
      (is (< (js/Math.abs (- m 5.097)) 0.5) "adapt-metric posterior mean ~ 5.1"))))

(deftest hmc-adapt-metric-test
  (testing "HMC adapt-metric"
    (let [samples (mcmc/hmc
                    {:samples 400 :burn 300 :step-size 0.01 :leapfrog-steps 10
                     :addresses [:mu] :adapt-step-size true :adapt-metric true
                     :compile? false :device :cpu}
                    model [3] obs)
          vals (mapv first samples)
          m (mean vals)]
      (is (= 400 (count samples)) "HMC adapt-metric returns 400 samples")
      (is (every? #(not (js/isNaN (first %))) samples) "HMC adapt-metric samples finite")
      (is (< (js/Math.abs (- m 5.097)) 0.7) "HMC adapt-metric posterior mean ~ 5.1"))))

(deftest default-behavior-unchanged-test
  (testing "default NUTS behavior unchanged"
    (let [samples (mcmc/nuts
                    {:samples 30 :burn 10 :step-size 0.05
                     :addresses [:mu] :compile? true :device :cpu}
                    model [3] obs)]
      (is (= 30 (count samples)) "default NUTS (no adaptation) returns 30 samples"))))

(cljs.test/run-tests)
