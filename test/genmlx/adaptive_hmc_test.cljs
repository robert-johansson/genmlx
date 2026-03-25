(ns genmlx.adaptive-hmc-test
  "Tests for HMC dual-averaging step-size adaptation."
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

(deftest adapted-step-size-validity-test
  (testing "adapted step-size is valid"
    (let [samples (mcmc/hmc
                    {:samples 50 :burn 100 :step-size 0.01 :leapfrog-steps 10
                     :addresses [:mu] :adapt-step-size true :compile? false
                     :device :cpu}
                    model [3] obs)]
      (is (= 50 (count samples)) "returns 50 samples")
      (is (every? #(not (js/isNaN (first %))) samples) "samples are finite"))))

(deftest posterior-accuracy-test
  (testing "posterior mean is correct"
    (let [samples (mcmc/hmc
                    {:samples 300 :burn 200 :step-size 0.01 :leapfrog-steps 10
                     :addresses [:mu] :adapt-step-size true :compile? false
                     :device :cpu}
                    model [3] obs)
          vals (mapv first samples)
          m (mean vals)
          expected 5.097]
      (is (< (js/Math.abs (- m expected)) 0.5) "posterior mean ~ 5.1"))))

(deftest adapted-vs-poorly-tuned-test
  (testing "adapted from bad init recovers"
    (let [adapted-samples (mcmc/hmc
                            {:samples 200 :burn 200 :step-size 0.5 :leapfrog-steps 10
                             :addresses [:mu] :adapt-step-size true :compile? false
                             :device :cpu}
                            model [3] obs)
          adapted-vals (mapv first adapted-samples)
          adapted-unique (count (distinct (map #(.toFixed % 2) adapted-vals)))
          m (mean adapted-vals)]
      (is (> adapted-unique 20) "adapted from bad init has diversity")
      (is (< (js/Math.abs (- m 5.097)) 0.5) "adapted posterior mean correct"))))

(deftest adaptive-loop-compilation-test
  (testing "adaptive + loop compilation"
    (let [samples (mcmc/hmc
                    {:samples 50 :burn 100 :step-size 0.01 :leapfrog-steps 10
                     :addresses [:mu] :adapt-step-size true :compile? true
                     :device :cpu}
                    model [3] obs)]
      (is (= 50 (count samples)) "compiled path returns 50 samples")
      (is (every? #(not (js/isNaN (first %))) samples) "compiled samples are finite"))))

(deftest default-behavior-unchanged-test
  (testing "default (no adaptation) is unchanged"
    (let [samples (mcmc/hmc
                    {:samples 30 :burn 20 :step-size 0.05 :leapfrog-steps 5
                     :addresses [:mu] :compile? true :device :cpu}
                    model [3] obs)]
      (is (= 30 (count samples)) "default (no adaptation) returns 30 samples"))))

(cljs.test/run-tests)
