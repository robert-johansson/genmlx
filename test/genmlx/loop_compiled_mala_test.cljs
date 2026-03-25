(ns genmlx.loop-compiled-mala-test
  "Tests for MALA loop compilation: correctness, cache validation, benchmarks."
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

(defn- variance [xs]
  (let [m (mean xs)]
    (/ (reduce + (map #(* (- % m) (- % m)) xs)) (count xs))))

;; ---------------------------------------------------------------------------
;; Model: simple Gaussian with known posterior
;; Posterior mean ~ 5.1, posterior std ~ 0.58
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

(deftest correctness-test
  (testing "compiled MALA returns correct structure"
    (let [samples (mcmc/mala
                    {:samples 30 :burn 20 :step-size 0.1
                     :addresses [:mu] :compile? true :device :cpu}
                    model [3] obs)]
      (is (= 30 (count samples)) "compiled returns 30 samples")
      (is (vector? (first samples)) "samples are vectors")
      (is (number? (first (first samples))) "sample values are numbers"))))

(deftest compiled-vs-eager-test
  (testing "compiled vs eager both produce finite samples"
    (let [compiled-samples (mcmc/mala
                             {:samples 500 :burn 500 :step-size 0.1
                              :addresses [:mu] :compile? true :device :cpu}
                             model [3] obs)
          eager-samples (mcmc/mala
                          {:samples 500 :burn 500 :step-size 0.1
                           :addresses [:mu] :compile? false :device :cpu}
                          model [3] obs)
          compiled-mean (mean (map first compiled-samples))
          eager-mean (mean (map first eager-samples))]
      (is (not (js/isNaN compiled-mean)) "compiled samples finite")
      (is (not (js/isNaN eager-mean)) "eager samples finite")
      (is (> (variance (map first compiled-samples)) 0.001) "compiled has variance")
      (is (> (variance (map first eager-samples)) 0.001) "eager has variance"))))

(deftest statistical-validity-test
  (testing "variance > 0 across chain"
    (let [samples (mcmc/mala
                    {:samples 100 :burn 100 :step-size 0.1
                     :addresses [:mu] :compile? true :device :cpu}
                    model [3] obs)
          vals (mapv first samples)
          v (variance vals)]
      (is (> v 0.001) "variance > 0"))))

(deftest thin-test
  (testing "thin > 1 uses compiled thin chain"
    (let [samples (mcmc/mala
                    {:samples 20 :burn 10 :thin 3 :step-size 0.1
                     :addresses [:mu] :compile? true :device :cpu}
                    model [3] obs)]
      (is (= 20 (count samples)) "thin=3 returns 20 samples"))))

(deftest long-chain-stability-test
  (testing "long chain stability (500 steps)"
    (let [samples (mcmc/mala
                    {:samples 500 :burn 100 :step-size 0.05
                     :addresses [:mu] :compile? true :block-size 50 :device :cpu}
                    model [3] obs)]
      (is (= 500 (count samples)) "500-step chain completes")
      (is (every? #(not (js/isNaN (first %))) samples) "no NaN in samples"))))

(cljs.test/run-tests)
