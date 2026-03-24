(ns genmlx.dist-test
  "Distribution tests: sampling, log-prob, GFI bridge, statistical validation."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]))

(deftest gaussian-test
  (testing "Gaussian(0,1) sampling"
    (let [d (dist/gaussian 0 1)
          v (dist/sample d)]
      (mx/eval! v)
      (let [val (mx/item v)]
        (is (number? val) "Gaussian(0,1) sample is number")))
    (let [d (dist/gaussian 0 1)
          v (dist/sample d)
          lp (dist/log-prob d v)]
      (mx/eval! lp)
      (let [lp-val (mx/item lp)]
        (is (number? lp-val) "Gaussian(0,1) log-prob is number")
        (is (js/isFinite lp-val) "Gaussian(0,1) log-prob is finite"))))
  (testing "Gaussian(0,1) log-prob at 0"
    (let [d (dist/gaussian 0 1)
          lp (dist/log-prob d (mx/scalar 0.0))]
      (mx/eval! lp)
      ;; log(1/sqrt(2*pi)) ~ -0.9189
      (is (h/close? -0.9189 (mx/item lp) 0.01) "Gaussian(0,1) log-prob at 0"))))

(deftest uniform-test
  (testing "Uniform(0,1) sampling"
    (let [d (dist/uniform 0 1)
          v (dist/sample d)]
      (mx/eval! v)
      (let [val (mx/item v)]
        (is (number? val) "Uniform(0,1) sample is number")
        (is (and (>= val 0) (<= val 1)) "Uniform(0,1) sample in range")))
    (let [d (dist/uniform 0 1)
          v (dist/sample d)
          lp (dist/log-prob d v)]
      (mx/eval! lp)
      (let [lp-val (mx/item lp)]
        (is (number? lp-val) "Uniform(0,1) log-prob is number")
        (is (js/isFinite lp-val) "Uniform(0,1) log-prob is finite"))))
  (testing "Uniform(0,1) log-prob at 0.5"
    (let [d (dist/uniform 0 1)
          lp (dist/log-prob d (mx/scalar 0.5))]
      (mx/eval! lp)
      (is (h/close? 0.0 (mx/item lp) 0.001) "Uniform(0,1) log-prob"))))

(deftest bernoulli-test
  (testing "Bernoulli(0.5) sampling"
    (let [d (dist/bernoulli 0.5)
          v (dist/sample d)]
      (mx/eval! v)
      (let [val (mx/item v)]
        (is (number? val) "Bernoulli(0.5) sample is number")
        (is (and (>= val 0) (<= val 1)) "Bernoulli(0.5) sample in range")))
    (let [d (dist/bernoulli 0.5)
          v (dist/sample d)
          lp (dist/log-prob d v)]
      (mx/eval! lp)
      (let [lp-val (mx/item lp)]
        (is (number? lp-val) "Bernoulli(0.5) log-prob is number")
        (is (js/isFinite lp-val) "Bernoulli(0.5) log-prob is finite"))))
  (testing "Bernoulli(0.7) log-prob values"
    (let [d (dist/bernoulli 0.7)
          lp1 (dist/log-prob d (mx/scalar 1.0))
          lp0 (dist/log-prob d (mx/scalar 0.0))]
      (mx/eval! lp1 lp0)
      (is (h/close? (js/Math.log 0.7) (mx/item lp1) 0.001) "Bernoulli log-prob(1)")
      (is (h/close? (js/Math.log 0.3) (mx/item lp0) 0.001) "Bernoulli log-prob(0)"))))

(deftest exponential-test
  (testing "Exponential(1) sampling"
    (let [d (dist/exponential 1)
          v (dist/sample d)]
      (mx/eval! v)
      (let [val (mx/item v)]
        (is (number? val) "Exponential(1) sample is number")
        (is (and (>= val 0) (<= val js/Infinity)) "Exponential(1) sample in range")))
    (let [d (dist/exponential 1)
          v (dist/sample d)
          lp (dist/log-prob d v)]
      (mx/eval! lp)
      (let [lp-val (mx/item lp)]
        (is (number? lp-val) "Exponential(1) log-prob is number")
        (is (js/isFinite lp-val) "Exponential(1) log-prob is finite"))))
  (testing "Exponential(2) log-prob at 1.0"
    (let [d (dist/exponential 2)
          lp (dist/log-prob d (mx/scalar 1.0))]
      (mx/eval! lp)
      ;; log(2) - 2*1 = 0.693 - 2 = -1.307
      (is (h/close? -1.307 (mx/item lp) 0.01) "Exponential log-prob"))))

(deftest beta-test
  (testing "Beta(2,5) sampling"
    (let [d (dist/beta-dist 2 5)
          v (dist/sample d)]
      (mx/eval! v)
      (let [val (mx/item v)]
        (is (number? val) "Beta(2,5) sample is number")
        (is (and (>= val 0) (<= val 1)) "Beta(2,5) sample in range")))
    (let [d (dist/beta-dist 2 5)
          v (dist/sample d)
          lp (dist/log-prob d v)]
      (mx/eval! lp)
      (let [lp-val (mx/item lp)]
        (is (number? lp-val) "Beta(2,5) log-prob is number")
        (is (js/isFinite lp-val) "Beta(2,5) log-prob is finite")))))

(deftest gamma-test
  (testing "Gamma(2,1) sampling"
    (let [d (dist/gamma-dist 2 1)
          v (dist/sample d)]
      (mx/eval! v)
      (let [val (mx/item v)]
        (is (number? val) "Gamma(2,1) sample is number")
        (is (and (>= val 0) (<= val js/Infinity)) "Gamma(2,1) sample in range")))
    (let [d (dist/gamma-dist 2 1)
          v (dist/sample d)
          lp (dist/log-prob d v)]
      (mx/eval! lp)
      (let [lp-val (mx/item lp)]
        (is (number? lp-val) "Gamma(2,1) log-prob is number")
        (is (js/isFinite lp-val) "Gamma(2,1) log-prob is finite")))))

(deftest laplace-test
  (testing "Laplace(0,1) sampling"
    (let [d (dist/laplace 0 1)
          v (dist/sample d)]
      (mx/eval! v)
      (let [val (mx/item v)]
        (is (number? val) "Laplace(0,1) sample is number")))
    (let [d (dist/laplace 0 1)
          v (dist/sample d)
          lp (dist/log-prob d v)]
      (mx/eval! lp)
      (let [lp-val (mx/item lp)]
        (is (number? lp-val) "Laplace(0,1) log-prob is number")
        (is (js/isFinite lp-val) "Laplace(0,1) log-prob is finite")))))

(deftest log-normal-test
  (testing "LogNormal(0,1) sampling"
    (let [d (dist/log-normal 0 1)
          v (dist/sample d)]
      (mx/eval! v)
      (let [val (mx/item v)]
        (is (number? val) "LogNormal(0,1) sample is number")
        (is (and (>= val 0) (<= val js/Infinity)) "LogNormal(0,1) sample in range")))
    (let [d (dist/log-normal 0 1)
          v (dist/sample d)
          lp (dist/log-prob d v)]
      (mx/eval! lp)
      (let [lp-val (mx/item lp)]
        (is (number? lp-val) "LogNormal(0,1) log-prob is number")
        (is (js/isFinite lp-val) "LogNormal(0,1) log-prob is finite")))))

(deftest poisson-test
  (testing "Poisson(3) sampling"
    (let [d (dist/poisson 3)
          v (dist/sample d)]
      (mx/eval! v)
      (let [val (mx/item v)]
        (is (number? val) "Poisson(3) sample is number")
        (is (and (>= val 0) (<= val js/Infinity)) "Poisson(3) sample in range")))
    (let [d (dist/poisson 3)
          v (dist/sample d)
          lp (dist/log-prob d v)]
      (mx/eval! lp)
      (let [lp-val (mx/item lp)]
        (is (number? lp-val) "Poisson(3) log-prob is number")
        (is (js/isFinite lp-val) "Poisson(3) log-prob is finite")))))

(deftest delta-test
  (testing "Delta"
    (let [d (dist/delta 5.0)
          v (dist/sample d)]
      (mx/eval! v)
      (is (h/close? 5.0 (mx/item v) 0.001) "Delta always returns value"))))

(deftest categorical-test
  (testing "Categorical"
    (let [logits (mx/array [(js/Math.log 0.2) (js/Math.log 0.3) (js/Math.log 0.5)])
          d (dist/categorical logits)
          v (dist/sample d)]
      (mx/eval! v)
      (let [val (mx/item v)]
        (is (and (>= val 0) (<= val 2)) "Categorical sample is integer-valued")))))

(deftest gfi-bridge-test
  (testing "GFI bridge"
    (let [d (dist/gaussian 0 1)
          trace (p/simulate d [])]
      (is (some? trace) "dist implements GFI simulate"))))

(deftest statistical-validation-test
  (testing "Statistical validation"
    (let [d (dist/gaussian 5 2)
          samples (mapv (fn [_]
                          (let [v (dist/sample d)]
                            (mx/eval! v)
                            (mx/item v)))
                        (range 1000))
          mean (/ (reduce + samples) (count samples))
          variance (/ (reduce + (map #(let [d (- % mean)] (* d d)) samples))
                      (count samples))]
      (is (h/close? 5.0 mean 0.3) "Gaussian mean ~ 5")
      (is (h/close? 4.0 variance 1.0) "Gaussian variance ~ 4"))))

(cljs.test/run-tests)
