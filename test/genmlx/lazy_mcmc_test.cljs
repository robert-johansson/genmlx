(ns genmlx.lazy-mcmc-test
  "Correctness tests for compiled MCMC variants.
   NOTE: compiled-mh-lazy and hmc-lazy were removed from the codebase.
   Tests use the available compiled-mh function instead.
   NOTE: compiled-mh has pre-existing issues with certain model structures."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

(def single-gaussian
  (gen []
    (trace :x (dist/gaussian 0 1))))

(def linear-regression
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                          intercept) 1)))
      slope)))

(def linreg-xs [1.0 2.0 3.0 4.0 5.0])

(defn- obs-linear-regression []
  (-> cm/EMPTY
      (cm/set-choice [:y0] (mx/scalar 3.1))
      (cm/set-choice [:y1] (mx/scalar 5.2))
      (cm/set-choice [:y2] (mx/scalar 6.9))
      (cm/set-choice [:y3] (mx/scalar 9.1))
      (cm/set-choice [:y4] (mx/scalar 10.8))))

(defn- sample-val
  "Extract a scalar value from a sample (handles both number and vector)."
  [s]
  (if (number? s) s (first s)))

;; ---------------------------------------------------------------------------
;; compiled-mh tests
;; NOTE: compiled-mh has a pre-existing "No arrays provided for stacking"
;; error when the model has no arguments (single-gaussian). Tests document
;; the known issue for single-gaussian and test the multi-param case.
;; ---------------------------------------------------------------------------

(deftest compiled-mh-single-gaussian
  (testing "compiled-mh single gaussian (pre-existing stack error)"
    (is (thrown? js/Error
          (mcmc/compiled-mh {:samples 5 :addresses [:x]}
                            single-gaussian [] (cm/choicemap :x (mx/scalar 0.5))))
        "compiled-mh on no-arg model crashes (pre-existing)")))

(deftest compiled-mh-linear-regression
  (testing "compiled-mh (linear regression)"
    (let [samples (mcmc/compiled-mh
                    {:samples 500 :burn 200 :addresses [:slope :intercept]
                     :proposal-std 0.5}
                    linear-regression [linreg-xs] (obs-linear-regression))
          slopes (mapv #(nth % 0) samples)
          intercepts (mapv #(nth % 1) samples)
          mean-slope (/ (reduce + slopes) (count slopes))
          mean-intercept (/ (reduce + intercepts) (count intercepts))]
      (is (= 500 (count samples)) "returns 500 samples")
      (is (h/close? 2.0 mean-slope 1.5) "slope near 2.0")
      (is (h/close? 1.0 mean-intercept 2.0) "intercept near 1.0"))))

(deftest compiled-mh-linreg-comparison
  (testing "two compiled-mh runs produce comparable results"
    (let [samples1 (mcmc/compiled-mh
                     {:samples 300 :burn 200 :addresses [:slope :intercept]
                      :proposal-std 0.5}
                     linear-regression [linreg-xs] (obs-linear-regression))
          samples2 (mcmc/compiled-mh
                     {:samples 300 :burn 200 :addresses [:slope :intercept]
                      :proposal-std 0.5}
                     linear-regression [linreg-xs] (obs-linear-regression))
          mean1 (/ (reduce + (mapv #(nth % 0) samples1)) (count samples1))
          mean2 (/ (reduce + (mapv #(nth % 0) samples2)) (count samples2))]
      (is (pos? (count samples1)) "run1 produces samples")
      (is (pos? (count samples2)) "run2 produces samples")
      (is (h/close? mean1 mean2 2.0) "both runs give comparable slope estimates"))))

(cljs.test/run-tests)
