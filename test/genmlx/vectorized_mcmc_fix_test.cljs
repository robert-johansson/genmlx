(ns genmlx.vectorized-mcmc-fix-test
  "Regression test: vectorized MCMC on multi-parameter models.
   Previously crashed with unordered_map::at on 2+ parameters."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Model B: 2 inferred parameters (slope + intercept)
(def model-b
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                          intercept) 1)))
      slope)))

(def xs [1.0 2.0 3.0])
(def obs (cm/choicemap [:y0 (mx/scalar 2.1)]
                       [:y1 (mx/scalar 4.0)]
                       [:y2 (mx/scalar 5.9)]))
(def n-chains 4)
(def n-samples 5)

(deftest vectorized-compiled-mh-multi-param
  (testing "vectorized-compiled-mh (2 params)"
    (let [result (mcmc/vectorized-compiled-mh
                   {:samples n-samples :burn 0 :addresses [:slope :intercept]
                    :proposal-std 0.5 :n-chains n-chains :device :cpu}
                   model-b [xs] obs)]
      (is (vector? result) "returns vector")
      (is (= (count result) n-samples) "correct sample count"))))

(deftest vectorized-mala-multi-param
  (testing "vectorized-mala (2 params)"
    (let [result (mcmc/vectorized-mala
                   {:samples n-samples :burn 0 :step-size 0.01
                    :addresses [:slope :intercept] :n-chains n-chains :device :cpu}
                   model-b [xs] obs)]
      (is (vector? result) "returns vector")
      (is (= (count result) n-samples) "correct sample count"))))

(deftest vectorized-hmc-multi-param
  (testing "vectorized-hmc (2 params)"
    (let [result (mcmc/vectorized-hmc
                   {:samples n-samples :burn 0 :step-size 0.01
                    :leapfrog-steps 5 :addresses [:slope :intercept]
                    :n-chains n-chains :device :cpu}
                   model-b [xs] obs)]
      (is (vector? result) "returns vector")
      (is (= (count result) n-samples) "correct sample count"))))

(cljs.test/run-tests)
