(ns genmlx.verify-test
  "Tests for validate-gen-fn static validator."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.verify :as verify]
            [genmlx.trace :as tr]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn- has-violation? [result type]
  (some #(= type (:type %)) (:violations result)))

(deftest valid-model
  (testing "valid model"
    (let [model (gen [mu]
                  (trace :x (dist/gaussian mu 1))
                  (trace :y (dist/gaussian 0 1)))
          result (verify/validate-gen-fn model [0.0])]
      (is (:valid? result) "valid model returns valid")
      (is (= 0 (count (:violations result))) "no violations")
      (is (instance? tr/Trace (:trace result)) "trace returned"))))

(deftest duplicate-address
  (testing "duplicate address"
    (let [model (gen []
                  (trace :x (dist/gaussian 0 1))
                  (trace :x (dist/gaussian 0 1)))
          result (verify/validate-gen-fn model [])]
      (is (not (:valid? result)) "duplicate detected as invalid")
      (is (has-violation? result :duplicate-address) "has duplicate-address violation")
      (is (= :x (:addr (first (filter #(= :duplicate-address (:type %)) (:violations result)))))
          "violation has addr"))))

(deftest non-finite-score
  (testing "non-finite score"
    (let [model (gen []
                  (trace :x (dist/gaussian 0 (mx/scalar 0.0))))
          result (verify/validate-gen-fn model [])]
      (is (not (:valid? result)) "non-finite detected as invalid")
      (is (has-violation? result :non-finite-score) "has non-finite-score violation"))))

(deftest empty-model
  (testing "empty model"
    (let [model (gen []
                  (mx/scalar 42.0))
          result (verify/validate-gen-fn model [])]
      (is (:valid? result) "empty model is valid (warning only)")
      (is (has-violation? result :empty-model) "has empty-model warning")
      (is (= :warning (:severity (first (filter #(= :empty-model (:type %)) (:violations result)))))
          "warning severity"))))

(deftest materialization-in-body
  (testing "materialization in body"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 1))]
                    (mx/eval! x)
                    (mx/item x)))
          result (verify/validate-gen-fn model [])]
      (is (:valid? result) "materialization model is valid (warning only)")
      (is (has-violation? result :materialization-in-body) "has materialization warning")
      (is (>= (count (filter #(= :materialization-in-body (:type %)) (:violations result))) 1)
          "at least 1 materialization warning"))))

(deftest model-that-throws
  (testing "model that throws"
    (let [model (gen []
                  (throw (js/Error. "intentional error")))
          result (verify/validate-gen-fn model [])]
      (is (not (:valid? result)) "throwing model is invalid")
      (is (has-violation? result :execution-error) "has execution-error")
      (is (nil? (:trace result)) "no trace on error"))))

(deftest conditional-duplicate-multi-trial
  (testing "conditional duplicate (multi-trial)"
    (let [model (gen []
                  (let [flip (trace :flip (dist/bernoulli 0.5))]
                    (mx/eval! flip)
                    (if (> (mx/item flip) 0.5)
                      (do (trace :a (dist/gaussian 0 1))
                          (trace :a (dist/gaussian 0 1)))
                      (trace :b (dist/gaussian 0 1)))))
          result (verify/validate-gen-fn model [] {:n-trials 20})]
      (is (has-violation? result :duplicate-address) "conditional dup caught with multi-trial"))))

(deftest multi-site-valid-model
  (testing "multi-site valid model"
    (let [model (gen [xs]
                  (let [slope (trace :slope (dist/gaussian 0 10))
                        intercept (trace :intercept (dist/gaussian 0 10))]
                    (doseq [[j x] (map-indexed vector xs)]
                      (trace (keyword (str "y" j))
                                 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                                        intercept) 1)))
                    slope))
          result (verify/validate-gen-fn model [(mapv float [1 2 3 4 5])])]
      (is (:valid? result) "multi-site model is valid")
      (is (= 0 (count (:violations result))) "no violations")
      (is (instance? tr/Trace (:trace result)) "trace returned"))))

(cljs.test/run-tests)
