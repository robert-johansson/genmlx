(ns genmlx.contracts-test
  "Contract registry tests."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.contracts :as contracts])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Standard model (with eval!/item -- works for scalar GFI ops)
(def two-site
  (dyn/auto-key (gen []
    (let [x (trace :x (dist/gaussian 0 1))
          y (trace :y (dist/gaussian 0 1))]
      (mx/eval! x y)
      (mx/item x)))))

;; Vectorization-compatible model (no eval!/item in body)
(def two-site-vec
  (dyn/auto-key (gen []
    (let [x (trace :x (dist/gaussian 0 1))
          y (trace :y (dist/gaussian 0 1))]
      x))))

(deftest contract-checks
  (testing "contract checks on 2-site gaussian"
    (let [model two-site
          args []
          trace (p/simulate model args)
          vec-model two-site-vec
          vec-trace (p/simulate vec-model args)
          scalar-ctx {:model model :args args :trace trace}
          vec-ctx {:model vec-model :args args :trace vec-trace}]
      (doseq [[k {:keys [theorem check]}] (sort-by (comp str key) contracts/contracts)]
        (let [ctx (if (= k :broadcast-equivalence) vec-ctx scalar-ctx)
              result (check ctx)]
          (is result (str (name k) " -- " theorem)))))))

(deftest registry-structure
  (testing "registry structure"
    (is (map? contracts/contracts) "contracts is a map")
    (is (= 11 (count contracts/contracts)) "11 contracts defined")
    (is (every? (fn [[_ v]]
                  (and (string? (:theorem v))
                       (fn? (:check v))))
                contracts/contracts)
        "every contract has :theorem and :check")))

(deftest verify-gfi-contracts-scalar
  (testing "verify-gfi-contracts on scalar model"
    (let [scalar-keys (disj (set (keys contracts/contracts)) :broadcast-equivalence)
          report (contracts/verify-gfi-contracts two-site [] :n-trials 5 :contract-keys scalar-keys)]
      (is (:all-pass? report) "all-pass? true for scalar contracts")
      (is (= 50 (:total-pass report)) "total-pass = 50 (10 contracts x 5 trials)")
      (is (= 0 (:total-fail report)) "total-fail = 0")
      (is (every? (fn [[_ v]]
                    (and (integer? (:pass v))
                         (integer? (:fail v))
                         (string? (:theorem v))))
                  (:results report))
          "each result has :pass :fail :theorem"))))

(deftest verify-gfi-contracts-broadcast
  (testing "verify-gfi-contracts broadcast-equivalence"
    (let [report (contracts/verify-gfi-contracts two-site-vec []
                   :n-trials 3
                   :contract-keys #{:broadcast-equivalence})]
      (is (:all-pass? report) "broadcast-equivalence passes on vec model")
      (is (= 3 (:total-pass report)) "broadcast total-pass = 3"))))

(cljs.test/run-tests)
