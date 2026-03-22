(ns genmlx.gfi-contracts-test
  "Run all GFI contracts from contracts.cljs against canonical test models.
   Each contract encodes a measure-theoretic invariant of the GFI."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.contracts :as contracts]
            [genmlx.test-models :as models]))

;; ---------------------------------------------------------------------------
;; Run all 10 contracts against each model
;; ---------------------------------------------------------------------------

(def ^:private test-models
  [["single-gaussian"  models/single-gaussian  []]
   ["two-gaussians"    models/two-gaussians    []]
   ["dependent-model"  models/dependent-model  []]
   ["multi-dist-model" models/multi-dist-model []]])

(deftest gfi-contracts-all-models
  (doseq [[model-name model args] test-models]
    (testing model-name
      (let [tr (p/simulate model args)
            ctx {:model model :args args :trace tr}]
        (doseq [[contract-key {:keys [theorem check]}] contracts/contracts]
          (testing (name contract-key)
            (is (check ctx) theorem)))))))

;; ---------------------------------------------------------------------------
;; Statistical contract verification (multiple trials)
;; ---------------------------------------------------------------------------

(deftest gfi-contracts-statistical-robustness
  (testing "contracts pass over 10 independent trials"
    (doseq [[model-name model args] test-models]
      (testing model-name
        (let [{:keys [all-pass? total-fail]}
              (contracts/verify-gfi-contracts model args :n-trials 10)]
          (is all-pass?
              (str model-name " failed " total-fail " contract checks")))))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
