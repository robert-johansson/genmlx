(ns genmlx.gfi-contracts-test
  "Run core GFI algebraic laws against canonical test models.
   Tests the 11 laws that correspond to the original GFI contracts."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.gfi :as gfi]
            [genmlx.test-models :as models]))

;; The 11 laws that correspond to the original GFI contracts
(def ^:private core-laws
  #{:generate-full-weight-equals-score
    :update-identity
    :update-density-ratio
    :update-round-trip
    :regenerate-empty-identity
    :project-all-equals-score
    :project-none-equals-zero
    :assess-equals-generate-score
    :propose-weight-equals-generate
    :score-full-decomposition
    :vsimulate-shape-correctness})

(def ^:private test-models
  [["single-gaussian"  models/single-gaussian  []]
   ["two-gaussians"    models/two-gaussians    []]
   ["dependent-model"  models/dependent-model  []]
   ["multi-dist-model" models/multi-dist-model []]])

(deftest gfi-laws-all-models
  (doseq [[model-name model args] test-models]
    (testing model-name
      (let [{:keys [all-pass? total-fail]}
            (gfi/verify model args :law-names core-laws :n-trials 1)]
        (is all-pass?
            (str model-name " failed " total-fail " law checks"))))))

(deftest gfi-laws-statistical-robustness
  (testing "laws pass over 10 independent trials"
    (doseq [[model-name model args] test-models]
      (testing model-name
        (let [{:keys [all-pass? total-fail]}
              (gfi/verify model args :law-names core-laws :n-trials 10)]
          (is all-pass?
              (str model-name " failed " total-fail " law checks")))))))

(cljs.test/run-tests)
