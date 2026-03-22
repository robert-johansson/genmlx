(ns genmlx.combinator-switch-test2
  "Switch combinator: score = selected branch score only."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.combinators :as comb]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

(def branch-a
  (dyn/auto-key
   (gen []
     (trace :x (dist/gaussian 0 1)))))

(def branch-b
  (dyn/auto-key
   (gen []
     (trace :x (dist/gaussian 10 1)))))

(def sw (comb/switch-combinator branch-a branch-b))

;; ---------------------------------------------------------------------------
;; Score = selected branch score only
;; ---------------------------------------------------------------------------

(deftest switch-score-is-selected-branch-score
  (testing "branch 0: score = log N(x; 0, 1)"
    (let [constraints (cm/choicemap :x (mx/scalar 0.5))
          {:keys [trace]} (p/generate sw [0] constraints)
          score (h/realize (:score trace))
          expected (h/gaussian-lp 0.5 0 1)]
      (is (h/close? expected score 1e-4)
          "score matches branch 0's gaussian(0,1)")))

  (testing "branch 1: score = log N(x; 10, 1)"
    (let [constraints (cm/choicemap :x (mx/scalar 10.5))
          {:keys [trace]} (p/generate sw [1] constraints)
          score (h/realize (:score trace))
          expected (h/gaussian-lp 10.5 10 1)]
      (is (h/close? expected score 1e-4)
          "score matches branch 1's gaussian(10,1)"))))

;; ---------------------------------------------------------------------------
;; Branch index metadata
;; ---------------------------------------------------------------------------

(deftest switch-records-branch-index
  (let [tr0 (p/simulate sw [0])
        tr1 (p/simulate sw [1])]
    (is (= 0 (::comb/switch-idx (meta tr0))))
    (is (= 1 (::comb/switch-idx (meta tr1))))))

;; ---------------------------------------------------------------------------
;; Generate weight
;; ---------------------------------------------------------------------------

(deftest switch-generate-weight-equals-score-when-constrained
  (let [constraints (cm/choicemap :x (mx/scalar 5.0))
        {:keys [trace weight]} (p/generate sw [0] constraints)]
    (is (h/close? (h/realize (:score trace)) (h/realize weight) 1e-6)
        "fully constrained: weight = score")))

;; ---------------------------------------------------------------------------
;; Update within same branch
;; ---------------------------------------------------------------------------

(deftest switch-update-same-branch
  (testing "update within same branch: weight = new_score - old_score"
    (let [constraints (cm/choicemap :x (mx/scalar 1.0))
          {:keys [trace]} (p/generate sw [0] constraints)
          old-score (h/realize (:score trace))
          new-constraints (cm/choicemap :x (mx/scalar 2.0))
          {:keys [trace weight]} (p/update sw trace new-constraints)
          new-score (h/realize (:score trace))]
      (is (h/close? (- new-score old-score) (h/realize weight) 1e-4)))))

;; ---------------------------------------------------------------------------
;; Address structure matches kernel
;; ---------------------------------------------------------------------------

(deftest switch-address-structure
  (let [tr (p/simulate sw [0])
        addrs (cm/addresses (:choices tr))]
    (is (= #{[:x]} (set addrs))
        "switch choices have the kernel's address structure")))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
