(ns genmlx.combinator-unfold-test2
  "Unfold combinator: sequential application with state threading.
   Score = sum of step scores, addresses are integer-keyed by step."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.combinators :as comb]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Kernel: x ~ N(state, 1), returns x as new state (random walk)
(def random-walk-step
  (dyn/auto-key
   (gen [t state]
     (trace :x (dist/gaussian state 1)))))

(def unfold (comb/unfold-combinator random-walk-step))

;; ---------------------------------------------------------------------------
;; Score = sum of step scores
;; ---------------------------------------------------------------------------

(deftest unfold-score-is-sum-of-step-scores
  (testing "score = Σ step scores across T steps"
    (let [tr (p/simulate unfold [3 0.0])
          total-score (h/realize (:score tr))
          step-scores (mapv h/realize (::comb/step-scores (meta tr)))
          expected-sum (reduce + step-scores)]
      (is (= 3 (count step-scores)))
      (is (h/close? expected-sum total-score 1e-4)
          "total score = sum of step scores"))))

;; ---------------------------------------------------------------------------
;; State threading: each step receives previous state
;; ---------------------------------------------------------------------------

(deftest unfold-state-threading
  (testing "each step's sampled value becomes next step's state"
    (let [constraints (-> cm/EMPTY
                          (cm/set-choice [0 :x] (mx/scalar 1.0))
                          (cm/set-choice [1 :x] (mx/scalar 2.0))
                          (cm/set-choice [2 :x] (mx/scalar 3.0)))
          {:keys [trace]} (p/generate unfold [3 0.0] constraints)
          states (:retval trace)]
      ;; State at step 0 = x sampled at step 0 = 1.0
      ;; State at step 1 = x sampled at step 1 = 2.0
      ;; etc.
      (is (h/close? 1.0 (h/realize (nth states 0)) 1e-6))
      (is (h/close? 2.0 (h/realize (nth states 1)) 1e-6))
      (is (h/close? 3.0 (h/realize (nth states 2)) 1e-6)))))

(deftest unfold-score-with-state-threading
  (testing "score accounts for state dependence"
    (let [constraints (-> cm/EMPTY
                          (cm/set-choice [0 :x] (mx/scalar 1.0))
                          (cm/set-choice [1 :x] (mx/scalar 2.0)))
          {:keys [trace]} (p/generate unfold [2 0.0] constraints)
          score (h/realize (:score trace))
          ;; Step 0: x=1.0, state=0.0 → log N(1; 0, 1)
          ;; Step 1: x=2.0, state=1.0 → log N(2; 1, 1)
          expected (+ (h/gaussian-lp 1.0 0.0 1) (h/gaussian-lp 2.0 1.0 1))]
      (is (h/close? expected score 1e-4)
          "score correctly reflects state-dependent conditionals"))))

;; ---------------------------------------------------------------------------
;; Address structure: integer-keyed by step
;; ---------------------------------------------------------------------------

(deftest unfold-address-structure
  (testing "addresses are [t :x] for each step t"
    (let [tr (p/simulate unfold [3 0.0])
          addrs (set (cm/addresses (:choices tr)))]
      (is (contains? addrs [0 :x]))
      (is (contains? addrs [1 :x]))
      (is (contains? addrs [2 :x])))))

;; ---------------------------------------------------------------------------
;; Generate weight
;; ---------------------------------------------------------------------------

(deftest unfold-generate-fully-constrained-weight-equals-score
  (let [constraints (-> cm/EMPTY
                        (cm/set-choice [0 :x] (mx/scalar 0.5))
                        (cm/set-choice [1 :x] (mx/scalar 1.0)))
        {:keys [trace weight]} (p/generate unfold [2 0.0] constraints)]
    (is (h/close? (h/realize (:score trace)) (h/realize weight) 1e-6)
        "fully constrained: weight = score")))

;; ---------------------------------------------------------------------------
;; Retval structure
;; ---------------------------------------------------------------------------

(deftest unfold-retval-is-vector-of-states
  (let [tr (p/simulate unfold [5 0.0])]
    (is (= 5 (count (:retval tr)))
        "retval has one entry per step")))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
