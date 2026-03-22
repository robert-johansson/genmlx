(ns genmlx.combinator-scan-test2
  "Scan combinator: sequential with carry-state and per-step inputs.
   Score = sum of step scores, carry threads through steps."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.combinators :as comb]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Kernel: takes [carry input], samples x ~ N(carry + input, 1),
;; returns [x x] (new carry = x, output = x)
(def scan-step
  (dyn/auto-key
   (gen [carry input]
     (let [x (trace :x (dist/gaussian (mx/add carry input) 1))]
       [x x]))))

(def scanned (comb/scan-combinator scan-step))

;; ---------------------------------------------------------------------------
;; Score = sum of step scores
;; ---------------------------------------------------------------------------

(deftest scan-score-is-sum-of-step-scores
  (testing "score = Σ step scores"
    (let [inputs [(mx/scalar 0.0) (mx/scalar 1.0) (mx/scalar 2.0)]
          tr (p/simulate scanned [(mx/scalar 0.0) inputs])
          total-score (h/realize (:score tr))
          step-scores (mapv h/realize (::comb/step-scores (meta tr)))
          expected-sum (reduce + step-scores)]
      (is (= 3 (count step-scores)))
      (is (h/close? expected-sum total-score 1e-4)))))

;; ---------------------------------------------------------------------------
;; Carry threading and input dependence
;; ---------------------------------------------------------------------------

(deftest scan-carry-threading-with-inputs
  (testing "score reflects carry + input dependence"
    (let [inputs [(mx/scalar 0.0) (mx/scalar 0.0)]
          constraints (-> cm/EMPTY
                          (cm/set-choice [0 :x] (mx/scalar 1.0))
                          (cm/set-choice [1 :x] (mx/scalar 2.0)))
          {:keys [trace]} (p/generate scanned [(mx/scalar 0.0) inputs] constraints)
          score (h/realize (:score trace))
          ;; Step 0: carry=0, input=0 → mu=0, x=1 → log N(1; 0, 1)
          ;; Step 1: carry=1, input=0 → mu=1, x=2 → log N(2; 1, 1)
          expected (+ (h/gaussian-lp 1.0 0.0 1) (h/gaussian-lp 2.0 1.0 1))]
      (is (h/close? expected score 1e-4)
          "score accounts for carry-threaded means"))))

;; ---------------------------------------------------------------------------
;; Address structure
;; ---------------------------------------------------------------------------

(deftest scan-address-structure
  (testing "addresses are [t :x] for each step"
    (let [inputs [(mx/scalar 0.0) (mx/scalar 0.0) (mx/scalar 0.0)]
          tr (p/simulate scanned [(mx/scalar 0.0) inputs])
          addrs (set (cm/addresses (:choices tr)))]
      (is (contains? addrs [0 :x]))
      (is (contains? addrs [1 :x]))
      (is (contains? addrs [2 :x])))))

;; ---------------------------------------------------------------------------
;; Generate weight
;; ---------------------------------------------------------------------------

(deftest scan-generate-fully-constrained-weight-equals-score
  (let [inputs [(mx/scalar 0.0) (mx/scalar 0.0)]
        constraints (-> cm/EMPTY
                        (cm/set-choice [0 :x] (mx/scalar 0.5))
                        (cm/set-choice [1 :x] (mx/scalar 1.0)))
        {:keys [trace weight]} (p/generate scanned [(mx/scalar 0.0) inputs] constraints)]
    (is (h/close? (h/realize (:score trace)) (h/realize weight) 1e-6)
        "fully constrained: weight = score")))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
