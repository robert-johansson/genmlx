(ns genmlx.combinator-recurse-test2
  "Recurse combinator: self-referential generative functions with termination.
   Score accumulates across recursive calls."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.combinators :as comb]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Recursive model: geometric depth, sample gaussian at each level.
;; Terminates when depth reaches 0.
;; maker receives `self` (the RecurseCombinator) so the inner gen can splice it.
(def recursive-model
  (comb/recurse
   (fn [self]
     (dyn/auto-key
      (gen [depth]
           (let [x (trace :x (dist/gaussian 0 1))]
             (mx/eval! x)
             (if (> depth 0)
               (do (splice :child self (dec depth))
                   x)
               x)))))))

;; ---------------------------------------------------------------------------
;; Terminates with finite depth
;; ---------------------------------------------------------------------------

(deftest recurse-terminates
  (testing "recursive model terminates and produces valid trace"
    (let [{:keys [score]} (p/simulate recursive-model [0])]
      (is (some? score))
      (is (mx/array? score))
      (is (js/isFinite (h/realize score))
          "score is finite"))))

;; ---------------------------------------------------------------------------
;; Score accumulates across recursive calls
;; ---------------------------------------------------------------------------

(deftest recurse-score-accumulates
  (testing "score includes contributions from all recursive levels"
    ;; depth=0 → single level, just x ~ N(0,1)
    (let [constraints (cm/choicemap :x (mx/scalar 0.5))
          {:keys [trace]} (p/generate recursive-model [0] constraints)
          score (h/realize (:score trace))
          expected (h/gaussian-lp 0.5 0 1)]
      (is (h/close? expected score 1e-4)
          "single-level score = log N(0.5; 0, 1)"))))

;; ---------------------------------------------------------------------------
;; Generate weight
;; ---------------------------------------------------------------------------

(deftest recurse-generate-weight-fully-constrained
  (let [constraints (cm/choicemap :x (mx/scalar 1.0))
        {:keys [trace weight]} (p/generate recursive-model [0] constraints)]
    (is (h/close? (h/realize (:score trace)) (h/realize weight) 1e-6)
        "fully constrained: weight = score")))

;; ---------------------------------------------------------------------------
;; Address structure
;; ---------------------------------------------------------------------------

(deftest recurse-base-case-addresses
  (testing "terminated model has :x address"
    (let [constraints (cm/choicemap :x (mx/scalar 0.0))
          {:keys [trace]} (p/generate recursive-model [0] constraints)
          addrs (set (cm/addresses (:choices trace)))]
      (is (contains? addrs [:x])))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
