(ns genmlx.combinator-mix-test2
  "Mix combinator: mixture model.
   Score = component score + log(mixture weight) for the selected component."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.combinators :as comb]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Two component gaussians with equal mixing weights
(def comp-a (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1)))))
(def comp-b (dyn/auto-key (gen [] (trace :x (dist/gaussian 10 1)))))

;; Equal weights: log(0.5) each
(def mix-equal (comb/mix-combinator [comp-a comp-b]
                                    (mx/array [(js/Math.log 0.5) (js/Math.log 0.5)])))

;; ---------------------------------------------------------------------------
;; Score = component score + log(mixture weight)
;; ---------------------------------------------------------------------------

(deftest mix-score-includes-component-weight
  (testing "score = component log-prob + log(mixture weight)"
    ;; Force component 0 (x ~ N(0,1)), constrain x=0.5
    (let [constraints (-> cm/EMPTY
                          (cm/set-choice [:component-idx] (mx/scalar 0 mx/int32))
                          (cm/set-choice [:x] (mx/scalar 0.5)))
          {:keys [trace]} (p/generate mix-equal [] constraints)
          score (h/realize (:score trace))
          ;; component score = log N(0.5; 0, 1)
          ;; categorical score = log(0.5) = -log(2)
          expected (+ (h/gaussian-lp 0.5 0 1) (js/Math.log 0.5))]
      (is (h/close? expected score 1e-4)
          "score = gaussian log-prob + log(0.5)"))))

(deftest mix-score-second-component
  (testing "component 1: score = log N(x; 10, 1) + log(0.5)"
    (let [constraints (-> cm/EMPTY
                          (cm/set-choice [:component-idx] (mx/scalar 1 mx/int32))
                          (cm/set-choice [:x] (mx/scalar 10.0)))
          {:keys [trace]} (p/generate mix-equal [] constraints)
          score (h/realize (:score trace))
          expected (+ (h/gaussian-lp 10.0 10 1) (js/Math.log 0.5))]
      (is (h/close? expected score 1e-4)))))

;; ---------------------------------------------------------------------------
;; Address structure includes component-idx
;; ---------------------------------------------------------------------------

(deftest mix-address-structure
  (let [tr (p/simulate mix-equal [])
        addrs (set (cm/addresses (:choices tr)))]
    (is (contains? addrs [:component-idx])
        "choices contain :component-idx")
    (is (contains? addrs [:x])
        "choices contain the component's :x")))

;; ---------------------------------------------------------------------------
;; Generate weight
;; ---------------------------------------------------------------------------

(deftest mix-generate-fully-constrained-weight-equals-score
  (let [constraints (-> cm/EMPTY
                        (cm/set-choice [:component-idx] (mx/scalar 0 mx/int32))
                        (cm/set-choice [:x] (mx/scalar 1.0)))
        {:keys [trace weight]} (p/generate mix-equal [] constraints)]
    (is (h/close? (h/realize (:score trace)) (h/realize weight) 1e-6)
        "fully constrained: weight = score")))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
