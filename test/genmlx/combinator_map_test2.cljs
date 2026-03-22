(ns genmlx.combinator-map-test2
  "Map combinator: score = sum of element scores, integer-keyed address structure."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.combinators :as comb]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

;; A simple kernel: x ~ N(mu, 1) where mu is the argument
(def gaussian-kernel
  (dyn/auto-key
   (gen [mu]
     (trace :x (dist/gaussian mu 1)))))

(def mapped (comb/map-combinator gaussian-kernel))

;; ---------------------------------------------------------------------------
;; Score = sum of element scores
;; ---------------------------------------------------------------------------

(deftest map-score-is-sum-of-element-scores
  (testing "Map score = Σ kernel scores"
    (let [tr (p/simulate mapped [[1.0 2.0 3.0]])
          total-score (h/realize (:score tr))
          element-scores (mapv h/realize
                               (::comb/element-scores (meta tr)))
          expected-sum (reduce + element-scores)]
      (is (= 3 (count element-scores)))
      (is (h/close? expected-sum total-score 1e-4)
          "total score = sum of element scores"))))

(deftest map-score-analytically-correct
  (testing "each element score = log N(x; mu, 1)"
    (let [mus [1.0 2.0 3.0]
          constraints (reduce (fn [cm [i mu]]
                                (cm/set-choice cm [i :x] (mx/scalar mu)))
                              cm/EMPTY
                              (map-indexed vector mus))
          {:keys [trace]} (p/generate mapped [mus] constraints)
          score (h/realize (:score trace))
          ;; Each x_i = mu_i, so log N(mu_i; mu_i, 1) = -0.5*log(2π)
          expected (* 3 (* -0.5 h/LOG-2PI))]
      (is (h/close? expected score 1e-4)
          "score with x=mu is 3 * (-0.5*log(2π))"))))

;; ---------------------------------------------------------------------------
;; Address structure: integer-keyed submaps
;; ---------------------------------------------------------------------------

(deftest map-address-structure
  (testing "addresses are integer-keyed with kernel addresses nested"
    (let [tr (p/simulate mapped [[1.0 2.0 3.0]])
          addrs (cm/addresses (:choices tr))
          addr-set (set addrs)]
      (is (= 3 (count addrs)))
      (is (contains? addr-set [0 :x]))
      (is (contains? addr-set [1 :x]))
      (is (contains? addr-set [2 :x])))))

;; ---------------------------------------------------------------------------
;; Generate weight = sum of element weights
;; ---------------------------------------------------------------------------

(deftest map-generate-weight-fully-constrained
  (testing "fully constrained: weight = score"
    (let [mus [1.0 2.0]
          constraints (-> cm/EMPTY
                          (cm/set-choice [0 :x] (mx/scalar 0.5))
                          (cm/set-choice [1 :x] (mx/scalar 1.5)))
          {:keys [trace weight]} (p/generate mapped [mus] constraints)]
      (is (h/close? (h/realize (:score trace)) (h/realize weight) 1e-6)
          "fully constrained weight = score"))))

(deftest map-generate-weight-partial
  (testing "partial constraint: weight = constrained element's log-prob"
    (let [mus [1.0 2.0]
          constraints (cm/set-choice cm/EMPTY [0 :x] (mx/scalar 1.0))
          {:keys [weight]} (p/generate mapped [mus] constraints)
          ;; Only element 0 constrained: x=1.0 with mu=1.0
          ;; log N(1; 1, 1) = -0.5*log(2π)
          expected (* -0.5 h/LOG-2PI)]
      (is (h/close? expected (h/realize weight) 1e-4)))))

;; ---------------------------------------------------------------------------
;; Update: weight = new_score - old_score
;; ---------------------------------------------------------------------------

(deftest map-update-weight-is-score-difference
  (let [mus [1.0 2.0]
        constraints (-> cm/EMPTY
                        (cm/set-choice [0 :x] (mx/scalar 0.0))
                        (cm/set-choice [1 :x] (mx/scalar 0.0)))
        {:keys [trace]} (p/generate mapped [mus] constraints)
        old-score (h/realize (:score trace))
        new-constraints (cm/set-choice cm/EMPTY [0 :x] (mx/scalar 1.0))
        {:keys [trace weight]} (p/update mapped trace new-constraints)
        new-score (h/realize (:score trace))]
    (is (h/close? (- new-score old-score) (h/realize weight) 1e-4)
        "update weight = new_score - old_score")))

;; ---------------------------------------------------------------------------
;; Edge case: single element
;; ---------------------------------------------------------------------------

(deftest map-single-element
  (testing "Map with one element behaves like the kernel"
    (let [tr (p/simulate mapped [[5.0]])
          x (h/realize (cm/get-choice (:choices tr) [0 :x]))
          score (h/realize (:score tr))
          expected (h/gaussian-lp x 5.0 1)]
      (is (h/close? expected score 1e-4)))))

;; ---------------------------------------------------------------------------
;; Retval structure
;; ---------------------------------------------------------------------------

(deftest map-retval-is-vector-of-kernel-retvals
  (let [tr (p/simulate mapped [[1.0 2.0 3.0]])]
    (is (= 3 (count (:retval tr))))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
