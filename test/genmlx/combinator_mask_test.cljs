(ns genmlx.combinator-mask-test
  "Mask combinator: active passes through, inactive contributes zero."
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

(def inner-gf
  (dyn/auto-key
   (gen [mu]
     (trace :x (dist/gaussian mu 1)))))

(def masked (comb/mask-combinator inner-gf))

;; ---------------------------------------------------------------------------
;; Active: passes through inner GF's score
;; ---------------------------------------------------------------------------

(deftest mask-active-score-equals-inner-score
  (testing "active mask: score = inner GF score"
    (let [constraints (cm/choicemap :x (mx/scalar 1.5))
          {:keys [trace]} (p/generate masked [true 0.0] constraints)
          score (h/realize (:score trace))
          expected (h/gaussian-lp 1.5 0.0 1)]
      (is (h/close? expected score 1e-4)
          "active mask score = inner score"))))

(deftest mask-active-has-choices
  (let [tr (p/simulate masked [true 5.0])]
    (is (not= cm/EMPTY (:choices tr))
        "active mask produces choices")))

;; ---------------------------------------------------------------------------
;; Inactive: zero contribution
;; ---------------------------------------------------------------------------

(deftest mask-inactive-score-is-zero
  (testing "inactive mask: score = 0"
    (let [tr (p/simulate masked [false 5.0])]
      (is (h/close? 0.0 (h/realize (:score tr)) 1e-6)
          "inactive mask has zero score"))))

(deftest mask-inactive-empty-choices
  (let [tr (p/simulate masked [false 5.0])]
    (is (= cm/EMPTY (:choices tr))
        "inactive mask has empty choices")))

(deftest mask-inactive-nil-retval
  (let [tr (p/simulate masked [false 5.0])]
    (is (nil? (:retval tr))
        "inactive mask returns nil")))

;; ---------------------------------------------------------------------------
;; Generate weight
;; ---------------------------------------------------------------------------

(deftest mask-active-generate-weight-equals-score
  (let [constraints (cm/choicemap :x (mx/scalar 2.0))
        {:keys [trace weight]} (p/generate masked [true 0.0] constraints)]
    (is (h/close? (h/realize (:score trace)) (h/realize weight) 1e-6)
        "active generate: weight = score")))

(deftest mask-inactive-generate-weight-is-zero
  (let [{:keys [weight]} (p/generate masked [false 0.0] cm/EMPTY)]
    (is (h/close? 0.0 (h/realize weight) 1e-6)
        "inactive generate: weight = 0")))

;; ---------------------------------------------------------------------------
;; Update
;; ---------------------------------------------------------------------------

(deftest mask-active-update-weight-is-score-difference
  (let [constraints (cm/choicemap :x (mx/scalar 1.0))
        {:keys [trace]} (p/generate masked [true 0.0] constraints)
        old-score (h/realize (:score trace))
        new-constraints (cm/choicemap :x (mx/scalar 2.0))
        {:keys [trace weight]} (p/update masked trace new-constraints)
        new-score (h/realize (:score trace))]
    (is (h/close? (- new-score old-score) (h/realize weight) 1e-4))))

(deftest mask-inactive-update-weight-is-zero
  (let [tr (p/simulate masked [false 5.0])
        {:keys [weight]} (p/update masked tr cm/EMPTY)]
    (is (h/close? 0.0 (h/realize weight) 1e-6))))

;; ---------------------------------------------------------------------------
;; Regenerate
;; ---------------------------------------------------------------------------

(deftest mask-inactive-regenerate-weight-is-zero
  (let [tr (p/simulate masked [false 5.0])
        {:keys [weight]} (p/regenerate masked tr (sel/select :x))]
    (is (h/close? 0.0 (h/realize weight) 1e-6)
        "regenerating on inactive mask: weight = 0")))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
