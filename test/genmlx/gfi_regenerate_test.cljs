(ns genmlx.gfi-regenerate-test
  "GFI regenerate: resample selected addresses, preserve unselected.
   weight = new_score - old_score - proposal_ratio (MH acceptance weight)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.test-helpers :as h]
            [genmlx.test-models :as models]))

;; ---------------------------------------------------------------------------
;; Unselected addresses preserved
;; ---------------------------------------------------------------------------

(deftest regenerate-preserves-unselected
  (testing "unselected address retains its value"
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [trace]} (p/generate models/two-gaussians [] constraints)
          {:keys [trace]} (p/regenerate models/two-gaussians trace (sel/select :x))
          y (h/realize (cm/get-value (cm/get-submap (:choices trace) :y)))]
      (is (h/close? 2.0 y 1e-6)
          ":y preserved when only :x is selected"))))

;; ---------------------------------------------------------------------------
;; Independent model: weight = 0 (proposal = prior)
;; ---------------------------------------------------------------------------

(deftest regenerate-independent-model-weight-zero
  (testing "independent model: MH weight = 0 because proposal = prior"
    ;; Derivation for two independent N(0,1):
    ;; weight = new_score - old_score - proposal_ratio
    ;; new_score = lp(x_new) + lp(y)
    ;; old_score = lp(x_old) + lp(y)
    ;; proposal_ratio = lp(x_new) - lp(x_old)
    ;; weight = [lp(x_new) + lp(y)] - [lp(x_old) + lp(y)] - [lp(x_new) - lp(x_old)] = 0
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [trace]} (p/generate models/two-gaussians [] constraints)
          {:keys [weight]} (p/regenerate models/two-gaussians trace (sel/select :x))]
      (is (h/close? 0.0 (h/realize weight) 1e-4)
          "MH weight is 0 for independent model"))))

;; ---------------------------------------------------------------------------
;; Dependent model: weight = likelihood ratio
;; ---------------------------------------------------------------------------

(deftest regenerate-dependent-model-weight-is-likelihood-ratio
  (testing "regenerating x with y fixed: weight = log N(y;x_new,1) - log N(y;x_old,1)"
    ;; Derivation for dependent model x~N(0,1), y~N(x,1):
    ;; Regenerate x (selected), keep y (unselected).
    ;; weight = new_score - old_score - proposal_ratio
    ;; new_score = lp(x_new;0,1) + lp(y;x_new,1)
    ;; old_score = lp(x_old;0,1) + lp(y;x_old,1)
    ;; proposal_ratio = lp(x_new;0,1) - lp(x_old;0,1)
    ;; weight = lp(y;x_new,1) - lp(y;x_old,1)
    (let [x-old 1.0
          y-val 2.0
          constraints (cm/choicemap :x (mx/scalar x-old) :y (mx/scalar y-val))
          {:keys [trace]} (p/generate models/dependent-model [] constraints)
          {:keys [trace weight]} (p/regenerate models/dependent-model trace (sel/select :x))
          x-new (h/realize (cm/get-value (cm/get-submap (:choices trace) :x)))
          expected (- (h/gaussian-lp y-val x-new 1) (h/gaussian-lp y-val x-old 1))]
      (is (h/close? expected (h/realize weight) 1e-4)
          "weight is likelihood ratio for child given new vs old parent"))))

;; ---------------------------------------------------------------------------
;; Empty selection: identity (weight = 0, choices unchanged)
;; ---------------------------------------------------------------------------

(deftest regenerate-empty-selection-is-identity
  (testing "sel/none → weight = 0, all choices unchanged"
    (let [constraints (cm/choicemap :x (mx/scalar 1.5))
          {:keys [trace]} (p/generate models/single-gaussian [] constraints)
          {:keys [trace weight]} (p/regenerate models/single-gaussian trace sel/none)]
      (is (h/close? 0.0 (h/realize weight) 1e-6)
          "empty selection has zero weight")
      (is (h/close? 1.5 (h/realize (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6)
          "value unchanged with empty selection"))))

;; ---------------------------------------------------------------------------
;; Regenerate all: new score is valid
;; ---------------------------------------------------------------------------

(deftest regenerate-all-produces-valid-score
  (testing "selecting all addresses: new trace has valid score"
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [trace]} (p/generate models/two-gaussians [] constraints)
          {:keys [trace]} (p/regenerate models/two-gaussians trace sel/all)
          x (h/realize (cm/get-value (cm/get-submap (:choices trace) :x)))
          y (h/realize (cm/get-value (cm/get-submap (:choices trace) :y)))
          expected (+ (h/gaussian-lp x 0 1) (h/gaussian-lp y 0 1))]
      (is (h/close? expected (h/realize (:score trace)) 1e-4)
          "regenerated trace score is analytically correct"))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
