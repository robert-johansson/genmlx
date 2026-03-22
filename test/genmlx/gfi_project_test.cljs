(ns genmlx.gfi-project-test
  "GFI project: log-probability of selected subset of choices.
   project(all) = score, project(none) = 0, additive decomposition."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.test-helpers :as h]
            [genmlx.test-models :as models]))

;; ---------------------------------------------------------------------------
;; project(all) = score
;; ---------------------------------------------------------------------------

(deftest project-all-equals-score
  (testing "project with sel/all returns full trace score"
    (let [tr (p/simulate models/two-gaussians [])
          projected (h/realize (p/project models/two-gaussians tr sel/all))
          score (h/realize (:score tr))]
      (is (h/close? score projected 1e-6)
          "project(all) = score"))))

;; ---------------------------------------------------------------------------
;; project(none) = 0
;; ---------------------------------------------------------------------------

(deftest project-none-equals-zero
  (testing "project with sel/none returns 0"
    (let [tr (p/simulate models/two-gaussians [])
          projected (h/realize (p/project models/two-gaussians tr sel/none))]
      (is (h/close? 0.0 projected 1e-6)
          "project(none) = 0"))))

;; ---------------------------------------------------------------------------
;; Additive decomposition: project(:x) + project(:y) = score
;; ---------------------------------------------------------------------------

(deftest project-additive-decomposition
  (testing "project(:x) + project(:y) = score for independent model"
    (let [tr (p/simulate models/two-gaussians [])
          px (h/realize (p/project models/two-gaussians tr (sel/select :x)))
          py (h/realize (p/project models/two-gaussians tr (sel/select :y)))
          score (h/realize (:score tr))]
      (is (h/close? score (+ px py) 1e-4)
          "score decomposes additively over independent addresses"))))

;; ---------------------------------------------------------------------------
;; project individual address = that address's log-prob
;; ---------------------------------------------------------------------------

(deftest project-single-address-is-log-prob
  (testing "project(:x) = log N(x; 0, 1) for single-gaussian"
    (let [constraints (cm/choicemap :x (mx/scalar 1.5))
          {:keys [trace]} (p/generate models/single-gaussian [] constraints)
          px (h/realize (p/project models/single-gaussian trace (sel/select :x)))
          expected (h/gaussian-lp 1.5 0 1)]
      (is (h/close? expected px 1e-4)
          "project(:x) is analytically correct log-prob"))))

;; ---------------------------------------------------------------------------
;; Dependent model decomposition
;; ---------------------------------------------------------------------------

(deftest project-dependent-model-decomposition
  (testing "project(:x) + project(:y) = score for dependent model"
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [trace]} (p/generate models/dependent-model [] constraints)
          px (h/realize (p/project models/dependent-model trace (sel/select :x)))
          py (h/realize (p/project models/dependent-model trace (sel/select :y)))
          score (h/realize (:score trace))]
      (is (h/close? score (+ px py) 1e-4)
          "score decomposes even for dependent model"))))

(deftest project-dependent-model-individual-terms
  (testing "project(:x) = log N(x;0,1), project(:y) = log N(y;x,1)"
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [trace]} (p/generate models/dependent-model [] constraints)
          px (h/realize (p/project models/dependent-model trace (sel/select :x)))
          py (h/realize (p/project models/dependent-model trace (sel/select :y)))]
      (is (h/close? (h/gaussian-lp 1.0 0 1) px 1e-4)
          "project(:x) = log N(1; 0, 1)")
      (is (h/close? (h/gaussian-lp 2.0 1.0 1) py 1e-4)
          "project(:y) = log N(2; 1, 1)"))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
