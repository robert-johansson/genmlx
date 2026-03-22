(ns genmlx.gfi-generate-test
  "GFI generate: constrained execution with analytically verified weights.
   weight = log P(constrained choices | args) for the constrained subset."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.test-helpers :as h]
            [genmlx.test-models :as models]))

;; ---------------------------------------------------------------------------
;; Fully constrained: weight = score
;; ---------------------------------------------------------------------------

(deftest fully-constrained-weight-equals-score
  (testing "all addresses constrained → weight = score"
    (let [constraints (cm/choicemap :x (mx/scalar 1.5) :y (mx/scalar -0.5))
          {:keys [trace weight]} (p/generate models/two-gaussians [] constraints)]
      (is (h/close? (h/realize (:score trace)) (h/realize weight) 1e-6)
          "weight equals score when fully constrained"))))

(deftest fully-constrained-weight-is-analytically-correct
  (testing "weight = log N(1.5; 0,1) + log N(-0.5; 0,1)"
    (let [constraints (cm/choicemap :x (mx/scalar 1.5) :y (mx/scalar -0.5))
          {:keys [weight]} (p/generate models/two-gaussians [] constraints)
          ;; Derivation: joint = product of independent marginals
          expected (+ (h/gaussian-lp 1.5 0 1) (h/gaussian-lp -0.5 0 1))]
      (is (h/close? expected (h/realize weight) 1e-4)))))

;; ---------------------------------------------------------------------------
;; No constraints: weight = 0
;; ---------------------------------------------------------------------------

(deftest no-constraints-weight-is-zero
  (testing "empty constraints → weight = 0 (nothing constrained)"
    (let [{:keys [weight]} (p/generate models/two-gaussians [] cm/EMPTY)]
      (is (h/close? 0.0 (h/realize weight) 1e-6)
          "weight = 0 when no constraints applied"))))

;; ---------------------------------------------------------------------------
;; Partial constraints: constrained value exact, weight correct
;; ---------------------------------------------------------------------------

(deftest partial-constraints-value-exact
  (testing "constrained address takes the exact constrained value"
    (let [constraints (cm/choicemap :x (mx/scalar 2.0))
          {:keys [trace]} (p/generate models/two-gaussians [] constraints)
          x (h/realize (cm/get-value (cm/get-submap (:choices trace) :x)))]
      (is (h/close? 2.0 x 1e-6)
          "constrained :x is exactly 2.0"))))

(deftest partial-constraints-weight-is-constrained-log-prob
  (testing "weight = log N(x_obs; 0, 1) for single constrained gaussian"
    (let [x-obs 2.0
          constraints (cm/choicemap :x (mx/scalar x-obs))
          {:keys [weight]} (p/generate models/two-gaussians [] constraints)
          ;; Derivation: weight = log-prob of constrained addresses only
          expected (h/gaussian-lp x-obs 0 1)]
      (is (h/close? expected (h/realize weight) 1e-4)
          "weight is log-prob of constrained value"))))

;; ---------------------------------------------------------------------------
;; Single gaussian: weight = log N(x; 0, 1)
;; ---------------------------------------------------------------------------

(deftest single-gaussian-generate-weight
  (testing "constraining x=1.5: weight = -0.5*log(2π) - 0.5*1.5²"
    (let [constraints (cm/choicemap :x (mx/scalar 1.5))
          {:keys [weight]} (p/generate models/single-gaussian [] constraints)
          ;; Derivation: log N(1.5; 0, 1) = -0.5*log(2π) - 0.5*(2.25)
          ;;           = -0.91894 - 1.125 = -2.04394
          expected (h/gaussian-lp 1.5 0 1)]
      (is (h/close? expected (h/realize weight) 1e-4)))))

;; ---------------------------------------------------------------------------
;; Dependent model: constrained value propagates to conditional
;; ---------------------------------------------------------------------------

(deftest dependent-model-generate-fully-constrained
  (testing "weight = log N(x;0,1) + log N(y;x,1) with both constrained"
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [trace weight]} (p/generate models/dependent-model [] constraints)
          ;; Derivation: y|x ~ N(x,1), so log p(y|x) uses (y-x)
          expected (+ (h/gaussian-lp 1.0 0 1) (h/gaussian-lp 2.0 1.0 1))]
      (is (h/close? expected (h/realize weight) 1e-4)
          "weight includes conditional log-prob"))))

(deftest dependent-model-generate-partial
  (testing "constraining only x: weight = log N(x; 0, 1)"
    (let [constraints (cm/choicemap :x (mx/scalar 1.0))
          {:keys [weight]} (p/generate models/dependent-model [] constraints)
          expected (h/gaussian-lp 1.0 0 1)]
      (is (h/close? expected (h/realize weight) 1e-4)))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
