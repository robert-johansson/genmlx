(ns genmlx.gfi-assess-test
  "GFI assess: weight = log P(choices | args) for fully specified choices."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.test-helpers :as h]
            [genmlx.test-models :as models]))

;; ---------------------------------------------------------------------------
;; assess weight = log-joint
;; ---------------------------------------------------------------------------

(deftest assess-single-gaussian-log-prob
  (testing "assess returns log N(x; 0, 1)"
    (let [choices (cm/choicemap :x (mx/scalar 1.0))
          {:keys [weight]} (p/assess models/single-gaussian [] choices)
          ;; Derivation: log N(1; 0, 1) = -0.5*log(2π) - 0.5 = -1.41894
          expected (h/gaussian-lp 1.0 0 1)]
      (is (h/close? expected (h/realize weight) 1e-4)))))

(deftest assess-two-gaussians-log-joint
  (testing "assess returns log N(x;0,1) + log N(y;0,1)"
    (let [choices (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [weight]} (p/assess models/two-gaussians [] choices)
          ;; Derivation: independent joint
          ;; lp(x=1) = -0.5*log(2π) - 0.5
          ;; lp(y=2) = -0.5*log(2π) - 2.0
          expected (+ (h/gaussian-lp 1.0 0 1) (h/gaussian-lp 2.0 0 1))]
      (is (h/close? expected (h/realize weight) 1e-4)
          "assess weight = log-joint"))))

(deftest assess-dependent-model-log-joint
  (testing "assess includes conditional: log N(x;0,1) + log N(y;x,1)"
    (let [choices (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [weight]} (p/assess models/dependent-model [] choices)
          ;; Derivation: y|x ~ N(x,1)
          expected (+ (h/gaussian-lp 1.0 0 1) (h/gaussian-lp 2.0 1.0 1))]
      (is (h/close? expected (h/realize weight) 1e-4)))))

;; ---------------------------------------------------------------------------
;; assess agrees with generate on same choices
;; ---------------------------------------------------------------------------

(deftest assess-equals-generate-score
  (testing "assess weight = generate trace score for same choices"
    (let [choices (cm/choicemap :x (mx/scalar 0.5))
          {:keys [weight]} (p/assess models/single-gaussian [] choices)
          {:keys [trace]} (p/generate models/single-gaussian [] choices)]
      (is (h/close? (h/realize weight) (h/realize (:score trace)) 1e-6)
          "assess weight = generate score"))))

(deftest assess-equals-generate-two-gaussians
  (testing "assess and generate agree on two-gaussians"
    (let [choices (cm/choicemap :x (mx/scalar -1.0) :y (mx/scalar 0.5))
          {:keys [weight]} (p/assess models/two-gaussians [] choices)
          {:keys [trace]} (p/generate models/two-gaussians [] choices)]
      (is (h/close? (h/realize weight) (h/realize (:score trace)) 1e-6)))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
