(ns genmlx.gfi-update-test
  "GFI update: weight = new_score - old_score, discard contains old values."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.test-helpers :as h]
            [genmlx.test-models :as models]))

;; ---------------------------------------------------------------------------
;; weight = new_score - old_score
;; ---------------------------------------------------------------------------

(deftest update-weight-is-score-difference
  (testing "changing one address: weight = new_score - old_score"
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.0))
          {:keys [trace]} (p/generate models/two-gaussians [] constraints)
          old-score (h/realize (:score trace))
          new-constraints (cm/choicemap :x (mx/scalar 2.0))
          {:keys [trace weight]} (p/update models/two-gaussians trace new-constraints)
          new-score (h/realize (:score trace))
          w (h/realize weight)]
      (is (h/close? (- new-score old-score) w 1e-4)
          "weight = new_score - old_score"))))

(deftest update-weight-analytically-correct
  (testing "changing x from 1.0 to 2.0 in N(0,1): weight = Δlog-prob"
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.0))
          {:keys [trace]} (p/generate models/two-gaussians [] constraints)
          new-constraints (cm/choicemap :x (mx/scalar 2.0))
          {:keys [weight]} (p/update models/two-gaussians trace new-constraints)
          ;; Derivation:
          ;; old log-prob(:x) = log N(1; 0, 1) = -0.5*log(2π) - 0.5
          ;; new log-prob(:x) = log N(2; 0, 1) = -0.5*log(2π) - 2.0
          ;; weight = (-0.5*log(2π) - 2.0) - (-0.5*log(2π) - 0.5) = -1.5
          expected -1.5]
      (is (h/close? expected (h/realize weight) 1e-4)))))

;; ---------------------------------------------------------------------------
;; Identity update: weight = 0
;; ---------------------------------------------------------------------------

(deftest update-identity-has-zero-weight
  (testing "update with same values → weight = 0"
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.0))
          {:keys [trace]} (p/generate models/two-gaussians [] constraints)
          {:keys [weight]} (p/update models/two-gaussians trace constraints)]
      (is (h/close? 0.0 (h/realize weight) 1e-4)
          "identity update has zero weight"))))

;; ---------------------------------------------------------------------------
;; Discard contains old values
;; ---------------------------------------------------------------------------

(deftest update-discard-contains-old-value
  (testing "discard holds the replaced value"
    (let [constraints (cm/choicemap :x (mx/scalar 1.0))
          {:keys [trace]} (p/generate models/single-gaussian [] constraints)
          new-constraints (cm/choicemap :x (mx/scalar 2.0))
          {:keys [discard]} (p/update models/single-gaussian trace new-constraints)]
      (is (h/close? 1.0 (h/realize (cm/get-value (cm/get-submap discard :x))) 1e-6)
          "discard at :x is the old value 1.0"))))

(deftest update-new-trace-has-new-value
  (testing "updated trace contains the new constrained value"
    (let [constraints (cm/choicemap :x (mx/scalar 1.0))
          {:keys [trace]} (p/generate models/single-gaussian [] constraints)
          new-constraints (cm/choicemap :x (mx/scalar 2.0))
          {:keys [trace]} (p/update models/single-gaussian trace new-constraints)]
      (is (h/close? 2.0 (h/realize (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6)))))

;; ---------------------------------------------------------------------------
;; Update round-trip: update then update back recovers original
;; ---------------------------------------------------------------------------

(deftest update-round-trip-recovers-original
  (testing "update(trace, c) then update(trace', discard) → original values"
    (let [constraints (cm/choicemap :x (mx/scalar 1.0))
          {:keys [trace]} (p/generate models/single-gaussian [] constraints)
          orig-score (h/realize (:score trace))
          new-constraints (cm/choicemap :x (mx/scalar 42.0))
          {:keys [trace discard]} (p/update models/single-gaussian trace new-constraints)
          {:keys [trace weight]} (p/update models/single-gaussian trace discard)
          recovered-x (h/realize (cm/get-value (cm/get-submap (:choices trace) :x)))]
      (is (h/close? 1.0 recovered-x 1e-6)
          "round-trip recovers original value")
      (is (h/close? orig-score (h/realize (:score trace)) 1e-4)
          "round-trip recovers original score"))))

;; ---------------------------------------------------------------------------
;; Dependent model: update cascades through conditional
;; ---------------------------------------------------------------------------

(deftest update-dependent-model-score-correctness
  (testing "changing x in dependent model updates conditional score"
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [trace]} (p/generate models/dependent-model [] constraints)
          new-constraints (cm/choicemap :x (mx/scalar 0.0))
          {:keys [trace weight]} (p/update models/dependent-model trace new-constraints)
          new-score (h/realize (:score trace))
          ;; Derivation: new score = log N(0;0,1) + log N(2;0,1)
          ;; x=0: log N(0;0,1) = -0.5*log(2π)
          ;; y=2|x=0: log N(2;0,1) = -0.5*log(2π) - 2.0
          expected (+ (h/gaussian-lp 0 0 1) (h/gaussian-lp 2 0 1))]
      (is (h/close? expected new-score 1e-4)
          "new score reflects updated conditional"))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
