(ns genmlx.gfi-simulate-test
  "GFI simulate: score = joint log-probability of all choices.
   Every expected value derived from the mathematical formula."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.test-helpers :as h]
            [genmlx.test-models :as models]))

;; ---------------------------------------------------------------------------
;; simulate: score = Σ log p(choice_i | parents)
;; ---------------------------------------------------------------------------

(deftest simulate-single-gaussian-score
  (testing "score = log N(x; 0, 1) for x ~ N(0,1)"
    (let [{:keys [choices score]} (p/simulate models/single-gaussian [])
          x (h/realize (cm/get-value (cm/get-submap choices :x)))
          ;; Derivation: log N(x; 0, 1) = -0.5*log(2π) - 0.5*x²
          expected (h/gaussian-lp x 0 1)]
      (is (h/close? expected (h/realize score) 1e-4)
          "score equals analytically computed log N(x; 0, 1)"))))

(deftest simulate-two-gaussians-score-additivity
  (testing "score = log N(x; 0,1) + log N(y; 0,1) for independent gaussians"
    (let [{:keys [choices score]} (p/simulate models/two-gaussians [])
          x (h/realize (cm/get-value (cm/get-submap choices :x)))
          y (h/realize (cm/get-value (cm/get-submap choices :y)))
          ;; Derivation: independent → joint log-prob = sum of marginals
          expected (+ (h/gaussian-lp x 0 1) (h/gaussian-lp y 0 1))]
      (is (h/close? expected (h/realize score) 1e-4)
          "score is sum of individual log-probs"))))

(deftest simulate-dependent-model-conditional-score
  (testing "score includes conditional: log N(x;0,1) + log N(y;x,1)"
    (let [{:keys [choices score]} (p/simulate models/dependent-model [])
          x (h/realize (cm/get-value (cm/get-submap choices :x)))
          y (h/realize (cm/get-value (cm/get-submap choices :y)))
          ;; Derivation: y|x ~ N(x,1), so log-prob uses (y-x) as deviation
          expected (+ (h/gaussian-lp x 0 1) (h/gaussian-lp y x 1))]
      (is (h/close? expected (h/realize score) 1e-4)
          "score includes conditional log-prob log N(y; x, 1)"))))

(deftest simulate-returns-valid-trace-structure
  (testing "trace has all required fields"
    (let [tr (p/simulate models/two-gaussians [])]
      (is (some? (:gen-fn tr)))
      (is (= [] (:args tr)))
      (is (instance? cm/Node (:choices tr)))
      (is (some? (:retval tr)))
      (is (mx/array? (:score tr))))))

(deftest simulate-multi-dist-score
  (testing "score is sum across different distribution types"
    (let [{:keys [choices score]} (p/simulate models/multi-dist-model [])
          a (h/realize (cm/get-value (cm/get-submap choices :a)))
          b (h/realize (cm/get-value (cm/get-submap choices :b)))
          c (h/realize (cm/get-value (cm/get-submap choices :c)))
          ;; a ~ N(0,1): log N(a; 0, 1)
          lp-a (h/gaussian-lp a 0 1)
          ;; b ~ Exp(1): log p(b) = log(1) - 1*b = -b (for b >= 0)
          lp-b (- b)
          ;; c ~ Bernoulli(0.5): log(0.5) regardless of outcome
          lp-c (js/Math.log 0.5)
          expected (+ lp-a lp-b lp-c)]
      (is (h/close? expected (h/realize score) 1e-4)
          "score sums log-probs across gaussian, exponential, bernoulli"))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
