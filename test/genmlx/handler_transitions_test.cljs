(ns genmlx.handler-transitions-test
  "Phase 3.2-3.3: Handler transition tests with analytical score verification.
   All 7 GFI operations tested: simulate, generate, update, regenerate,
   assess, project, and batched variants.
   Every score and weight is verified against the analytical formula."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.handler :as handler]
            [genmlx.runtime :as rt]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private LOG-2PI (js/Math.log (* 2 js/Math.PI)))

(defn- gaussian-logprob
  "Analytical log N(v; mu, sigma)."
  [v mu sigma]
  (let [z (/ (- v mu) sigma)]
    (- (* -0.5 LOG-2PI) (js/Math.log sigma) (* 0.5 z z))))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def single-gaussian
  (dyn/auto-key
   (gen []
        (trace :x (dist/gaussian 0 1)))))

(def two-gaussians
  (dyn/auto-key
   (gen []
        (let [x (trace :x (dist/gaussian 0 1))
              y (trace :y (dist/gaussian 0 1))]
          (mx/add x y)))))

(def dependent-model
  (dyn/auto-key
   (gen []
        (let [x (trace :x (dist/gaussian 0 1))]
          (trace :y (dist/gaussian x 1))))))

;; ==========================================================================
;; 1. Simulate: score = joint log-prob
;; ==========================================================================
;; simulate score = sum of log p(v_i | parents(v_i)) for all trace sites

(deftest simulate-score-is-joint-log-prob
  (testing "single gaussian: score = log N(x; 0, 1)"
    (let [tr (p/simulate single-gaussian [])
          x (h/realize (cm/get-value (cm/get-submap (:choices tr) :x)))
          score (h/realize (:score tr))
          expected (gaussian-logprob x 0 1)]
      (is (h/close? expected score 1e-4)
          "score equals analytically computed log-prob")))

  (testing "two gaussians: score = log N(x;0,1) + log N(y;0,1)"
    (let [tr (p/simulate two-gaussians [])
          x (h/realize (cm/get-value (cm/get-submap (:choices tr) :x)))
          y (h/realize (cm/get-value (cm/get-submap (:choices tr) :y)))
          score (h/realize (:score tr))
          expected (+ (gaussian-logprob x 0 1) (gaussian-logprob y 0 1))]
      (is (h/close? expected score 1e-4)
          "score is sum of individual log-probs")))

  (testing "dependent model: score includes conditional log-prob"
    ;; x ~ N(0,1), y ~ N(x,1)
    ;; score = log N(x;0,1) + log N(y;x,1)
    (let [tr (p/simulate dependent-model [])
          x (h/realize (cm/get-value (cm/get-submap (:choices tr) :x)))
          y (h/realize (cm/get-value (cm/get-submap (:choices tr) :y)))
          score (h/realize (:score tr))
          expected (+ (gaussian-logprob x 0 1)
                      (gaussian-logprob y x 1))]
      (is (h/close? expected score 1e-4)
          "score includes conditional log-prob"))))

;; ==========================================================================
;; 2. Generate: weight = log-likelihood of observations
;; ==========================================================================
;; When all addresses constrained: weight = score (since proposal = prior)
;; When partially constrained: weight = sum of log-probs of constrained sites

(deftest generate-fully-constrained
  (testing "all addresses constrained: weight = score"
    (let [constraints (cm/choicemap :x (mx/scalar 1.5) :y (mx/scalar -0.5))
          {:keys [trace weight]} (p/generate two-gaussians [] constraints)
          w (h/realize weight)
          s (h/realize (:score trace))]
      (is (h/close? s w 1e-4) "weight equals score for full constraints")))

  (testing "weight is analytically correct"
    ;; constrain x=1.5, y=-0.5 in two independent N(0,1)
    ;; weight = log N(1.5;0,1) + log N(-0.5;0,1)
    (let [constraints (cm/choicemap :x (mx/scalar 1.5) :y (mx/scalar -0.5))
          {:keys [weight]} (p/generate two-gaussians [] constraints)
          w (h/realize weight)
          expected (+ (gaussian-logprob 1.5 0 1) (gaussian-logprob -0.5 0 1))]
      (is (h/close? expected w 1e-4)
          "weight = sum of log-probs of constrained values"))))

(deftest generate-unconstrained
  (testing "no constraints: weight = 0 (prior proposal)"
    ;; When proposal = prior and no observations, weight should be 0
    ;; Actually, with EMPTY constraints, generate falls through to simulate
    ;; for all addresses, so weight = 0 (score accumulates, but weight stays 0
    ;; because only constrained sites add to weight)
    (let [{:keys [weight]} (p/generate two-gaussians [] cm/EMPTY)
          w (h/realize weight)]
      ;; weight should be 0 because no sites were constrained
      (is (h/close? 0.0 w 1e-4)
          "weight = 0 when no constraints"))))

(deftest generate-partial-constraints
  (testing "partial constraints: constrained value exact, weight = log-prob of constrained"
    (let [constraints (cm/choicemap :x (mx/scalar 2.0))
          {:keys [trace weight]} (p/generate two-gaussians [] constraints)
          x (h/realize (cm/get-value (cm/get-submap (:choices trace) :x)))
          w (h/realize weight)]
      (is (h/close? 2.0 x 1e-6) "constrained value is exact")
      ;; weight = log N(2;0,1) since only :x was constrained
      (is (h/close? (gaussian-logprob 2.0 0 1) w 1e-4)
          "weight = log-prob of constrained site only"))))

(deftest generate-single-constrained
  (testing "single constrained gaussian: weight = log N(x; 0, 1)"
    ;; Constrain x=1.5 in single-gaussian model
    ;; weight = log N(1.5; 0, 1) = -0.5*log(2pi) - 0.5*2.25 = -2.04394
    (let [constraints (cm/choicemap :x (mx/scalar 1.5))
          {:keys [weight]} (p/generate single-gaussian [] constraints)
          w (h/realize weight)
          expected (gaussian-logprob 1.5 0 1)]
      (is (h/close? expected w 1e-4)
          "weight equals log-prob of constrained value"))))

;; ==========================================================================
;; 3. Update: weight = new_score - old_score for changed addresses
;; ==========================================================================

(deftest update-weight-is-score-difference
  (testing "changing one address: weight = new_lp - old_lp"
    (let [;; Generate initial trace with known values
          constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.0))
          {:keys [trace]} (p/generate two-gaussians [] constraints)
          old-score (h/realize (:score trace))
          ;; Update x from 1.0 to 2.0
          new-constraints (cm/choicemap :x (mx/scalar 2.0))
          {:keys [trace weight]} (p/update two-gaussians trace new-constraints)
          new-score (h/realize (:score trace))
          w (h/realize weight)]
      ;; weight = new_score - old_score
      (is (h/close? (- new-score old-score) w 1e-4)
          "weight = new_score - old_score")
      ;; Analytically: weight = log N(2;0,1) - log N(1;0,1)
      ;; = (-0.5*log(2pi) - 2) - (-0.5*log(2pi) - 0.5) = -1.5
      (is (h/close? -1.5 w 1e-4)
          "weight analytically equals -1.5"))))

(deftest update-identity
  (testing "update with same values: weight = 0"
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.0))
          {:keys [trace]} (p/generate two-gaussians [] constraints)
          ;; Re-constrain with same values
          {:keys [weight]} (p/update two-gaussians trace constraints)]
      (is (h/close? 0.0 (h/realize weight) 1e-4)
          "identity update has zero weight"))))

(deftest update-discard-contains-old-values
  (testing "discard holds the replaced value"
    (let [constraints (cm/choicemap :x (mx/scalar 1.0))
          {:keys [trace]} (p/generate single-gaussian [] constraints)
          new-constraints (cm/choicemap :x (mx/scalar 2.0))
          {:keys [discard]} (p/update single-gaussian trace new-constraints)]
      (is (h/close? 1.0
                    (h/realize (cm/get-value (cm/get-submap discard :x)))
                    1e-6)
          "discard contains old value"))))

;; ==========================================================================
;; 4. Regenerate: weight = log p(new) - log p(old) for selected addresses
;; ==========================================================================

(deftest regenerate-preserves-unselected
  (testing "unselected address preserved"
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [trace]} (p/generate two-gaussians [] constraints)
          selection (sel/select :x)
          {:keys [trace]} (p/regenerate two-gaussians trace selection)
          y (h/realize (cm/get-value (cm/get-submap (:choices trace) :y)))]
      (is (h/close? 2.0 y 1e-6) "unselected address preserved"))))

(deftest regenerate-weight-is-log-ratio
  (testing "single-site: weight = 0 (proposal = prior, no downstream)"
    ;; For single-gaussian, regenerating :x from prior gives:
    ;; weight = (new_score - old_score) - (new_lp - old_lp) = 0
    ;; because new_score = new_lp, old_score = old_lp
    (let [constraints (cm/choicemap :x (mx/scalar 1.0))
          {:keys [trace]} (p/generate single-gaussian [] constraints)
          selection (sel/select :x)
          {:keys [weight]} (p/regenerate single-gaussian trace selection)]
      (is (h/close? 0.0 (h/realize weight) 1e-4)
          "single-site regen from prior has zero weight")))

  (testing "dependent model: weight = change in likelihood of downstream"
    ;; x ~ N(0,1), y ~ N(x,1). Regenerate :x, keep :y fixed.
    ;; weight = (new_score - old_score) - proposal_ratio
    ;;        = [log N(new_x;0,1) + log N(y;new_x,1) - log N(old_x;0,1) - log N(y;old_x,1)]
    ;;          - [log N(new_x;0,1) - log N(old_x;0,1)]
    ;;        = log N(y; new_x, 1) - log N(y; old_x, 1)
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [trace]} (p/generate dependent-model [] constraints)
          selection (sel/select :x)
          {:keys [trace weight]} (p/regenerate dependent-model trace selection)
          w (h/realize weight)
          new-x (h/realize (cm/get-value (cm/get-submap (:choices trace) :x)))
          y 2.0
          old-x 1.0
          expected-w (- (gaussian-logprob y new-x 1)
                        (gaussian-logprob y old-x 1))]
      (is (h/close? expected-w w 1e-4)
          "weight = change in likelihood of y given new x"))))

(deftest regenerate-empty-selection-is-identity
  (testing "empty selection: weight = 0, choices unchanged"
    (let [constraints (cm/choicemap :x (mx/scalar 1.5))
          {:keys [trace]} (p/generate single-gaussian [] constraints)
          {:keys [trace weight]} (p/regenerate single-gaussian trace sel/none)]
      (is (h/close? 0.0 (h/realize weight) 1e-6)
          "empty regen has zero weight")
      (is (h/close? 1.5
                    (h/realize (cm/get-value (cm/get-submap (:choices trace) :x)))
                    1e-6)
          "value unchanged"))))

;; ==========================================================================
;; 5. Assess: weight = log-joint
;; ==========================================================================
;; assess requires ALL addresses constrained. weight = sum of all log-probs.

(deftest assess-score-equals-log-joint
  (testing "fully specified: weight = log P(choices)"
    ;; two-gaussians: x=1, y=2
    ;; weight = log N(1;0,1) + log N(2;0,1)
    (let [choices (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [weight]} (p/assess two-gaussians [] choices)
          w (h/realize weight)
          expected (+ (gaussian-logprob 1.0 0 1) (gaussian-logprob 2.0 0 1))]
      (is (h/close? expected w 1e-4)
          "assess weight = log-joint"))))

(deftest assess-equals-generate-weight
  (testing "assess and generate agree on fully constrained choices"
    (let [choices (cm/choicemap :x (mx/scalar 0.5))
          {:keys [weight]} (p/assess single-gaussian [] choices)
          {:keys [trace]} (p/generate single-gaussian [] choices)]
      (is (h/close? (h/realize weight) (h/realize (:score trace)) 1e-6)
          "assess weight = generate score"))))

(deftest assess-dependent-model
  (testing "dependent model: weight includes conditional"
    ;; x=1, y=2 in model where x~N(0,1), y~N(x,1)
    ;; weight = log N(1;0,1) + log N(2;1,1)
    (let [choices (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [weight]} (p/assess dependent-model [] choices)
          w (h/realize weight)
          expected (+ (gaussian-logprob 1.0 0 1)
                      (gaussian-logprob 2.0 1 1))]
      (is (h/close? expected w 1e-4)
          "assess includes conditional log-prob"))))

;; ==========================================================================
;; 6. Project: weight = sum of log-probs for selected addresses
;; ==========================================================================

(deftest project-all-equals-score
  (testing "project with sel/all = trace score"
    (let [tr (p/simulate two-gaussians [])
          projected (p/project two-gaussians tr sel/all)]
      (is (h/close? (h/realize (:score tr)) (h/realize projected) 1e-6)
          "project(all) = score"))))

(deftest project-none-equals-zero
  (testing "project with sel/none = 0"
    (let [tr (p/simulate two-gaussians [])
          projected (p/project two-gaussians tr sel/none)]
      (is (h/close? 0.0 (h/realize projected) 1e-6)
          "project(none) = 0"))))

(deftest project-decomposition
  (testing "project(:x) + project(:y) = score (additive decomposition)"
    (let [tr (p/simulate two-gaussians [])
          px (h/realize (p/project two-gaussians tr (sel/select :x)))
          py (h/realize (p/project two-gaussians tr (sel/select :y)))
          score (h/realize (:score tr))]
      (is (h/close? score (+ px py) 1e-4)
          "score decomposes additively"))))

(deftest project-single-analytically-correct
  (testing "project(:x) = log N(x; 0, 1)"
    (let [constraints (cm/choicemap :x (mx/scalar 1.5) :y (mx/scalar 0.0))
          {:keys [trace]} (p/generate two-gaussians [] constraints)
          px (h/realize (p/project two-gaussians trace (sel/select :x)))
          expected (gaussian-logprob 1.5 0 1)]
      (is (h/close? expected px 1e-4)
          "project(:x) equals log-prob at x"))))

;; ==========================================================================
;; 7. Batched simulate: [N]-shaped scores
;; ==========================================================================

(deftest batched-simulate-shapes
  (testing "batched simulate produces [N]-shaped score"
    (let [n 50
          key (h/deterministic-key)
          d (dist/gaussian 0 1)
          init {:key key :choices cm/EMPTY :score (mx/zeros [n])
                :batch-size n :batched? true :executor nil}
          [v state] (handler/batched-simulate-transition init :x d)]
      (mx/eval! v)
      (mx/eval! (:score state))
      (is (= [n] (mx/shape v)) "values are [N]-shaped")
      (is (= [n] (mx/shape (:score state))) "scores are [N]-shaped"))))

(deftest batched-generate-shapes
  (testing "batched generate with constraints: scalar observation broadcasts"
    (let [n 50
          d (dist/gaussian 0 1)
          constraints (cm/choicemap :x (mx/scalar 1.5))
          init {:key (h/deterministic-key) :choices cm/EMPTY :score (mx/zeros [n])
                :weight (mx/zeros [n]) :batch-size n :batched? true
                :constraints constraints :executor nil}
          [v state] (handler/batched-generate-transition init :x d)]
      (mx/eval! (:weight state))
      ;; Weight should broadcast the scalar log-prob to [N]
      ;; All elements should be the same: log N(1.5; 0, 1)
      (let [w (mx/->clj (:weight state))
            expected (gaussian-logprob 1.5 0 1)]
        (is (every? #(h/close? expected % 1e-4) w)
            "all weight elements equal scalar log-prob")))))

;; ==========================================================================
;; Run tests
;; ==========================================================================

(cljs.test/run-tests)
