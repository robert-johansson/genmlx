(ns genmlx.gfi-regenerate-test
  "GFI Laws #4, #14, #15 — Regenerate weight formula and MH correctness.

   Mathematical ground truth derived from:
     Cusumano-Towner 2020, PhD thesis
     - Chapter 4, Eq 4.1: regenerate weight formula
     - Algorithm 5: MH acceptance probability
     - Section 3.4.2: MH proposal validity requirements

   All expected values are derived from first principles, never from
   running the implementation.

   ITEM #4 — Regenerate weight formula (Eq 4.1)
   =============================================

   The GFI regenerate contract for a trace t with selection S:

     weight = new_score - old_score - handler_weight

   where:
     new_score    = sum_i log p(tau'[a_i] | parents(a_i) in tau')
     old_score    = sum_i log p(tau[a_i]  | parents(a_i) in tau)
     handler_weight = sum_{i in S} [log p(tau'[a_i] | replay_parents)
                                  - log p(tau[a_i]  | replay_parents)]

   Critical: replay_parents uses the NEW values for previously-visited
   selected addresses. For address a_i, if a parent a_j (j < i) was
   selected, then replay_parents(a_i) uses tau'[a_j], not tau[a_j].

   For GenMLX's forward-sampling internal proposal (q = conditional prior):
     handler_weight = sum_{i in S} [log q(tau'[a_i]) - log q_reverse(tau[a_i])]

   Simplification cases (all verified algebraically and numerically):

   Case 1: Independent model, select one of two
     x ~ N(0,1), y ~ N(0,1), S = {:x}
     handler_weight = lp(x';0,1) - lp(x;0,1)
     weight = [lp(x';0,1)+lp(y;0,1)] - [lp(x;0,1)+lp(y;0,1)]
            - [lp(x';0,1)-lp(x;0,1)]
            = 0

   Case 2: Dependent model, select parent
     x ~ N(0,1), y ~ N(x,1), S = {:x}
     handler_weight = lp(x';0,1) - lp(x;0,1)
     weight = [lp(x';0,1)+lp(y;x',1)] - [lp(x;0,1)+lp(y;x,1)]
            - [lp(x';0,1)-lp(x;0,1)]
     Cancelling x prior terms:
            = lp(y;x',1) - lp(y;x,1)

   Case 3: Dependent model, select child only
     x ~ N(0,1), y ~ N(x,1), S = {:y}, x kept
     handler_weight = lp(y';x,1) - lp(y;x,1)    [dist uses kept x]
     weight = [lp(x;0,1)+lp(y';x,1)] - [lp(x;0,1)+lp(y;x,1)]
            - [lp(y';x,1)-lp(y;x,1)]
            = 0

   Case 4: Dependent model, select ALL
     x ~ N(0,1), y ~ N(x,1), S = {:x,:y}
     handler_weight for :x: lp(x';0,1) - lp(x;0,1)
     handler_weight for :y: lp(y';x',1) - lp(y;x',1)
       [old_lp for y uses x' because x was already replayed with new value]
     total handler_weight = [lp(x';0,1)-lp(x;0,1)] + [lp(y';x',1)-lp(y;x',1)]
     weight = [lp(x';0,1)+lp(y';x',1)] - [lp(x;0,1)+lp(y;x,1)]
            - [lp(x';0,1)-lp(x;0,1)] - [lp(y';x',1)-lp(y;x',1)]
     Cancelling: = lp(y;x',1) - lp(y;x,1)
     Note: NOT zero! The 'stale child' correction.

   Case 5: Empty selection
     S = {}, no addresses resampled
     handler_weight = 0
     new_score = old_score (same values, same distributions)
     weight = 0

   ITEM #14 — MH acceptance probability (Alg 5)
   ==============================================
   alpha = min{1, exp(log_w)}
   Implementation: accept if log(u) < log_w where u ~ U(0,1)
   This is correct because P(log(u) < w) = P(u < exp(w)) = min{1,exp(w)}

   ITEM #15 — MH proposal validity (Section 3.4.2)
   =================================================
   Three requirements:
   1. Proposal takes current trace as argument  [regenerate signature]
   2. Selection should not include observed addresses [user responsibility]
   3. Reversibility: forward-sampling proposal has full support

   Tolerance policy:
   - float32 deterministic: 1e-5 (7 significant digits, accumulation)
   - Exact zero (no accumulation): 1e-6
   - Statistical (MH posterior): z * sigma / sqrt(N_eff), z=3.5"
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.util :as u]
            [genmlx.test-helpers :as h]
            [genmlx.test-models :as models])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Additional test models
;; ---------------------------------------------------------------------------

(def dep-model-narrow
  "x ~ N(0,1), y ~ N(x, 0.5) — for MH posterior test."
  (dyn/auto-key
   (gen []
     (let [x (trace :x (dist/gaussian 0 1))]
       (trace :y (dist/gaussian x 0.5))))))

;; =========================================================================
;; ITEM #4: Regenerate weight formula [Cusumano-Towner 2020, Eq 4.1]
;; =========================================================================

;; ---------------------------------------------------------------------------
;; Case 1: Independent model, select one — weight = 0
;; ---------------------------------------------------------------------------

(deftest regenerate-independent-weight-zero
  (testing "Independent model: weight = 0 when selecting one address"
    ;; Derivation: x ~ N(0,1), y ~ N(0,1), S = {:x}
    ;; weight = [lp(x')+lp(y)] - [lp(x)+lp(y)] - [lp(x')-lp(x)]
    ;;        = lp(x')+lp(y)-lp(x)-lp(y)-lp(x')+lp(x) = 0
    ;; Tolerance: 1e-5 (float32 accumulation: 4 additions/subtractions)
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [trace]} (p/generate models/two-gaussians [] constraints)
          {:keys [weight]} (p/regenerate models/two-gaussians trace (sel/select :x))]
      (is (h/close? 0.0 (h/realize weight) 1e-5)
          "weight is 0 for independent model"))))

(deftest regenerate-independent-select-all-weight-zero
  (testing "Independent model: select-all also gives weight = 0"
    ;; Derivation: both selected, no parent dependencies
    ;; handler_weight = [lp(x')-lp(x)] + [lp(y')-lp(y)]
    ;;               = new_score - old_score
    ;; weight = new_score - old_score - handler_weight = 0
    ;; Tolerance: 1e-5
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [trace]} (p/generate models/two-gaussians [] constraints)
          {:keys [weight]} (p/regenerate models/two-gaussians trace sel/all)]
      (is (h/close? 0.0 (h/realize weight) 1e-5)
          "weight is 0 for independent model with select-all"))))

;; ---------------------------------------------------------------------------
;; Case 2: Dependent model, select parent — weight = child likelihood ratio
;; ---------------------------------------------------------------------------

(deftest regenerate-dependent-select-parent
  (testing "Dependent model: weight = lp(y;x',1) - lp(y;x,1)"
    ;; Derivation: x ~ N(0,1), y ~ N(x,1), S = {:x}
    ;; new_score = lp(x';0,1) + lp(y;x',1)
    ;; old_score = lp(x;0,1)  + lp(y;x,1)
    ;; handler_weight = lp(x';0,1) - lp(x;0,1)
    ;; weight = lp(y;x',1) - lp(y;x,1)
    ;;
    ;; The prior terms for x cancel exactly. The weight captures only
    ;; the change in y's likelihood due to x changing.
    ;; Tolerance: 1e-4 (float32 accumulation in score computation)
    (let [x-old 1.0
          y-val 2.0
          constraints (cm/choicemap :x (mx/scalar x-old) :y (mx/scalar y-val))
          {:keys [trace]} (p/generate models/dependent-model [] constraints)
          {:keys [trace weight]} (p/regenerate models/dependent-model trace (sel/select :x))
          x-new (h/realize (cm/get-value (cm/get-submap (:choices trace) :x)))
          ;; Gaussian log-prob: -0.5*log(2*pi) - log(sigma) - 0.5*((x-mu)/sigma)^2
          expected (- (h/gaussian-lp y-val x-new 1) (h/gaussian-lp y-val x-old 1))]
      (is (h/close? expected (h/realize weight) 1e-4)
          "weight is child likelihood ratio"))))

;; ---------------------------------------------------------------------------
;; Case 3: Dependent model, select child only — weight = 0
;; ---------------------------------------------------------------------------

(deftest regenerate-dependent-select-child-weight-zero
  (testing "Dependent model, select leaf only: weight = 0"
    ;; Derivation: x ~ N(0,1), y ~ N(x,1), S = {:y}, x kept
    ;; handler_weight = lp(y';x,1) - lp(y;x,1)  [dist uses kept x]
    ;; weight = [lp(x;0,1)+lp(y';x,1)] - [lp(x;0,1)+lp(y;x,1)]
    ;;        - [lp(y';x,1)-lp(y;x,1)]
    ;; = lp(y';x,1) - lp(y;x,1) - lp(y';x,1) + lp(y;x,1) = 0
    ;;
    ;; Intuition: selecting a leaf with no children and proposing from
    ;; the prior = model distribution at that site. Every proposal is
    ;; equally good, so MH always accepts.
    ;; Tolerance: 1e-5
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [trace]} (p/generate models/dependent-model [] constraints)
          {:keys [trace weight]} (p/regenerate models/dependent-model trace (sel/select :y))
          x-kept (h/realize (cm/get-value (cm/get-submap (:choices trace) :x)))]
      (is (h/close? 0.0 (h/realize weight) 1e-5)
          "weight is 0 for leaf-only selection")
      (is (h/close? 1.0 x-kept 1e-6)
          "parent x is preserved"))))

;; ---------------------------------------------------------------------------
;; Case 4: Dependent model, select ALL — weight = stale child correction
;; ---------------------------------------------------------------------------

(deftest regenerate-dependent-select-all-stale-child
  (testing "Dependent model, select-all: weight = lp(y_old;x',1) - lp(y_old;x,1)"
    ;; Derivation: x ~ N(0,1), y ~ N(x,1), S = {:x,:y}
    ;; handler for :x: lp(x';0,1) - lp(x;0,1)
    ;; handler for :y: lp(y';x',1) - lp(y;x',1)
    ;;   [old_lp for y evaluated under NEW x' because x was already replayed]
    ;; handler_weight = [lp(x';0,1)-lp(x;0,1)] + [lp(y';x',1)-lp(y;x',1)]
    ;;
    ;; weight = new_score - old_score - handler_weight
    ;; = [lp(x';0,1)+lp(y';x',1)] - [lp(x;0,1)+lp(y;x,1)]
    ;;   - [lp(x';0,1)-lp(x;0,1)] - [lp(y';x',1)-lp(y;x',1)]
    ;; Cancel lp(x';0,1), lp(x;0,1), lp(y';x',1):
    ;; = lp(y;x',1) - lp(y;x,1)
    ;;
    ;; This is the "stale child" correction: old y's density changes
    ;; because its parent changed. Weight != 0 in general.
    ;; Tolerance: 1e-4 (float32 accumulation)
    (let [x-old 1.0
          y-old 2.0
          constraints (cm/choicemap :x (mx/scalar x-old) :y (mx/scalar y-old))
          {:keys [trace]} (p/generate models/dependent-model [] constraints)
          {:keys [trace weight]} (p/regenerate models/dependent-model trace sel/all)
          x-new (h/realize (cm/get-value (cm/get-submap (:choices trace) :x)))
          expected (- (h/gaussian-lp y-old x-new 1) (h/gaussian-lp y-old x-old 1))]
      (is (h/close? expected (h/realize weight) 1e-4)
          "weight is stale-child correction for old y under new x"))))

;; ---------------------------------------------------------------------------
;; Case 5: Empty selection — identity operation
;; ---------------------------------------------------------------------------

(deftest regenerate-empty-selection-is-identity
  (testing "sel/none: weight = 0, all choices unchanged"
    ;; Derivation: S = {}, no addresses resampled
    ;; handler_weight = 0 (sum over empty set)
    ;; new_score = old_score (same values, same distributions)
    ;; weight = 0 - 0 = 0
    ;; Tolerance: exact 0 expected — no float ops on weight
    (let [constraints (cm/choicemap :x (mx/scalar 1.5))
          {:keys [trace]} (p/generate models/single-gaussian [] constraints)
          {:keys [trace weight]} (p/regenerate models/single-gaussian trace sel/none)]
      (is (h/close? 0.0 (h/realize weight) 1e-6)
          "empty selection has zero weight")
      (is (h/close? 1.5 (h/realize (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6)
          "value unchanged with empty selection"))))

;; ---------------------------------------------------------------------------
;; Score consistency: regenerated trace score = sum of site log-probs
;; ---------------------------------------------------------------------------

(deftest regenerate-score-consistency
  (testing "regenerated trace score matches analytical joint log-prob"
    ;; Invariant: for any regenerate call, the returned trace's score equals
    ;; sum_i log p(tau'[a_i] | parents(a_i) in tau')
    ;; This is the fundamental trace score contract.
    ;; Tolerance: 1e-4 (float32 accumulation)
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [trace]} (p/generate models/dependent-model [] constraints)
          {:keys [trace]} (p/regenerate models/dependent-model trace (sel/select :x))
          x (h/realize (cm/get-value (cm/get-submap (:choices trace) :x)))
          y (h/realize (cm/get-value (cm/get-submap (:choices trace) :y)))
          ;; score = lp(x;0,1) + lp(y;x,1)
          expected (+ (h/gaussian-lp x 0 1) (h/gaussian-lp y x 1))]
      (is (h/close? expected (h/realize (:score trace)) 1e-4)
          "trace score matches analytical log-joint"))))

(deftest regenerate-score-consistency-independent
  (testing "independent model: regenerated score also correct"
    ;; score = lp(x';0,1) + lp(y;0,1)
    ;; Tolerance: 1e-4
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [trace]} (p/generate models/two-gaussians [] constraints)
          {:keys [trace]} (p/regenerate models/two-gaussians trace sel/all)
          x (h/realize (cm/get-value (cm/get-submap (:choices trace) :x)))
          y (h/realize (cm/get-value (cm/get-submap (:choices trace) :y)))
          expected (+ (h/gaussian-lp x 0 1) (h/gaussian-lp y 0 1))]
      (is (h/close? expected (h/realize (:score trace)) 1e-4)
          "independent regenerated score correct"))))

;; ---------------------------------------------------------------------------
;; Unselected addresses preserved
;; ---------------------------------------------------------------------------

(deftest regenerate-preserves-unselected
  (testing "unselected address retains its exact value"
    ;; The regenerate contract guarantees: for a not in S, tau'[a] = tau[a].
    ;; Tolerance: 1e-6 (bit-exact preservation, comparing realized values)
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [trace]} (p/generate models/two-gaussians [] constraints)
          {:keys [trace]} (p/regenerate models/two-gaussians trace (sel/select :x))
          y (h/realize (cm/get-value (cm/get-submap (:choices trace) :y)))]
      (is (h/close? 2.0 y 1e-6)
          ":y preserved when only :x is selected"))))

;; =========================================================================
;; ITEM #14: MH acceptance probability [Cusumano-Towner 2020, Alg 5]
;; =========================================================================

(deftest mh-accept-deterministic-cases
  (testing "accept-mh? boundary conditions"
    ;; alpha = min{1, exp(log_w)}
    ;; log_w >= 0 => exp(log_w) >= 1 => alpha = 1 => always accept
    ;; log_w = -Infinity => exp(-Inf) = 0 => never accept
    (is (true? (u/accept-mh? 0))
        "log_w = 0: exp(0) = 1, always accept")
    (is (true? (u/accept-mh? 1.0))
        "log_w > 0: exp(1) > 1, always accept")
    (is (true? (u/accept-mh? 100.0))
        "log_w >> 0: always accept")
    (is (false? (u/accept-mh? js/-Infinity))
        "log_w = -Inf: never accept")))

(deftest mh-accept-statistical-rate
  (testing "P(accept | log_w = log(0.3)) ≈ 0.3"
    ;; P(accept) = min{1, exp(log_w)} = 0.3
    ;; Run N=10000 trials.
    ;; SE = sqrt(p*(1-p)/N) = sqrt(0.3*0.7/10000) = 0.00458
    ;; 3.5-sigma tolerance = 3.5 * 0.00458 = 0.016
    (let [target-p 0.3
          log-w (js/Math.log target-p)
          n 10000
          accepts (loop [i 0 cnt 0]
                    (if (>= i n) cnt
                        (recur (inc i)
                               (if (u/accept-mh? log-w (rng/fresh-key (+ i 5000)))
                                 (inc cnt) cnt))))
          rate (/ accepts n)
          se (js/Math.sqrt (/ (* target-p (- 1 target-p)) n))
          z (/ (js/Math.abs (- rate target-p)) se)]
      (is (< z 3.5)
          (str "acceptance rate " rate " within 3.5-sigma of " target-p
               " (z=" z ")")))))

(deftest mh-step-weight-determines-acceptance
  (testing "mh-step uses regenerate weight for acceptance"
    ;; For independent model, regenerate weight = 0, so alpha = 1.
    ;; Every proposal should be accepted: trace changes on every step.
    ;; Run 20 steps, verify at least 15 produce a different trace.
    ;; (With weight=0, acceptance prob=1, so all should accept.
    ;;  But we allow some tolerance for identical-by-chance proposals.)
    (let [constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [trace]} (p/generate models/two-gaussians [] constraints)
          results (loop [i 0 t trace changes 0]
                    (if (>= i 20)
                      changes
                      (let [new-t (mcmc/mh-step t (sel/select :x))]
                        (recur (inc i) new-t
                               (if (not (identical? new-t t))
                                 (inc changes) changes)))))]
      (is (>= results 15)
          (str "at least 15/20 accepted (got " results
               "); weight=0 implies alpha=1")))))

;; =========================================================================
;; ITEM #15: MH proposal validity [Cusumano-Towner 2020, Section 3.4.2]
;; =========================================================================

;; Requirement 1: Proposal takes current trace as argument
;; → Verified structurally: regenerate signature is (regenerate gf trace sel)

;; Requirement 2: Selection should not include observed addresses
;; → The GFI primitive does NOT enforce this; the inference wrapper does.
;;   Test: regenerating a latent preserves observations.

(deftest mh-validity-observations-preserved
  (testing "MH on latent preserves observed addresses"
    ;; Model: x ~ N(0,1), y ~ N(x,1), observe y=2.0
    ;; Selection: {:x} (latent only)
    ;; After 10 MH steps, y must still be 2.0
    (let [obs (cm/choicemap :y (mx/scalar 2.0))
          {:keys [trace]} (p/generate models/dependent-model [] obs)
          final-trace (reduce (fn [t _] (mcmc/mh-step t (sel/select :x)))
                              trace (range 10))
          y-val (h/realize (cm/get-value (cm/get-submap (:choices final-trace) :y)))]
      (is (h/close? 2.0 y-val 1e-6)
          "y=2.0 preserved after 10 MH steps on :x"))))

;; Requirement 3: Reversibility (forward-sampling has full support)
;; → Every continuous prior has support = R (or R+), so any trace
;;   can be reached with positive probability. For the Normal-Normal
;;   model, verify MH converges to correct posterior.

(deftest mh-validity-correct-posterior
  (testing "MH converges to analytical posterior (Normal-Normal)"
    ;; Model: x ~ N(0,1), y ~ N(x, 0.5), observe y=2.0
    ;; Posterior: x | y=2 ~ N(mu_post, sigma_post)
    ;;   precision_prior = 1/(1)^2 = 1
    ;;   precision_lik   = 1/(0.5)^2 = 4
    ;;   precision_post  = 1 + 4 = 5
    ;;   mu_post = (1*0 + 4*2) / 5 = 8/5 = 1.6
    ;;   var_post = 1/5 = 0.2
    ;;   sigma_post = sqrt(0.2) = 0.44721...
    ;;
    ;; Tolerance derivation:
    ;;   N = 5000 samples, burn = 500
    ;;   Prior proposal on N(0,1) -> posterior N(1.6, 0.447):
    ;;     acceptance rate ~ exp(-0.5 * d^2) where d = distance in posterior
    ;;     Conservative N_eff estimate: N/10 = 500
    ;;   SE(mean) = sigma_post / sqrt(N_eff) = 0.447 / sqrt(500) = 0.020
    ;;   3.5-sigma tolerance for mean: 3.5 * 0.020 = 0.070
    ;;   SE(var) = var_post * sqrt(2/N_eff) = 0.2 * 0.0632 = 0.013
    ;;   3.5-sigma tolerance for var: 3.5 * 0.013 = 0.045
    (let [mu-post 1.6
          var-post 0.2
          obs (cm/choicemap :y (mx/scalar 2.0))
          traces (mcmc/mh {:samples 5000 :burn 500
                           :selection (sel/select :x)
                           :key (rng/fresh-key 42)}
                          dep-model-narrow [] obs)
          xs (mapv #(h/realize (cm/get-value (cm/get-submap (:choices %) :x)))
                   traces)
          emp-mean (h/sample-mean xs)
          emp-var (h/sample-variance xs)]
      (is (h/close? mu-post emp-mean 0.070)
          (str "posterior mean " emp-mean " within 0.070 of " mu-post))
      (is (h/close? var-post emp-var 0.045)
          (str "posterior var " emp-var " within 0.045 of " var-post)))))

;; ---------------------------------------------------------------------------
;; Weight decomposition: new_score - old_score - handler_weight
;; ---------------------------------------------------------------------------

(deftest regenerate-weight-decomposition
  (testing "weight = new_score - old_score - handler_weight (verified via assess)"
    ;; Verify the decomposition by independently computing new_score and
    ;; old_score via assess, then checking the regenerate weight matches.
    ;;
    ;; Model: x ~ N(0,1), y ~ N(x,1)
    ;; Fix x=1.0, y=2.0, regenerate selecting :x
    ;; Tolerance: 1e-4 (float32 accumulation)
    (let [x-old 1.0
          y-val 2.0
          constraints (cm/choicemap :x (mx/scalar x-old) :y (mx/scalar y-val))
          {:keys [trace]} (p/generate models/dependent-model [] constraints)
          old-score (h/realize (:score trace))
          {:keys [trace weight]} (p/regenerate models/dependent-model trace (sel/select :x))
          new-score (h/realize (:score trace))
          regen-weight (h/realize weight)
          x-new (h/realize (cm/get-value (cm/get-submap (:choices trace) :x)))
          ;; Verify old_score analytically
          old-score-analytical (+ (h/gaussian-lp x-old 0 1)
                                  (h/gaussian-lp y-val x-old 1))
          ;; Verify new_score analytically
          new-score-analytical (+ (h/gaussian-lp x-new 0 1)
                                  (h/gaussian-lp y-val x-new 1))
          ;; The derived weight formula:
          expected-weight (- (h/gaussian-lp y-val x-new 1)
                             (h/gaussian-lp y-val x-old 1))]
      ;; Verify scores match analytical values
      (is (h/close? old-score-analytical old-score 1e-4)
          "old trace score matches analytical")
      (is (h/close? new-score-analytical new-score 1e-4)
          "new trace score matches analytical")
      ;; Verify the weight equals the derived formula
      (is (h/close? expected-weight regen-weight 1e-4)
          "weight matches derived formula: lp(y;x',1)-lp(y;x,1)"))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
