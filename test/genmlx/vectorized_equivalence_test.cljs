(ns genmlx.vectorized-equivalence-test
  "Vectorized equivalence: vsimulate produces N independent samples
   with correct distributional properties. ESS computation.
   Statistical equivalence tests verify vectorized paths match
   analytical moments using z-tests at 3.5-sigma threshold."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vec]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test model: x ~ N(0,1), known analytical score = log N(x; 0, 1)
;; ---------------------------------------------------------------------------

(def gaussian-model
  (dyn/auto-key
   (gen []
        (trace :x (dist/gaussian 0 1)))))

(def two-site-model
  (dyn/auto-key
   (gen []
        (let [x (trace :x (dist/gaussian 0 1))
              y (trace :y (dist/gaussian x 1))]
          y))))

(def N 200)

;; ---------------------------------------------------------------------------
;; vsimulate produces N distinct samples
;; ---------------------------------------------------------------------------

(deftest vsimulate-produces-n-scores
  (let [scores (h/realize-vec (:score (dyn/vsimulate gaussian-model [] N
                                                     (h/deterministic-key))))]
    (is (= N (count scores)))))

(deftest vsimulate-scores-have-positive-variance
  (testing "N particles produce diverse scores (not all identical)"
    (let [scores (h/realize-vec (:score (dyn/vsimulate gaussian-model [] N
                                                       (h/deterministic-key))))
          variance (h/sample-variance scores)]
      (is (pos? variance)
          "vsimulate scores are not all identical"))))

(deftest vsimulate-choices-have-positive-variance
  (testing "N particles produce diverse choice values"
    (let [vtr (dyn/vsimulate gaussian-model [] N (h/deterministic-key))
          x-vals (h/realize-vec (cm/get-value (cm/get-submap (:choices vtr) :x)))
          variance (h/sample-variance x-vals)]
      (is (pos? variance)
          "choice values are diverse"))))

;; ---------------------------------------------------------------------------
;; Score consistency: each particle's score matches analytical log-prob
;; ---------------------------------------------------------------------------

(deftest vsimulate-scores-match-analytical-log-prob
  (testing "each particle score = log N(x_i; 0, 1)"
    (let [vtr (dyn/vsimulate gaussian-model [] 16 (h/deterministic-key))
          x-vals (h/realize-vec (cm/get-value (cm/get-submap (:choices vtr) :x)))
          scores (h/realize-vec (:score vtr))]
      (doseq [[x-val score] (map vector x-vals scores)]
        (is (h/close? (h/gaussian-lp x-val 0 1) score 1e-4)
            (str "score matches log N(" x-val "; 0, 1)"))))))

(deftest vsimulate-two-site-scores-match-analytical
  (testing "each particle score = log N(x; 0,1) + log N(y; x,1)"
    (let [vtr (dyn/vsimulate two-site-model [] 16 (h/deterministic-key))
          {:keys [choices score]} vtr
          x-vals (h/realize-vec (cm/get-value (cm/get-submap choices :x)))
          y-vals (h/realize-vec (cm/get-value (cm/get-submap choices :y)))
          scores (h/realize-vec score)]
      (doseq [[x-val y-val sc] (map vector x-vals y-vals scores)]
        (let [expected (+ (h/gaussian-lp x-val 0 1)
                          (h/gaussian-lp y-val x-val 1))]
          (is (h/close? expected sc 1e-4)
              "two-site score = joint log-prob"))))))

;; ---------------------------------------------------------------------------
;; ESS computation
;; ---------------------------------------------------------------------------

(defn log-ess
  "Effective sample size from log-weights (seq of JS numbers).
   ESS = 1 / Σ(w_i²) where w_i = exp(lw_i - max) / Σ exp(lw_j - max)."
  [log-weights]
  (let [lw (vec log-weights)
        max-lw (apply max lw)
        shifted (mapv #(js/Math.exp (- % max-lw)) lw)
        total (reduce + shifted)
        normalized (mapv #(/ % total) shifted)
        sum-sq (reduce + (mapv #(* % %) normalized))]
    (/ 1.0 sum-sq)))

(deftest ess-uniform-weights-equal-n
  (let [ess (log-ess (repeat 50 0.0))]
    (is (h/close? 50.0 ess 1e-6)
        "uniform weights → ESS = N")))

(deftest ess-single-dominant-weight-near-one
  (let [ess (log-ess (into [100.0] (repeat 49 -100.0)))]
    (is (< ess 1.01)
        "single dominant weight → ESS ≈ 1")))

(deftest ess-bounded-by-n
  (testing "ESS from vgenerate importance weights ≤ N"
    (let [constraints (cm/choicemap :x (mx/scalar 0.0))
          vtr (dyn/vgenerate two-site-model [] constraints N (h/deterministic-key))
          ;; Use scores as proxy log-weights (since weight is scalar)
          scores (h/realize-vec (:score vtr))
          ess (log-ess scores)]
      (is (<= ess (+ N 0.01))
          "ESS ≤ N"))))

;; ---------------------------------------------------------------------------
;; Deterministic reproducibility
;; ---------------------------------------------------------------------------

(deftest vsimulate-same-key-reproduces-scores
  (testing "same PRNG key → identical scores"
    (let [scores1 (h/realize-vec (:score (dyn/vsimulate gaussian-model [] 8
                                                        (rng/fresh-key 42))))
          scores2 (h/realize-vec (:score (dyn/vsimulate gaussian-model [] 8
                                                        (rng/fresh-key 42))))]
      (is (h/all-close? scores1 scores2 1e-6)))))

;; ---------------------------------------------------------------------------
;; Statistical equivalence: vectorized vs analytical moments (z-test)
;; ---------------------------------------------------------------------------
;;
;; Tolerance derivation for all tests below:
;; N=500 samples. z=3.5 gives P(false positive) < 0.0005.
;;
;; Mean z-test: |sample-mean - expected| < 3.5 * SE
;;   where SE = sigma/sqrt(N). Handled by h/z-test-passes?.
;;
;; Variance check: |S^2 - sigma^2| < tol
;;   For normal data: SE(S^2) = sigma^2 * sqrt(2/(N-1))
;;   tol = 3.5 * SE(S^2). For sigma=1, N=500: tol ≈ 0.222.
;;   For non-normal data we derive SE analytically per distribution.
;; ---------------------------------------------------------------------------

(def ^:private stat-N
  "Sample size for statistical equivalence tests.
   N=500 gives 3.5-sigma mean tolerance of ~0.157 for N(0,1)."
  500)

(defn- normal-variance-tol
  "3.5-sigma tolerance for sample variance of a normal distribution.
   Var(S^2) = 2*sigma^4/(n-1), so tol = z * sigma^2 * sqrt(2/(n-1))."
  [true-var n]
  (* 3.5 true-var (js/Math.sqrt (/ 2.0 (dec n)))))

(defn- extract-choice-vals
  "Realize vectorized choice values at addr from a VectorizedTrace."
  [vtr addr]
  (h/realize-vec (cm/get-value (cm/get-submap (:choices vtr) addr))))

(defn- collect-scalar-choices
  "Run N scalar simulates, return realized choice values at addr."
  [model args addr n]
  (mapv (fn [_]
          (h/realize (cm/get-value (cm/get-submap (:choices (p/simulate model args)) addr))))
        (range n)))

(defn- collect-scalar-scores
  "Run N scalar simulates, return realized scores."
  [model args n]
  (mapv (fn [_] (h/realize (:score (p/simulate model args))))
        (range n)))

;; ---------------------------------------------------------------------------
;; Test 1: vsimulate choice moments match analytical N(0,1)
;; ---------------------------------------------------------------------------

(deftest vsimulate-choice-moments-match-analytical
  (testing "vsimulate N(0,1) choices: mean≈0, var≈1 by z-test"
    ;; Analytical: X ~ N(0,1) => E[X]=0, Var[X]=1
    ;; Mean tolerance: 3.5 * 1/sqrt(500) ≈ 0.157
    ;; Variance tolerance: 3.5 * sqrt(2/499) ≈ 0.222
    (let [v-vals (extract-choice-vals
                  (dyn/vsimulate gaussian-model [] stat-N (h/deterministic-key 1))
                  :x)
          s-vals (collect-scalar-choices gaussian-model [] :x stat-N)
          var-tol (normal-variance-tol 1.0 stat-N)]
      (is (h/z-test-passes? 0.0 v-vals)
          "vectorized mean consistent with E[X]=0")
      (is (h/z-test-passes? 0.0 s-vals)
          "scalar mean consistent with E[X]=0")
      (is (h/close? 1.0 (h/sample-variance v-vals) var-tol)
          "vectorized variance consistent with Var[X]=1")
      (is (h/close? 1.0 (h/sample-variance s-vals) var-tol)
          "scalar variance consistent with Var[X]=1"))))

;; ---------------------------------------------------------------------------
;; Test 2: vsimulate score moments match analytical
;; ---------------------------------------------------------------------------

(deftest vsimulate-score-moments-match-analytical
  (testing "vsimulate N(0,1) scores: E[score]=-0.5*log(2pi)-0.5, Var[score]=0.5"
    ;; For X ~ N(0,1): score = log N(x;0,1) = -0.5*log(2pi) - 0.5*x^2
    ;; E[score] = -0.5*log(2pi) - 0.5*E[x^2] = -0.5*log(2pi) - 0.5
    ;; Var[score] = 0.25*Var[x^2] = 0.25*2 = 0.5
    ;;
    ;; Score is a nonlinear transform of chi-squared, so variance SE
    ;; uses the fourth central moment: E[(S-E[S])^4] = 3.75
    ;; SE(S^2) = sqrt((3.75 - 0.25)/N) ≈ 0.084 for N=500
    ;; tol = 3.5 * SE ≈ 0.293
    (let [expected-mean (- (* -0.5 h/LOG-2PI) 0.5)
          expected-var 0.5
          score-var-tol (* 3.5 (js/Math.sqrt (/ (- 3.75 (* expected-var expected-var))
                                                stat-N)))
          v-scores (h/realize-vec (:score (dyn/vsimulate gaussian-model [] stat-N
                                                         (h/deterministic-key 2))))
          s-scores (collect-scalar-scores gaussian-model [] stat-N)]
      (is (h/z-test-passes? expected-mean v-scores)
          "vectorized score mean matches analytical")
      (is (h/z-test-passes? expected-mean s-scores)
          "scalar score mean matches analytical")
      (is (h/close? expected-var (h/sample-variance v-scores) score-var-tol)
          "vectorized score variance matches analytical")
      (is (h/close? expected-var (h/sample-variance s-scores) score-var-tol)
          "scalar score variance matches analytical"))))

;; ---------------------------------------------------------------------------
;; Test 3: vgenerate weight moments for partial constraints
;; ---------------------------------------------------------------------------

(deftest vgenerate-weight-moments-match-analytical
  (testing "vgenerate weight for x~N(0,1),y~N(x,1) constrained at y=0.5"
    ;; Weight_i = log N(0.5; x_i, 1) where x_i ~ N(0,1)
    ;; = -0.5*log(2pi) - 0.5*(0.5-x_i)^2
    ;; E[weight] = -0.5*log(2pi) - 0.5*E[(0.5-x)^2]
    ;;           = -0.5*log(2pi) - 0.5*(0.25 + 1)  [E[(a-X)^2] = a^2 + Var[X]]
    ;;           = -0.5*log(2pi) - 0.625
    ;; Var[weight] = 0.25*Var[(0.5-x)^2]
    ;; (0.5-x)^2 = x^2 - x + 0.25, Var = Var[x^2-x] = Var[x^2]+Var[x]-2Cov[x^2,x]
    ;; For N(0,1): Var[x^2]=2, Var[x]=1, Cov[x^2,x]=E[x^3]=0 => Var=3
    ;; Var[weight] = 0.25*3 = 0.75
    ;;
    ;; Note: scalar p/generate on this conjugate model computes the exact
    ;; analytical marginal (constant weight), so scalar comparison is omitted.
    ;; The vectorized path does importance sampling with N proposal particles.
    (let [expected-mean (- (* -0.5 h/LOG-2PI) 0.625)
          expected-var 0.75
          constraints (cm/choicemap :y (mx/scalar 0.5))
          v-weights (h/realize-vec
                     (:weight (dyn/vgenerate two-site-model [] constraints
                                             stat-N (h/deterministic-key 3))))
          var-tol (normal-variance-tol expected-var stat-N)]
      (is (h/z-test-passes? expected-mean v-weights)
          "vectorized weight mean matches E[log N(0.5; x, 1)]")
      (is (h/close? expected-var (h/sample-variance v-weights) var-tol)
          "vectorized weight variance matches analytical"))))

;; ---------------------------------------------------------------------------
;; Test 4: vupdate weight moments
;; ---------------------------------------------------------------------------

(deftest vupdate-weight-moments-match-analytical
  (testing "vupdate weight for constraining y=1.0 on two-site model"
    ;; Start: vsimulate two-site (x~N(0,1), y~N(x,1)), no constraints.
    ;; Update: constrain y=1.0.
    ;; weight_i = log N(1;x_i,1) - log N(y_i;x_i,1)
    ;;         = -0.5*(1-x_i)^2 + 0.5*(y_i-x_i)^2
    ;; E[weight] = -0.5*E[(1-x)^2] + 0.5*E[(y-x)^2]
    ;;           = -0.5*(1+1) + 0.5*1 = -0.5
    ;; (since E[(1-x)^2] = 1^2 + Var[x] = 2, E[(y-x)^2] = Var[y-x] = Var[eps] = 1)
    (let [expected-mean -0.5
          vt (dyn/vsimulate two-site-model [] stat-N (h/deterministic-key 4))
          {:keys [weight]} (dyn/vupdate two-site-model vt
                                        (cm/choicemap :y (mx/scalar 1.0))
                                        (h/deterministic-key 40))
          v-weights (h/realize-vec weight)
          s-weights (mapv (fn [_]
                            (let [tr (p/simulate two-site-model [])]
                              (h/realize (:weight (p/update two-site-model tr
                                                            (cm/choicemap :y (mx/scalar 1.0)))))))
                          (range stat-N))]
      (is (h/z-test-passes? expected-mean v-weights)
          "vectorized update weight mean ≈ -0.5")
      (is (h/z-test-passes? expected-mean s-weights)
          "scalar update weight mean ≈ -0.5"))))

;; ---------------------------------------------------------------------------
;; Test 5: vregenerate weight moments
;; ---------------------------------------------------------------------------

(deftest vregenerate-weight-moments-match-analytical
  (testing "vregenerate weight for resampling :x in two-site model"
    ;; Start: vsimulate two-site, then regenerate :x.
    ;; weight_i = log N(y_i; x'_i, 1) - log N(y_i; x_i, 1)
    ;;         = -0.5*(y_i - x'_i)^2 + 0.5*(y_i - x_i)^2
    ;; where x ~ N(0,1), y ~ N(x,1), x' ~ N(0,1) independent of x,y.
    ;; E[weight] = -0.5*E[(y-x')^2] + 0.5*E[(y-x)^2]
    ;; E[(y-x')^2] = Var[y] + Var[x'] = 2 + 1 = 3 (independent)
    ;; E[(y-x)^2] = Var[y-x] = Var[eps] = 1
    ;; E[weight] = -0.5*3 + 0.5*1 = -1.0
    (let [expected-mean -1.0
          vt (dyn/vsimulate two-site-model [] stat-N (h/deterministic-key 5))
          {:keys [weight]} (dyn/vregenerate two-site-model vt
                                            (sel/select :x)
                                            (h/deterministic-key 50))
          v-weights (h/realize-vec weight)
          s-weights (mapv (fn [_]
                            (let [tr (p/simulate two-site-model [])
                                  {:keys [weight]} (p/regenerate two-site-model tr (sel/select :x))]
                              (h/realize weight)))
                          (range stat-N))]
      (is (h/z-test-passes? expected-mean v-weights)
          "vectorized regenerate weight mean ≈ -1.0")
      (is (h/z-test-passes? expected-mean s-weights)
          "scalar regenerate weight mean ≈ -1.0"))))

;; ---------------------------------------------------------------------------
;; Test 6: multi-distribution model joint moments
;; ---------------------------------------------------------------------------

(def ^:private multi-dist-model
  "Three independent sites: gaussian + exponential + uniform."
  (dyn/auto-key
   (gen []
        (let [x (trace :x (dist/gaussian 2 0.5))
              y (trace :y (dist/exponential 1))
              z (trace :z (dist/uniform 0 1))]
          [x y z]))))

(deftest multi-distribution-joint-moments-match-analytical
  (testing "joint moments of gaussian+exponential+uniform via vsimulate"
    ;; Analytical moments:
    ;; x ~ N(2, 0.5): E[x]=2, Var[x]=0.25
    ;; y ~ Exp(1):    E[y]=1, Var[y]=1
    ;; z ~ U(0,1):    E[z]=0.5, Var[z]=1/12 ≈ 0.0833
    ;;
    ;; Mean tolerances (3.5 * sigma/sqrt(500)):
    ;;   x: 3.5 * 0.5/sqrt(500)  ≈ 0.0783
    ;;   y: 3.5 * 1/sqrt(500)    ≈ 0.1565
    ;;   z: 3.5 * 0.2887/sqrt(500) ≈ 0.0452
    ;;
    ;; Variance tolerances (via SE of S^2):
    ;;   x (normal): 3.5 * 0.25 * sqrt(2/499) ≈ 0.0554
    ;;   y (Exp(1)): SE(S^2) = sqrt((mu4 - sigma^4)/n) where mu4=9, sigma^4=1
    ;;              = sqrt(8/500) ≈ 0.1265, tol = 3.5*0.1265 ≈ 0.443
    ;;   z (Uniform): SE(S^2) = sqrt((1/80 - 1/144)/500) ≈ 0.00333, tol ≈ 0.0117
    (let [vt (dyn/vsimulate multi-dist-model [] stat-N (h/deterministic-key 6))
          x-vals (extract-choice-vals vt :x)
          y-vals (extract-choice-vals vt :y)
          z-vals (extract-choice-vals vt :z)]
      ;; Mean z-tests (3.5-sigma, automatic SE from samples)
      (is (h/z-test-passes? 2.0 x-vals) "gaussian mean ≈ 2")
      (is (h/z-test-passes? 1.0 y-vals) "exponential mean ≈ 1")
      (is (h/z-test-passes? 0.5 z-vals) "uniform mean ≈ 0.5")
      ;; Variance checks with analytically-derived tolerances
      (is (h/close? 0.25 (h/sample-variance x-vals)
                    (normal-variance-tol 0.25 stat-N))
          "gaussian variance ≈ 0.25")
      (is (h/close? 1.0 (h/sample-variance y-vals) 0.45)
          "exponential variance ≈ 1")
      (is (h/close? (/ 1.0 12) (h/sample-variance z-vals) 0.012)
          "uniform variance ≈ 1/12"))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
