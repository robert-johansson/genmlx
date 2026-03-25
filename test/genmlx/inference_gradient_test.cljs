(ns genmlx.inference-gradient-test
  "Phase 4.3: Inference gradient magnitude tests.
   Verifies gradient correctness through three inference mechanisms:

   A. ADEV gradient estimation
      - Exact gradient via reparameterization (zero MC variance)
      - Statistical gradient test via z-test
      - Gradient sign consistency

   B. VI ELBO optimization
      - Gradient direction pushes toward posterior
      - Convergence to analytical posterior (Normal-Normal conjugate)

   C. Differentiable resampling (Gumbel-softmax)
      - Gradient flows through soft resampling (finite, non-uniform)
      - Particle gradients match FD
      - Weight gradients match FD, sum to zero"
  (:require [cljs.test :refer [deftest is testing run-tests]]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.inference.adev :as adev]
            [genmlx.inference.vi :as vi]
            [genmlx.inference.differentiable-resample :as dr]
            [genmlx.test-helpers :as h]
            [genmlx.gradient-fd-test :as fd])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ==========================================================================
;; Shared model: parameterized Gaussian
;; ==========================================================================
;; x ~ N(mu, 1) where mu is a learnable parameter.
;; Used across ADEV tests A1-A3.

(def ^:private gaussian-param-model
  (dyn/auto-key
    (gen []
      (let [mu (param :mu 0.0)]
        (trace :x (dist/gaussian mu 1))))))

;; ==========================================================================
;; A. ADEV gradient estimation
;; ==========================================================================

;; --------------------------------------------------------------------------
;; A1. Exact gradient (zero-variance case)
;; --------------------------------------------------------------------------
;; cost(trace) = retval = x = mu + eps  (reparameterization)
;; d/d(mu) E[x] = d/d(mu) mu = 1.0 exactly.
;; A single sample suffices because reparameterization gives zero MC variance.

(deftest adev-exact-gradient-for-identity-cost
  (testing "d/d(mu) E[x] = 1.0 via reparameterization"
    (let [{:keys [grad]} (adev/adev-gradient
                           {:n-samples 1}
                           gaussian-param-model []
                           (fn [tr] (:retval tr))
                           [:mu]
                           (mx/array [0.0]))]
      (mx/eval! grad)
      (is (h/close? 1.0 (first (mx/->clj grad)) 1e-4)
          "gradient is exactly 1 with zero variance"))))

;; --------------------------------------------------------------------------
;; A2. Statistical gradient via z-test
;; --------------------------------------------------------------------------
;; cost(trace) = x^2, E[x^2] = mu^2 + sigma^2 = mu^2 + 1
;; d/d(mu) E[x^2] = 2*mu
;; At mu=2.0: expected gradient = 4.0
;; Single-sample gradient variance = Var[2x] = 4*sigma^2 = 4
;; With n=50 per estimate: per-estimate variance = 4/50 = 0.08
;; 10 estimates, z-test against analytical at z=3.5

(deftest adev-quadratic-cost-gradient-matches-analytical
  (testing "d/d(mu) E[x^2] at mu=2 is 4.0 by z-test"
    (let [cost-fn (fn [tr] (mx/square (:retval tr)))
          grads (mapv (fn [_]
                        (let [{:keys [grad]} (adev/adev-gradient
                                               {:n-samples 50}
                                               gaussian-param-model []
                                               cost-fn
                                               [:mu]
                                               (mx/array [2.0]))]
                          (mx/eval! grad)
                          (first (mx/->clj grad))))
                      (range 10))]
      (is (h/z-test-passes? 4.0 grads 3.5)
          (str "mean gradient " (h/sample-mean grads) " should be ~4.0")))))

;; --------------------------------------------------------------------------
;; A3. Gradient sign
;; --------------------------------------------------------------------------
;; cost = (x - target)^2, target = 5
;; At mu=0 (below target): d/d(mu) = 2*(mu - target) = -10 => gradient < 0
;; At mu=10 (above target): d/d(mu) = 2*(mu - target) = 10 => gradient > 0
;; Averaging over 50 samples makes sign reliable.

(deftest adev-gradient-sign-points-toward-target
  (let [target (mx/scalar 5.0)
        cost-fn (fn [tr] (mx/square (mx/subtract (:retval tr) target)))
        grad-at (fn [mu-val]
                  (let [{:keys [grad]} (adev/adev-gradient
                                         {:n-samples 50}
                                         gaussian-param-model []
                                         cost-fn
                                         [:mu]
                                         (mx/array [mu-val]))]
                    (mx/eval! grad)
                    (first (mx/->clj grad))))]
    (testing "below target: gradient is negative (increase mu)"
      (is (neg? (grad-at 0.0))))
    (testing "above target: gradient is positive (decrease mu)"
      (is (pos? (grad-at 10.0))))))

;; ==========================================================================
;; B. VI ELBO gradient
;; ==========================================================================
;; Conjugate Normal-Normal model:
;;   mu ~ N(0, 1),  x ~ N(mu, 1),  observe x = 1.5
;; Posterior: mu | x ~ N(0.75, 1/sqrt(2))
;; log-density(mu) = log N(mu; 0, 1) + log N(1.5; mu, 1)
;;   = -0.5*mu^2 - 0.5*(1.5 - mu)^2  + const

(defn- normal-normal-log-density
  "Joint log-density (up to constant) for the Normal-Normal model."
  [mu-arr]
  (mx/add (mx/multiply (mx/scalar -0.5) (mx/square mu-arr))
          (mx/multiply (mx/scalar -0.5)
                       (mx/square (mx/subtract (mx/scalar 1.5) mu-arr)))))

;; --------------------------------------------------------------------------
;; B1. VI gradient direction
;; --------------------------------------------------------------------------
;; Starting at m=0, the neg-ELBO gradient should push m toward 0.75.
;; After a few VI steps, m should have increased.

(deftest vi-gradient-pushes-toward-posterior
  (testing "VI moves mean toward posterior 0.75 from initial 0"
    (let [{:keys [mu]} (vi/vi {:iterations 10
                               :learning-rate 0.05
                               :elbo-samples 50}
                              normal-normal-log-density
                              (mx/array [0.0]))]
      (mx/eval! mu)
      (is (> (mx/item mu) 0.0)
          "mean increased from 0 toward posterior 0.75"))))

;; --------------------------------------------------------------------------
;; B2. VI convergence to analytical posterior
;; --------------------------------------------------------------------------
;; After 500 iterations: m should converge to ~0.75, s to ~0.7071
;; Tolerance: 0.15 (ELBO has stochastic gradients)

(deftest vi-converges-to-analytical-posterior
  (testing "VI converges to m=0.75, s=0.7071"
    (let [{:keys [mu sigma]} (vi/vi {:iterations 500
                                     :learning-rate 0.03
                                     :elbo-samples 10}
                                    normal-normal-log-density
                                    (mx/array [0.0]))]
      (mx/eval! mu sigma)
      (is (h/close? 0.75 (mx/item mu) 0.15)
          (str "posterior mean " (mx/item mu) " should be ~0.75"))
      (is (h/close? 0.7071 (mx/item sigma) 0.15)
          (str "posterior std " (mx/item sigma) " should be ~0.7071")))))

;; ==========================================================================
;; C. Differentiable resampling (Gumbel-softmax)
;; ==========================================================================
;; particles = [1, 2, 3, 4], log_weights = [-1, 0, -0.5, -2]
;; Zero Gumbel noise makes softmax deterministic for FD comparison.

(def ^:private test-particles (mx/reshape (mx/array [1 2 3 4]) [4 1]))
(def ^:private test-log-weights (mx/array [-1 0 -0.5 -2]))
(def ^:private zero-gumbel (mx/zeros [4 4]))
(def ^:private test-tau (mx/scalar 0.5))

;; --------------------------------------------------------------------------
;; C1. Gradient flows through soft resampling
;; --------------------------------------------------------------------------

(deftest gumbel-softmax-gradients-are-finite-and-nonuniform
  (testing "gradients w.r.t. particles are finite and non-uniform"
    (let [f (fn [p]
              (-> (dr/gumbel-softmax p test-log-weights zero-gumbel test-tau)
                  :particles mx/sum))
          g ((mx/grad f) test-particles)]
      (mx/eval! g)
      (let [gv (mx/->clj (mx/reshape g [4]))]
        (is (every? js/isFinite gv) "all gradients finite")
        (is (not (apply = gv)) "gradients are non-uniform")))))

;; --------------------------------------------------------------------------
;; C2. Particle gradients match FD
;; --------------------------------------------------------------------------
;; d(sum output)/d(particle_i) should equal softmax(log_weights / tau) summed
;; over output rows. With zero noise: deterministic.

(deftest gumbel-softmax-particle-gradient-matches-fd
  (testing "analytical particle gradient matches central-difference FD"
    (let [f (fn [p]
              (-> (dr/gumbel-softmax (mx/reshape p [4 1])
                                     test-log-weights zero-gumbel test-tau)
                  :particles mx/sum))
          g ((mx/grad f) (mx/array [1 2 3 4]))
          _ (mx/eval! g)
          analytical (mx/->clj g)
          fd-vals (fd/fd-vector-gradient f [1.0 2.0 3.0 4.0])]
      (is (every? true? (map fd/gradient-close? analytical fd-vals))
          (str "analytical " analytical " vs FD " fd-vals)))))

;; --------------------------------------------------------------------------
;; C3. Weight gradients match FD
;; --------------------------------------------------------------------------
;; d(sum output)/d(log_weight_i) with fixed particles.
;; Sum of weight gradients should be ~0 (softmax row sums are constant).

(deftest gumbel-softmax-weight-gradient-matches-fd
  (testing "analytical weight gradient matches FD"
    (let [f (fn [lw]
              (-> (dr/gumbel-softmax test-particles lw zero-gumbel test-tau)
                  :particles mx/sum))
          g ((mx/grad f) (mx/array [-1 0 -0.5 -2]))
          _ (mx/eval! g)
          analytical (mx/->clj g)
          fd-vals (fd/fd-vector-gradient f [-1.0 0.0 -0.5 -2.0])]
      (is (every? true? (map fd/gradient-close? analytical fd-vals))
          (str "analytical " analytical " vs FD " fd-vals))))

  (testing "weight gradient sum is approximately zero"
    (let [f (fn [lw]
              (-> (dr/gumbel-softmax test-particles lw zero-gumbel test-tau)
                  :particles mx/sum))
          g ((mx/grad f) (mx/array [-1 0 -0.5 -2]))]
      (mx/eval! g)
      (is (h/close? 0.0 (reduce + (mx/->clj g)) 1e-4)
          "weight gradients sum to zero"))))

;; ==========================================================================
;; Run
;; ==========================================================================

(run-tests)
