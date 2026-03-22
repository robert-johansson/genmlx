(ns genmlx.inference-adev-test
  "Phase 4.7: ADEV gradient estimation tests.
   ADEV computes unbiased gradients ∇_θ E_{p(·;θ)}[cost(trace)] by
   automatically choosing reparameterization or REINFORCE at each trace site.

   Test 1: For cost(x) = x with x ~ N(mu, 1):
     ∂/∂mu E[x] = 1 (exact, zero variance via reparameterization).

   Test 2: For cost(x) = x^2 with x ~ N(mu, 1):
     E[x^2] = mu^2 + 1
     ∂/∂mu E[x^2] = 2*mu
     At mu=0: gradient = 0. At mu=2: gradient = 4.
     Single-sample gradient = 2*x = 2*(mu + eps), variance = 4*sigma^2 = 4.
     Multi-sample averaging reduces this variance.

   Test 3: Gradient estimates are finite."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.adev :as adev]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test model: parameterized Gaussian
;; ---------------------------------------------------------------------------

(def gaussian-model
  (dyn/auto-key
    (gen []
      (let [mu (param :mu 0.0)]
        (trace :x (genmlx.dist/gaussian mu 1))))))

;; ==========================================================================
;; 1. ADEV gradient for E[x] w.r.t. mu is exactly 1
;; ==========================================================================
;; x = mu + eps via reparameterization, ∂x/∂mu = 1 exactly.
;; cost = x, so ∂cost/∂mu = 1 with zero variance.

(deftest adev-gradient-identity
  (testing "gradient of E[x] w.r.t. mu = 1 (exact via reparam)"
    (let [{:keys [grad]} (adev/adev-gradient
                           {:n-samples 1}
                           gaussian-model []
                           (fn [trace] (:retval trace))
                           [:mu]
                           (mx/array [0.0]))]
      (mx/eval! grad)
      ;; Exact: no averaging needed
      (is (h/close? 1.0 (first (mx/->clj grad)) 1e-4)
          "gradient is exactly 1"))))

;; ==========================================================================
;; 2. ADEV gradient for E[x^2] w.r.t. mu
;; ==========================================================================
;; E[x^2] = mu^2 + sigma^2. At mu=2, sigma=1: E[x^2] = 5.
;; ∂/∂mu E[x^2] = 2*mu = 4.
;; Single-sample: ∂x^2/∂mu = 2*x = 2*(mu + eps), var = 4*sigma^2 = 4.
;; With n=20 samples: std_error = 2/sqrt(20) = 0.447
;; 5 runs averaged: std_error = 2/sqrt(100) = 0.2
;; 3.5-sigma = 0.7

(deftest adev-gradient-quadratic
  (testing "gradient of E[x^2] w.r.t. mu at mu=2 ≈ 4"
    (let [cost-fn (fn [trace] (mx/square (:retval trace)))
          grads (mapv (fn [_]
                        (let [{:keys [grad]} (adev/adev-gradient
                                               {:n-samples 20}
                                               gaussian-model []
                                               cost-fn
                                               [:mu]
                                               (mx/array [2.0]))]
                          (mx/eval! grad) (first (mx/->clj grad))))
                      (range 5))
          mean-grad (h/sample-mean grads)]
      (is (h/close? 4.0 mean-grad 1.0)
          (str "gradient " mean-grad " ≈ 4.0")))))

;; ==========================================================================
;; 3. ADEV gradient is finite
;; ==========================================================================

(deftest adev-gradient-finite
  (testing "single-sample ADEV gradient is finite"
    (let [{:keys [loss grad]} (adev/adev-gradient
                                {:n-samples 1}
                                gaussian-model []
                                (fn [trace] (:retval trace))
                                [:mu]
                                (mx/array [0.0]))]
      (mx/eval! grad)
      (mx/eval! loss)
      (is (js/isFinite (mx/item loss)) "loss is finite")
      (is (js/isFinite (first (mx/->clj grad))) "gradient is finite"))))

;; ==========================================================================
;; 4. Multi-sample reduces variance for quadratic cost
;; ==========================================================================
;; cost = x^2, single-sample gradient variance = 4.
;; n=10: variance ~ 4/10 = 0.4.

(deftest adev-variance-reduction
  (testing "n-samples=10 has lower gradient variance than n-samples=1 for x^2 cost"
    (let [cost-fn (fn [trace] (mx/square (:retval trace)))
          run-grads (fn [n-samp n-runs]
                      (mapv (fn [_]
                              (let [{:keys [grad]} (adev/adev-gradient
                                                     {:n-samples n-samp}
                                                     gaussian-model []
                                                     cost-fn
                                                     [:mu]
                                                     (mx/array [2.0]))]
                                (mx/eval! grad) (first (mx/->clj grad))))
                            (range n-runs)))
          grads-1 (run-grads 1 20)
          grads-10 (run-grads 10 20)
          var-1 (h/sample-variance grads-1)
          var-10 (h/sample-variance grads-10)]
      ;; var-1 should be ~4, var-10 should be ~0.4
      (is (> var-1 0) "single-sample has nonzero variance")
      (is (< var-10 var-1)
          (str "var(n=10)=" var-10 " < var(n=1)=" var-1)))))

;; ==========================================================================
;; Run tests
;; ==========================================================================

(cljs.test/run-tests)
