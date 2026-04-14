(ns genmlx.sequential-convergence-test
  "Sequential inference convergence tests on a linear-Gaussian SSM.

   SSM:      x_t ~ N(x_{t-1}, 1),  y_t ~ N(x_t, 1),  x_0 ~ N(0, 1)
   Ground truth: Kalman filter analytical log marginal likelihood.

   Three kernel-level sequential algorithms tested:
   1. batched-smc-unfold  — batched bootstrap PF via vgenerate (one per step)
   2. compiled-smc        — Level 2 compiled bootstrap PF via noise transforms
   3. smc-unfold          — incremental PF via unfold-extend

   Two single-step algorithms tested on the Normal-Normal conjugate model:
   4. smcp3               — bootstrap mode on conjugate (sequential unfold
                             requires custom proposals, not tested here)
   5. vsmc-init           — batched importance sampling"
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.combinators :as comb]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.smcp3 :as smcp3]
            [genmlx.inference.compiled-smc :as compiled-smc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ===========================================================================
;; Kalman filter — analytical log marginal likelihood
;; ===========================================================================
;;
;; Linear-Gaussian SSM:
;;   x_0 ~ N(prior-mean, q)          (prior = transition from prior-mean)
;;   x_t ~ N(x_{t-1}, q)  for t > 0  (transition)
;;   y_t ~ N(x_t, r)       for all t  (observation)
;;
;; Kalman recursion gives exact log p(y_0, ..., y_{T-1}).
;; ===========================================================================

(def ^:private LOG-2PI (js/Math.log (* 2 js/Math.PI)))

(defn- kalman-log-ml
  "Exact log marginal likelihood via Kalman filter.
   q = transition std, r = observation std.
   Returns log p(y_0, ..., y_{T-1})."
  [observations prior-mean q r]
  (let [q2 (* q q), r2 (* r r)]
    (loop [t 0, mu (double prior-mean), P q2, acc 0.0]
      (if (>= t (count observations))
        acc
        (let [y   (nth observations t)
              S   (+ P r2)
              v   (- y mu)
              ll  (- (* -0.5 LOG-2PI) (* 0.5 (js/Math.log S)) (/ (* v v) (* 2 S)))
              K   (/ P S)]
          (recur (inc t)
                 (+ mu (* K v))
                 (+ (* (- 1.0 K) P) q2)
                 (+ acc ll)))))))

;; ===========================================================================
;; Shared SSM fixture
;; ===========================================================================

(def ^:private ssm-kernel
  "x_t ~ N(x_{t-1}, 1),  y_t ~ N(x_t, 1).  Returns x_t."
  (gen [t prev-state]
       (let [x (trace :x (dist/gaussian prev-state 1))]
         (trace :y (dist/gaussian x 1))
         x)))

(def ^:private init-state (mx/scalar 0.0))
(def ^:private T 5)
(def ^:private observations [1.0 0.5 -0.3 1.2 0.8])

;; Kernel-level obs choicemaps (for batched-smc-unfold, compiled-smc, smc-unfold)
(def ^:private kernel-obs-seq
  (mapv #(cm/choicemap :y (mx/scalar %)) observations))

;; Analytical ground truth: ≈ -7.484
(def ^:private ssm-analytical-log-ml
  (kalman-log-ml observations 0.0 1.0 1.0))

;; Tolerance for 500 particles on 5-step model (single run).
;; Empirically: estimates within ~0.3 of analytical. Use 1.5 for ~5σ margin.
(def ^:private ssm-log-ml-tol 1.5)

;; ===========================================================================
;; Normal-Normal conjugate fixture (for non-sequential tests)
;; ===========================================================================

(def ^:private conjugate-model
  (gen []
       (let [x (trace :x (dist/gaussian 0 1))]
         (trace :y (dist/gaussian x 1))
         x)))

(def ^:private conjugate-obs (cm/choicemap :y (mx/scalar 2.0)))

;; log N(2; 0, sqrt(2)) = -0.5*log(2π) - 0.5*log(2) - 0.5*(4/2) ≈ -2.2655
(def ^:private conjugate-log-ml (h/gaussian-lp 2.0 0.0 (js/Math.sqrt 2.0)))
(def ^:private conjugate-log-ml-tol 0.5)

;; ===========================================================================
;; 1. batched-smc-unfold — batched bootstrap PF
;; ===========================================================================

(deftest batched-smc-unfold-converges
  (testing "batched-smc-unfold log-ML converges to Kalman analytical value"
    (let [result (smc/batched-smc-unfold
                   {:particles 500 :key (rng/fresh-key 42)}
                   ssm-kernel init-state kernel-obs-seq)
          estimated (h/realize (:log-ml result))]
      (is (h/finite? estimated) "log-ML is finite")
      (is (h/close? ssm-analytical-log-ml estimated ssm-log-ml-tol)
          (str "batched-smc-unfold log-ML=" (.toFixed estimated 3)
               " expected=" (.toFixed ssm-analytical-log-ml 3))))))

;; ===========================================================================
;; 2. compiled-smc — Level 2 compiled bootstrap PF
;; ===========================================================================

(deftest compiled-smc-converges
  (testing "compiled-smc log-ML converges to Kalman analytical value"
    (let [result (compiled-smc/compiled-smc
                   {:particles 500 :key (rng/fresh-key 42)}
                   ssm-kernel init-state kernel-obs-seq)
          estimated (h/realize (:log-ml result))]
      (is (h/finite? estimated) "log-ML is finite")
      (is (h/close? ssm-analytical-log-ml estimated ssm-log-ml-tol)
          (str "compiled-smc log-ML=" (.toFixed estimated 3)
               " expected=" (.toFixed ssm-analytical-log-ml 3))))))

;; ===========================================================================
;; 3. smc-unfold — incremental PF via unfold-extend
;; ===========================================================================

(deftest smc-unfold-converges
  (testing "smc-unfold log-ML converges to Kalman analytical value"
    (let [result (smc/smc-unfold
                   {:particles 500 :key (rng/fresh-key 42)}
                   ssm-kernel init-state kernel-obs-seq)
          estimated (h/realize (:log-ml result))]
      (is (h/finite? estimated) "log-ML is finite")
      (is (h/close? ssm-analytical-log-ml estimated ssm-log-ml-tol)
          (str "smc-unfold log-ML=" (.toFixed estimated 3)
               " expected=" (.toFixed ssm-analytical-log-ml 3))))))

;; ===========================================================================
;; 4. smcp3 — bootstrap mode on conjugate model
;; ===========================================================================
;;
;; Note: smcp3 on unfold models via progressive constraining gives degenerate
;; weights (free→constrained update has infinite-variance importance weights).
;; Sequential unfold inference with smcp3 requires custom forward/backward
;; proposal kernels via the edit interface. Here we test bootstrap mode on
;; the single-observation conjugate model to verify the algorithm converges.
;; ===========================================================================

(deftest smcp3-converges-conjugate
  (testing "SMCP3 (bootstrap mode) log-ML on conjugate model"
    (let [result (smcp3/smcp3
                   {:particles 500 :key (rng/fresh-key 42)}
                   conjugate-model [] [conjugate-obs])
          estimated (h/realize (:log-ml-estimate result))]
      (is (h/finite? estimated) "log-ML is finite")
      (is (h/close? conjugate-log-ml estimated conjugate-log-ml-tol)
          (str "smcp3 log-ML=" (.toFixed estimated 3)
               " expected=" (.toFixed conjugate-log-ml 3))))))

;; ===========================================================================
;; 5. vsmc-init — batched importance sampling
;; ===========================================================================
;;
;; Note: vsmc (multi-step) requires vgenerate/vupdate on a DynamicGF.
;; UnfoldCombinator is not a DynamicGF, so vsmc cannot operate on unfold
;; models directly. Progressive constraining via vupdate also suffers from
;; infinite-variance weights (same issue as smcp3). Here we test vsmc-init
;; (single-step batched IS) on the conjugate model.
;; ===========================================================================

(deftest vsmc-init-converges-conjugate
  (testing "vsmc-init log-ML on conjugate model"
    (let [result (smc/vsmc-init
                   conjugate-model [] conjugate-obs 500 (rng/fresh-key 42))
          estimated (h/realize (:log-ml-estimate result))]
      (is (h/finite? estimated) "log-ML is finite")
      (is (h/close? conjugate-log-ml estimated conjugate-log-ml-tol)
          (str "vsmc-init log-ML=" (.toFixed estimated 3)
               " expected=" (.toFixed conjugate-log-ml 3))))))

(cljs.test/run-tests)
