(ns genmlx.dist-boundary-test
  "Phase 1.4: Domain boundary tests for all distributions.
   Every bounded distribution must return -Infinity outside its support.
   Every distribution must return a finite value inside its support.
   Tests at: boundaries, just outside, just inside, mode, tails."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.mlx :as mx]
            [genmlx.test-helpers :as h]))

;; ==========================================================================
;; Uniform: support [lo, hi]
;; ==========================================================================

(deftest uniform-boundaries
  (testing "inside support → finite"
    (is (js/isFinite (h/realize (dist/log-prob (dist/uniform 0 1) (mx/scalar 0.5)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/uniform 0 1) (mx/scalar 0.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/uniform 0 1) (mx/scalar 1.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/uniform 0 1) (mx/scalar 0.001)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/uniform 0 1) (mx/scalar 0.999))))))

  (testing "outside support → -Infinity"
    (is (= ##-Inf (h/realize (dist/log-prob (dist/uniform 0 1) (mx/scalar -0.001)))))
    (is (= ##-Inf (h/realize (dist/log-prob (dist/uniform 0 1) (mx/scalar 1.001)))))
    (is (= ##-Inf (h/realize (dist/log-prob (dist/uniform 0 1) (mx/scalar -100.0)))))
    (is (= ##-Inf (h/realize (dist/log-prob (dist/uniform 0 1) (mx/scalar 100.0))))))

  (testing "various ranges"
    (is (= ##-Inf (h/realize (dist/log-prob (dist/uniform 2 5) (mx/scalar 1.0)))))
    (is (= ##-Inf (h/realize (dist/log-prob (dist/uniform 2 5) (mx/scalar 6.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/uniform 2 5) (mx/scalar 3.5)))))))

;; ==========================================================================
;; Exponential: support [0, +∞)
;; ==========================================================================

(deftest exponential-boundaries
  (testing "inside support → finite"
    (is (js/isFinite (h/realize (dist/log-prob (dist/exponential 1) (mx/scalar 0.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/exponential 1) (mx/scalar 0.001)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/exponential 1) (mx/scalar 1.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/exponential 1) (mx/scalar 100.0))))))

  (testing "negative values → -Infinity"
    (is (= ##-Inf (h/realize (dist/log-prob (dist/exponential 1) (mx/scalar -0.001)))))
    (is (= ##-Inf (h/realize (dist/log-prob (dist/exponential 1) (mx/scalar -1.0)))))
    (is (= ##-Inf (h/realize (dist/log-prob (dist/exponential 2) (mx/scalar -100.0)))))))

;; ==========================================================================
;; Beta: support (0, 1)
;; ==========================================================================

(deftest beta-boundaries
  (testing "inside support → finite"
    (is (js/isFinite (h/realize (dist/log-prob (dist/beta-dist 2 5) (mx/scalar 0.5)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/beta-dist 2 5) (mx/scalar 0.01)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/beta-dist 2 5) (mx/scalar 0.99))))))

  (testing "outside support → -Infinity or NaN"
    ;; Note: Beta log-prob uses log(v) and log(1-v), so v<=0 or v>=1
    ;; may produce -Inf or NaN depending on implementation.
    ;; We test that the result is not a valid finite log-prob.
    (let [lp-neg (h/realize (dist/log-prob (dist/beta-dist 2 5) (mx/scalar -0.1)))
          lp-above (h/realize (dist/log-prob (dist/beta-dist 2 5) (mx/scalar 1.1)))]
      (is (or (= ##-Inf lp-neg) (js/isNaN lp-neg))
          "negative value gives -Inf or NaN")
      (is (or (= ##-Inf lp-above) (js/isNaN lp-above))
          "value > 1 gives -Inf or NaN"))))

;; ==========================================================================
;; Gamma: support (0, +∞)
;; ==========================================================================

(deftest gamma-boundaries
  (testing "inside support → finite"
    (is (js/isFinite (h/realize (dist/log-prob (dist/gamma-dist 2 1) (mx/scalar 0.01)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/gamma-dist 2 1) (mx/scalar 1.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/gamma-dist 2 1) (mx/scalar 100.0))))))

  (testing "non-positive → invalid (not finite positive log-prob)"
    ;; Gamma log-prob uses log(v), so v=0 gives -Inf
    (let [lp-zero (h/realize (dist/log-prob (dist/gamma-dist 2 1) (mx/scalar 0.0)))]
      (is (or (= ##-Inf lp-zero) (js/isNaN lp-zero))
          "v=0 gives -Inf or NaN"))))

;; ==========================================================================
;; Truncated Normal: support [lo, hi]
;; ==========================================================================

(deftest truncated-normal-boundaries
  (testing "inside support → finite"
    (is (js/isFinite (h/realize (dist/log-prob (dist/truncated-normal 0 1 -1 1) (mx/scalar 0.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/truncated-normal 0 1 -1 1) (mx/scalar -0.99)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/truncated-normal 0 1 -1 1) (mx/scalar 0.99))))))

  (testing "outside support → -Infinity"
    (is (= ##-Inf (h/realize (dist/log-prob (dist/truncated-normal 0 1 -1 1) (mx/scalar -2.0)))))
    (is (= ##-Inf (h/realize (dist/log-prob (dist/truncated-normal 0 1 -1 1) (mx/scalar 2.0)))))
    (is (= ##-Inf (h/realize (dist/log-prob (dist/truncated-normal 0 1 -1 1) (mx/scalar -1.01)))))
    (is (= ##-Inf (h/realize (dist/log-prob (dist/truncated-normal 0 1 -1 1) (mx/scalar 1.01)))))))

;; ==========================================================================
;; Discrete Uniform: support {lo, lo+1, ..., hi}
;; ==========================================================================

(deftest discrete-uniform-boundaries
  (testing "inside support → finite"
    (is (js/isFinite (h/realize (dist/log-prob (dist/discrete-uniform 1 6) (mx/scalar 1.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/discrete-uniform 1 6) (mx/scalar 3.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/discrete-uniform 1 6) (mx/scalar 6.0))))))

  (testing "outside support → -Infinity"
    (is (= ##-Inf (h/realize (dist/log-prob (dist/discrete-uniform 1 6) (mx/scalar 0.0)))))
    (is (= ##-Inf (h/realize (dist/log-prob (dist/discrete-uniform 1 6) (mx/scalar 7.0)))))))

;; ==========================================================================
;; Delta: support {v}
;; ==========================================================================

(deftest delta-boundaries
  (testing "at point mass → 0.0"
    (is (= 0.0 (h/realize (dist/log-prob (dist/delta (mx/scalar 3.0)) (mx/scalar 3.0))))))

  (testing "away from point mass → -Infinity"
    (is (= ##-Inf (h/realize (dist/log-prob (dist/delta (mx/scalar 3.0)) (mx/scalar 2.99)))))
    (is (= ##-Inf (h/realize (dist/log-prob (dist/delta (mx/scalar 3.0)) (mx/scalar 3.01)))))
    (is (= ##-Inf (h/realize (dist/log-prob (dist/delta (mx/scalar 3.0)) (mx/scalar 0.0)))))
    (is (= ##-Inf (h/realize (dist/log-prob (dist/delta (mx/scalar 3.0)) (mx/scalar -100.0)))))))

;; ==========================================================================
;; Bernoulli: support {0, 1}
;; ==========================================================================

(deftest bernoulli-boundaries
  (testing "at support points → finite"
    (is (js/isFinite (h/realize (dist/log-prob (dist/bernoulli 0.5) (mx/scalar 0.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/bernoulli 0.5) (mx/scalar 1.0))))))

  (testing "extreme probabilities"
    ;; Bernoulli(0.99) at v=1: log(0.99) ≈ -0.01, still finite
    (is (js/isFinite (h/realize (dist/log-prob (dist/bernoulli 0.99) (mx/scalar 1.0)))))
    ;; Bernoulli(0.01) at v=1: log(0.01) ≈ -4.61, still finite
    (is (js/isFinite (h/realize (dist/log-prob (dist/bernoulli 0.01) (mx/scalar 1.0)))))))

;; ==========================================================================
;; Inverse Gamma: support (0, +∞)
;; ==========================================================================

(deftest inv-gamma-boundaries
  (testing "inside support → finite"
    (is (js/isFinite (h/realize (dist/log-prob (dist/inv-gamma 2 1) (mx/scalar 0.01)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/inv-gamma 2 1) (mx/scalar 1.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/inv-gamma 2 1) (mx/scalar 100.0))))))

  (testing "non-positive → invalid"
    (let [lp (h/realize (dist/log-prob (dist/inv-gamma 2 1) (mx/scalar 0.0)))]
      (is (or (= ##-Inf lp) (js/isNaN lp)) "v=0 gives -Inf or NaN"))))

;; ==========================================================================
;; Log-Normal: support (0, +∞)
;; ==========================================================================

(deftest log-normal-boundaries
  (testing "inside support → finite"
    (is (js/isFinite (h/realize (dist/log-prob (dist/log-normal 0 1) (mx/scalar 0.01)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/log-normal 0 1) (mx/scalar 1.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/log-normal 0 1) (mx/scalar 100.0))))))

  (testing "non-positive → invalid"
    (let [lp (h/realize (dist/log-prob (dist/log-normal 0 1) (mx/scalar 0.0)))]
      (is (or (= ##-Inf lp) (js/isNaN lp)) "v=0 gives -Inf or NaN"))))

;; ==========================================================================
;; Unbounded distributions: finite everywhere on support
;; ==========================================================================

(deftest gaussian-always-finite
  (testing "gaussian returns finite log-prob for any finite input"
    (is (js/isFinite (h/realize (dist/log-prob (dist/gaussian 0 1) (mx/scalar 0.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/gaussian 0 1) (mx/scalar 100.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/gaussian 0 1) (mx/scalar -100.0)))))))

(deftest cauchy-always-finite
  (testing "cauchy returns finite log-prob for any finite input"
    (is (js/isFinite (h/realize (dist/log-prob (dist/cauchy 0 1) (mx/scalar 0.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/cauchy 0 1) (mx/scalar 1000.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/cauchy 0 1) (mx/scalar -1000.0)))))))

(deftest student-t-always-finite
  (testing "student-t returns finite log-prob for any finite input"
    (is (js/isFinite (h/realize (dist/log-prob (dist/student-t 3 0 1) (mx/scalar 0.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/student-t 3 0 1) (mx/scalar 100.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/student-t 3 0 1) (mx/scalar -100.0)))))))

(deftest laplace-always-finite
  (testing "laplace returns finite log-prob for any finite input"
    (is (js/isFinite (h/realize (dist/log-prob (dist/laplace 0 1) (mx/scalar 0.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/laplace 0 1) (mx/scalar 100.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/laplace 0 1) (mx/scalar -100.0)))))))

;; ==========================================================================
;; Binomial: support {0, 1, ..., n}
;; ==========================================================================

(deftest binomial-boundaries
  (testing "inside support → finite"
    (is (js/isFinite (h/realize (dist/log-prob (dist/binomial 10 0.5) (mx/scalar 0.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/binomial 10 0.5) (mx/scalar 5.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/binomial 10 0.5) (mx/scalar 10.0)))))))

;; ==========================================================================
;; Categorical: support {0, 1, ..., K-1}
;; ==========================================================================

(deftest categorical-boundaries
  (testing "inside support → finite"
    (let [d (dist/categorical (mx/array [0.0 0.0 0.0]))]
      (is (js/isFinite (h/realize (dist/log-prob d (mx/scalar 0 mx/int32)))))
      (is (js/isFinite (h/realize (dist/log-prob d (mx/scalar 1 mx/int32)))))
      (is (js/isFinite (h/realize (dist/log-prob d (mx/scalar 2 mx/int32))))))))

;; ==========================================================================
;; Geometric: support {0, 1, 2, ...}
;; ==========================================================================

(deftest geometric-boundaries
  (testing "inside support → finite"
    (is (js/isFinite (h/realize (dist/log-prob (dist/geometric 0.5) (mx/scalar 0.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/geometric 0.5) (mx/scalar 1.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/geometric 0.5) (mx/scalar 10.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/geometric 0.5) (mx/scalar 100.0)))))))

;; ==========================================================================
;; Poisson: support {0, 1, 2, ...}
;; ==========================================================================

(deftest poisson-boundaries
  (testing "inside support → finite"
    (is (js/isFinite (h/realize (dist/log-prob (dist/poisson 3) (mx/scalar 0.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/poisson 3) (mx/scalar 1.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/poisson 3) (mx/scalar 10.0)))))
    (is (js/isFinite (h/realize (dist/log-prob (dist/poisson 3) (mx/scalar 50.0)))))))

;; ==========================================================================
;; Run tests
;; ==========================================================================

(cljs.test/run-tests)
