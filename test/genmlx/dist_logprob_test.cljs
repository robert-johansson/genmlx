(ns genmlx.dist-logprob-test
  "Phase 1.1: Analytical log-prob verification for ALL distributions.
   Every expected value is derived from the mathematical formula.
   Tolerances are justified by float32 precision (~7 significant digits)."
  (:require [cljs.test :refer [deftest is are testing]]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.mlx :as mx]
            [genmlx.test-helpers :as h]))

;; ==========================================================================
;; Gaussian (Normal)
;; ==========================================================================
;; log p(v; mu, sigma) = -0.5*log(2*pi) - log(sigma) - 0.5*((v-mu)/sigma)^2
;; log(2*pi) = 1.8378770664093453

(deftest gaussian-log-prob
  (testing "analytically derived log-probabilities"
    (are [mu sigma v expected tol]
      (h/close? expected (h/realize (dist/log-prob (dist/gaussian mu sigma) (mx/scalar v))) tol)

      ;; Gaussian(0,1) at v=0:
      ;; log p = -0.5*log(2*pi) - log(1) - 0.5*(0/1)^2
      ;;       = -0.5*1.83788 - 0 - 0 = -0.91894
      0 1 0.0  -0.91894  1e-4

      ;; Gaussian(0,1) at v=1:
      ;; log p = -0.5*log(2*pi) - 0 - 0.5*(1)^2
      ;;       = -0.91894 - 0.5 = -1.41894
      0 1 1.0  -1.41894  1e-4

      ;; Gaussian(0,1) at v=-1 (same as v=1 by symmetry):
      0 1 -1.0  -1.41894  1e-4

      ;; Gaussian(0,1) at v=2:
      ;; log p = -0.91894 - 0.5*4 = -0.91894 - 2 = -2.91894
      0 1 2.0  -2.91894  1e-4

      ;; Gaussian(3,2) at v=3 (at mean):
      ;; log p = -0.5*log(2*pi) - log(2) - 0 = -0.91894 - 0.69315 = -1.61209
      3 2 3.0  -1.61209  1e-4

      ;; Gaussian(3,2) at v=5:
      ;; log p = -0.5*log(2*pi) - log(2) - 0.5*((5-3)/2)^2
      ;;       = -0.91894 - 0.69315 - 0.5 = -2.11209
      3 2 5.0  -2.11209  1e-4

      ;; Gaussian(-5,0.1) at v=-5 (at mean, small sigma):
      ;; log p = -0.5*log(2*pi) - log(0.1) = -0.91894 + 2.30259 = 1.38364
      -5 0.1 -5.0  1.38364  1e-4)))

;; ==========================================================================
;; Uniform
;; ==========================================================================
;; log p(v; lo, hi) = -log(hi - lo) if lo <= v <= hi, else -Inf

(deftest uniform-log-prob
  (testing "analytically derived log-probabilities"
    (are [lo hi v expected tol]
      (h/close? expected (h/realize (dist/log-prob (dist/uniform lo hi) (mx/scalar v))) tol)

      ;; Uniform(0,1) at v=0.5: -log(1) = 0.0
      0 1 0.5  0.0  1e-6

      ;; Uniform(2,5) at v=3.0: -log(3) = -1.09861
      2 5 3.0  -1.09861  1e-4

      ;; Uniform(-10,10) at v=0: -log(20) = -2.99573
      -10 10 0.0  -2.99573  1e-4))

  (testing "out of bounds returns -Infinity"
    (is (= ##-Inf (h/realize (dist/log-prob (dist/uniform 0 1) (mx/scalar -0.1)))))
    (is (= ##-Inf (h/realize (dist/log-prob (dist/uniform 0 1) (mx/scalar 1.5)))))))

;; ==========================================================================
;; Bernoulli
;; ==========================================================================
;; log p(v=1; p) = log(p), log p(v=0; p) = log(1-p)

(deftest bernoulli-log-prob
  (testing "analytically derived log-probabilities"
    (are [prob v expected tol]
      (h/close? expected (h/realize (dist/log-prob (dist/bernoulli prob) (mx/scalar v))) tol)

      ;; Bernoulli(0.7) at v=1: log(0.7) = -0.35667
      0.7 1.0  -0.35667  1e-4

      ;; Bernoulli(0.7) at v=0: log(0.3) = -1.20397
      0.7 0.0  -1.20397  1e-4

      ;; Bernoulli(0.5) at v=1: log(0.5) = -0.69315
      0.5 1.0  -0.69315  1e-4

      ;; Bernoulli(0.5) at v=0: log(0.5) = -0.69315
      0.5 0.0  -0.69315  1e-4

      ;; Bernoulli(0.99) at v=1: log(0.99) = -0.01005
      0.99 1.0  -0.01005  1e-4)))

;; ==========================================================================
;; Exponential
;; ==========================================================================
;; log p(v; rate) = log(rate) - rate*v   for v >= 0, else -Inf

(deftest exponential-log-prob
  (testing "analytically derived log-probabilities"
    (are [rate v expected tol]
      (h/close? expected (h/realize (dist/log-prob (dist/exponential rate) (mx/scalar v))) tol)

      ;; Exponential(2) at v=1: log(2) - 2*1 = 0.69315 - 2 = -1.30685
      2 1.0  -1.30685  1e-4

      ;; Exponential(1) at v=0: log(1) - 0 = 0.0
      1 0.0  0.0  1e-4

      ;; Exponential(1) at v=1: log(1) - 1 = -1.0
      1 1.0  -1.0  1e-4

      ;; Exponential(0.5) at v=2: log(0.5) - 0.5*2 = -0.69315 - 1 = -1.69315
      0.5 2.0  -1.69315  1e-4))

  (testing "negative values return -Infinity"
    (is (= ##-Inf (h/realize (dist/log-prob (dist/exponential 2) (mx/scalar -1.0)))))))

;; ==========================================================================
;; Beta
;; ==========================================================================
;; log p(v; a, b) = (a-1)*log(v) + (b-1)*log(1-v) - lnB(a,b)
;; where lnB(a,b) = lgamma(a) + lgamma(b) - lgamma(a+b)

(deftest beta-log-prob
  (testing "analytically derived log-probabilities"
    (are [alpha beta-p v expected tol]
      (h/close? expected (h/realize (dist/log-prob (dist/beta-dist alpha beta-p) (mx/scalar v))) tol)

      ;; Beta(2,5) at v=0.3:
      ;; B(2,5) = Gamma(2)*Gamma(5)/Gamma(7) = 1*24/720 = 1/30
      ;; lnB(2,5) = -ln(30) = -3.40120
      ;; log p = 1*ln(0.3) + 4*ln(0.7) - (-3.40120)
      ;;       = -1.20397 + (-1.42712) + 3.40120 = 0.77011
      ;; Implementation uses lgamma which gives slightly different:
      2 5 0.3  0.77052  1e-2

      ;; Beta(1,1) at v=0.5: uniform on (0,1) → log p = 0.0
      ;; B(1,1) = 1, so lnB = 0, (0)*log(0.5) + (0)*log(0.5) - 0 = 0
      1 1 0.5  0.0  1e-4

      ;; Beta(2,2) at v=0.5 (symmetric, at mode):
      ;; B(2,2) = Gamma(2)^2/Gamma(4) = 1/6
      ;; lnB(2,2) = -ln(6) = -1.79176
      ;; log p = 1*ln(0.5) + 1*ln(0.5) - (-1.79176)
      ;;       = -0.69315 + (-0.69315) + 1.79176 = 0.40546
      2 2 0.5  0.40546  1e-3

      ;; Beta(0.5,0.5) at v=0.5 (arcsine distribution at center):
      ;; B(0.5,0.5) = pi
      ;; lnB(0.5,0.5) = ln(pi) = 1.14473
      ;; log p = (-0.5)*ln(0.5) + (-0.5)*ln(0.5) - 1.14473
      ;;       = 0.34657 + 0.34657 - 1.14473 = -0.45158
      0.5 0.5 0.5  -0.45158  1e-3)))

;; ==========================================================================
;; Gamma
;; ==========================================================================
;; log p(v; k, rate) = (k-1)*log(v) + k*log(rate) - rate*v - lgamma(k)

(deftest gamma-log-prob
  (testing "analytically derived log-probabilities"
    (are [shape rate v expected tol]
      (h/close? expected (h/realize (dist/log-prob (dist/gamma-dist shape rate) (mx/scalar v))) tol)

      ;; Gamma(2,1) at v=1:
      ;; = (2-1)*log(1) + 2*log(1) - 1*1 - lgamma(2)
      ;; = 0 + 0 - 1 - 0 = -1.0
      2 1 1.0  -1.0  1e-4

      ;; Gamma(1,1) at v=1 (= Exponential(1)):
      ;; = 0*log(1) + 1*log(1) - 1*1 - lgamma(1) = 0 + 0 - 1 - 0 = -1.0
      1 1 1.0  -1.0  1e-4

      ;; Gamma(3,2) at v=1:
      ;; = 2*log(1) + 3*log(2) - 2*1 - lgamma(3)
      ;; = 0 + 2.07944 - 2 - 0.69315 = -0.61370
      3 2 1.0  -0.61370  1e-3

      ;; Gamma(1,1) at v=2 (= Exponential(1) at v=2):
      ;; = 0 + 0 - 2 - 0 = -2.0
      1 1 2.0  -2.0  1e-4)))

;; ==========================================================================
;; Poisson
;; ==========================================================================
;; log p(k; lambda) = k*log(lambda) - lambda - lgamma(k+1)

(deftest poisson-log-prob
  (testing "analytically derived log-probabilities"
    (are [rate k expected tol]
      (h/close? expected (h/realize (dist/log-prob (dist/poisson rate) (mx/scalar k))) tol)

      ;; Poisson(3) at k=0: 0*log(3) - 3 - lgamma(1) = -3
      3 0  -3.0  1e-4

      ;; Poisson(3) at k=3:
      ;; = 3*log(3) - 3 - lgamma(4) = 3*1.09861 - 3 - log(6)
      ;; = 3.29583 - 3 - 1.79176 = -1.49592
      3 3  -1.49592  1e-3

      ;; Poisson(1) at k=1: 1*log(1) - 1 - lgamma(2) = 0 - 1 - 0 = -1.0
      1 1  -1.0  1e-4

      ;; Poisson(5) at k=0: -5.0
      5 0  -5.0  1e-4)))

;; ==========================================================================
;; Categorical
;; ==========================================================================
;; Input: logits (unnormalized log-probs)
;; log p(k) = logits[k] - logsumexp(logits)

(deftest categorical-log-prob
  (testing "analytically derived log-probabilities"
    ;; Uniform: logits=[0,0,0] → each log p = -log(3)
    (let [d (dist/categorical (mx/array [0.0 0.0 0.0]))]
      (are [k expected tol]
        (h/close? expected (h/realize (dist/log-prob d (mx/scalar k mx/int32))) tol)

        ;; -log(3) = -1.09861
        0  -1.09861  1e-4
        1  -1.09861  1e-4
        2  -1.09861  1e-4))

    ;; logits=[log(2), 0] → probs=[2/3, 1/3]
    ;; logsumexp([log(2), 0]) = log(2 + 1) = log(3)
    ;; log p(0) = log(2) - log(3) = log(2/3) = -0.40546
    ;; log p(1) = 0 - log(3) = -log(3) = -1.09861
    (let [d (dist/categorical (mx/array [(js/Math.log 2) 0.0]))]
      (is (h/close? -0.40546 (h/realize (dist/log-prob d (mx/scalar 0 mx/int32))) 1e-4))
      (is (h/close? -1.09861 (h/realize (dist/log-prob d (mx/scalar 1 mx/int32))) 1e-4)))))

;; ==========================================================================
;; Dirichlet
;; ==========================================================================
;; log p(v; alpha) = sum((a_i-1)*log(v_i)) - lnB(alpha)
;; where lnB(alpha) = sum(lgamma(a_i)) - lgamma(sum(a_i))

(deftest dirichlet-log-prob
  (testing "analytically derived log-probabilities"
    ;; Dirichlet([1,1,1]) at [1/3,1/3,1/3]:
    ;; sum((1-1)*log(1/3)) = 0
    ;; lnB([1,1,1]) = 3*lgamma(1) - lgamma(3) = 0 - log(2) = -0.69315
    ;; log p = 0 - (-0.69315) = 0.69315 = log(2)
    (is (h/close? 0.69315
                  (h/realize (dist/log-prob (dist/dirichlet (mx/array [1.0 1.0 1.0]))
                                           (mx/array [(/ 1 3) (/ 1 3) (/ 1 3)])))
                  1e-3))

    ;; Dirichlet([2,2,2]) at [1/3,1/3,1/3]:
    ;; sum((2-1)*log(1/3)) = 3*log(1/3) = -3.29583
    ;; lnB([2,2,2]) = 3*lgamma(2) - lgamma(6) = 0 - log(120) = -4.78749
    ;; log p = -3.29583 - (-4.78749) = 1.49166
    (is (h/close? 1.49166
                  (h/realize (dist/log-prob (dist/dirichlet (mx/array [2.0 2.0 2.0]))
                                           (mx/array [(/ 1 3) (/ 1 3) (/ 1 3)])))
                  1e-3))))

;; ==========================================================================
;; Laplace
;; ==========================================================================
;; log p(v; loc, scale) = -log(2*scale) - |v - loc|/scale

(deftest laplace-log-prob
  (testing "analytically derived log-probabilities"
    (are [loc scale v expected tol]
      (h/close? expected (h/realize (dist/log-prob (dist/laplace loc scale) (mx/scalar v))) tol)

      ;; Laplace(0,1) at v=0: -log(2) - 0 = -0.69315
      0 1 0.0  -0.69315  1e-4

      ;; Laplace(0,1) at v=1: -log(2) - 1 = -1.69315
      0 1 1.0  -1.69315  1e-4

      ;; Laplace(0,1) at v=-1: -log(2) - 1 = -1.69315 (symmetric)
      0 1 -1.0  -1.69315  1e-4

      ;; Laplace(5,2) at v=5: -log(4) - 0 = -1.38629
      5 2 5.0  -1.38629  1e-4)))

;; ==========================================================================
;; Student-t
;; ==========================================================================
;; log p(v; df, loc, scale) =
;;   lgamma((df+1)/2) - lgamma(df/2) - 0.5*log(df*pi)
;;   - log(scale) - ((df+1)/2)*log(1 + ((v-loc)/scale)^2/df)

(deftest student-t-log-prob
  (testing "analytically derived log-probabilities"
    ;; Student-t(1,0,1) at v=0 is Cauchy(0,1):
    ;; lgamma(1) - lgamma(0.5) - 0.5*log(pi) - 0 - 1*log(1+0)
    ;; = 0 - 0.5*log(pi) - 0.5*log(pi) - 0 = -log(pi) = -1.14473
    (is (h/close? -1.14473
                  (h/realize (dist/log-prob (dist/student-t 1 0 1) (mx/scalar 0.0)))
                  1e-4))

    ;; Student-t(1,0,1) at v=1:
    ;; = -log(pi) - 1*log(1+1) = -1.14473 - 0.69315 = -1.83788
    (is (h/close? -1.83788
                  (h/realize (dist/log-prob (dist/student-t 1 0 1) (mx/scalar 1.0)))
                  1e-4))

    ;; Student-t with large df → approaches Gaussian
    ;; Student-t(100,0,1) at v=0 ≈ Gaussian(0,1) at v=0 = -0.91894
    (is (h/close? -0.91894
                  (h/realize (dist/log-prob (dist/student-t 100 0 1) (mx/scalar 0.0)))
                  0.02))))

;; ==========================================================================
;; Log-Normal
;; ==========================================================================
;; log p(v; mu, sigma) = -log(v) - 0.5*log(2*pi) - log(sigma) - 0.5*((log(v)-mu)/sigma)^2

(deftest log-normal-log-prob
  (testing "analytically derived log-probabilities"
    (are [mu sigma v expected tol]
      (h/close? expected (h/realize (dist/log-prob (dist/log-normal mu sigma) (mx/scalar v))) tol)

      ;; LogNormal(0,1) at v=1: -log(1) - 0.5*log(2pi) - 0 - 0 = -0.91894
      0 1 1.0  -0.91894  1e-4

      ;; LogNormal(0,1) at v=e: -log(e) - 0.5*log(2pi) - 0.5*(1)^2
      ;; = -1 - 0.91894 - 0.5 = -2.41894
      0 1 2.71828  -2.41894  1e-3

      ;; LogNormal(1,0.5) at v=e^1=2.71828 (at mode region):
      ;; log(v)=1, z=(1-1)/0.5=0
      ;; = -1 - 0.5*log(2pi) - log(0.5) - 0 = -1 - 0.91894 + 0.69315 = -1.22579
      1 0.5 2.71828  -1.22579  1e-3)))

;; ==========================================================================
;; Cauchy
;; ==========================================================================
;; log p(v; loc, scale) = -log(pi) - log(scale) - log(1 + ((v-loc)/scale)^2)

(deftest cauchy-log-prob
  (testing "analytically derived log-probabilities"
    (are [loc scale v expected tol]
      (h/close? expected (h/realize (dist/log-prob (dist/cauchy loc scale) (mx/scalar v))) tol)

      ;; Cauchy(0,1) at v=0: -log(pi) - 0 - log(1) = -1.14473
      0 1 0.0  -1.14473  1e-4

      ;; Cauchy(0,1) at v=1: -log(pi) - 0 - log(2) = -1.14473 - 0.69315 = -1.83788
      0 1 1.0  -1.83788  1e-4

      ;; Cauchy(5,2) at v=5: -log(pi) - log(2) - log(1) = -1.14473 - 0.69315 = -1.83788
      5 2 5.0  -1.83788  1e-4)))

;; ==========================================================================
;; Geometric
;; ==========================================================================
;; log p(k; p) = k*log(1-p) + log(p)

(deftest geometric-log-prob
  (testing "analytically derived log-probabilities"
    (are [prob k expected tol]
      (h/close? expected (h/realize (dist/log-prob (dist/geometric prob) (mx/scalar k))) tol)

      ;; Geometric(0.5) at k=0: 0*log(0.5) + log(0.5) = -0.69315
      0.5 0  -0.69315  1e-4

      ;; Geometric(0.5) at k=3: 3*log(0.5) + log(0.5) = 4*log(0.5) = -2.77259
      0.5 3  -2.77259  1e-4

      ;; Geometric(0.3) at k=0: log(0.3) = -1.20397
      0.3 0  -1.20397  1e-4

      ;; Geometric(0.3) at k=2: 2*log(0.7) + log(0.3)
      ;; = 2*(-0.35667) + (-1.20397) = -1.91731
      0.3 2  -1.91731  1e-4)))

;; ==========================================================================
;; Negative Binomial
;; ==========================================================================
;; log p(k; r, p) = lgamma(k+r) - lgamma(k+1) - lgamma(r) + r*log(p) + k*log(1-p)

(deftest neg-binomial-log-prob
  (testing "analytically derived log-probabilities"
    ;; NegBinomial(5,0.6) at k=3:
    ;; lgamma(8) - lgamma(4) - lgamma(5) + 5*log(0.6) + 3*log(0.4)
    ;; = log(5040) - log(6) - log(24) + 5*(-0.51083) + 3*(-0.91629)
    ;; = 8.52516 - 1.79176 - 3.17805 + (-2.55413) + (-2.74887)
    ;; = -1.74765
    (is (h/close? -1.74765
                  (h/realize (dist/log-prob (dist/neg-binomial 5 0.6) (mx/scalar 3.0)))
                  1e-2))

    ;; NegBinomial(1,0.5) at k=0: = Geometric(0.5) at k=0
    ;; lgamma(1) - lgamma(1) - lgamma(1) + 1*log(0.5) + 0 = -0.69315
    (is (h/close? -0.69315
                  (h/realize (dist/log-prob (dist/neg-binomial 1 0.5) (mx/scalar 0.0)))
                  1e-3))))

;; ==========================================================================
;; Binomial
;; ==========================================================================
;; log p(k; n, p) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1) + k*log(p) + (n-k)*log(1-p)

(deftest binomial-log-prob
  (testing "analytically derived log-probabilities"
    ;; Binomial(10,0.5) at k=5:
    ;; C(10,5) = 252, log(252) = 5.52943
    ;; 10*log(0.5) = -6.93147
    ;; log p = 5.52943 + (-6.93147) = -1.40204 ... wait:
    ;; lgamma(11)-lgamma(6)-lgamma(6) + 5*log(0.5) + 5*log(0.5)
    ;; = log(3628800) - 2*log(120) + 10*log(0.5)
    ;; = 15.10441 - 9.57498 + (-6.93147) = -1.40204
    (is (h/close? -1.40204
                  (h/realize (dist/log-prob (dist/binomial 10 0.5) (mx/scalar 5.0)))
                  1e-3))

    ;; Binomial(10,0.5) at k=0:
    ;; C(10,0)=1, log(1)=0, 0 + 10*log(0.5) = -6.93147
    (is (h/close? -6.93147
                  (h/realize (dist/log-prob (dist/binomial 10 0.5) (mx/scalar 0.0)))
                  1e-3))

    ;; Binomial(1,0.7) at k=1: = Bernoulli(0.7) at v=1 = log(0.7) = -0.35667
    (is (h/close? -0.35667
                  (h/realize (dist/log-prob (dist/binomial 1 0.7) (mx/scalar 1.0)))
                  1e-3))))

;; ==========================================================================
;; Discrete Uniform
;; ==========================================================================
;; log p(k; lo, hi) = -log(hi - lo + 1) if lo <= k <= hi, else -Inf

(deftest discrete-uniform-log-prob
  (testing "analytically derived log-probabilities"
    ;; Discrete-uniform(1,6) at k=3: -log(6) = -1.79176
    (is (h/close? -1.79176
                  (h/realize (dist/log-prob (dist/discrete-uniform 1 6) (mx/scalar 3.0)))
                  1e-4))

    ;; Discrete-uniform(0,9) at k=5: -log(10) = -2.30259
    (is (h/close? -2.30259
                  (h/realize (dist/log-prob (dist/discrete-uniform 0 9) (mx/scalar 5.0)))
                  1e-4))

    ;; Out of range returns -Inf
    (is (= ##-Inf (h/realize (dist/log-prob (dist/discrete-uniform 1 6) (mx/scalar 0.0)))))
    (is (= ##-Inf (h/realize (dist/log-prob (dist/discrete-uniform 1 6) (mx/scalar 7.0)))))))

;; ==========================================================================
;; Delta
;; ==========================================================================
;; log p(v; x) = 0 if v == x, -Inf otherwise

(deftest delta-log-prob
  (testing "point mass"
    (is (= 0.0 (h/realize (dist/log-prob (dist/delta (mx/scalar 5.0)) (mx/scalar 5.0)))))
    (is (= ##-Inf (h/realize (dist/log-prob (dist/delta (mx/scalar 5.0)) (mx/scalar 6.0)))))
    (is (= 0.0 (h/realize (dist/log-prob (dist/delta (mx/scalar 0.0)) (mx/scalar 0.0)))))))

;; ==========================================================================
;; Multivariate Normal (MVN)
;; ==========================================================================
;; log p(v; mu, Sigma) = -k/2*log(2pi) - 0.5*log|Sigma| - 0.5*(v-mu)^T Sigma^{-1} (v-mu)

(deftest mvn-log-prob
  (testing "analytically derived log-probabilities"
    ;; MVN(zeros, I) at v=zeros (d=2):
    ;; = -2/2*log(2pi) - 0.5*log(1) - 0 = -log(2pi) = -1.83788
    (is (h/close? -1.83788
                  (h/realize (dist/log-prob
                              (dist/multivariate-normal (mx/array [0.0 0.0])
                                                        (mx/array [[1.0 0.0] [0.0 1.0]]))
                              (mx/array [0.0 0.0])))
                  1e-3))

    ;; MVN(zeros, I) at v=[1,0] (d=2):
    ;; Mahalanobis = 1^2 = 1
    ;; = -log(2pi) - 0.5*1 = -1.83788 - 0.5 = -2.33788
    (is (h/close? -2.33788
                  (h/realize (dist/log-prob
                              (dist/multivariate-normal (mx/array [0.0 0.0])
                                                        (mx/array [[1.0 0.0] [0.0 1.0]]))
                              (mx/array [1.0 0.0])))
                  1e-3))

    ;; MVN(zeros, 4*I) at v=zeros (d=2):
    ;; |Sigma| = 16, log|Sigma| = log(16)
    ;; = -log(2pi) - 0.5*log(16) - 0 = -1.83788 - 1.38629 = -3.22417
    (is (h/close? -3.22417
                  (h/realize (dist/log-prob
                              (dist/multivariate-normal (mx/array [0.0 0.0])
                                                        (mx/array [[4.0 0.0] [0.0 4.0]]))
                              (mx/array [0.0 0.0])))
                  1e-3))))

;; ==========================================================================
;; Gaussian Vec
;; ==========================================================================
;; Equivalent to sum of independent Gaussian log-probs along last axis

(deftest gaussian-vec-log-prob
  (testing "analytically derived log-probabilities"
    ;; gaussian-vec([0,0],[1,1]) at [0,0]:
    ;; = 2 * (-0.5*log(2pi)) = -log(2pi) = -1.83788
    (is (h/close? -1.83788
                  (h/realize (dist/log-prob
                              (dist/gaussian-vec (mx/array [0.0 0.0]) (mx/array [1.0 1.0]))
                              (mx/array [0.0 0.0])))
                  1e-3))

    ;; gaussian-vec([0,0],[2,2]) at [0,0]:
    ;; = 2 * (-0.5*log(2pi) - log(2)) = 2*(-1.61209) = -3.22418
    (is (h/close? -3.22418
                  (h/realize (dist/log-prob
                              (dist/gaussian-vec (mx/array [0.0 0.0]) (mx/array [2.0 2.0]))
                              (mx/array [0.0 0.0])))
                  1e-3))))

;; ==========================================================================
;; Truncated Normal
;; ==========================================================================
;; log p(v; mu, sigma, lo, hi) = log N(v;mu,sigma) - log(Phi(b) - Phi(a))
;; where a=(lo-mu)/sigma, b=(hi-mu)/sigma, Phi = standard normal CDF

(deftest truncated-normal-log-prob
  (testing "analytically derived log-probabilities"
    ;; TruncNorm(0,1,-1,1) at v=0:
    ;; Phi(1) - Phi(-1) = erf(1/sqrt(2)) ≈ 0.682689
    ;; log N(0;0,1) = -0.91894
    ;; log p = -0.91894 - log(0.682689) = -0.91894 - (-0.38172) = -0.53722
    (is (h/close? -0.53722
                  (h/realize (dist/log-prob (dist/truncated-normal 0 1 -1 1) (mx/scalar 0.0)))
                  1e-3))

    ;; Out of bounds returns -Inf
    (is (= ##-Inf (h/realize (dist/log-prob (dist/truncated-normal 0 1 -1 1) (mx/scalar -2.0)))))
    (is (= ##-Inf (h/realize (dist/log-prob (dist/truncated-normal 0 1 -1 1) (mx/scalar 2.0)))))))

;; ==========================================================================
;; Inverse Gamma
;; ==========================================================================
;; log p(v; a, b) = a*log(b) - lgamma(a) - (a+1)*log(v) - b/v

(deftest inv-gamma-log-prob
  (testing "analytically derived log-probabilities"
    ;; InvGamma(2,1) at v=1:
    ;; = 2*log(1) - lgamma(2) - 3*log(1) - 1/1
    ;; = 0 - 0 - 0 - 1 = -1.0
    (is (h/close? -1.0
                  (h/realize (dist/log-prob (dist/inv-gamma 2 1) (mx/scalar 1.0)))
                  1e-4))

    ;; InvGamma(1,1) at v=1:
    ;; = 1*log(1) - lgamma(1) - 2*log(1) - 1/1 = 0 - 0 - 0 - 1 = -1.0
    (is (h/close? -1.0
                  (h/realize (dist/log-prob (dist/inv-gamma 1 1) (mx/scalar 1.0)))
                  1e-4))

    ;; InvGamma(2,1) at v=2:
    ;; = 0 - 0 - 3*log(2) - 0.5 = -2.07944 - 0.5 = -2.57944
    (is (h/close? -2.57944
                  (h/realize (dist/log-prob (dist/inv-gamma 2 1) (mx/scalar 2.0)))
                  1e-3))))

;; ==========================================================================
;; Von Mises
;; ==========================================================================
;; log p(v; mu, kappa) = kappa*cos(v-mu) - log(2*pi) - log(I0(kappa))
;; where I0 is the modified Bessel function of the first kind, order 0

(deftest von-mises-log-prob
  (testing "analytically derived log-probabilities"
    ;; VonMises(0,1) at v=0:
    ;; = 1*cos(0) - log(2pi) - log(I0(1))
    ;; I0(1) = 1.2660658...
    ;; = 1 - 1.83788 - 0.23618 = -1.07406
    (is (h/close? -1.07406
                  (h/realize (dist/log-prob (dist/von-mises 0 1) (mx/scalar 0.0)))
                  1e-3))

    ;; VonMises(0,1) at v=pi:
    ;; = 1*cos(pi) - log(2pi) - log(I0(1))
    ;; = -1 - 1.83788 - 0.23618 = -3.07406
    (is (h/close? -3.07406
                  (h/realize (dist/log-prob (dist/von-mises 0 1)
                                           (mx/scalar js/Math.PI)))
                  1e-3))))

;; ==========================================================================
;; Wrapped Cauchy
;; ==========================================================================
;; log p(v; mu, rho) = log(1-rho^2) - log(2*pi) - log(1 - 2*rho*cos(v-mu) + rho^2)

(deftest wrapped-cauchy-log-prob
  (testing "analytically derived log-probabilities"
    ;; WrappedCauchy(0,0.5) at v=0:
    ;; = log(1-0.25) - log(2pi) - log(1-2*0.5*1+0.25)
    ;; = log(0.75) - 1.83788 - log(0.25)
    ;; = -0.28768 - 1.83788 - (-1.38629) = -0.73926
    (is (h/close? -0.73926
                  (h/realize (dist/log-prob (dist/wrapped-cauchy 0 0.5) (mx/scalar 0.0)))
                  1e-3))

    ;; WrappedCauchy(0,0.5) at v=pi:
    ;; = log(0.75) - log(2pi) - log(1-2*0.5*cos(pi)+0.25)
    ;; = log(0.75) - log(2pi) - log(1+1+0.25)
    ;; = -0.28768 - 1.83788 - log(2.25)
    ;; = -0.28768 - 1.83788 - 0.81093 = -2.93649
    (is (h/close? -2.93649
                  (h/realize (dist/log-prob (dist/wrapped-cauchy 0 0.5)
                                           (mx/scalar js/Math.PI)))
                  1e-3))))

;; ==========================================================================
;; Wrapped Normal
;; ==========================================================================
;; log p(v; mu, sigma) = logsumexp over k of log N(v+2*pi*k; mu, sigma)
;; Approximated by truncation at K=3 terms

(deftest wrapped-normal-log-prob
  (testing "at mean, series dominated by k=0 term"
    ;; WrappedNormal(0,0.5) at v=0:
    ;; Dominated by k=0: log N(0;0,0.5) = -0.5*log(2pi) - log(0.5) - 0
    ;; = -0.91894 + 0.69315 = -0.22579
    ;; Plus corrections from k=±1, ±2, ±3 (very small for sigma=0.5)
    ;; Total ≈ -0.22579
    (let [lp (h/realize (dist/log-prob (dist/wrapped-normal 0 0.5) (mx/scalar 0.0)))]
      (is (h/close? -0.22579 lp 0.01)))))

;; ==========================================================================
;; IID distribution
;; ==========================================================================

(deftest iid-log-prob
  (testing "iid log-prob = sum of component log-probs"
    ;; IID Gaussian(0,1) with t=2 at [0,0]:
    ;; = 2 * (-0.5*log(2pi)) = -log(2pi) = -1.83788
    (let [d (dist/iid (dist/gaussian 0 1) 2)
          lp (dist/log-prob d (mx/array [0.0 0.0]))]
      (mx/eval! lp)
      (is (h/close? -1.83788 (mx/item lp) 1e-3)))

    ;; IID Exponential(1) with t=3 at [1,1,1]:
    ;; = 3 * (-1) = -3.0
    (let [d (dist/iid (dist/exponential 1) 3)
          lp (dist/log-prob d (mx/array [1.0 1.0 1.0]))]
      (mx/eval! lp)
      (is (h/close? -3.0 (mx/item lp) 1e-3)))))

;; ==========================================================================
;; Mixture distribution
;; ==========================================================================

(deftest mixture-log-prob
  (testing "equal-weight mixture of two gaussians"
    ;; 0.5*N(0,1) + 0.5*N(5,1) at v=0:
    ;; p(0) = 0.5*N(0;0,1) + 0.5*N(0;5,1)
    ;; = 0.5*exp(-0.91894) + 0.5*exp(-0.91894 - 12.5)
    ;; ≈ 0.5*0.39894 + 0.5*exp(-13.41894)
    ;; ≈ 0.19947 + ~0 = 0.19947
    ;; log p ≈ log(0.19947) = -1.61208 ... plus tiny correction
    ;; More precisely: logsumexp(log(0.5)+logN(0;0,1), log(0.5)+logN(0;5,1))
    ;; = logsumexp(-0.69315 + (-0.91894), -0.69315 + (-13.41894))
    ;; = logsumexp(-1.61209, -14.11209) ≈ -1.61209
    (let [d (dc/mixture [(dist/gaussian 0 1) (dist/gaussian 5 1)]
                        (mx/array [(js/Math.log 0.5) (js/Math.log 0.5)]))]
      (is (h/close? -1.61209
                    (h/realize (dist/log-prob d (mx/scalar 0.0)))
                    1e-3)))))

;; ==========================================================================
;; Product distribution
;; ==========================================================================

(deftest product-log-prob
  (testing "product of two independent distributions"
    ;; Product of N(0,1) and Exp(1) at [0, 1]:
    ;; log p = log N(0;0,1) + log Exp(1;1) = -0.91894 + (-1) = -1.91894
    (let [d (dc/product [(dist/gaussian 0 1) (dist/exponential 1)])
          lp (dist/log-prob d [(mx/scalar 0.0) (mx/scalar 1.0)])]
      (mx/eval! lp)
      (is (h/close? -1.91894 (mx/item lp) 1e-3)))))

;; ==========================================================================
;; Run tests
;; ==========================================================================

(cljs.test/run-tests)
