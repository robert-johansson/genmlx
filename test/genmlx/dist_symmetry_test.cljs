(ns genmlx.dist-symmetry-test
  "Phase 1.2: Symmetry and invariant tests for symmetric distributions.
   For distributions symmetric around a location parameter:
     log-prob(d, loc + v) = log-prob(d, loc - v)
   Tolerance: 1e-10 (algebraic identity — only floating-point rounding error)."
  (:require [cljs.test :refer [deftest is are testing]]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.test-helpers :as h]))

;; ==========================================================================
;; Gaussian: symmetric around mu
;; ==========================================================================
;; N(mu, sigma): log p(mu+v) = log p(mu-v) for all v, sigma

(deftest gaussian-symmetry
  (testing "log-prob(mu+v) = log-prob(mu-v) for gaussian centered at 0"
    (are [sigma v]
      (h/close? (h/realize (dist/log-prob (dist/gaussian 0 sigma) (mx/scalar v)))
                (h/realize (dist/log-prob (dist/gaussian 0 sigma) (mx/scalar (- v))))
                1e-10)
      ;; Various sigma and v combinations
      1.0  0.5
      1.0  2.0
      1.0  5.0
      0.5  0.5
      2.0  0.5
      5.0  0.5
      0.1  0.01
      10.0 3.0))

  (testing "log-prob(mu+v) = log-prob(mu-v) for gaussian centered at mu=3"
    (are [sigma v]
      (h/close? (h/realize (dist/log-prob (dist/gaussian 3 sigma) (mx/scalar (+ 3 v))))
                (h/realize (dist/log-prob (dist/gaussian 3 sigma) (mx/scalar (- 3 v))))
                1e-10)
      1.0  0.5
      1.0  2.0
      2.0  1.0
      0.5  0.1)))

;; ==========================================================================
;; Cauchy: symmetric around loc
;; ==========================================================================
;; Cauchy(loc, scale): log p(loc+v) = log p(loc-v)

(deftest cauchy-symmetry
  (testing "log-prob(loc+v) = log-prob(loc-v) for cauchy centered at 0"
    (are [scale v]
      (h/close? (h/realize (dist/log-prob (dist/cauchy 0 scale) (mx/scalar v)))
                (h/realize (dist/log-prob (dist/cauchy 0 scale) (mx/scalar (- v))))
                1e-10)
      1.0  0.5
      1.0  2.0
      1.0  10.0
      0.5  1.0
      2.0  1.0
      5.0  3.0))

  (testing "log-prob(loc+v) = log-prob(loc-v) for cauchy centered at loc=-2"
    (are [scale v]
      (h/close? (h/realize (dist/log-prob (dist/cauchy -2 scale) (mx/scalar (+ -2 v))))
                (h/realize (dist/log-prob (dist/cauchy -2 scale) (mx/scalar (- -2 v))))
                1e-10)
      1.0  0.5
      1.0  5.0
      2.0  1.0)))

;; ==========================================================================
;; Student-t: symmetric around loc
;; ==========================================================================
;; Student-t(df, loc, scale): log p(loc+v) = log p(loc-v)

(deftest student-t-symmetry
  (testing "log-prob(loc+v) = log-prob(loc-v) for student-t centered at 0"
    (are [df scale v]
      (h/close? (h/realize (dist/log-prob (dist/student-t df 0 scale) (mx/scalar v)))
                (h/realize (dist/log-prob (dist/student-t df 0 scale) (mx/scalar (- v))))
                1e-10)
      ;; df=1 (Cauchy)
      1  1.0  0.5
      1  1.0  2.0
      1  2.0  1.0
      ;; df=3
      3  1.0  0.5
      3  1.0  2.0
      3  2.0  1.0
      ;; df=30
      30 1.0  0.5
      30 1.0  2.0))

  (testing "student-t centered at loc=5"
    (are [df scale v]
      (h/close? (h/realize (dist/log-prob (dist/student-t df 5 scale) (mx/scalar (+ 5 v))))
                (h/realize (dist/log-prob (dist/student-t df 5 scale) (mx/scalar (- 5 v))))
                1e-10)
      1  1.0  0.5
      3  1.0  1.0
      10 2.0  3.0)))

;; ==========================================================================
;; Laplace: symmetric around loc
;; ==========================================================================
;; Laplace(loc, scale): log p(loc+v) = log p(loc-v)

(deftest laplace-symmetry
  (testing "log-prob(loc+v) = log-prob(loc-v) for laplace centered at 0"
    (are [scale v]
      (h/close? (h/realize (dist/log-prob (dist/laplace 0 scale) (mx/scalar v)))
                (h/realize (dist/log-prob (dist/laplace 0 scale) (mx/scalar (- v))))
                1e-10)
      1.0  0.5
      1.0  2.0
      1.0  5.0
      0.5  0.5
      2.0  0.5
      5.0  0.5
      0.1  0.01
      10.0 3.0))

  (testing "laplace centered at loc=7"
    (are [scale v]
      (h/close? (h/realize (dist/log-prob (dist/laplace 7 scale) (mx/scalar (+ 7 v))))
                (h/realize (dist/log-prob (dist/laplace 7 scale) (mx/scalar (- 7 v))))
                1e-10)
      1.0  0.5
      2.0  1.0
      0.5  3.0)))

;; ==========================================================================
;; Von Mises: symmetric around mu on the circle
;; ==========================================================================
;; VonMises(mu, kappa): log p(mu+v) = log p(mu-v)

(deftest von-mises-symmetry
  (testing "log-prob(mu+v) = log-prob(mu-v) for von-mises centered at 0"
    (are [kappa v]
      (h/close? (h/realize (dist/log-prob (dist/von-mises 0 kappa) (mx/scalar v)))
                (h/realize (dist/log-prob (dist/von-mises 0 kappa) (mx/scalar (- v))))
                1e-10)
      1.0  0.5
      1.0  1.0
      1.0  2.0
      5.0  0.5
      0.5  1.0)))

;; ==========================================================================
;; Wrapped Cauchy: symmetric around mu on the circle
;; ==========================================================================
;; WrappedCauchy(mu, rho): log p(mu+v) = log p(mu-v)

(deftest wrapped-cauchy-symmetry
  (testing "log-prob(mu+v) = log-prob(mu-v) for wrapped-cauchy centered at 0"
    (are [rho v]
      (h/close? (h/realize (dist/log-prob (dist/wrapped-cauchy 0 rho) (mx/scalar v)))
                (h/realize (dist/log-prob (dist/wrapped-cauchy 0 rho) (mx/scalar (- v))))
                1e-10)
      0.5  0.5
      0.5  1.0
      0.5  2.0
      0.3  0.5
      0.8  0.5)))

;; ==========================================================================
;; Beta: symmetric when alpha = beta
;; ==========================================================================
;; Beta(a,a) at v and 1-v should give equal log-prob

(deftest beta-symmetry
  (testing "Beta(a,a) at v = Beta(a,a) at 1-v"
    (are [a v]
      (h/close? (h/realize (dist/log-prob (dist/beta-dist a a) (mx/scalar v)))
                (h/realize (dist/log-prob (dist/beta-dist a a) (mx/scalar (- 1 v))))
                1e-6)
      ;; Tolerance: 1e-6 rather than 1e-10 because beta log-prob involves
      ;; lgamma computations with more float32 rounding
      1.0  0.3
      2.0  0.3
      2.0  0.1
      0.5  0.3
      5.0  0.2)))

;; ==========================================================================
;; Wrapped Normal: symmetric around mu
;; ==========================================================================

(deftest wrapped-normal-symmetry
  (testing "log-prob(mu+v) = log-prob(mu-v) for wrapped-normal"
    (are [sigma v]
      (h/close? (h/realize (dist/log-prob (dist/wrapped-normal 0 sigma) (mx/scalar v)))
                (h/realize (dist/log-prob (dist/wrapped-normal 0 sigma) (mx/scalar (- v))))
                1e-6)
      ;; Tolerance: 1e-6 because wrapped normal uses series truncation
      0.5  0.5
      0.5  1.0
      1.0  0.5
      2.0  1.0)))

;; ==========================================================================
;; Run tests
;; ==========================================================================

(cljs.test/run-tests)
