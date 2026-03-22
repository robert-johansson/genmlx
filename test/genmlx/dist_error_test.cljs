(ns genmlx.dist-error-test
  "Phase 1.8: Error condition tests for distribution parameter validation.
   Every distribution must throw on invalid parameters.
   Documents which validations exist and which are missing."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]))

;; ==========================================================================
;; Gaussian: sigma must be positive
;; ==========================================================================

(deftest gaussian-errors
  (testing "sigma=0 throws"
    (is (thrown? js/Error (dist/gaussian 0 0))))
  (testing "sigma<0 throws"
    (is (thrown? js/Error (dist/gaussian 0 -1))))
  (testing "valid parameters don't throw"
    (is (some? (dist/gaussian 0 1)))
    (is (some? (dist/gaussian -5 0.01)))))

;; ==========================================================================
;; Uniform: lo must be less than hi
;; ==========================================================================

(deftest uniform-errors
  (testing "lo >= hi throws"
    (is (thrown? js/Error (dist/uniform 5 3)))
    (is (thrown? js/Error (dist/uniform 5 5))))
  (testing "valid parameters don't throw"
    (is (some? (dist/uniform 0 1)))
    (is (some? (dist/uniform -10 10)))))

;; ==========================================================================
;; Bernoulli: p must be in [0, 1]
;; ==========================================================================

(deftest bernoulli-errors
  (testing "p > 1 throws"
    (is (thrown? js/Error (dist/bernoulli 1.5))))
  (testing "p < 0 throws"
    (is (thrown? js/Error (dist/bernoulli -0.1))))
  (testing "valid parameters don't throw"
    (is (some? (dist/bernoulli 0.5)))
    (is (some? (dist/bernoulli 0.0)))
    (is (some? (dist/bernoulli 1.0)))))

;; ==========================================================================
;; Exponential: rate must be positive
;; ==========================================================================

(deftest exponential-errors
  (testing "rate <= 0 throws"
    (is (thrown? js/Error (dist/exponential -1)))
    (is (thrown? js/Error (dist/exponential 0))))
  (testing "valid parameters don't throw"
    (is (some? (dist/exponential 1)))
    (is (some? (dist/exponential 0.01)))))

;; ==========================================================================
;; Beta: alpha and beta must be positive
;; ==========================================================================

(deftest beta-errors
  (testing "alpha <= 0 throws"
    (is (thrown? js/Error (dist/beta-dist 0 1)))
    (is (thrown? js/Error (dist/beta-dist -1 1))))
  (testing "beta <= 0 throws"
    (is (thrown? js/Error (dist/beta-dist 1 0)))
    (is (thrown? js/Error (dist/beta-dist 1 -1))))
  (testing "valid parameters don't throw"
    (is (some? (dist/beta-dist 1 1)))
    (is (some? (dist/beta-dist 0.5 0.5)))))

;; ==========================================================================
;; Gamma: shape and rate must be positive
;; ==========================================================================

(deftest gamma-errors
  (testing "shape <= 0 throws"
    (is (thrown? js/Error (dist/gamma-dist 0 1)))
    (is (thrown? js/Error (dist/gamma-dist -1 1))))
  (testing "rate <= 0 throws"
    (is (thrown? js/Error (dist/gamma-dist 1 0)))
    (is (thrown? js/Error (dist/gamma-dist 1 -1))))
  (testing "valid parameters don't throw"
    (is (some? (dist/gamma-dist 1 1)))
    (is (some? (dist/gamma-dist 0.5 2)))))

;; ==========================================================================
;; Poisson: rate must be positive
;; ==========================================================================

(deftest poisson-errors
  (testing "rate <= 0 throws"
    (is (thrown? js/Error (dist/poisson -1)))
    (is (thrown? js/Error (dist/poisson 0))))
  (testing "valid parameters don't throw"
    (is (some? (dist/poisson 1)))
    (is (some? (dist/poisson 0.1)))))

;; ==========================================================================
;; Student-t: df and scale must be positive
;; ==========================================================================

(deftest student-t-errors
  (testing "df <= 0 throws"
    (is (thrown? js/Error (dist/student-t 0 0 1)))
    (is (thrown? js/Error (dist/student-t -1 0 1))))
  (testing "scale <= 0 throws"
    (is (thrown? js/Error (dist/student-t 3 0 0)))
    (is (thrown? js/Error (dist/student-t 3 0 -1))))
  (testing "valid parameters don't throw"
    (is (some? (dist/student-t 1 0 1)))
    (is (some? (dist/student-t 30 5 2)))))

;; ==========================================================================
;; Cauchy: scale must be positive
;; ==========================================================================

(deftest cauchy-errors
  (testing "scale <= 0 throws"
    (is (thrown? js/Error (dist/cauchy 0 0)))
    (is (thrown? js/Error (dist/cauchy 0 -1))))
  (testing "valid parameters don't throw"
    (is (some? (dist/cauchy 0 1)))
    (is (some? (dist/cauchy 5 0.1)))))

;; ==========================================================================
;; Laplace: scale must be positive
;; ==========================================================================

(deftest laplace-errors
  (testing "scale <= 0 throws"
    (is (thrown? js/Error (dist/laplace 0 0)))
    (is (thrown? js/Error (dist/laplace 0 -1))))
  (testing "valid parameters don't throw"
    (is (some? (dist/laplace 0 1)))))

;; ==========================================================================
;; Log-Normal: sigma must be positive
;; ==========================================================================

(deftest log-normal-errors
  (testing "sigma <= 0 throws"
    (is (thrown? js/Error (dist/log-normal 0 0)))
    (is (thrown? js/Error (dist/log-normal 0 -1))))
  (testing "valid parameters don't throw"
    (is (some? (dist/log-normal 0 1)))))

;; ==========================================================================
;; Inverse Gamma: shape and scale must be positive
;; ==========================================================================

(deftest inv-gamma-errors
  (testing "shape <= 0 throws"
    (is (thrown? js/Error (dist/inv-gamma 0 1)))
    (is (thrown? js/Error (dist/inv-gamma -1 1))))
  (testing "scale <= 0 throws"
    (is (thrown? js/Error (dist/inv-gamma 2 0)))
    (is (thrown? js/Error (dist/inv-gamma 2 -1))))
  (testing "valid parameters don't throw"
    (is (some? (dist/inv-gamma 2 1)))))

;; ==========================================================================
;; Geometric: p must be in (0, 1)
;; ==========================================================================

(deftest geometric-errors
  (testing "p out of [0,1] throws"
    (is (thrown? js/Error (dist/geometric 1.5)))
    (is (thrown? js/Error (dist/geometric -0.1))))
  (testing "valid parameters don't throw"
    (is (some? (dist/geometric 0.5)))
    (is (some? (dist/geometric 0.01)))))

;; ==========================================================================
;; Negative Binomial: r > 0, p in [0, 1]
;; ==========================================================================

(deftest neg-binomial-errors
  (testing "r <= 0 throws"
    (is (thrown? js/Error (dist/neg-binomial 0 0.5)))
    (is (thrown? js/Error (dist/neg-binomial -1 0.5))))
  (testing "p out of [0,1] throws"
    (is (thrown? js/Error (dist/neg-binomial 5 1.5)))
    (is (thrown? js/Error (dist/neg-binomial 5 -0.1))))
  (testing "valid parameters don't throw"
    (is (some? (dist/neg-binomial 5 0.5)))))

;; ==========================================================================
;; Binomial: p must be in [0, 1]
;; ==========================================================================

(deftest binomial-errors
  (testing "p out of [0,1] throws"
    (is (thrown? js/Error (dist/binomial 10 1.5)))
    (is (thrown? js/Error (dist/binomial 10 -0.1))))
  (testing "valid parameters don't throw"
    (is (some? (dist/binomial 10 0.5)))
    (is (some? (dist/binomial 1 0.0)))
    (is (some? (dist/binomial 1 1.0)))))

;; ==========================================================================
;; Discrete Uniform: lo < hi
;; ==========================================================================

(deftest discrete-uniform-errors
  (testing "lo >= hi throws"
    (is (thrown? js/Error (dist/discrete-uniform 6 3)))
    (is (thrown? js/Error (dist/discrete-uniform 5 5))))
  (testing "valid parameters don't throw"
    (is (some? (dist/discrete-uniform 1 6)))
    (is (some? (dist/discrete-uniform 0 1)))))

;; ==========================================================================
;; Von Mises: kappa must be positive
;; ==========================================================================

(deftest von-mises-errors
  (testing "kappa <= 0 throws"
    (is (thrown? js/Error (dist/von-mises 0 0)))
    (is (thrown? js/Error (dist/von-mises 0 -1))))
  (testing "valid parameters don't throw"
    (is (some? (dist/von-mises 0 1)))))

;; ==========================================================================
;; Wrapped Cauchy: rho must be in (0, 1)
;; ==========================================================================

(deftest wrapped-cauchy-errors
  (testing "rho out of (0,1) throws"
    (is (thrown? js/Error (dist/wrapped-cauchy 0 0)))
    (is (thrown? js/Error (dist/wrapped-cauchy 0 1)))
    (is (thrown? js/Error (dist/wrapped-cauchy 0 -0.1)))
    (is (thrown? js/Error (dist/wrapped-cauchy 0 1.5))))
  (testing "valid parameters don't throw"
    (is (some? (dist/wrapped-cauchy 0 0.5)))))

;; ==========================================================================
;; Wrapped Normal: sigma must be positive
;; ==========================================================================

(deftest wrapped-normal-errors
  (testing "sigma <= 0 throws"
    (is (thrown? js/Error (dist/wrapped-normal 0 0)))
    (is (thrown? js/Error (dist/wrapped-normal 0 -1))))
  (testing "valid parameters don't throw"
    (is (some? (dist/wrapped-normal 0 1)))))

;; ==========================================================================
;; Truncated Normal: sigma > 0, lo < hi
;; ==========================================================================

(deftest truncated-normal-errors
  (testing "sigma <= 0 throws"
    (is (thrown? js/Error (dist/truncated-normal 0 0 -1 1)))
    (is (thrown? js/Error (dist/truncated-normal 0 -1 -1 1))))
  (testing "lo >= hi throws"
    (is (thrown? js/Error (dist/truncated-normal 0 1 5 3)))
    (is (thrown? js/Error (dist/truncated-normal 0 1 5 5))))
  (testing "valid parameters don't throw"
    (is (some? (dist/truncated-normal 0 1 -1 1)))))

;; ==========================================================================
;; Run tests
;; ==========================================================================

(cljs.test/run-tests)
