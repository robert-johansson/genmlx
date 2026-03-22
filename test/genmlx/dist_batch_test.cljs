(ns genmlx.dist-batch-test
  "Phase 1.5: Batch shape tests for all distributions.
   Every distribution's sample-n must return correctly-shaped arrays.
   Every distribution's log-prob with batched input must return batched output."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.mlx :as mx]
            [genmlx.test-helpers :as h]))

(def ^:private N 50)

;; ==========================================================================
;; Scalar distributions: sample-n → [N], log-prob([N]) → [N]
;; ==========================================================================

(deftest gaussian-batch-shapes
  (let [d (dist/gaussian 0 1)
        samples (dc/dist-sample-n d (h/deterministic-key) N)]
    (mx/eval! samples)
    (is (= [N] (mx/shape samples)) "sample-n returns [N]")
    (let [lps (dist/log-prob d samples)]
      (mx/eval! lps)
      (is (= [N] (mx/shape lps)) "log-prob broadcasts to [N]"))))

(deftest uniform-batch-shapes
  (let [d (dist/uniform 0 1)
        samples (dc/dist-sample-n d (h/deterministic-key) N)]
    (mx/eval! samples)
    (is (= [N] (mx/shape samples)) "sample-n returns [N]")
    (let [lps (dist/log-prob d samples)]
      (mx/eval! lps)
      (is (= [N] (mx/shape lps)) "log-prob broadcasts to [N]"))))

(deftest bernoulli-batch-shapes
  (let [d (dist/bernoulli 0.5)
        samples (dc/dist-sample-n d (h/deterministic-key) N)]
    (mx/eval! samples)
    (is (= [N] (mx/shape samples)) "sample-n returns [N]")))

(deftest exponential-batch-shapes
  (let [d (dist/exponential 2)
        samples (dc/dist-sample-n d (h/deterministic-key) N)]
    (mx/eval! samples)
    (is (= [N] (mx/shape samples)) "sample-n returns [N]")
    (let [lps (dist/log-prob d samples)]
      (mx/eval! lps)
      (is (= [N] (mx/shape lps)) "log-prob broadcasts to [N]"))))

(deftest beta-batch-shapes
  (let [d (dist/beta-dist 2 5)
        samples (dc/dist-sample-n d (h/deterministic-key) N)]
    (mx/eval! samples)
    (is (= [N] (mx/shape samples)) "sample-n returns [N]")))

(deftest gamma-batch-shapes
  (let [d (dist/gamma-dist 2 1)
        samples (dc/dist-sample-n d (h/deterministic-key) N)]
    (mx/eval! samples)
    (is (= [N] (mx/shape samples)) "sample-n returns [N]")))

(deftest laplace-batch-shapes
  (let [d (dist/laplace 0 1)
        samples (dc/dist-sample-n d (h/deterministic-key) N)]
    (mx/eval! samples)
    (is (= [N] (mx/shape samples)) "sample-n returns [N]")
    (let [lps (dist/log-prob d samples)]
      (mx/eval! lps)
      (is (= [N] (mx/shape lps)) "log-prob broadcasts to [N]"))))

(deftest cauchy-batch-shapes
  (let [d (dist/cauchy 0 1)
        samples (dc/dist-sample-n d (h/deterministic-key) N)]
    (mx/eval! samples)
    (is (= [N] (mx/shape samples)) "sample-n returns [N]")
    (let [lps (dist/log-prob d samples)]
      (mx/eval! lps)
      (is (= [N] (mx/shape lps)) "log-prob broadcasts to [N]"))))

(deftest student-t-batch-shapes
  (let [d (dist/student-t 3 0 1)
        samples (dc/dist-sample-n d (h/deterministic-key) N)]
    (mx/eval! samples)
    (is (= [N] (mx/shape samples)) "sample-n returns [N]")))

(deftest log-normal-batch-shapes
  (let [d (dist/log-normal 0 1)
        samples (dc/dist-sample-n d (h/deterministic-key) N)]
    (mx/eval! samples)
    (is (= [N] (mx/shape samples)) "sample-n returns [N]")
    (let [lps (dist/log-prob d samples)]
      (mx/eval! lps)
      (is (= [N] (mx/shape lps)) "log-prob broadcasts to [N]"))))

(deftest categorical-batch-shapes
  (let [d (dist/categorical (mx/array [0.0 0.0 0.0]))
        samples (dc/dist-sample-n d (h/deterministic-key) N)]
    (mx/eval! samples)
    (is (= [N] (mx/shape samples)) "sample-n returns [N]")))

(deftest geometric-batch-shapes
  (let [d (dist/geometric 0.5)
        samples (dc/dist-sample-n d (h/deterministic-key) N)]
    (mx/eval! samples)
    (is (= [N] (mx/shape samples)) "sample-n returns [N]")))

(deftest binomial-batch-shapes
  (let [d (dist/binomial 10 0.5)
        samples (dc/dist-sample-n d (h/deterministic-key) N)]
    (mx/eval! samples)
    (is (= [N] (mx/shape samples)) "sample-n returns [N]")))

(deftest discrete-uniform-batch-shapes
  (let [d (dist/discrete-uniform 1 6)
        samples (dc/dist-sample-n d (h/deterministic-key) N)]
    (mx/eval! samples)
    (is (= [N] (mx/shape samples)) "sample-n returns [N]")))

(deftest von-mises-batch-shapes
  (let [d (dist/von-mises 0 1)
        samples (dc/dist-sample-n d (h/deterministic-key) N)]
    (mx/eval! samples)
    (is (= [N] (mx/shape samples)) "sample-n returns [N]")))

;; ==========================================================================
;; Multi-dimensional distributions: sample-n → [N, D]
;; ==========================================================================

(deftest mvn-batch-shapes
  (testing "MVN(d=2): sample-n → [N, 2]"
    (let [d (dist/multivariate-normal (mx/array [0.0 0.0])
                                       (mx/array [[1.0 0.0] [0.0 1.0]]))
          samples (dc/dist-sample-n d (h/deterministic-key) N)]
      (mx/eval! samples)
      (is (= [N 2] (mx/shape samples)) "sample-n returns [N, 2]"))))

(deftest gaussian-vec-batch-shapes
  (testing "gaussian-vec(d=2): sample-n → [N, 2]"
    (let [d (dist/gaussian-vec (mx/array [0.0 0.0]) (mx/array [1.0 1.0]))
          samples (dc/dist-sample-n d (h/deterministic-key) N)]
      (mx/eval! samples)
      (is (= [N 2] (mx/shape samples)) "sample-n returns [N, 2]"))))

(deftest dirichlet-batch-shapes
  (testing "Dirichlet(k=3): sample-n → [N, 3]"
    (let [d (dist/dirichlet (mx/array [1.0 1.0 1.0]))
          samples (dc/dist-sample-n d (h/deterministic-key) N)]
      (mx/eval! samples)
      (is (= [N 3] (mx/shape samples)) "sample-n returns [N, 3]"))))

(deftest iid-batch-shapes
  (testing "IID gaussian(t=3): sample-n → [N, 3]"
    (let [d (dist/iid (dist/gaussian 0 1) 3)
          samples (dc/dist-sample-n d (h/deterministic-key) N)]
      (mx/eval! samples)
      (is (= [N 3] (mx/shape samples)) "sample-n returns [N, 3]"))))

;; ==========================================================================
;; Delta: sample-n → [N]
;; ==========================================================================

(deftest delta-batch-shapes
  (let [d (dist/delta (mx/scalar 5.0))
        samples (dc/dist-sample-n d (h/deterministic-key) N)]
    (mx/eval! samples)
    (is (= [N] (mx/shape samples)) "sample-n returns [N]")
    ;; All values should be 5.0
    (is (every? #(= 5.0 %) (mx/->clj samples)) "all samples equal point mass")))

;; ==========================================================================
;; Run tests
;; ==========================================================================

(cljs.test/run-tests)
