(ns genmlx.validation-test
  "Parameter validation tests for distributions."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]))

(defn- throws? [f]
  (try (f) false (catch :default _ true)))

(defn- no-throw? [f]
  (try (f) true (catch :default _ false)))

(deftest bernoulli-validation
  (testing "bernoulli"
    (is (no-throw? #(dist/bernoulli 0.5)) "valid p=0.5")
    (is (no-throw? #(dist/bernoulli 0)) "valid p=0")
    (is (no-throw? #(dist/bernoulli 1)) "valid p=1")
    (is (throws? #(dist/bernoulli -0.1)) "invalid p=-0.1")
    (is (throws? #(dist/bernoulli 1.5)) "invalid p=1.5")
    (is (no-throw? #(dist/bernoulli (mx/scalar 0.5))) "mlx array skips")))

(deftest poisson-validation
  (testing "poisson"
    (is (no-throw? #(dist/poisson 5)) "valid rate=5")
    (is (throws? #(dist/poisson 0)) "invalid rate=0")
    (is (throws? #(dist/poisson -1)) "invalid rate=-1")
    (is (no-throw? #(dist/poisson (mx/scalar 5))) "mlx array skips")))

(deftest laplace-validation
  (testing "laplace"
    (is (no-throw? #(dist/laplace 0 1)) "valid scale=1")
    (is (throws? #(dist/laplace 0 0)) "invalid scale=0")
    (is (throws? #(dist/laplace 0 -1)) "invalid scale=-1")
    (is (no-throw? #(dist/laplace 0 (mx/scalar 1))) "mlx array skips")))

(deftest student-t-validation
  (testing "student-t"
    (is (no-throw? #(dist/student-t 3 0 1)) "valid df=3 scale=1")
    (is (throws? #(dist/student-t 0 0 1)) "invalid df=0")
    (is (throws? #(dist/student-t 3 0 -1)) "invalid scale=-1")
    (is (no-throw? #(dist/student-t (mx/scalar 3) 0 (mx/scalar 1))) "mlx array skips")))

(deftest log-normal-validation
  (testing "log-normal"
    (is (no-throw? #(dist/log-normal 0 1)) "valid sigma=1")
    (is (throws? #(dist/log-normal 0 0)) "invalid sigma=0")
    (is (throws? #(dist/log-normal 0 -1)) "invalid sigma=-1")
    (is (no-throw? #(dist/log-normal 0 (mx/scalar 1))) "mlx array skips")))

(deftest cauchy-validation
  (testing "cauchy"
    (is (no-throw? #(dist/cauchy 0 1)) "valid scale=1")
    (is (throws? #(dist/cauchy 0 0)) "invalid scale=0")
    (is (throws? #(dist/cauchy 0 -1)) "invalid scale=-1")
    (is (no-throw? #(dist/cauchy 0 (mx/scalar 1))) "mlx array skips")))

(deftest inv-gamma-validation
  (testing "inv-gamma"
    (is (no-throw? #(dist/inv-gamma 2 1)) "valid shape=2 scale=1")
    (is (throws? #(dist/inv-gamma 0 1)) "invalid shape=0")
    (is (throws? #(dist/inv-gamma 2 -1)) "invalid scale=-1")
    (is (no-throw? #(dist/inv-gamma (mx/scalar 2) (mx/scalar 1))) "mlx array skips")))

(deftest geometric-validation
  (testing "geometric"
    (is (no-throw? #(dist/geometric 0.5)) "valid p=0.5")
    (is (throws? #(dist/geometric -0.1)) "invalid p=-0.1")
    (is (throws? #(dist/geometric 1.5)) "invalid p=1.5")
    (is (no-throw? #(dist/geometric (mx/scalar 0.5))) "mlx array skips")))

(deftest neg-binomial-validation
  (testing "neg-binomial"
    (is (no-throw? #(dist/neg-binomial 5 0.5)) "valid r=5 p=0.5")
    (is (throws? #(dist/neg-binomial 0 0.5)) "invalid r=0")
    (is (throws? #(dist/neg-binomial 5 -0.1)) "invalid p=-0.1")
    (is (throws? #(dist/neg-binomial 5 1.5)) "invalid p=1.5")
    (is (no-throw? #(dist/neg-binomial (mx/scalar 5) (mx/scalar 0.5))) "mlx array skips")))

(deftest binomial-validation
  (testing "binomial"
    (is (no-throw? #(dist/binomial 10 0.5)) "valid n=10 p=0.5")
    (is (throws? #(dist/binomial 10 -0.1)) "invalid p=-0.1")
    (is (throws? #(dist/binomial 10 1.5)) "invalid p=1.5")
    (is (no-throw? #(dist/binomial 10 (mx/scalar 0.5))) "mlx array skips")))

(deftest discrete-uniform-validation
  (testing "discrete-uniform"
    (is (no-throw? #(dist/discrete-uniform 0 10)) "valid lo=0 hi=10")
    (is (throws? #(dist/discrete-uniform 10 0)) "invalid lo=10 hi=0")
    (is (throws? #(dist/discrete-uniform 5 5)) "invalid lo=hi=5")
    (is (no-throw? #(dist/discrete-uniform (mx/scalar 0) (mx/scalar 10))) "mlx array skips")))

(deftest truncated-normal-validation
  (testing "truncated-normal"
    (is (no-throw? #(dist/truncated-normal 0 1 -1 1)) "valid sigma=1 lo=-1 hi=1")
    (is (throws? #(dist/truncated-normal 0 0 -1 1)) "invalid sigma=0")
    (is (throws? #(dist/truncated-normal 0 1 1 -1)) "invalid lo>hi")
    (is (no-throw? #(dist/truncated-normal 0 (mx/scalar 1) -1 1)) "mlx array skips")))

(cljs.test/run-tests)
