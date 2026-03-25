(ns genmlx.batch-sample-n-test
  "Smoke tests for newly added dist-sample-n implementations."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]))

(deftest discrete-uniform-sample-n
  (testing "discrete-uniform dist-sample-n"
    (let [d (dist/discrete-uniform (mx/scalar 1 mx/int32) (mx/scalar 6 mx/int32))
          key (rng/fresh-key 42)
          samples (dc/dist-sample-n d key 100)]
      (mx/eval! samples)
      (is (= [100] (mx/shape samples)) "shape is [100]")
      (let [vals (mx/->clj samples)
            mn (apply min vals)
            mx-val (apply max vals)]
        (is (>= mn 1) "min >= 1")
        (is (<= mx-val 6) "max <= 6")))))

(deftest geometric-sample-n
  (testing "geometric dist-sample-n"
    (let [d (dist/geometric (mx/scalar 0.3))
          key (rng/fresh-key 42)
          samples (dc/dist-sample-n d key 100)]
      (mx/eval! samples)
      (is (= [100] (mx/shape samples)) "shape is [100]")
      (let [vals (mx/->clj samples)
            mn (apply min vals)]
        (is (>= mn 0) "all >= 0")
        (let [mean-val (/ (reduce + vals) (count vals))]
          (is (h/close? 2.33 mean-val 1.5) "mean ~ 2.33"))))))

(deftest categorical-sample-n
  (testing "categorical dist-sample-n"
    (let [logits (mx/array [0.0 0.0 0.0])
          d (dist/categorical logits)
          key (rng/fresh-key 42)
          samples (dc/dist-sample-n d key 100)]
      (mx/eval! samples)
      (is (= [100] (mx/shape samples)) "shape is [100]")
      (let [vals (mx/->clj samples)
            mn (apply min vals)
            mx-val (apply max vals)]
        (is (>= mn 0) "min >= 0")
        (is (<= mx-val 2) "max <= 2")))))

(deftest multivariate-normal-sample-n
  (testing "multivariate-normal dist-sample-n"
    (let [mu (mx/array [1.0 2.0 3.0])
          cov (mx/array [[1.0 0.0 0.0] [0.0 1.0 0.0] [0.0 0.0 1.0]])
          d (dist/multivariate-normal mu cov)
          key (rng/fresh-key 42)
          samples (dc/dist-sample-n d key 100)]
      (mx/eval! samples)
      (is (= [100 3] (mx/shape samples)) "shape is [100, 3]")
      (let [mean-vec (mx/->clj (mx/mean samples [0]))]
        (is (h/close? 1.0 (nth mean-vec 0) 0.5) "mean[0] ~ 1")
        (is (h/close? 2.0 (nth mean-vec 1) 0.5) "mean[1] ~ 2")
        (is (h/close? 3.0 (nth mean-vec 2) 0.5) "mean[2] ~ 3")))))

(deftest binomial-sample-n
  (testing "binomial dist-sample-n"
    (let [d (dist/binomial (mx/scalar 20) (mx/scalar 0.5))
          key (rng/fresh-key 42)
          samples (dc/dist-sample-n d key 100)]
      (mx/eval! samples)
      (is (= [100] (mx/shape samples)) "shape is [100]")
      (let [vals (mx/->clj samples)
            mn (apply min vals)
            mx-val (apply max vals)
            mean-val (/ (reduce + vals) (count vals))]
        (is (>= mn 0) "min >= 0")
        (is (<= mx-val 20) "max <= 20")
        (is (h/close? 10.0 mean-val 2.0) "mean ~ 10")))))

(deftest student-t-sample-n
  (testing "student-t dist-sample-n"
    (let [d (dist/student-t (mx/scalar 10) (mx/scalar 0.0) (mx/scalar 1.0))
          key (rng/fresh-key 42)
          samples (dc/dist-sample-n d key 200)]
      (mx/eval! samples)
      (is (= [200] (mx/shape samples)) "shape is [200]")
      (let [vals (mx/->clj samples)
            mean-val (/ (reduce + vals) (count vals))]
        (is (h/close? 0.0 mean-val 0.5) "mean ~ 0")))))

(cljs.test/run-tests)
