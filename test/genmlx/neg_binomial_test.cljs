(ns genmlx.neg-binomial-test
  "Tests for the negative binomial distribution."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm])
  (:require-macros [genmlx.gen :refer [gen]]))

(deftest log-prob-spot-checks
  (testing "NB(r=5, p=0.5) log-prob spot checks"
    (let [d (dist/neg-binomial 5 0.5)]
      (let [lp0 (dist/log-prob d (mx/scalar 0))]
        (mx/eval! lp0)
        (is (h/close? -3.4657 (mx/item lp0) 0.01) "log-prob at k=0"))
      (let [lp3 (dist/log-prob d (mx/scalar 3))]
        (mx/eval! lp3)
        (is (h/close? -1.989 (mx/item lp3) 0.02) "log-prob at k=3"))
      (let [lp5 (dist/log-prob d (mx/scalar 5))]
        (mx/eval! lp5)
        (is (h/close? -2.096 (mx/item lp5) 0.02) "log-prob at k=5")))))

(deftest sample-validity
  (testing "samples are non-negative integers"
    (let [d (dist/neg-binomial 5 0.5)
          samples (repeatedly 20 #(let [v (dist/sample d)]
                                     (mx/eval! v)
                                     (mx/item v)))]
      (is (every? #(>= % 0) samples) "all samples non-negative")
      (is (every? #(== % (js/Math.floor %)) samples) "all samples are integers"))))

(deftest sample-mean-test
  (testing "sample mean ~ 5.0"
    (let [d (dist/neg-binomial 5 0.5)
          n 2000
          samples (repeatedly n #(let [v (dist/sample d)]
                                    (mx/eval! v)
                                    (mx/item v)))
          mean (/ (reduce + samples) n)]
      (is (h/close? 5.0 mean 1.5) "sample mean ~ 5.0"))))

(deftest sample-variance-test
  (testing "sample variance ~ 10.0"
    (let [d (dist/neg-binomial 5 0.5)
          n 2000
          samples (vec (repeatedly n #(let [v (dist/sample d)]
                                         (mx/eval! v)
                                         (mx/item v))))
          mean (/ (reduce + samples) n)
          variance (/ (reduce + (map #(* (- % mean) (- % mean)) samples)) n)]
      (is (h/close? 10.0 variance 4.0) "sample variance ~ 10.0"))))

(deftest generate-weight
  (testing "generate weight = log-prob"
    (let [model (gen []
                  (trace :k (dist/neg-binomial 5 0.5)))
          obs (cm/choicemap :k (mx/scalar 3))
          {:keys [trace weight]} (p/generate (dyn/auto-key model) [] obs)]
      (mx/eval! weight)
      (let [w (mx/item weight)
            lp (dist/log-prob (dist/neg-binomial 5 0.5) (mx/scalar 3))]
        (mx/eval! lp)
        (is (h/close? (mx/item lp) w 0.01) "generate weight = log-prob")))))

(cljs.test/run-tests)
