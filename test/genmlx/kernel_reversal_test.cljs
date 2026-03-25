(ns genmlx.kernel-reversal-test
  "Tests for kernel reversal declarations and auto-reversal of composites."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.selection :as sel]
            [genmlx.inference.kernel :as kern])
  (:require-macros [genmlx.gen :refer [gen]]))

(deftest with-reversal-test
  (testing "with-reversal"
    (let [k-fwd (fn [trace key] trace)
          k-bwd (fn [trace key] trace)
          k (kern/with-reversal k-fwd k-bwd)]
      (is (some? (kern/reversal k)) "reversal of forward is backward")
      (is (fn? (kern/reversal k)) "reversal is a function")
      (is (not (kern/symmetric? k)) "not marked symmetric"))))

(deftest symmetric-kernel-test
  (testing "symmetric-kernel"
    (let [k-raw (fn [trace key] trace)
          k (kern/symmetric-kernel k-raw)]
      (is (kern/symmetric? k) "symmetric? returns true")
      (is (some? (kern/reversal k)) "reversal exists")
      (is (fn? (kern/reversal k)) "reversal is a function")
      (is (kern/symmetric? (kern/reversal k)) "reversal of symmetric-kernel is symmetric")
      (is (identical? k (kern/reversal k)) "reversal returns the decorated kernel itself (identical?)"))))

(deftest double-reversal-test
  (testing "with-reversal: double reversal"
    (let [k-fwd (fn [trace key] trace)
          k-bwd (fn [trace key] trace)
          k (kern/with-reversal k-fwd k-bwd)
          rev (kern/reversal k)
          rev-rev (kern/reversal rev)]
      (is (fn? rev-rev) "double reversal is callable")
      (is (some? (kern/reversal rev-rev)) "double reversal has a reversal itself"))))

(deftest reversed-throws-test
  (testing "reversed throws on undecorated"
    (let [k (fn [trace key] trace)
          threw? (try
                   (kern/reversed k)
                   false
                   (catch :default _ true))]
      (is threw? "throws on kernel without reversal"))))

(deftest reversed-returns-test
  (testing "reversed returns reversal"
    (let [k-fwd (fn [trace key] trace)
          k-bwd (fn [trace key] trace)
          k (kern/with-reversal k-fwd k-bwd)]
      (is (fn? (kern/reversed k)) "reversed returns the backward kernel"))))

(deftest builtin-symmetric-test
  (testing "built-in kernels are symmetric"
    (let [k (kern/mh-kernel (sel/select :x))]
      (is (kern/symmetric? k) "mh-kernel is symmetric"))
    (let [k (kern/prior :x)]
      (is (kern/symmetric? k) "prior is symmetric"))
    (let [k (kern/random-walk :x 0.5)]
      (is (kern/symmetric? k) "random-walk (single addr) is symmetric"))))

(deftest chain-reversal-test
  (testing "composite reversal: chain"
    (let [k1 (kern/symmetric-kernel (fn [t k] t))
          k2 (kern/symmetric-kernel (fn [t k] t))
          k3 (kern/symmetric-kernel (fn [t k] t))
          composed (kern/chain k1 k2 k3)]
      (is (some? (kern/reversal composed))
          "chain of symmetric kernels has reversal"))

    (let [k1 (fn [t k] t)
          k2 (kern/symmetric-kernel (fn [t k] t))
          composed (kern/chain k1 k2)]
      (is (nil? (kern/reversal composed))
          "chain with un-decorated kernel has no reversal"))))

(deftest repeat-kernel-reversal-test
  (testing "composite reversal: repeat-kernel"
    (let [k (kern/symmetric-kernel (fn [t key] t))
          repeated (kern/repeat-kernel 5 k)]
      (is (some? (kern/reversal repeated))
          "repeat of symmetric kernel has reversal"))

    (let [k (fn [t key] t)
          repeated (kern/repeat-kernel 5 k)]
      (is (nil? (kern/reversal repeated))
          "repeat of undecorated kernel has no reversal"))))

(deftest cycle-kernels-reversal-test
  (testing "composite reversal: cycle-kernels"
    (let [k1 (kern/symmetric-kernel (fn [t k] t))
          k2 (kern/symmetric-kernel (fn [t k] t))
          cycled (kern/cycle-kernels 6 [k1 k2])]
      (is (some? (kern/reversal cycled))
          "cycle of symmetric kernels has reversal"))))

(deftest mix-kernels-reversal-test
  (testing "composite reversal: mix-kernels"
    (let [k1 (kern/symmetric-kernel (fn [t k] t))
          k2 (kern/symmetric-kernel (fn [t k] t))
          mixed (kern/mix-kernels [[k1 0.5] [k2 0.5]])]
      (is (some? (kern/reversal mixed))
          "mix of symmetric kernels has reversal"))))

(deftest seed-reversal-test
  (testing "composite reversal: seed"
    (let [k (kern/symmetric-kernel (fn [t key] t))
          seeded (kern/seed k (rng/fresh-key))]
      (is (some? (kern/reversal seeded))
          "seed of symmetric kernel has reversal"))))

(deftest reversed-kernel-valid-traces-test
  (testing "reversed kernel produces valid traces"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 1))]
                    x))
          model (dyn/auto-key model)
          trace (:trace (p/generate model [] (cm/choicemap :x (mx/scalar 1.0))))
          k (kern/mh-kernel (sel/select :x))
          rev-k (kern/reversed k)
          key (rng/fresh-key)
          result-trace (rev-k trace key)]
      (is (some? (:gen-fn result-trace))
          "reversed kernel returns a trace with gen-fn")
      (is (some? (:choices result-trace))
          "reversed kernel returns a trace with choices"))))

(deftest run-kernel-reversed-symmetric-test
  (testing "run-kernel with reversed symmetric kernel (10 samples)"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 10))]
                    x))
          model (dyn/auto-key model)
          init-trace (:trace (p/generate model [] (cm/choicemap :x (mx/scalar 0.0))))
          sk (kern/mh-kernel (sel/select :x))
          rev (kern/reversal sk)]
      (is (kern/symmetric? sk) "mh-kernel is symmetric")
      (is (kern/symmetric? rev) "reversal of mh-kernel is symmetric")
      (let [samples (kern/run-kernel {:samples 10 :burn 5} rev init-trace)]
        (is (= 10 (count samples))
            "run-kernel with reversed symmetric kernel produces 10 samples")))))

(cljs.test/run-tests)
