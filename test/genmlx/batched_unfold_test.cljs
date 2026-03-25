(ns genmlx.batched-unfold-test
  "Batched unfold combinator tests."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.combinators :as comb]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ar-kernel
  (gen [t state]
    (let [x (trace :x (dist/gaussian (mx/multiply state (mx/scalar 0.9))
                                      (mx/scalar 1.0)))]
      x)))

(deftest scalar-unfold-sanity
  (testing "scalar unfold sanity check"
    (let [unfold (comb/unfold-combinator (dyn/auto-key ar-kernel))
          trace (p/simulate unfold [5 (mx/scalar 0.0)])]
      (is (some? trace) "scalar unfold returns trace")
      (is (= 5 (count (:retval trace))) "scalar unfold has 5 states")
      (mx/eval! (:score trace))
      (is (js/isFinite (mx/item (:score trace))) "scalar unfold has finite score"))))

(def model-unfold
  (gen [n-steps]
    (let [unfold (comb/unfold-combinator ar-kernel)
          result (splice :temporal unfold n-steps (mx/scalar 0.0))]
      result)))

(deftest batched-unfold-vsimulate
  (testing "batched unfold via vsimulate"
    (let [key (rng/fresh-key)
          vtrace (dyn/vsimulate model-unfold [5] 100 key)]
      (is (some? vtrace) "batched unfold returns vtrace")
      (mx/eval! (:score vtrace))
      (is (= [100] (mx/shape (:score vtrace))) "score is [100]-shaped")
      (let [choices (:choices vtrace)
            step0 (cm/get-submap (cm/get-submap choices :temporal) 0)
            x0 (cm/get-value (cm/get-submap step0 :x))]
        (mx/eval! x0)
        (is (= [100] (mx/shape x0)) "step 0 :x is [100]-shaped")))))

(deftest batched-unfold-vgenerate
  (testing "batched unfold with constraints (vgenerate)"
    (let [key (rng/fresh-key)
          obs (cm/set-value
                (cm/set-value cm/EMPTY
                  [:temporal 2 :x] (mx/scalar 1.0))
                [:temporal 4 :x] (mx/scalar -0.5))
          vtrace (dyn/vgenerate model-unfold [5] obs 100 key)]
      (is (some? vtrace) "vgenerate returns vtrace")
      (mx/eval! (:score vtrace))
      (mx/eval! (:weight vtrace))
      (is (= [100] (mx/shape (:score vtrace))) "score is [100]-shaped")
      (is (= [100] (mx/shape (:weight vtrace))) "weight is [100]-shaped")
      (let [choices (:choices vtrace)
            step0 (cm/get-submap (cm/get-submap choices :temporal) 0)
            x0 (cm/get-value (cm/get-submap step0 :x))]
        (mx/eval! x0)
        (is (= [100] (mx/shape x0)) "unconstrained step 0 :x is [100]-shaped")))))

(deftest batched-unfold-score-consistency
  (testing "score consistency: batched vs N scalar runs"
    (let [key (rng/fresh-key)
          vtrace (dyn/vsimulate model-unfold [3] 50 key)
          _ (mx/eval! (:score vtrace))
          batched-mean (mx/item (mx/mean (:score vtrace)))
          scalar-unfold (comb/unfold-combinator (dyn/auto-key ar-kernel))
          scalar-scores (mapv (fn [_]
                                (let [trace (p/simulate scalar-unfold [3 (mx/scalar 0.0)])]
                                  (mx/eval! (:score trace))
                                  (mx/item (:score trace))))
                              (range 200))
          scalar-mean (/ (reduce + scalar-scores) (count scalar-scores))]
      (is (h/close? scalar-mean batched-mean 1.0) "mean scores similar"))))

(def ar-kernel-with-drift
  (gen [t state drift]
    (let [x (trace :x (dist/gaussian (mx/add (mx/multiply state (mx/scalar 0.9))
                                              drift)
                                      (mx/scalar 1.0)))]
      x)))

(def model-unfold-drift
  (gen [n-steps drift]
    (let [unfold (comb/unfold-combinator ar-kernel-with-drift)
          result (splice :temporal unfold n-steps (mx/scalar 0.0) drift)]
      result)))

(deftest multi-argument-kernel
  (testing "multi-argument kernel"
    (let [key (rng/fresh-key)
          vtrace (dyn/vsimulate model-unfold-drift [4 (mx/scalar 0.5)] 50 key)]
      (is (some? vtrace) "multi-arg batched unfold works")
      (mx/eval! (:score vtrace))
      (is (= [50] (mx/shape (:score vtrace))) "multi-arg score is [50]-shaped"))))

(cljs.test/run-tests)
