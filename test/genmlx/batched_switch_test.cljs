(ns genmlx.batched-switch-test
  "Batched switch combinator tests."
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

(def branch-low
  (gen []
    (let [x (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))]
      x)))

(def branch-high
  (gen []
    (let [x (trace :x (dist/gaussian (mx/scalar 10.0) (mx/scalar 1.0)))]
      x)))

(deftest scalar-switch-sanity
  (testing "scalar switch sanity check"
    (let [sw (comb/switch-combinator (dyn/auto-key branch-low) (dyn/auto-key branch-high))
          trace0 (p/simulate sw [0])
          trace1 (p/simulate sw [1])]
      (mx/eval! (:retval trace0))
      (mx/eval! (:retval trace1))
      (is (< (js/Math.abs (mx/item (:retval trace0))) 5) "branch 0 returns value near 0")
      (is (< (js/Math.abs (- 10 (mx/item (:retval trace1)))) 5) "branch 1 returns value near 10"))))

(def model-switch
  (gen [index]
    (let [sw (comb/switch-combinator branch-low branch-high)
          result (splice :choice sw index)]
      result)))

(deftest batched-switch-vsimulate
  (testing "batched switch via vsimulate"
    (let [key (rng/fresh-key)
          index (mx/array (vec (concat (repeat 50 0) (repeat 50 1))) mx/int32)
          vtrace (dyn/vsimulate model-switch [index] 100 key)]
      (is (some? vtrace) "batched switch returns vtrace")
      (mx/eval! (:score vtrace))
      (is (= [100] (mx/shape (:score vtrace))) "score is [100]-shaped")
      (let [choices (:choices vtrace)
            x-vals (cm/get-value (cm/get-submap (cm/get-submap choices :choice) :x))]
        (mx/eval! x-vals)
        (is (= [100] (mx/shape x-vals)) ":x is [100]-shaped")
        (let [first-half-mean (mx/item (mx/mean (mx/slice x-vals 0 50)))
              second-half-mean (mx/item (mx/mean (mx/slice x-vals 50 100)))]
          (is (< (js/Math.abs first-half-mean) 3) "branch 0 particles near 0")
          (is (< (js/Math.abs (- 10 second-half-mean)) 3) "branch 1 particles near 10"))))))

(deftest batched-switch-vgenerate
  (testing "batched switch with constraints (vgenerate)"
    (let [key (rng/fresh-key)
          index (mx/array (vec (concat (repeat 50 0) (repeat 50 1))) mx/int32)
          obs (cm/set-value cm/EMPTY [:choice :x] (mx/scalar 5.0))
          vtrace (dyn/vgenerate model-switch [index] obs 100 key)]
      (is (some? vtrace) "vgenerate returns vtrace")
      (mx/eval! (:score vtrace))
      (mx/eval! (:weight vtrace))
      (is (= [100] (mx/shape (:score vtrace))) "score is [100]-shaped")
      (is (= [100] (mx/shape (:weight vtrace))) "weight is [100]-shaped"))))

(def model-mixture
  (gen []
    (let [z (trace :z (dist/bernoulli (mx/scalar 0.5)))
          idx (mx/multiply z (mx/scalar 1 mx/int32))
          sw (comb/switch-combinator branch-low branch-high)
          result (splice :comp sw idx)]
      result)))

(deftest random-per-particle-branch
  (testing "random per-particle branch selection"
    (let [key (rng/fresh-key)
          vtrace (dyn/vsimulate model-mixture [] 200 key)]
      (is (some? vtrace) "random branch vtrace exists")
      (mx/eval! (:score vtrace))
      (is (= [200] (mx/shape (:score vtrace))) "score is [200]-shaped")
      (let [choices (:choices vtrace)
            x-vals (cm/get-value (cm/get-submap (cm/get-submap choices :comp) :x))]
        (mx/eval! x-vals)
        (let [overall-mean (mx/item (mx/mean x-vals))]
          (is (< (js/Math.abs (- 5 overall-mean)) 3) "mixture mean near 5"))))))

(cljs.test/run-tests)
