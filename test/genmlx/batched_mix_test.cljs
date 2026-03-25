(ns genmlx.batched-mix-test
  "Batched mix combinator tests."
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

(def comp-low
  (gen [x]
    (let [y (trace :y (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))]
      y)))

(def comp-high
  (gen [x]
    (let [y (trace :y (dist/gaussian (mx/scalar 10.0) (mx/scalar 1.0)))]
      y)))

(deftest scalar-mix-sanity
  (testing "scalar mix sanity check"
    (let [log-w (mx/array [0.0 0.0])
          mix (comb/mix-combinator [(dyn/auto-key comp-low) (dyn/auto-key comp-high)]
                                    log-w)
          trace (p/simulate mix [(mx/scalar 0.0)])]
      (mx/eval! (:retval trace))
      (mx/eval! (:score trace))
      (is (some? trace) "scalar mix returns trace")
      (is (js/isFinite (mx/item (:score trace))) "score is finite"))))

(def model-mix
  (gen [x]
    (let [log-w (mx/array [0.0 0.0])
          mix (comb/mix-combinator [comp-low comp-high] log-w)
          result (splice :mixture mix x)]
      result)))

(deftest batched-mix-vsimulate
  (testing "batched mix via vsimulate"
    (let [key (rng/fresh-key)
          vtrace (dyn/vsimulate model-mix [(mx/scalar 0.0)] 200 key)]
      (is (some? vtrace) "batched mix returns vtrace")
      (mx/eval! (:score vtrace))
      (is (= [200] (mx/shape (:score vtrace))) "score is [200]-shaped")
      (let [choices (:choices vtrace)
            mix-sub (cm/get-submap choices :mixture)
            y-vals (cm/get-value (cm/get-submap mix-sub :y))
            idx-vals (cm/get-value (cm/get-submap mix-sub :component-idx))]
        (mx/eval! y-vals)
        (mx/eval! idx-vals)
        (is (= [200] (mx/shape y-vals)) ":y is [200]-shaped")
        (is (= [200] (mx/shape idx-vals)) ":component-idx is [200]-shaped")
        (let [overall-mean (mx/item (mx/mean y-vals))]
          (is (< (js/Math.abs (- 5 overall-mean)) 3) "mixture mean near 5"))))))

(deftest batched-mix-vgenerate
  (testing "batched mix with constraints (vgenerate)"
    (let [key (rng/fresh-key)
          obs (cm/set-value cm/EMPTY [:mixture :y] (mx/scalar 5.0))
          vtrace (dyn/vgenerate model-mix [(mx/scalar 0.0)] obs 100 key)]
      (is (some? vtrace) "vgenerate returns vtrace")
      (mx/eval! (:score vtrace))
      (mx/eval! (:weight vtrace))
      (is (= [100] (mx/shape (:score vtrace))) "score is [100]-shaped")
      (is (= [100] (mx/shape (:weight vtrace))) "weight is [100]-shaped"))))

(deftest batched-mix-score-consistency
  (testing "score consistency: batched vs scalar"
    (let [key (rng/fresh-key)
          vtrace (dyn/vsimulate model-mix [(mx/scalar 0.0)] 100 key)
          _ (mx/eval! (:score vtrace))
          batched-mean (mx/item (mx/mean (:score vtrace)))
          log-w (mx/array [0.0 0.0])
          mix (comb/mix-combinator [(dyn/auto-key comp-low) (dyn/auto-key comp-high)]
                                    log-w)
          scalar-scores (mapv (fn [_]
                                (let [trace (p/simulate mix [(mx/scalar 0.0)])]
                                  (mx/eval! (:score trace))
                                  (mx/item (:score trace))))
                              (range 200))
          scalar-mean (/ (reduce + scalar-scores) (count scalar-scores))]
      (is (h/close? scalar-mean batched-mean 1.5) "mean scores similar"))))

(def model-mix-weighted
  (gen [x]
    (let [log-w (mx/array [(js/Math.log 0.9) (js/Math.log 0.1)])
          mix (comb/mix-combinator [comp-low comp-high] log-w)
          result (splice :mixture mix x)]
      result)))

(deftest batched-mix-unequal-weights
  (testing "unequal weights"
    (let [key (rng/fresh-key)
          vtrace (dyn/vsimulate model-mix-weighted [(mx/scalar 0.0)] 500 key)]
      (mx/eval! (:score vtrace))
      (let [choices (:choices vtrace)
            mix-sub (cm/get-submap choices :mixture)
            y-vals (cm/get-value (cm/get-submap mix-sub :y))]
        (mx/eval! y-vals)
        (let [overall-mean (mx/item (mx/mean y-vals))]
          (is (< (js/Math.abs (- 1.0 overall-mean)) 3) "weighted mean near 1"))))))

(cljs.test/run-tests)
