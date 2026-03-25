(ns genmlx.batched-scan-test
  "Batched scan combinator tests."
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

(def scan-kernel
  (gen [carry input]
    (let [x (trace :x (dist/gaussian (mx/add carry input) (mx/scalar 1.0)))]
      [x (mx/multiply x (mx/scalar 2.0))])))

(def model-scan
  (gen [inputs]
    (let [sc (comb/scan-combinator scan-kernel)
          result (splice :scan sc (mx/scalar 0.0) inputs)]
      result)))

(deftest batched-scan-vsimulate
  (testing "batched scan via vsimulate"
    (let [key (rng/fresh-key)
          inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
          vtrace (dyn/vsimulate model-scan [inputs] 100 key)]
      (is (some? vtrace) "batched scan returns vtrace")
      (mx/eval! (:score vtrace))
      (is (= [100] (mx/shape (:score vtrace))) "score is [100]-shaped")
      (let [choices (:choices vtrace)
            step0 (cm/get-submap (cm/get-submap choices :scan) 0)
            x0 (cm/get-value (cm/get-submap step0 :x))]
        (mx/eval! x0)
        (is (= [100] (mx/shape x0)) "step 0 :x is [100]-shaped")))))

(deftest batched-scan-vgenerate
  (testing "batched scan with constraints"
    (let [key (rng/fresh-key)
          inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
          obs (cm/set-value cm/EMPTY [:scan 1 :x] (mx/scalar 5.0))
          vtrace (dyn/vgenerate model-scan [inputs] obs 100 key)]
      (is (some? vtrace) "vgenerate returns vtrace")
      (mx/eval! (:weight vtrace))
      (is (= [100] (mx/shape (:weight vtrace))) "weight is [100]-shaped"))))

(deftest batched-scan-score-consistency
  (testing "score consistency"
    (let [key (rng/fresh-key)
          inputs [(mx/scalar 1.0) (mx/scalar 2.0)]
          vtrace (dyn/vsimulate model-scan [inputs] 50 key)
          _ (mx/eval! (:score vtrace))
          batched-mean (mx/item (mx/mean (:score vtrace)))
          scalar-scores (mapv (fn [_]
                                (let [sc (comb/scan-combinator (dyn/auto-key scan-kernel))
                                      trace (p/simulate sc [(mx/scalar 0.0) inputs])]
                                  (mx/eval! (:score trace))
                                  (mx/item (:score trace))))
                              (range 200))
          scalar-mean (/ (reduce + scalar-scores) (count scalar-scores))]
      (is (< (js/Math.abs (- batched-mean scalar-mean)) 1.5) "means similar"))))

(cljs.test/run-tests)
