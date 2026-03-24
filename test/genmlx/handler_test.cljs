(ns genmlx.handler-test
  "Handler state-transition tests: simulate and generate via runtime."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.handler :as hdlr]
            [genmlx.runtime :as rt]
            [genmlx.choicemap :as cm]
            [genmlx.mlx.random :as rng]))

(deftest simulate-handler-test
  (testing "simulate handler"
    (let [key (rng/fresh-key 42)
          result (rt/run-handler hdlr/simulate-transition
                   {:choices cm/EMPTY :score (mx/scalar 0.0) :key key
                    :executor nil}
                   (fn [rt]
                     (let [trace (.-trace rt)
                           x (trace :x (dist/gaussian 0 1))
                           y (trace :y (dist/gaussian 0 1))]
                       (mx/eval! x y)
                       (+ (mx/item x) (mx/item y)))))]
      (is (number? (:retval result)) "simulate returns retval")
      (is (cm/has-value? (cm/get-submap (:choices result) :x)) "simulate has choices for :x")
      (is (cm/has-value? (cm/get-submap (:choices result) :y)) "simulate has choices for :y")
      (let [score (:score result)]
        (mx/eval! score)
        (is (< (mx/item score) 0) "simulate has negative score")))))

(deftest generate-handler-test
  (testing "generate handler with constraints"
    (let [key (rng/fresh-key 42)
          constraints (cm/choicemap :x (mx/scalar 1.5))
          result (rt/run-handler hdlr/generate-transition
                   {:choices cm/EMPTY :score (mx/scalar 0.0)
                    :weight (mx/scalar 0.0)
                    :key key :constraints constraints
                    :executor nil}
                   (fn [rt]
                     (let [trace (.-trace rt)
                           x (trace :x (dist/gaussian 0 1))]
                       (mx/eval! x)
                       (mx/item x))))]
      (let [x-val (cm/get-value (cm/get-submap (:choices result) :x))]
        (mx/eval! x-val)
        (is (h/close? 1.5 (mx/item x-val) 0.001) "generate constrains x"))
      (let [weight (:weight result)]
        (mx/eval! weight)
        (is (not= 0 (mx/item weight)) "generate has nonzero weight")))))

(cljs.test/run-tests)
