(ns genmlx.bugfix-test
  "Tests for three bug fixes:
   1. Update weight correctness for dependent variables
   2. Vectorized switch produces distinct samples
   3. Conditional SMC uses the reference trace"
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.combinators :as comb]
            [genmlx.inference.smc :as smc]))

(deftest update-weight-dependent-variables
  (testing "update weight: dependent variables"
    (let [model (dyn/auto-key (gen [obs-val]
                  (let [mu (trace :mu (dist/gaussian 0 10))]
                    (trace :obs (dist/gaussian mu 1))
                    mu)))
          obs-val 5.0
          init-constraints (cm/choicemap :mu (mx/scalar 0.0) :obs (mx/scalar obs-val))
          {:keys [trace]} (p/generate model [obs-val] init-constraints)
          old-score (:score trace)
          new-constraints (cm/choicemap :mu (mx/scalar 5.0))
          {:keys [trace weight]} (p/update model trace new-constraints)
          new-score (:score trace)
          expected-weight (- (mx/realize new-score) (mx/realize old-score))
          actual-weight (mx/realize weight)]
      (is (h/close? expected-weight actual-weight 0.001) "update weight = new_score - old_score")
      (is (> actual-weight 0) "update weight > 0 (mu moved toward obs)"))))

(deftest update-weight-map-combinator
  (testing "update weight: MapCombinator"
    (let [kernel (gen [x]
                  (let [mu (trace :mu (dist/gaussian 0 10))]
                    (trace :obs (dist/gaussian mu 1))
                    mu))
          model (comb/map-combinator (dyn/auto-key kernel))
          args [[0.0 0.0]]
          constraints (cm/choicemap
                        0 (cm/choicemap :mu (mx/scalar 1.0) :obs (mx/scalar 5.0))
                        1 (cm/choicemap :mu (mx/scalar 2.0) :obs (mx/scalar 5.0)))
          {:keys [trace]} (p/generate model args constraints)
          old-score (:score trace)
          new-constraints (cm/choicemap
                            0 (cm/choicemap :mu (mx/scalar 5.0))
                            1 (cm/choicemap :mu (mx/scalar 5.0)))
          {:keys [trace weight]} (p/update model trace new-constraints)
          new-score (:score trace)
          expected-weight (- (mx/realize new-score) (mx/realize old-score))
          actual-weight (mx/realize weight)]
      (is (h/close? expected-weight actual-weight 0.001) "map update weight = new_score - old_score")
      (is (> actual-weight 0) "map update weight > 0 (mu moved toward obs)"))))

(deftest update-weight-scan-combinator
  (testing "update weight: ScanCombinator"
    (let [kernel (gen [carry input]
                  (let [x (trace :x (dist/gaussian carry 1))]
                    (trace :obs (dist/gaussian x 0.5))
                    [x x]))
          model (comb/scan-combinator (dyn/auto-key kernel))
          args [(mx/scalar 0.0) [1 2]]
          constraints (cm/choicemap
                        0 (cm/choicemap :x (mx/scalar 1.0) :obs (mx/scalar 3.0))
                        1 (cm/choicemap :x (mx/scalar 2.0) :obs (mx/scalar 3.0)))
          {:keys [trace]} (p/generate model args constraints)
          old-score (:score trace)
          new-constraints (cm/choicemap
                            0 (cm/choicemap :x (mx/scalar 3.0)))
          {:keys [trace weight]} (p/update model trace new-constraints)
          new-score (:score trace)
          expected-weight (- (mx/realize new-score) (mx/realize old-score))
          actual-weight (mx/realize weight)]
      (is (h/close? expected-weight actual-weight 0.001) "scan update weight = new_score - old_score"))))

(deftest vectorized-switch-distinct-samples
  (testing "vectorized switch: distinct samples"
    (let [branches [(dist/gaussian 0 1) (dist/gaussian 10 1)]
          n 20
          index (mx/zeros [n] mx/int32)
          result (comb/vectorized-switch branches index [])
          values (cm/get-value (:choices result))
          vals-list (mx/->clj values)]
      (is (= (mx/shape values) [n]) "vectorized switch: values are [N]-shaped")
      (let [unique-count (count (set vals-list))]
        (is (> unique-count 1) "vectorized switch: values are distinct (not identical)"))
      (let [mean-val (mx/realize (mx/mean values))]
        (is (< (js/Math.abs mean-val) 2.0) "vectorized switch: mean near 0 for branch 0")))))

(deftest vectorized-switch-mixed-branches
  (testing "vectorized switch: mixed branch selection"
    (let [branches [(dist/gaussian 0 0.1) (dist/gaussian 100 0.1)]
          n 10
          index (mx/array [0 0 0 0 0 1 1 1 1 1] mx/int32)
          result (comb/vectorized-switch branches index [])
          values (cm/get-value (:choices result))
          vals-list (mx/->clj values)
          branch0-vals (take 5 vals-list)
          branch1-vals (drop 5 vals-list)]
      (is (< (/ (reduce + branch0-vals) 5) 5.0) "branch 0 values near 0")
      (is (> (/ (reduce + branch1-vals) 5) 90.0) "branch 1 values near 100")
      (is (= (mx/shape (:score result)) [n]) "vectorized switch: scores are [N]-shaped"))))

(deftest csmc-reference-trace
  (testing "conditional SMC: reference trace used"
    (let [model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/gaussian 0 10))]
                    (trace :obs (dist/gaussian x 1))
                    x)))
          ref-x 42.0
          ref-constraints (cm/choicemap :x (mx/scalar ref-x) :obs (mx/scalar 42.5))
          {:keys [trace]} (p/generate model [] ref-constraints)
          reference-trace trace
          obs (cm/choicemap :obs (mx/scalar 42.5))
          result (smc/csmc {:particles 10 :key (rng/fresh-key)}
                           model [] [obs] reference-trace)
          ref-particle-trace (first (:traces result))
          ref-x-val (mx/realize (cm/get-choice (:choices ref-particle-trace) [:x]))]
      (is (h/close? ref-x ref-x-val 0.001) "cSMC: reference particle x = 42.0")
      (let [other-xs (mapv (fn [t]
                             (mx/realize (cm/get-choice (:choices t) [:x])))
                           (rest (:traces result)))
            different-count (count (filter #(> (js/Math.abs (- % ref-x)) 5) other-xs))]
        (is (> different-count 0) "cSMC: other particles differ from reference")))))

(cljs.test/run-tests)
