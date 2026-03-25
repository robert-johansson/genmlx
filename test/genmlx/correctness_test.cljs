(ns genmlx.correctness-test
  "Regression tests for Phase 16/17 correctness fixes."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.combinators :as comb]
            [genmlx.custom-gradient :as cg]
            [genmlx.nn :as nn]
            [genmlx.diff :as diff])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; 17.3/17.4/17.5: Missing protocols on deterministic GFs
;; ---------------------------------------------------------------------------

(deftest custom-gradient-gf-protocols
  (testing "CustomGradientGF update/regenerate/project"
    (let [gf (cg/custom-gradient-gf {:forward (fn [x] (mx/multiply x (mx/scalar 2.0)))
                                      :has-argument-grads [true]})
          trace (p/simulate gf [(mx/scalar 3.0)])]
      ;; update
      (let [{:keys [trace weight discard]} (p/update gf trace cm/EMPTY)]
        (mx/eval! weight)
        (is (instance? tr/Trace trace) "update returns trace")
        (is (h/close? 0.0 (mx/item weight) 1e-6) "update weight = 0")
        (is (= discard cm/EMPTY) "update discard is empty"))
      ;; regenerate
      (let [{:keys [trace weight]} (p/regenerate gf trace sel/all)]
        (mx/eval! weight)
        (is (instance? tr/Trace trace) "regenerate returns trace")
        (is (h/close? 0.0 (mx/item weight) 1e-6) "regenerate weight = 0"))
      ;; project
      (let [w (p/project gf trace sel/all)]
        (mx/eval! w)
        (is (h/close? 0.0 (mx/item w) 1e-6) "project = 0")))))

(deftest neural-net-gf-protocols
  (testing "NeuralNetGF update/regenerate/project"
    (let [module (nn/linear 2 1)
          _ (mx/eval! module)
          gf (nn/nn->gen-fn module)
          trace (p/simulate gf [(mx/ones [2])])]
      ;; update
      (let [{:keys [trace weight discard]} (p/update gf trace cm/EMPTY)]
        (mx/eval! weight)
        (is (instance? tr/Trace trace) "nn update returns trace")
        (is (h/close? 0.0 (mx/item weight) 1e-6) "nn update weight = 0")
        (is (= discard cm/EMPTY) "nn update discard is empty"))
      ;; regenerate
      (let [{:keys [trace weight]} (p/regenerate gf trace sel/all)]
        (mx/eval! weight)
        (is (instance? tr/Trace trace) "nn regenerate returns trace")
        (is (h/close? 0.0 (mx/item weight) 1e-6) "nn regenerate weight = 0"))
      ;; project
      (let [w (p/project gf trace sel/all)]
        (mx/eval! w)
        (is (h/close? 0.0 (mx/item w) 1e-6) "nn project = 0")))))

;; ---------------------------------------------------------------------------
;; 17.6: MixCombinator IUpdateWithDiffs
;; ---------------------------------------------------------------------------

(deftest mix-combinator-update-with-diffs
  (testing "MixCombinator update-with-diffs"
    (let [comp1 (gen [x] (trace :v (dist/gaussian (mx/scalar 0.0) 1)))
          comp2 (gen [x] (trace :v (dist/gaussian (mx/scalar 10.0) 1)))
          mix (comb/mix-combinator [(dyn/auto-key comp1) (dyn/auto-key comp2)]
                                    (mx/log (mx/array [0.5 0.5])))
          trace (p/simulate mix [(mx/scalar 0.0)])]
      ;; No-change path
      (let [{:keys [trace weight discard]}
            (p/update-with-diffs mix trace cm/EMPTY diff/no-change)]
        (mx/eval! weight)
        (is (some? trace) "mix no-change returns same trace")
        (is (h/close? 0.0 (mx/item weight) 1e-6) "mix no-change weight = 0")
        (is (= discard cm/EMPTY) "mix no-change discard empty"))
      ;; With constraints
      (let [constraints (cm/choicemap :v (mx/scalar 5.0))
            {:keys [trace weight]}
            (p/update-with-diffs mix trace constraints diff/no-change)]
        (mx/eval! weight)
        (is (some? trace) "mix with-diffs + constraints works")))))

;; ---------------------------------------------------------------------------
;; 16.1: Combinator sub-trace score propagation
;; ---------------------------------------------------------------------------

(deftest map-combinator-sub-trace-scores
  (testing "Map combinator sub-trace scores"
    (let [kernel (gen [x]
                   (let [v (trace :v (dist/gaussian x 1))]
                     v))
          mapper (comb/map-combinator (dyn/auto-key kernel))
          constraints (-> cm/EMPTY
                          (cm/set-choice [0 :v] (mx/scalar 1.0))
                          (cm/set-choice [1 :v] (mx/scalar 2.0)))
          {:keys [trace]} (p/generate mapper [[0.0 0.0]] constraints)]
      (is (some? (:genmlx.combinators/element-scores (meta trace)))
          "map trace has element-scores metadata")
      ;; Update with constraint on element 1 only
      (let [update-constraints (cm/set-choice cm/EMPTY [1 :v] (mx/scalar 3.0))
            {:keys [trace weight]} (p/update mapper trace update-constraints)]
        (mx/eval! weight)
        (is (h/close? -2.5 (mx/item weight) 0.1) "map update weight correct"))
      ;; Regenerate with selection on element 0
      (let [{:keys [weight]} (p/regenerate mapper trace (sel/select 0))]
        (mx/eval! weight)
        (is (js/isFinite (mx/item weight)) "map regenerate weight is finite")))))

(deftest unfold-combinator-sub-trace-scores
  (testing "Unfold combinator sub-trace scores"
    (let [kernel (gen [t state]
                   (let [v (trace :v (dist/gaussian state 1))]
                     (mx/eval! v)
                     (mx/item v)))
          unfolder (comb/unfold-combinator (dyn/auto-key kernel))
          constraints (-> cm/EMPTY
                          (cm/set-choice [0 :v] (mx/scalar 1.0))
                          (cm/set-choice [1 :v] (mx/scalar 2.0)))
          {:keys [trace]} (p/generate unfolder [2 0.0] constraints)]
      (is (some? (:genmlx.combinators/step-scores (meta trace)))
          "unfold trace has step-scores metadata")
      ;; Update step 1
      (let [update-constraints (cm/set-choice cm/EMPTY [1 :v] (mx/scalar 3.0))
            {:keys [weight]} (p/update unfolder trace update-constraints)]
        (mx/eval! weight)
        (is (js/isFinite (mx/item weight)) "unfold update weight is finite")))))

;; ---------------------------------------------------------------------------
;; 16.3: assess validates all choices constrained
;; ---------------------------------------------------------------------------

(deftest assess-validates-all-choices
  (testing "assess validates all choices"
    (let [model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/gaussian 0 1))
                        y (trace :y (dist/gaussian 0 1))]
                    [x y])))]
      ;; Complete choices: should succeed
      (let [choices (cm/choicemap :x (mx/scalar 1.0)
                                  :y (mx/scalar 2.0))
            {:keys [retval weight]} (p/assess model [] choices)]
        (mx/eval! weight)
        (is (number? (mx/item weight)) "assess with all choices succeeds")
        (let [expected (+ (* -0.5 (+ 1.0 (js/Math.log (* 2 js/Math.PI))))
                          (* -0.5 (+ 4.0 (js/Math.log (* 2 js/Math.PI)))))]
          (is (h/close? expected (mx/item weight) 0.01) "assess weight = sum of log-probs")))
      ;; Missing choice: should throw
      (is (thrown? js/Error (p/assess model [] (cm/choicemap :x (mx/scalar 1.0))))
          "assess with missing choice throws"))))

;; ---------------------------------------------------------------------------
;; 16.2: Splice score tracking
;; ---------------------------------------------------------------------------

(deftest splice-score-tracking
  (testing "splice score tracking"
    (let [sub-model (gen [mu]
                      (trace :z (dist/gaussian mu 1)))
          outer-model (dyn/auto-key (gen []
                        (let [x (trace :x (dist/gaussian 0 10))]
                          (splice :sub sub-model x)
                          x)))
          constraints (-> (cm/choicemap :x (mx/scalar 2.0))
                          (cm/set-choice [:sub :z] (mx/scalar 3.0)))
          {:keys [trace]} (p/generate outer-model [] constraints)]
      (is (some? (:genmlx.dynamic/splice-scores (meta trace)))
          "trace has splice-scores metadata")
      ;; Regenerate the :x address
      (let [{:keys [trace weight]} (p/regenerate outer-model trace (sel/select :x))]
        (mx/eval! weight)
        (is (js/isFinite (mx/item weight)) "splice regenerate weight is finite")
        (is (not (js/isNaN (mx/item weight))) "splice regenerate weight is not NaN")))))

(cljs.test/run-tests)
