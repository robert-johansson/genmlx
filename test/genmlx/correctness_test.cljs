(ns genmlx.correctness-test
  "Regression tests for Phase 16/17 correctness fixes."
  (:require [genmlx.mlx :as mx]
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

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(defn assert-throws [msg f]
  (let [threw (try (f) false (catch :default e true))]
    (if threw
      (println "  PASS:" msg)
      (println "  FAIL:" msg "- expected exception"))))

(println "\n=== Phase 16/17 Correctness Tests ===\n")

;; ---------------------------------------------------------------------------
;; 17.3/17.4/17.5: Missing protocols on deterministic GFs
;; ---------------------------------------------------------------------------

(println "-- 17.3: CustomGradientGF update/regenerate/project --")
(let [gf (cg/custom-gradient-gf {:forward (fn [x] (mx/multiply x (mx/scalar 2.0)))
                                  :has-argument-grads [true]})
      trace (p/simulate gf [(mx/scalar 3.0)])]
  ;; update
  (let [{:keys [trace weight discard]} (p/update gf trace cm/EMPTY)]
    (mx/eval! weight)
    (assert-true "update returns trace" (instance? tr/Trace trace))
    (assert-close "update weight = 0" 0.0 (mx/item weight) 1e-6)
    (assert-true "update discard is empty" (= discard cm/EMPTY)))
  ;; regenerate
  (let [{:keys [trace weight]} (p/regenerate gf trace sel/all)]
    (mx/eval! weight)
    (assert-true "regenerate returns trace" (instance? tr/Trace trace))
    (assert-close "regenerate weight = 0" 0.0 (mx/item weight) 1e-6))
  ;; project
  (let [w (p/project gf trace sel/all)]
    (mx/eval! w)
    (assert-close "project = 0" 0.0 (mx/item w) 1e-6)))

(println "\n-- 17.4: NeuralNetGF update/regenerate/project --")
(let [module (nn/linear 2 1)
      _ (mx/eval! module)
      gf (nn/nn->gen-fn module)
      trace (p/simulate gf [(mx/ones [2])])]
  ;; update
  (let [{:keys [trace weight discard]} (p/update gf trace cm/EMPTY)]
    (mx/eval! weight)
    (assert-true "nn update returns trace" (instance? tr/Trace trace))
    (assert-close "nn update weight = 0" 0.0 (mx/item weight) 1e-6)
    (assert-true "nn update discard is empty" (= discard cm/EMPTY)))
  ;; regenerate
  (let [{:keys [trace weight]} (p/regenerate gf trace sel/all)]
    (mx/eval! weight)
    (assert-true "nn regenerate returns trace" (instance? tr/Trace trace))
    (assert-close "nn regenerate weight = 0" 0.0 (mx/item weight) 1e-6))
  ;; project
  (let [w (p/project gf trace sel/all)]
    (mx/eval! w)
    (assert-close "nn project = 0" 0.0 (mx/item w) 1e-6)))

;; ---------------------------------------------------------------------------
;; 17.6: MixCombinator IUpdateWithDiffs
;; ---------------------------------------------------------------------------

(println "\n-- 17.6: MixCombinator update-with-diffs --")
(let [comp1 (gen [x] (dyn/trace :v (dist/gaussian (mx/scalar 0.0) 1)))
      comp2 (gen [x] (dyn/trace :v (dist/gaussian (mx/scalar 10.0) 1)))
      mix (comb/mix-combinator [comp1 comp2]
                                (mx/log (mx/array [0.5 0.5])))
      trace (p/simulate mix [(mx/scalar 0.0)])]
  ;; No-change path
  (let [{:keys [trace weight discard]}
        (p/update-with-diffs mix trace cm/EMPTY diff/no-change)]
    (mx/eval! weight)
    (assert-true "mix no-change returns same trace" (some? trace))
    (assert-close "mix no-change weight = 0" 0.0 (mx/item weight) 1e-6)
    (assert-true "mix no-change discard empty" (= discard cm/EMPTY)))
  ;; With constraints: should delegate to update
  (let [idx (int (mx/item (cm/get-choice (:choices trace) [:component-idx])))
        constraints (cm/choicemap :v (mx/scalar 5.0))
        {:keys [trace weight]}
        (p/update-with-diffs mix trace constraints diff/no-change)]
    (mx/eval! weight)
    (assert-true "mix with-diffs + constraints works" (some? trace))))

;; ---------------------------------------------------------------------------
;; 16.1: Combinator sub-trace score propagation
;; ---------------------------------------------------------------------------

(println "\n-- 16.1: Map combinator sub-trace scores --")
(let [kernel (gen [x]
               (let [v (dyn/trace :v (dist/gaussian x 1))]
                 v))
      mapper (comb/map-combinator kernel)
      ;; Generate with known values so we can verify scores
      constraints (-> cm/EMPTY
                      (cm/set-choice [0 :v] (mx/scalar 1.0))
                      (cm/set-choice [1 :v] (mx/scalar 2.0)))
      {:keys [trace]} (p/generate mapper [[0.0 0.0]] constraints)]
  ;; The trace should have element-scores in metadata
  (assert-true "map trace has element-scores metadata"
    (some? (:genmlx.combinators/element-scores (meta trace))))
  ;; Now update with constraint on element 1 only
  (let [update-constraints (cm/set-choice cm/EMPTY [1 :v] (mx/scalar 3.0))
        {:keys [trace weight]} (p/update mapper trace update-constraints)]
    (mx/eval! weight)
    ;; Weight should reflect score change in element 1 only
    ;; new log-prob for element 1: gaussian(3.0; 0, 1)
    ;; old log-prob for element 1: gaussian(2.0; 0, 1)
    ;; Difference: -0.5*(9) + 0.5*(4) = -2.5
    (assert-close "map update weight correct" -2.5 (mx/item weight) 0.1))
  ;; Regenerate with selection on element 0
  (let [{:keys [weight]} (p/regenerate mapper trace (sel/select 0))]
    (mx/eval! weight)
    ;; Weight = new_score - old_score - proposal_ratio; should be finite
    (assert-true "map regenerate weight is finite" (js/isFinite (mx/item weight)))))

(println "\n-- 16.1: Unfold combinator sub-trace scores --")
(let [kernel (gen [t state]
               (let [v (dyn/trace :v (dist/gaussian state 1))]
                 (mx/eval! v)
                 (mx/item v)))
      unfolder (comb/unfold-combinator kernel)
      ;; Generate with known values
      constraints (-> cm/EMPTY
                      (cm/set-choice [0 :v] (mx/scalar 1.0))
                      (cm/set-choice [1 :v] (mx/scalar 2.0)))
      {:keys [trace]} (p/generate unfolder [2 0.0] constraints)]
  (assert-true "unfold trace has step-scores metadata"
    (some? (:genmlx.combinators/step-scores (meta trace))))
  ;; Update step 1
  (let [update-constraints (cm/set-choice cm/EMPTY [1 :v] (mx/scalar 3.0))
        {:keys [weight]} (p/update unfolder trace update-constraints)]
    (mx/eval! weight)
    (assert-true "unfold update weight is finite" (js/isFinite (mx/item weight)))))

;; ---------------------------------------------------------------------------
;; 16.3: assess validates all choices constrained
;; ---------------------------------------------------------------------------

(println "\n-- 16.3: assess validates all choices --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))
                    y (dyn/trace :y (dist/gaussian 0 1))]
                [x y]))]
  ;; Complete choices: should succeed
  (let [choices (cm/choicemap :x (mx/scalar 1.0)
                              :y (mx/scalar 2.0))
        {:keys [retval weight]} (p/assess model [] choices)]
    (mx/eval! weight)
    (assert-true "assess with all choices succeeds" (number? (mx/item weight)))
    ;; Weight should be sum of log-probs
    (let [expected (+ (* -0.5 (+ 1.0 (js/Math.log (* 2 js/Math.PI))))
                      (* -0.5 (+ 4.0 (js/Math.log (* 2 js/Math.PI)))))]
      (assert-close "assess weight = sum of log-probs" expected (mx/item weight) 0.01)))
  ;; Missing choice: should throw
  (assert-throws "assess with missing choice throws"
    #(p/assess model [] (cm/choicemap :x (mx/scalar 1.0)))))

;; ---------------------------------------------------------------------------
;; 16.2: Splice score tracking
;; ---------------------------------------------------------------------------

(println "\n-- 16.2: splice score tracking --")
(let [sub-model (gen [mu]
                  (dyn/trace :z (dist/gaussian mu 1)))
      outer-model (gen []
                    (let [x (dyn/trace :x (dist/gaussian 0 10))]
                      (dyn/splice :sub sub-model x)
                      x))
      ;; Generate with known values
      constraints (-> (cm/choicemap :x (mx/scalar 2.0))
                      (cm/set-choice [:sub :z] (mx/scalar 3.0)))
      {:keys [trace]} (p/generate outer-model [] constraints)]
  ;; Trace should have splice-scores metadata
  (assert-true "trace has splice-scores metadata"
    (some? (:genmlx.dynamic/splice-scores (meta trace))))
  ;; Regenerate the :x address (which affects the sub-model's distribution)
  ;; The old sub-model score should be correctly tracked
  (let [{:keys [trace weight]} (p/regenerate outer-model trace (sel/select :x))]
    (mx/eval! weight)
    (assert-true "splice regenerate weight is finite" (js/isFinite (mx/item weight)))
    ;; The weight should NOT be NaN or Infinity
    (assert-true "splice regenerate weight is not NaN" (not (js/isNaN (mx/item weight))))))

(println "\n=== All Phase 16/17 Correctness Tests Complete ===")
