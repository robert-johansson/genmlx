(ns genmlx.m4-constraint-stacking-test
  "M4: Auto-constraint stacking tests.
   Verifies that flat loop constraints are detected and pre-stacked
   into [T]-shaped MLX tensors for M5 consumption."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.vectorized :as vec])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def pass-count (atom 0))
(def fail-count (atom 0))

(defn assert-true [msg pred]
  (if pred
    (do (swap! pass-count inc) (println "  PASS:" msg))
    (do (swap! fail-count inc) (println "  FAIL:" msg))))

(defn assert-equal [msg expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc) (println "  PASS:" msg))
    (do (swap! fail-count inc) (println "  FAIL:" msg "expected:" expected "got:" actual))))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass-count inc) (println "  PASS:" msg))
      (do (swap! fail-count inc) (println "  FAIL:" msg "expected:" expected "got:" actual "diff:" diff)))))

;; ---------------------------------------------------------------------------
;; Shared models
;; ---------------------------------------------------------------------------

(def linreg-model
  (gen [xs]
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept) 1)))
      slope)))

;; ---------------------------------------------------------------------------
;; 1. loop-obs basic
;; ---------------------------------------------------------------------------

(println "\n-- 1. loop-obs basic --")

(let [obs (dyn/loop-obs "y" [1.0 2.0 3.0])]
  (assert-true "loop-obs returns Node" (instance? cm/Node obs))
  (assert-equal "y0 value" 1 (cm/get-value (cm/get-submap obs :y0)))
  (assert-equal "y1 value" 2 (cm/get-value (cm/get-submap obs :y1)))
  (assert-equal "y2 value" 3 (cm/get-value (cm/get-submap obs :y2))))

;; ---------------------------------------------------------------------------
;; 2. merge-obs
;; ---------------------------------------------------------------------------

(println "\n-- 2. merge-obs --")

(let [static (cm/choicemap :slope 2.0 :intercept 1.0)
      loop-c (dyn/loop-obs "y" [10.0 20.0 30.0])
      merged (dyn/merge-obs static loop-c)]
  (assert-equal "merged has slope" 2 (cm/get-value (cm/get-submap merged :slope)))
  (assert-equal "merged has y0" 10 (cm/get-value (cm/get-submap merged :y0)))
  (assert-equal "merged has y2" 30 (cm/get-value (cm/get-submap merged :y2))))

;; ---------------------------------------------------------------------------
;; 3. prepare-loop-stacks with model
;; ---------------------------------------------------------------------------

(println "\n-- 3. prepare-loop-stacks with model --")

(let [obs (dyn/loop-obs "y" [1.0 2.0 3.0])
      schema (:schema linreg-model)
      stacks (dyn/prepare-loop-stacks obs schema)]
  (assert-equal "one stack" 1 (count stacks))
  (assert-equal "prefix is y" "y" (:prefix (first stacks)))
  (assert-equal "count is 3" 3 (:count (first stacks)))
  (assert-equal "values shape [3]" [3] (mx/shape (:values (first stacks))))
  (assert-true "fully constrained" (:fully-constrained? (first stacks))))

;; ---------------------------------------------------------------------------
;; 4. prepare-loop-stacks edge cases
;; ---------------------------------------------------------------------------

(println "\n-- 4. prepare-loop-stacks edge cases --")

(let [schema (:schema linreg-model)]
  ;; nil constraints
  (assert-true "nil constraints -> nil" (nil? (dyn/prepare-loop-stacks nil schema)))
  ;; EMPTY constraints
  (assert-true "EMPTY constraints -> nil" (nil? (dyn/prepare-loop-stacks cm/EMPTY schema)))
  ;; No matching prefix
  (assert-true "no match -> nil"
               (nil? (dyn/prepare-loop-stacks (cm/choicemap :slope 1.0 :intercept 2.0) schema)))
  ;; Sparse (non-contiguous) indices
  (let [sparse (cm/choicemap :y0 1.0 :y2 3.0)
        stacks (dyn/prepare-loop-stacks sparse schema)]
    (assert-true "sparse not fully-constrained"
                 (not (:fully-constrained? (first stacks))))))

;; ---------------------------------------------------------------------------
;; 5. vgenerate + loop model
;; ---------------------------------------------------------------------------

(println "\n-- 5. vgenerate + loop model --")

(let [xs [1.0 2.0 3.0]
      obs (dyn/loop-obs "y" [10.0 20.0 30.0])
      key (rng/fresh-key)
      vt (dyn/vgenerate linreg-model [xs] obs 10 key)]
  (assert-equal "n-particles" 10 (:n-particles vt))
  (assert-equal "weight shape" [10] (mx/shape (:weight vt)))
  (assert-equal "score shape" [10] (mx/shape (:score vt)))
  ;; Constrained sites should have scalar values (broadcast)
  (let [y0-val (cm/get-value (cm/get-submap (:choices vt) :y0))]
    (assert-true "y0 is constrained scalar" (= [] (mx/shape y0-val)))))

;; ---------------------------------------------------------------------------
;; 6. Equivalence: loop-obs == manual choicemap
;; ---------------------------------------------------------------------------

(println "\n-- 6. Equivalence: loop-obs vs manual choicemap --")

(let [xs [1.0 2.0 3.0]
      manual (cm/choicemap :y0 10.0 :y1 20.0 :y2 30.0)
      loop-c (dyn/loop-obs "y" [10.0 20.0 30.0])
      key (rng/fresh-key)
      vt-manual (dyn/vgenerate linreg-model [xs] manual 100 key)
      vt-loop (dyn/vgenerate linreg-model [xs] loop-c 100 key)
      w-manual (mx/item (mx/mean (:weight vt-manual)))
      w-loop (mx/item (mx/mean (:weight vt-loop)))]
  (assert-close "mean weights match" w-manual w-loop 0.01)
  (assert-equal "same n-particles" (:n-particles vt-manual) (:n-particles vt-loop)))

;; ---------------------------------------------------------------------------
;; 7. merge-obs + vgenerate end-to-end
;; ---------------------------------------------------------------------------

(println "\n-- 7. merge-obs + vgenerate end-to-end --")

(let [xs [1.0 2.0 3.0]
      static (cm/choicemap :slope 2.0 :intercept 1.0)
      loop-c (dyn/loop-obs "y" [3.0 5.0 7.0])
      merged (dyn/merge-obs static loop-c)
      key (rng/fresh-key)
      vt (dyn/vgenerate linreg-model [xs] merged 50 key)]
  (assert-equal "vt n-particles" 50 (:n-particles vt))
  ;; Slope should be constrained to scalar 2.0
  (let [slope-val (cm/get-value (cm/get-submap (:choices vt) :slope))
        slope-num (if (mx/array? slope-val) (mx/item slope-val) slope-val)]
    (assert-close "slope constrained" 2.0 slope-num 0.001)))

;; ---------------------------------------------------------------------------
;; 8. Summary
;; ---------------------------------------------------------------------------

(println "\n========================================")
(println (str "M4 Constraint Stacking: " @pass-count "/" (+ @pass-count @fail-count)
              " passed" (when (pos? @fail-count) (str ", " @fail-count " FAILED"))))
(println "========================================")

(when (pos? @fail-count)
  (js/process.exit 1))
