(ns genmlx.gen-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel])
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

(println "\n=== Gen Macro Tests ===\n")

;; Test basic gen macro
(println "-- basic gen --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))]
                (mx/eval! x)
                (mx/item x)))
      trace (p/simulate model [])]
  (assert-true "gen creates DynamicGF" (instance? dyn/DynamicGF model))
  (assert-true "simulate returns trace" (instance? tr/Trace trace))
  (assert-true "trace has retval" (number? (:retval trace)))
  (assert-true "trace has choices" (cm/has-value? (cm/get-submap (:choices trace) :x))))

;; Test gen with args
(println "\n-- gen with args --")
(let [model (gen [mu]
              (let [x (dyn/trace :x (dist/gaussian mu 1))]
                (mx/eval! x)
                (mx/item x)))
      trace (p/simulate model [5.0])]
  (assert-true "gen with args returns trace" (instance? tr/Trace trace))
  (assert-true "args stored in trace" (= [5.0] (:args trace))))

;; Test generate with constraints
(println "\n-- generate with constraints --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))]
                (mx/eval! x)
                (mx/item x)))
      constraints (cm/choicemap :x (mx/scalar 2.0))
      {:keys [trace weight]} (p/generate model [] constraints)]
  (let [x-val (cm/get-value (cm/get-submap (:choices trace) :x))]
    (mx/eval! x-val)
    (assert-close "generate constrains value" 2.0 (mx/item x-val) 0.001))
  (mx/eval! weight)
  (assert-true "generate returns weight" (number? (mx/item weight))))

;; Test update
(println "\n-- update --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))
                    y (dyn/trace :y (dist/gaussian 0 1))]
                (mx/eval! x y)
                (+ (mx/item x) (mx/item y))))
      trace (p/simulate model [])
      new-constraints (cm/choicemap :x (mx/scalar 3.0))
      {:keys [trace weight discard]} (p/update model trace new-constraints)]
  (let [x-val (cm/get-value (cm/get-submap (:choices trace) :x))]
    (mx/eval! x-val)
    (assert-close "update constrains x" 3.0 (mx/item x-val) 0.001))
  (mx/eval! weight)
  (assert-true "update returns weight" (number? (mx/item weight)))
  (assert-true "update returns discard" (some? discard)))

;; Test regenerate
(println "\n-- regenerate --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))
                    y (dyn/trace :y (dist/gaussian 0 1))]
                (mx/eval! x y)
                (+ (mx/item x) (mx/item y))))
      trace (p/simulate model [])
      old-y (let [v (cm/get-value (cm/get-submap (:choices trace) :y))]
              (mx/eval! v) (mx/item v))
      {:keys [trace weight]} (p/regenerate model trace (sel/select :x))]
  ;; y should be unchanged
  (let [new-y (cm/get-value (cm/get-submap (:choices trace) :y))]
    (mx/eval! new-y)
    (assert-close "regenerate keeps unselected" old-y (mx/item new-y) 0.001))
  (mx/eval! weight)
  (assert-true "regenerate returns weight" (number? (mx/item weight))))

;; Test call
(println "\n-- call --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))]
                (mx/eval! x)
                (mx/item x)))
      result (dyn/call model)]
  (assert-true "gen-fn is callable via dyn/call" (number? result)))

(println "\nAll gen tests complete.")
