(ns genmlx.handler-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.handler :as h]
            [genmlx.choicemap :as cm]
            [genmlx.mlx.random :as rng]))

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

(println "\n=== Handler Tests ===\n")

;; Test simulate handler
(println "-- simulate handler --")
(let [key (rng/fresh-key 42)
      result (h/run-handler h/simulate-handler
               {:choices cm/EMPTY :score (mx/scalar 0.0) :key key
                :executor nil}
               (fn []
                 (let [x (h/trace-choice! :x (dist/gaussian 0 1))
                       y (h/trace-choice! :y (dist/gaussian 0 1))]
                   (mx/eval! x y)
                   (+ (mx/item x) (mx/item y)))))]
  (assert-true "simulate returns retval" (number? (:retval result)))
  (assert-true "simulate has choices" (cm/has-value? (cm/get-submap (:choices result) :x)))
  (assert-true "simulate has choices" (cm/has-value? (cm/get-submap (:choices result) :y)))
  (let [score (:score result)]
    (mx/eval! score)
    (assert-true "simulate has negative score" (< (mx/item score) 0))))

;; Test generate handler with constraints
(println "\n-- generate handler with constraints --")
(let [key (rng/fresh-key 42)
      constraints (cm/choicemap :x (mx/scalar 1.5))
      result (h/run-handler h/generate-handler
               {:choices cm/EMPTY :score (mx/scalar 0.0)
                :weight (mx/scalar 0.0)
                :key key :constraints constraints
                :executor nil}
               (fn []
                 (let [x (h/trace-choice! :x (dist/gaussian 0 1))]
                   (mx/eval! x)
                   (mx/item x))))]
  (let [x-val (cm/get-value (cm/get-submap (:choices result) :x))]
    (mx/eval! x-val)
    (assert-close "generate constrains x" 1.5 (mx/item x-val) 0.001))
  (let [weight (:weight result)]
    (mx/eval! weight)
    (assert-true "generate has nonzero weight" (not= 0 (mx/item weight)))))

(println "\nAll handler tests complete.")
