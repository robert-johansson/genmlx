(ns genmlx.combinators-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert= [msg expected actual]
  (if (= expected actual)
    (println "  PASS:" msg)
    (do (println "  FAIL:" msg)
        (println "    expected:" (pr-str expected))
        (println "    actual:  " (pr-str actual)))))

(println "\n=== Combinator Tests ===\n")

;; Map combinator
(println "-- Map combinator --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 1))]
                 (mx/eval! y)
                 (mx/item y)))
      mapped (comb/map-combinator kernel)
      trace (p/simulate mapped [[1.0 2.0 3.0]])]
  (assert-true "map returns trace" (instance? tr/Trace trace))
  (assert= "map returns 3 values" 3 (count (tr/get-retval trace)))
  (assert-true "map retvals are numbers" (every? number? (tr/get-retval trace))))

;; Map combinator with generate
(println "\n-- Map combinator generate --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 (mx/eval! y)
                 (mx/item y)))
      mapped (comb/map-combinator kernel)
      constraints (cm/set-choice cm/EMPTY [0] (cm/choicemap :y (mx/scalar 1.5)))
      {:keys [trace weight]} (p/generate mapped [[1.0 2.0]] constraints)]
  (assert-true "map generate returns trace" (instance? tr/Trace trace))
  (mx/eval! weight)
  (assert-true "map generate has weight" (number? (mx/item weight))))

;; Unfold combinator
(println "\n-- Unfold combinator --")
(let [step (gen [t state]
             (let [next (dyn/trace :x (dist/gaussian state 0.1))]
               (mx/eval! next)
               (mx/item next)))
      unfold (comb/unfold-combinator step)
      trace (p/simulate unfold [5 0.0])]
  (assert-true "unfold returns trace" (instance? tr/Trace trace))
  (assert= "unfold returns 5 states" 5 (count (tr/get-retval trace)))
  (assert-true "unfold retvals are numbers" (every? number? (tr/get-retval trace))))

;; Switch combinator
(println "\n-- Switch combinator --")
(let [branch0 (gen []
                (let [x (dyn/trace :x (dist/gaussian 0 1))]
                  (mx/eval! x)
                  (mx/item x)))
      branch1 (gen []
                (let [x (dyn/trace :x (dist/gaussian 10 1))]
                  (mx/eval! x)
                  (mx/item x)))
      sw (comb/switch-combinator branch0 branch1)
      trace0 (p/simulate sw [0])
      trace1 (p/simulate sw [1])]
  (assert-true "switch branch 0 returns trace" (instance? tr/Trace trace0))
  (assert-true "switch branch 1 returns trace" (instance? tr/Trace trace1))
  (assert-true "branch 0 value near 0" (< (js/Math.abs (tr/get-retval trace0)) 5))
  (assert-true "branch 1 value near 10" (< (js/Math.abs (- (tr/get-retval trace1) 10)) 5)))

(println "\nAll combinator tests complete.")
