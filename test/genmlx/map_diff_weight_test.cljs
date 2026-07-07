;; @tier fast
(ns genmlx.map-diff-weight-test
  "genmlx-5a87: MapCombinator update-with-diffs weight must equal the
   p/update fallback's thesis weight on a structure-flipping kernel.

   The optimized vector-diff path computed weight = new_total - old_total (a
   raw score delta), which matches the thesis weight ONLY when per-element
   updates sample no fresh sites. A kernel with a data-dependent branch that
   flips under a constraint samples fresh unconstrained sites whose log-probs
   then land in the weight instead of canceling against the internal proposal
   (the fixed genmlx-zek9 Mix class). The thesis weight for such a flip is
   DETERMINISTIC (fresh sites cancel), so the diff path and the fallback must
   agree exactly — and a raw score delta cannot (it varies with the fresh
   draw).

   Run: bunx --bun nbb@1.4.208 test/genmlx/map_diff_weight_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.combinators :as comb]
            [genmlx.diff :as diff])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

;; structure-flipping kernel: constraining :b flips which site exists
(def kernel
  (dyn/auto-key
   (gen [x]
     (let [b (trace :b (dist/bernoulli 0.5))]
       (if (pos? (mx/item b))
         (trace :y (dist/gaussian x 1))
         (trace :z (dist/gaussian x 2)))))))

(def mgf (dyn/auto-key (comb/map-combinator kernel)))
(def xs [0.5 1.5 -0.5])

(defn- flip-constraint [trace i]
  ;; constrain element i's :b to the OPPOSITE of its current value
  (let [b (mx/realize (cm/get-choice (:choices trace) [i :b]))]
    (cm/set-choice cm/EMPTY [i :b] (mx/scalar (- 1.0 b)))))

(println "\n-- diff-path weight == p/update fallback weight on a branch flip --")
(dotimes [s 5]
  (let [t (p/simulate (dyn/with-key mgf (rng/fresh-key (+ 100 s))) [xs])
        constraints (flip-constraint t 1)
        diff-w (mx/realize (:weight (p/update-with-diffs mgf t constraints diff/no-change)))
        full-w (mx/realize (:weight (p/update mgf t constraints)))]
    (assert-true (str "seed " s ": diff-path weight " (.toFixed diff-w 6)
                      " == fallback " (.toFixed full-w 6))
                 (< (js/Math.abs (- diff-w full-w)) 1e-4))))

;; the no-fresh-sites case must stay exact too (regression on the fast path)
(println "\n-- plain (non-flipping) constraint stays exact --")
(let [t (p/simulate (dyn/with-key mgf (rng/fresh-key 7)) [xs])
      b1 (mx/realize (cm/get-choice (:choices t) [1 :b]))
      site (if (pos? b1) :y :z)
      constraints (cm/set-choice cm/EMPTY [1 site] (mx/scalar 0.25))
      diff-w (mx/realize (:weight (p/update-with-diffs mgf t constraints diff/no-change)))
      full-w (mx/realize (:weight (p/update mgf t constraints)))]
  (assert-true (str "same-structure constraint: diff " (.toFixed diff-w 6)
                    " == fallback " (.toFixed full-w 6))
               (< (js/Math.abs (- diff-w full-w)) 1e-4)))

(println (str "\n== map-diff-weight: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
