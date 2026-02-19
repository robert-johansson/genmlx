(ns genmlx.map-test
  "Tests for MAP (Maximum A Posteriori) optimization."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc])
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

(println "\n=== MAP Optimization Tests ===\n")

;; Gradient-friendly model: no mx/eval! or mx/item inside model body.
;; Pass MLX arrays directly to distributions so the gradient graph stays intact.

;; -- Gaussian posterior --
;; x ~ N(0, 10), observe y ~ N(x, 1) at y=5
;; Posterior: N(5 * 100/101, 1/101) ≈ N(4.95, 0.0099)
;; MAP estimate ≈ 4.95
(println "-- Gaussian posterior --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 10))]
                (dyn/trace :y (dist/gaussian x 1))
                x))
      obs (cm/choicemap :y (mx/scalar 5.0))
      result (mcmc/map-optimize
               {:iterations 2000 :lr 0.05 :addresses [:x]}
               model [] obs)]
  (assert-close "MAP x ≈ 4.95" 4.95 (first (:params result)) 0.2)
  (assert-true "score is finite" (js/isFinite (:score result)))
  (assert-true "score-history has entries" (= 2000 (count (:score-history result))))
  (assert-true "trace returned" (some? (:trace result))))

;; -- Multi-parameter --
;; x ~ N(0, 10), z ~ N(0, 10)
;; observe y1 ~ N(x, 0.5) at 3.0, y2 ~ N(z, 0.5) at -2.0
(println "\n-- multi-parameter --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 10))
                    z (dyn/trace :z (dist/gaussian 0 10))]
                (dyn/trace :y1 (dist/gaussian x 0.5))
                (dyn/trace :y2 (dist/gaussian z 0.5))
                [x z]))
      obs (cm/merge-cm (cm/choicemap :y1 (mx/scalar 3.0))
                       (cm/choicemap :y2 (mx/scalar -2.0)))
      result (mcmc/map-optimize
               {:iterations 2000 :lr 0.05 :addresses [:x :z]}
               model [] obs)
      [x-val z-val] (:params result)]
  (assert-close "MAP x ≈ 3.0" 3.0 x-val 0.3)
  (assert-close "MAP z ≈ -2.0" -2.0 z-val 0.3))

;; -- Score monotonicity --
;; Score should generally increase (non-decreasing after initial convergence)
(println "\n-- score monotonicity --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 10))]
                (dyn/trace :y (dist/gaussian x 1))
                x))
      obs (cm/choicemap :y (mx/scalar 5.0))
      result (mcmc/map-optimize
               {:iterations 200 :lr 0.01 :addresses [:x]}
               model [] obs)
      history (:score-history result)
      ;; Check that final score > initial score
      initial-score (first history)
      final-score (last history)]
  (assert-true "final score > initial score"
               (> final-score initial-score)))

;; -- SGD option --
(println "\n-- SGD optimizer --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 10))]
                (dyn/trace :y (dist/gaussian x 1))
                x))
      obs (cm/choicemap :y (mx/scalar 5.0))
      result (mcmc/map-optimize
               {:iterations 1000 :optimizer :sgd :lr 0.05 :addresses [:x]}
               model [] obs)]
  (assert-close "SGD MAP x ≈ 4.95" 4.95 (first (:params result)) 0.3))

(println "\nAll MAP optimization tests complete.")
