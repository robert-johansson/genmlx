(ns genmlx.vupdate-test
  "Tests for batched update (vupdate)."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.vectorized :as vec])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-equal [msg expected actual]
  (if (= expected actual)
    (println "  PASS:" msg)
    (do (println "  FAIL:" msg)
        (println "    expected:" expected)
        (println "    actual:  " actual))))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(println "\n=== vupdate Tests ===\n")

;; Model: x ~ N(0, 1), y ~ N(x, 0.5)
(def model
  (gen []
    (let [x (dyn/trace :x (dist/gaussian 0 1))]
      (dyn/trace :y (dist/gaussian x 0.5))
      x)))

(def n 50)

;; ---------------------------------------------------------------------------
;; 1. Shape correctness
;; ---------------------------------------------------------------------------

(println "-- vupdate shape correctness --")
(let [key (rng/fresh-key)
      [k1 k2] (rng/split key)
      obs1 (cm/choicemap :y (mx/scalar 2.0))
      vtrace (dyn/vgenerate model [] obs1 n k1)
      ;; Update observation from y=2 to y=3
      obs2 (cm/choicemap :y (mx/scalar 3.0))
      {:keys [vtrace weight]} (dyn/vupdate model vtrace obs2 k2)]

  ;; :x should still be [N]-shaped
  (let [x-val (cm/get-value (cm/get-submap (:choices vtrace) :x))]
    (mx/eval! x-val)
    (assert-equal "vupdate :x shape" [n] (mx/shape x-val)))

  ;; score should be [N]-shaped
  (let [score (:score vtrace)]
    (mx/eval! score)
    (assert-equal "vupdate score shape" [n] (mx/shape score)))

  ;; weight should be [N]-shaped
  (mx/eval! weight)
  (assert-equal "vupdate weight shape" [n] (mx/shape weight)))

;; ---------------------------------------------------------------------------
;; 2. Observation updated correctly
;; ---------------------------------------------------------------------------

(println "\n-- vupdate observation updated --")
(let [key (rng/fresh-key)
      [k1 k2] (rng/split key)
      obs1 (cm/choicemap :y (mx/scalar 2.0))
      vtrace (dyn/vgenerate model [] obs1 n k1)
      obs2 (cm/choicemap :y (mx/scalar 3.0))
      {:keys [vtrace]} (dyn/vupdate model vtrace obs2 k2)
      y-val (cm/get-value (cm/get-submap (:choices vtrace) :y))]
  (mx/eval! y-val)
  (assert-close "vupdate: y is now 3.0" 3.0 (mx/realize y-val) 1e-6))

;; ---------------------------------------------------------------------------
;; 3. Unconstrained :x preserved
;; ---------------------------------------------------------------------------

(println "\n-- vupdate :x unchanged --")
(let [key (rng/fresh-key)
      [k1 k2] (rng/split key)
      obs1 (cm/choicemap :y (mx/scalar 2.0))
      vtrace-before (dyn/vgenerate model [] obs1 n k1)
      x-before (cm/get-value (cm/get-submap (:choices vtrace-before) :x))
      _ (mx/eval! x-before)
      x-before-mean (mx/realize (mx/mean x-before))
      obs2 (cm/choicemap :y (mx/scalar 3.0))
      {:keys [vtrace]} (dyn/vupdate model vtrace-before obs2 k2)
      x-after (cm/get-value (cm/get-submap (:choices vtrace) :x))
      _ (mx/eval! x-after)
      x-after-mean (mx/realize (mx/mean x-after))]
  ;; x values should be identical (unchanged by update)
  (assert-close "vupdate: :x unchanged" x-before-mean x-after-mean 1e-6))

;; ---------------------------------------------------------------------------
;; 4. Weights finite
;; ---------------------------------------------------------------------------

(println "\n-- vupdate weights finite --")
(let [key (rng/fresh-key)
      [k1 k2] (rng/split key)
      obs1 (cm/choicemap :y (mx/scalar 2.0))
      vtrace (dyn/vgenerate model [] obs1 n k1)
      obs2 (cm/choicemap :y (mx/scalar 3.0))
      {:keys [weight]} (dyn/vupdate model vtrace obs2 k2)]
  (mx/eval! weight)
  (let [w-min (mx/realize (mx/amin weight))
        w-max (mx/realize (mx/amax weight))]
    (assert-true "vupdate: all weights finite"
      (and (js/isFinite w-min) (js/isFinite w-max)))))

;; ---------------------------------------------------------------------------
;; 5. Statistical equivalence with sequential update
;; ---------------------------------------------------------------------------

(println "\n-- vupdate statistical equivalence --")
(let [n-test 30
      obs1 (cm/choicemap :y (mx/scalar 2.0))
      obs2 (cm/choicemap :y (mx/scalar 3.0))
      ;; Sequential: generate n traces, then update each
      seq-weights (mapv (fn [_]
                          (let [{:keys [trace]} (p/generate model [] obs1)
                                {:keys [weight]} (p/update model trace obs2)]
                            (mx/realize weight)))
                        (range n-test))
      seq-mean-w (/ (reduce + seq-weights) n-test)
      ;; Batched: vgenerate + vupdate
      key (rng/fresh-key)
      [k1 k2] (rng/split key)
      vtrace (dyn/vgenerate model [] obs1 n-test k1)
      {:keys [weight]} (dyn/vupdate model vtrace obs2 k2)
      _ (mx/eval! weight)
      batch-mean-w (mx/realize (mx/mean weight))]
  ;; Mean weights should be in same ballpark
  ;; (different random seeds, so just check they're both finite and reasonable)
  (assert-true "sequential mean weight finite" (js/isFinite seq-mean-w))
  (assert-true "batched mean weight finite" (js/isFinite batch-mean-w))
  ;; Both should have similar sign/magnitude (weight = log p(y=3|x) - log p(y=2|x))
  (assert-close "mean weight similar" seq-mean-w batch-mean-w 2.0))

(println "\nAll vupdate tests complete.")
