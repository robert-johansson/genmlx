(ns genmlx.batched-switch-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.combinators :as comb]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (println "  PASS:" msg)
      (println "  FAIL:" msg "- expected" expected "got" actual "diff" diff))))

(println "\n=== Batched Switch Tests ===\n")

;; -- Two branch gen functions --
(def branch-low
  (gen []
    (let [x (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))]
      x)))

(def branch-high
  (gen []
    (let [x (trace :x (dist/gaussian (mx/scalar 10.0) (mx/scalar 1.0)))]
      x)))

;; -- 1. Scalar switch still works --
(println "-- 1. Scalar switch sanity check --")
(let [sw (comb/switch-combinator (dyn/auto-key branch-low) (dyn/auto-key branch-high))
      trace0 (p/simulate sw [0])
      trace1 (p/simulate sw [1])]
  (mx/eval! (:retval trace0))
  (mx/eval! (:retval trace1))
  (assert-true "branch 0 returns value near 0" (< (js/Math.abs (mx/item (:retval trace0))) 5))
  (assert-true "branch 1 returns value near 10" (< (js/Math.abs (- 10 (mx/item (:retval trace1)))) 5))
  (println "  branch 0 retval:" (mx/item (:retval trace0))
           "branch 1 retval:" (mx/item (:retval trace1))))

;; -- 2. Batched switch via vsimulate --
(println "\n-- 2. Batched switch via vsimulate --")
(def model-switch
  (gen [index]
    (let [sw (comb/switch-combinator branch-low branch-high)
          result (splice :choice sw index)]
      result)))

(let [key (rng/fresh-key)
      ;; 100 particles, half branch 0 half branch 1
      index (mx/array (vec (concat (repeat 50 0) (repeat 50 1))) mx/int32)
      vtrace (dyn/vsimulate model-switch [index] 100 key)]
  (assert-true "batched switch returns vtrace" (some? vtrace))
  (mx/eval! (:score vtrace))
  (let [score-shape (mx/shape (:score vtrace))]
    (assert-true "score is [100]-shaped" (= [100] score-shape))
    (println "  score shape:" score-shape))
  ;; Check choices structure
  (let [choices (:choices vtrace)
        x-vals (cm/get-value (cm/get-submap (cm/get-submap choices :choice) :x))]
    (mx/eval! x-vals)
    (assert-true ":x is [100]-shaped" (= [100] (mx/shape x-vals)))
    ;; First 50 should be near 0, last 50 near 10
    (let [first-half-mean (mx/item (mx/mean (mx/slice x-vals 0 50)))
          second-half-mean (mx/item (mx/mean (mx/slice x-vals 50 100)))]
      (println "  first 50 (branch 0) mean:" (.toFixed first-half-mean 2)
               "last 50 (branch 1) mean:" (.toFixed second-half-mean 2))
      (assert-true "branch 0 particles near 0" (< (js/Math.abs first-half-mean) 3))
      (assert-true "branch 1 particles near 10" (< (js/Math.abs (- 10 second-half-mean)) 3)))))

;; -- 3. Batched switch with constraints (vgenerate) --
(println "\n-- 3. Batched switch with constraints (vgenerate) --")
(let [key (rng/fresh-key)
      index (mx/array (vec (concat (repeat 50 0) (repeat 50 1))) mx/int32)
      obs (cm/set-value cm/EMPTY [:choice :x] (mx/scalar 5.0))
      vtrace (dyn/vgenerate model-switch [index] obs 100 key)]
  (assert-true "vgenerate returns vtrace" (some? vtrace))
  (mx/eval! (:score vtrace))
  (mx/eval! (:weight vtrace))
  (assert-true "score is [100]-shaped" (= [100] (mx/shape (:score vtrace))))
  (assert-true "weight is [100]-shaped" (= [100] (mx/shape (:weight vtrace))))
  ;; x=5 is more likely under branch-high (mean=10) than under branch-low (mean=0)
  ;; but closer to 0 so branch 0 weight should be lower
  (let [w (mx/item (mx/mean (:weight vtrace)))]
    (println "  mean weight:" (.toFixed w 3))))

;; -- 4. Per-particle branch selection with random index --
(println "\n-- 4. Random per-particle branch selection --")
(def model-mixture
  (gen []
    (let [;; Each particle samples its own branch
          z (trace :z (dist/bernoulli (mx/scalar 0.5)))
          idx (mx/multiply z (mx/scalar 1 mx/int32))
          sw (comb/switch-combinator branch-low branch-high)
          result (splice :comp sw idx)]
      result)))

(let [key (rng/fresh-key)
      vtrace (dyn/vsimulate model-mixture [] 200 key)]
  (assert-true "random branch vtrace exists" (some? vtrace))
  (mx/eval! (:score vtrace))
  (assert-true "score is [200]-shaped" (= [200] (mx/shape (:score vtrace))))
  ;; Get the branch indices and values
  (let [choices (:choices vtrace)
        z-vals (cm/get-value (cm/get-submap choices :z))
        x-vals (cm/get-value (cm/get-submap (cm/get-submap choices :comp) :x))]
    (mx/eval! z-vals)
    (mx/eval! x-vals)
    ;; Rough check: overall mean should be ~5 (mixture of 0 and 10)
    (let [overall-mean (mx/item (mx/mean x-vals))]
      (println "  overall mean (expect ~5):" (.toFixed overall-mean 2))
      (assert-true "mixture mean near 5" (< (js/Math.abs (- 5 overall-mean)) 3)))))

;; -- 5. Performance: batched vs scalar --
(println "\n-- 5. Performance: batched vs scalar --")
(let [key (rng/fresh-key)
      index (mx/array (vec (concat (repeat 250 0) (repeat 250 1))) mx/int32)
      ;; Batched
      t0 (js/Date.now)
      _ (dotimes [_ 20]
          (let [vt (dyn/vsimulate model-switch [index] 500 key)]
            (mx/eval! (:score vt))))
      t1 (js/Date.now)
      batched-ms (- t1 t0)
      ;; Scalar comparison: 50 runs
      _ (mx/clear-cache!)
      t2 (js/Date.now)
      _ (let [sw (comb/switch-combinator (dyn/auto-key branch-low) (dyn/auto-key branch-high))]
          (dotimes [_ 50]
            (let [tr (p/simulate sw [0])]
              (mx/eval! (:score tr)))))
      t3 (js/Date.now)
      scalar-ms (- t3 t2)
      scalar-equiv (* scalar-ms 10)]
  (println "  batched (20 × 500 particles):" batched-ms "ms")
  (println "  scalar  (50 runs, ×10 extrapolated):" scalar-equiv "ms")
  (when (pos? batched-ms)
    (println "  estimated speedup:" (.toFixed (/ scalar-equiv batched-ms) 1) "x")))

(println "\n=== Done ===")
