(ns genmlx.batched-mix-test
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

(println "\n=== Batched Mix Tests ===\n")

;; -- Two component gen functions --
(def comp-low
  (gen [x]
    (let [y (trace :y (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))]
      y)))

(def comp-high
  (gen [x]
    (let [y (trace :y (dist/gaussian (mx/scalar 10.0) (mx/scalar 1.0)))]
      y)))

;; -- 1. Scalar mix sanity check --
(println "-- 1. Scalar mix sanity check --")
(let [log-w (mx/array [0.0 0.0])  ;; equal weights
      mix (comb/mix-combinator [(dyn/auto-key comp-low) (dyn/auto-key comp-high)]
                                log-w)
      trace (p/simulate mix [(mx/scalar 0.0)])]
  (mx/eval! (:retval trace))
  (mx/eval! (:score trace))
  (assert-true "scalar mix returns trace" (some? trace))
  (assert-true "score is finite" (js/isFinite (mx/item (:score trace))))
  (println "  retval:" (mx/item (:retval trace))
           "score:" (.toFixed (mx/item (:score trace)) 3)))

;; -- 2. Batched mix via vsimulate --
(println "\n-- 2. Batched mix via vsimulate --")
(def model-mix
  (gen [x]
    (let [log-w (mx/array [0.0 0.0])
          mix (comb/mix-combinator [comp-low comp-high] log-w)
          result (splice :mixture mix x)]
      result)))

(let [key (rng/fresh-key)
      vtrace (dyn/vsimulate model-mix [(mx/scalar 0.0)] 200 key)]
  (assert-true "batched mix returns vtrace" (some? vtrace))
  (mx/eval! (:score vtrace))
  (let [score-shape (mx/shape (:score vtrace))]
    (assert-true "score is [200]-shaped" (= [200] score-shape))
    (println "  score shape:" score-shape))
  ;; Check choices: should have :y and :component-idx
  (let [choices (:choices vtrace)
        mix-sub (cm/get-submap choices :mixture)
        y-vals (cm/get-value (cm/get-submap mix-sub :y))
        idx-vals (cm/get-value (cm/get-submap mix-sub :component-idx))]
    (mx/eval! y-vals)
    (mx/eval! idx-vals)
    (assert-true ":y is [200]-shaped" (= [200] (mx/shape y-vals)))
    (assert-true ":component-idx is [200]-shaped" (= [200] (mx/shape idx-vals)))
    ;; With equal weights, ~half should be near 0, ~half near 10
    ;; Overall mean should be ~5
    (let [overall-mean (mx/item (mx/mean y-vals))]
      (println "  overall y mean (expect ~5):" (.toFixed overall-mean 2))
      (assert-true "mixture mean near 5" (< (js/Math.abs (- 5 overall-mean)) 3)))))

;; -- 3. Batched mix with constraints (vgenerate) --
(println "\n-- 3. Batched mix with constraints (vgenerate) --")
(let [key (rng/fresh-key)
      obs (cm/set-value cm/EMPTY [:mixture :y] (mx/scalar 5.0))
      vtrace (dyn/vgenerate model-mix [(mx/scalar 0.0)] obs 100 key)]
  (assert-true "vgenerate returns vtrace" (some? vtrace))
  (mx/eval! (:score vtrace))
  (mx/eval! (:weight vtrace))
  (assert-true "score is [100]-shaped" (= [100] (mx/shape (:score vtrace))))
  (assert-true "weight is [100]-shaped" (= [100] (mx/shape (:weight vtrace))))
  (println "  weight shape:" (mx/shape (:weight vtrace))
           "mean weight:" (.toFixed (mx/item (mx/mean (:weight vtrace))) 3)))

;; -- 4. Score consistency: batched vs scalar --
(println "\n-- 4. Score consistency --")
(let [key (rng/fresh-key)
      ;; Batched
      vtrace (dyn/vsimulate model-mix [(mx/scalar 0.0)] 100 key)
      _ (mx/eval! (:score vtrace))
      batched-mean (mx/item (mx/mean (:score vtrace)))
      ;; Scalar: 200 independent runs
      log-w (mx/array [0.0 0.0])
      mix (comb/mix-combinator [(dyn/auto-key comp-low) (dyn/auto-key comp-high)]
                                log-w)
      scalar-scores (mapv (fn [_]
                            (let [trace (p/simulate mix [(mx/scalar 0.0)])]
                              (mx/eval! (:score trace))
                              (mx/item (:score trace))))
                          (range 200))
      scalar-mean (/ (reduce + scalar-scores) (count scalar-scores))]
  (println "  batched mean score:" (.toFixed batched-mean 3))
  (println "  scalar mean score:" (.toFixed scalar-mean 3))
  (assert-close "mean scores similar" scalar-mean batched-mean 1.5))

;; -- 5. Unequal weights --
(println "\n-- 5. Unequal weights --")
(def model-mix-weighted
  (gen [x]
    (let [;; 90% comp-low, 10% comp-high
          log-w (mx/array [(js/Math.log 0.9) (js/Math.log 0.1)])
          mix (comb/mix-combinator [comp-low comp-high] log-w)
          result (splice :mixture mix x)]
      result)))

(let [key (rng/fresh-key)
      vtrace (dyn/vsimulate model-mix-weighted [(mx/scalar 0.0)] 500 key)]
  (mx/eval! (:score vtrace))
  (let [choices (:choices vtrace)
        mix-sub (cm/get-submap choices :mixture)
        idx-vals (cm/get-value (cm/get-submap mix-sub :component-idx))
        y-vals (cm/get-value (cm/get-submap mix-sub :y))]
    (mx/eval! idx-vals)
    (mx/eval! y-vals)
    ;; With 90/10 weights, mean should be closer to 0 than 10
    (let [overall-mean (mx/item (mx/mean y-vals))
          ;; Expected: 0.9*0 + 0.1*10 = 1.0
          idx-float (mx/multiply idx-vals (mx/scalar 1.0))
          frac-high (mx/item (mx/mean idx-float))]
      (println "  overall y mean (expect ~1):" (.toFixed overall-mean 2))
      (println "  fraction comp-high (expect ~0.1):" (.toFixed frac-high 3))
      (assert-true "weighted mean near 1" (< (js/Math.abs (- 1.0 overall-mean)) 3)))))

;; -- 6. Performance: batched vs scalar --
(println "\n-- 6. Performance: batched vs scalar --")
(let [key (rng/fresh-key)
      ;; Batched
      t0 (js/Date.now)
      _ (dotimes [_ 20]
          (let [vt (dyn/vsimulate model-mix [(mx/scalar 0.0)] 500 key)]
            (mx/eval! (:score vt))))
      t1 (js/Date.now)
      batched-ms (- t1 t0)
      ;; Scalar
      _ (mx/clear-cache!)
      log-w (mx/array [0.0 0.0])
      mix (comb/mix-combinator [(dyn/auto-key comp-low) (dyn/auto-key comp-high)]
                                log-w)
      t2 (js/Date.now)
      _ (dotimes [_ 50]
          (let [tr (p/simulate mix [(mx/scalar 0.0)])]
            (mx/eval! (:score tr))))
      t3 (js/Date.now)
      scalar-ms (- t3 t2)
      scalar-equiv (* scalar-ms 10)]
  (println "  batched (20 × 500 particles):" batched-ms "ms")
  (println "  scalar  (50 runs, ×10 extrapolated):" scalar-equiv "ms")
  (when (pos? batched-ms)
    (println "  estimated speedup:" (.toFixed (/ scalar-equiv batched-ms) 1) "x")))

(println "\n=== Done ===")
