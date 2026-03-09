(ns genmlx.batched-unfold-test
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

(println "\n=== Batched Unfold Tests ===\n")

;; -- Simple AR(1) kernel --
(def ar-kernel
  (gen [t state]
    (let [x (trace :x (dist/gaussian (mx/multiply state (mx/scalar 0.9))
                                      (mx/scalar 1.0)))]
      x)))

;; -- 1. Scalar unfold still works --
(println "-- 1. Scalar unfold sanity check --")
(let [unfold (comb/unfold-combinator (dyn/auto-key ar-kernel))
      trace (p/simulate unfold [5 (mx/scalar 0.0)])]
  (assert-true "scalar unfold returns trace" (some? trace))
  (assert-true "scalar unfold has 5 states" (= 5 (count (:retval trace))))
  (mx/eval! (:score trace))
  (assert-true "scalar unfold has finite score" (js/isFinite (mx/item (:score trace))))
  (println "  score:" (mx/item (:score trace))))

;; -- 2. Batched unfold via vsimulate --
(println "\n-- 2. Batched unfold via vsimulate --")
(def model-unfold
  (gen [n-steps]
    (let [unfold (comb/unfold-combinator ar-kernel)
          result (splice :temporal unfold n-steps (mx/scalar 0.0))]
      result)))

(let [key (rng/fresh-key)
      vtrace (dyn/vsimulate model-unfold [5] 100 key)]
  (assert-true "batched unfold returns vtrace" (some? vtrace))
  (mx/eval! (:score vtrace))
  (let [score-shape (mx/shape (:score vtrace))]
    (assert-true "score is [100]-shaped" (= [100] score-shape))
    (println "  score shape:" score-shape))
  ;; Check choices structure: :temporal -> {0: {x: [100]}, 1: {x: [100]}, ...}
  (let [choices (:choices vtrace)
        step0 (cm/get-submap (cm/get-submap choices :temporal) 0)
        x0 (cm/get-value (cm/get-submap step0 :x))]
    (mx/eval! x0)
    (assert-true "step 0 :x is [100]-shaped" (= [100] (mx/shape x0)))
    (println "  step 0 :x shape:" (mx/shape x0)
             "mean:" (.toFixed (mx/item (mx/mean x0)) 3))))

;; -- 3. Batched unfold with constraints (vgenerate) --
(println "\n-- 3. Batched unfold with constraints (vgenerate) --")
(let [key (rng/fresh-key)
      ;; Observe x=1.0 at step 2
      obs (cm/set-value
            (cm/set-value cm/EMPTY
              [:temporal 2 :x] (mx/scalar 1.0))
            [:temporal 4 :x] (mx/scalar -0.5))
      vtrace (dyn/vgenerate model-unfold [5] obs 100 key)]
  (assert-true "vgenerate returns vtrace" (some? vtrace))
  (mx/eval! (:score vtrace))
  (mx/eval! (:weight vtrace))
  (let [score-shape (mx/shape (:score vtrace))
        weight-shape (mx/shape (:weight vtrace))]
    (assert-true "score is [100]-shaped" (= [100] score-shape))
    (assert-true "weight is [100]-shaped" (= [100] weight-shape))
    (println "  score shape:" score-shape "weight shape:" weight-shape)
    (println "  mean weight:" (.toFixed (mx/item (mx/mean (:weight vtrace))) 3)))
  ;; Verify unconstrained step has [100]-shaped values
  (let [choices (:choices vtrace)
        step0 (cm/get-submap (cm/get-submap choices :temporal) 0)
        x0 (cm/get-value (cm/get-submap step0 :x))]
    (mx/eval! x0)
    (assert-true "unconstrained step 0 :x is [100]-shaped" (= [100] (mx/shape x0)))
    (println "  step 0 :x shape:" (mx/shape x0))))

;; -- 4. Score consistency: batched vs N scalar runs --
(println "\n-- 4. Score consistency --")
(let [key (rng/fresh-key)
      ;; Batched: 50 particles
      vtrace (dyn/vsimulate model-unfold [3] 50 key)
      _ (mx/eval! (:score vtrace))
      batched-mean (mx/item (mx/mean (:score vtrace)))
      ;; Scalar: 200 independent runs (using the unfold directly, not batched)
      scalar-unfold (comb/unfold-combinator (dyn/auto-key ar-kernel))
      scalar-scores (mapv (fn [_]
                            (let [trace (p/simulate scalar-unfold [3 (mx/scalar 0.0)])]
                              (mx/eval! (:score trace))
                              (mx/item (:score trace))))
                          (range 200))
      scalar-mean (/ (reduce + scalar-scores) (count scalar-scores))]
  ;; Both should be sampling from the same distribution
  ;; Mean score of AR(1) with rho=0.9, sigma=1 over 3 steps ≈ -3*log(sqrt(2*pi)) ≈ -2.76
  (println "  batched mean score:" (.toFixed batched-mean 3))
  (println "  scalar mean score:" (.toFixed scalar-mean 3))
  (assert-close "mean scores similar" scalar-mean batched-mean 1.0))

;; -- 5. Multi-argument kernel --
(println "\n-- 5. Multi-argument kernel --")
(def ar-kernel-with-drift
  (gen [t state drift]
    (let [x (trace :x (dist/gaussian (mx/add (mx/multiply state (mx/scalar 0.9))
                                              drift)
                                      (mx/scalar 1.0)))]
      x)))

(def model-unfold-drift
  (gen [n-steps drift]
    (let [unfold (comb/unfold-combinator ar-kernel-with-drift)
          result (splice :temporal unfold n-steps (mx/scalar 0.0) drift)]
      result)))

(let [key (rng/fresh-key)
      vtrace (dyn/vsimulate model-unfold-drift [4 (mx/scalar 0.5)] 50 key)]
  (assert-true "multi-arg batched unfold works" (some? vtrace))
  (mx/eval! (:score vtrace))
  (assert-true "multi-arg score is [50]-shaped" (= [50] (mx/shape (:score vtrace))))
  (println "  score shape:" (mx/shape (:score vtrace))))

;; -- 6. Performance comparison --
(println "\n-- 6. Performance: batched vs fallback --")
(let [key (rng/fresh-key)
      ;; Batched path (fast): 500 particles, 10 steps
      t0 (js/Date.now)
      _ (dotimes [_ 5]
          (let [vt (dyn/vsimulate model-unfold [10] 500 key)]
            (mx/eval! (:score vt))))
      t1 (js/Date.now)
      batched-ms (- t1 t0)
      ;; Scalar path: 50 separate runs (scaled down to avoid Metal exhaustion)
      _ (mx/clear-cache!)
      t2 (js/Date.now)
      _ (let [unfold (comb/unfold-combinator (dyn/auto-key ar-kernel))]
          (dotimes [_ 50]
            (let [tr (p/simulate unfold [10 (mx/scalar 0.0)])]
              (mx/eval! (:score tr)))))
      t3 (js/Date.now)
      scalar-ms (- t3 t2)
      ;; Scale scalar time to equivalent 500 particles
      scalar-equiv (* scalar-ms 10)]
  (println "  batched (5 × 500 particles × 10 steps):" batched-ms "ms")
  (println "  scalar  (50 runs × 10 steps, ×10 extrapolated):" scalar-equiv "ms")
  (when (pos? batched-ms)
    (println "  estimated speedup:" (.toFixed (/ scalar-equiv batched-ms) 1) "x")))

(println "\n=== Done ===")
