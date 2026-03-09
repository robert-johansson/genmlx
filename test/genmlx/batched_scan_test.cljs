(ns genmlx.batched-scan-test
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

(println "\n=== Batched Scan Tests ===\n")

;; Scan kernel: carry + input -> [new-carry output]
(def scan-kernel
  (gen [carry input]
    (let [x (trace :x (dist/gaussian (mx/add carry input) (mx/scalar 1.0)))]
      [x (mx/multiply x (mx/scalar 2.0))])))

;; -- 1. Batched scan via vsimulate --
(println "-- 1. Batched scan via vsimulate --")
(def model-scan
  (gen [inputs]
    (let [sc (comb/scan-combinator scan-kernel)
          result (splice :scan sc (mx/scalar 0.0) inputs)]
      result)))

(let [key (rng/fresh-key)
      inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
      vtrace (dyn/vsimulate model-scan [inputs] 100 key)]
  (assert-true "batched scan returns vtrace" (some? vtrace))
  (mx/eval! (:score vtrace))
  (assert-true "score is [100]-shaped" (= [100] (mx/shape (:score vtrace))))
  (let [choices (:choices vtrace)
        step0 (cm/get-submap (cm/get-submap choices :scan) 0)
        x0 (cm/get-value (cm/get-submap step0 :x))]
    (mx/eval! x0)
    (assert-true "step 0 :x is [100]-shaped" (= [100] (mx/shape x0)))
    (println "  score shape:" (mx/shape (:score vtrace))
             "step 0 :x shape:" (mx/shape x0))))

;; -- 2. Batched scan with constraints --
(println "\n-- 2. Batched scan with constraints --")
(let [key (rng/fresh-key)
      inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
      obs (cm/set-value cm/EMPTY [:scan 1 :x] (mx/scalar 5.0))
      vtrace (dyn/vgenerate model-scan [inputs] obs 100 key)]
  (assert-true "vgenerate returns vtrace" (some? vtrace))
  (mx/eval! (:weight vtrace))
  (assert-true "weight is [100]-shaped" (= [100] (mx/shape (:weight vtrace))))
  (println "  weight shape:" (mx/shape (:weight vtrace))
           "mean weight:" (.toFixed (mx/item (mx/mean (:weight vtrace))) 3)))

;; -- 3. Score consistency --
(println "\n-- 3. Score consistency --")
(let [key (rng/fresh-key)
      inputs [(mx/scalar 1.0) (mx/scalar 2.0)]
      vtrace (dyn/vsimulate model-scan [inputs] 50 key)
      _ (mx/eval! (:score vtrace))
      batched-mean (mx/item (mx/mean (:score vtrace)))
      scalar-scores (mapv (fn [_]
                            (let [sc (comb/scan-combinator (dyn/auto-key scan-kernel))
                                  trace (p/simulate sc [(mx/scalar 0.0) inputs])]
                              (mx/eval! (:score trace))
                              (mx/item (:score trace))))
                          (range 200))
      scalar-mean (/ (reduce + scalar-scores) (count scalar-scores))]
  (println "  batched mean:" (.toFixed batched-mean 3)
           "scalar mean:" (.toFixed scalar-mean 3))
  (assert-true "means similar" (< (js/Math.abs (- batched-mean scalar-mean)) 1.5)))

(println "\n=== Done ===")
