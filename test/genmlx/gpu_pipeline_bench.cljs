(ns genmlx.gpu-pipeline-bench
  "Benchmark: sequential vs vectorized IS and ADEV."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.inference.importance :as importance]
            [genmlx.inference.adev :as adev])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Linear regression model
(def xs (mapv #(mx/scalar %) (range 10)))

(def model
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1)))
      slope)))

(def observations
  (apply cm/choicemap
    (mapcat (fn [j]
              [(keyword (str "y" j))
               (mx/scalar (+ (* 2.5 j) 1.0))])
            (range 10))))

(defn bench [label f n-runs]
  (f) ;; warmup
  (mx/clear-cache!)
  (let [start (js/Date.now)]
    (dotimes [_ n-runs] (f))
    (let [elapsed (- (js/Date.now) start)
          per-run (/ elapsed n-runs)]
      (println (str "  " label ": " (.toFixed per-run 1) "ms/run"))
      per-run)))

;; ---------------------------------------------------------------------------
;; IS benchmark
;; ---------------------------------------------------------------------------

(println "\n=== Importance Sampling: Sequential vs Vectorized (GPU) ===")
(println "Model: linear regression, 10 obs, 2 latent\n")

(doseq [n [100 500 1000 5000]]
  (println (str "N=" n ":"))
  (let [t-seq (bench "sequential" #(importance/importance-sampling {:samples n} model [xs] observations) 3)
        t-vec (bench "vectorized" #(importance/vectorized-importance-sampling {:samples n} model [xs] observations) 3)]
    (println (str "  speedup: " (.toFixed (/ t-seq t-vec) 1) "x")))
  (mx/clear-cache!)
  (println))

;; ---------------------------------------------------------------------------
;; ADEV benchmark
;; ---------------------------------------------------------------------------

(println "=== ADEV Optimize: Sequential vs Vectorized ===")

(def simple-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (doseq [i (range 5)]
        (trace (keyword (str "obs" i)) (dist/gaussian mu 1)))
      mu)))

;; Sequential path: cost-fn receives a Trace record
(def cost-seq (fn [trace] (mx/multiply (:score trace) (mx/scalar -1.0))))
;; Vectorized path: cost-fn receives {:choices :score :reinforce-lp :retval}
(def cost-vec (fn [result] (mx/multiply (:score result) (mx/scalar -1.0))))

(doseq [n-samples [10 100]]
  (println (str "\n10 iterations, n-samples=" n-samples ":"))
  (let [t-seq (bench "sequential"
                #(adev/adev-optimize {:iterations 10 :n-samples n-samples :sequential true}
                   simple-model [] cost-seq [:mu] (mx/array [0.0]))
                3)
        t-vec (bench "vectorized"
                #(adev/adev-optimize {:iterations 10 :n-samples n-samples}
                   simple-model [] cost-vec [:mu] (mx/array [0.0]))
                3)]
    (println (str "  speedup: " (.toFixed (/ t-seq t-vec) 1) "x")))
  (mx/clear-cache!))

(println "\nDone.")
