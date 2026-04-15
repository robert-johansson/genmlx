(ns bench.vectorization
  "Vectorization benchmark — shape-based batching speedups.

   Measures sequential vs batched performance for:
   1. dist-sample-n (1000 samples of gaussian)
   2. Importance sampling (1000 particles sequential vs vectorized)
   3. SMC init (sequential generate vs vsmc-init)

   Output: results/vectorization/data.json

   Usage: bun run --bun nbb bench/vectorization.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.importance :as is]
            [genmlx.inference.smc :as smc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/vectorization")))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir out-dir)
  (let [filepath (str out-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  wrote: " filepath))))

(defn perf-now [] (js/performance.now))

(defn benchmark
  "Run f repeatedly, return timing statistics.
   outer-n independent runs (with cache clear between), each takes min of inner-n."
  [label f & {:keys [warmup-n outer-n inner-n]
              :or {warmup-n 3 outer-n 5 inner-n 3}}]
  (println (str "\n  [" label "] warming up..."))
  (dotimes [_ warmup-n] (f) (mx/materialize!))
  (mx/clear-cache!)
  (let [outer-times
        (vec (for [_ (range outer-n)]
               (let [inner-times
                     (vec (for [_ (range inner-n)]
                            (let [t0 (perf-now)]
                              (f)
                              (mx/materialize!)
                              (- (perf-now) t0))))]
                 (mx/clear-cache!)
                 (apply min inner-times))))
        mean-ms (/ (reduce + outer-times) (count outer-times))
        std-ms  (js/Math.sqrt (/ (reduce + (map #(* (- % mean-ms) (- % mean-ms))
                                                  outer-times))
                                   (max 1 (dec (count outer-times)))))]
    (println (str "  [" label "] " (.toFixed mean-ms 3) " +/- " (.toFixed std-ms 3) " ms"))
    {:label label :mean-ms mean-ms :std-ms std-ms
     :min-ms (apply min outer-times) :max-ms (apply max outer-times)
     :raw outer-times}))

;; ---------------------------------------------------------------------------
;; Model: linear regression (same as compilation-ladder)
;; ---------------------------------------------------------------------------

(def sigma-obs 1.0)
(def sigma-prior 10.0)

(def linreg
  (dyn/auto-key
    (gen [xs]
      (let [slope     (trace :slope (dist/gaussian 0 sigma-prior))
            intercept (trace :intercept (dist/gaussian 0 sigma-prior))]
        (doseq [[j x] (map-indexed vector xs)]
          (trace (keyword (str "y" j))
                 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                        intercept)
                                sigma-obs)))
        slope))))

(def xs-raw [1.0 2.0 3.0 4.0 5.0])

(def obs
  (-> cm/EMPTY
      (cm/set-choice [:y0] (mx/scalar 2.3))
      (cm/set-choice [:y1] (mx/scalar 4.7))
      (cm/set-choice [:y2] (mx/scalar 6.1))
      (cm/set-choice [:y3] (mx/scalar 8.9))
      (cm/set-choice [:y4] (mx/scalar 10.2))))

;; ---------------------------------------------------------------------------
;; Benchmark 1: N x dist-sample vs dist-sample-n (gaussian, N=1000)
;; ---------------------------------------------------------------------------

(println "\n=== Vectorization Benchmarks ===")
(println "\n--- Benchmark 1: dist-sample vs dist-sample-n (gaussian, N=1000) ---")

(def N-sample 1000)

(def b1-sequential
  (benchmark "sequential-sample"
    (fn []
      (let [d (dist/gaussian 0 1)
            keys (rng/split-n (rng/fresh-key) N-sample)]
        (doseq [k keys]
          (let [v (dc/dist-sample d k)]
            (mx/eval! v)))))
    :warmup-n 2 :outer-n 4 :inner-n 3))

(def b1-batched
  (benchmark "batched-sample-n"
    (fn []
      (let [d (dist/gaussian 0 1)
            v (dc/dist-sample-n d (rng/fresh-key) N-sample)]
        (mx/eval! v)))
    :warmup-n 2 :outer-n 4 :inner-n 3))

(def b1-speedup
  (if (pos? (:mean-ms b1-batched))
    (/ (:mean-ms b1-sequential) (:mean-ms b1-batched))
    ##Inf))

(println (str "\n  Speedup (dist-sample-n): " (.toFixed b1-speedup 1) "x"))

;; ---------------------------------------------------------------------------
;; Benchmark 2: Sequential IS vs Vectorized IS (N=1000)
;; ---------------------------------------------------------------------------

(println "\n--- Benchmark 2: Sequential IS vs Vectorized IS (N=1000) ---")

(def N-is 1000)

(def b2-sequential
  (benchmark "sequential-IS"
    (fn []
      (let [r (is/importance-sampling {:samples N-is} linreg [xs-raw] obs)]
        (mx/eval! (:log-ml-estimate r))))
    :warmup-n 2 :outer-n 3 :inner-n 2))

(def b2-vectorized
  (benchmark "vectorized-IS"
    (fn []
      (let [r (is/vectorized-importance-sampling {:samples N-is} linreg [xs-raw] obs)]
        (mx/eval! (:log-ml-estimate r))))
    :warmup-n 2 :outer-n 3 :inner-n 2))

(def b2-speedup
  (if (pos? (:mean-ms b2-vectorized))
    (/ (:mean-ms b2-sequential) (:mean-ms b2-vectorized))
    ##Inf))

(println (str "\n  Speedup (vectorized IS): " (.toFixed b2-speedup 1) "x"))

;; Sanity check: both should give similar log-ml estimates
(let [seq-r (is/importance-sampling {:samples N-is} linreg [xs-raw] obs)
      vec-r (is/vectorized-importance-sampling {:samples N-is} linreg [xs-raw] obs)]
  (mx/materialize! (:log-ml-estimate seq-r) (:log-ml-estimate vec-r))
  (println (str "  Sequential log-ml: " (.toFixed (mx/item (:log-ml-estimate seq-r)) 2)))
  (println (str "  Vectorized log-ml: " (.toFixed (mx/item (:log-ml-estimate vec-r)) 2))))

;; ---------------------------------------------------------------------------
;; Benchmark 3: Sequential SMC init vs batched SMC init
;; ---------------------------------------------------------------------------

(println "\n--- Benchmark 3: Sequential SMC init vs batched SMC init (N=100) ---")
(mx/clear-cache!)

(def N-smc 100)

(def b3-sequential
  (benchmark "sequential-SMC-init"
    (fn []
      (let [results (mapv (fn [_] (p/generate linreg [xs-raw] obs)) (range N-smc))
            weights (mapv :weight results)]
        (mx/eval! (mx/array (mapv mx/item weights)))))
    :warmup-n 1 :outer-n 3 :inner-n 2))

(mx/clear-cache!)
(def b3-batched
  (benchmark "batched-SMC-init"
    (fn []
      (let [{:keys [vtrace]} (smc/vsmc-init linreg [xs-raw] obs N-smc nil)]
        (mx/eval! (:weight vtrace))))
    :warmup-n 1 :outer-n 3 :inner-n 2))

(def b3-speedup
  (if (pos? (:mean-ms b3-batched))
    (/ (:mean-ms b3-sequential) (:mean-ms b3-batched))
    ##Inf))

(println (str "\n  Speedup (batched SMC init): " (.toFixed b3-speedup 1) "x"))

;; ---------------------------------------------------------------------------
;; Summary table
;; ---------------------------------------------------------------------------

(println "\n\n========================================")
(println "      VECTORIZATION BENCHMARK RESULTS")
(println "========================================\n")

(println "| Benchmark | Sequential (ms) | Batched (ms) | Speedup |")
(println "|-----------|-----------------|--------------|---------|")

(doseq [[label seq-t bat-t spd]
        [["dist-sample-n (N=1000)" b1-sequential b1-batched b1-speedup]
         ["Vectorized IS (N=1000)" b2-sequential b2-vectorized b2-speedup]
         ["Batched SMC init (N=1000)" b3-sequential b3-batched b3-speedup]]]
  (println (str "| " label
                " | " (.toFixed (:mean-ms seq-t) 1) " +/- " (.toFixed (:std-ms seq-t) 1)
                " | " (.toFixed (:mean-ms bat-t) 1) " +/- " (.toFixed (:std-ms bat-t) 1)
                " | " (.toFixed spd 1) "x |")))

;; ---------------------------------------------------------------------------
;; Write data.json
;; ---------------------------------------------------------------------------

(write-json "data.json"
  {:experiment "vectorization"
   :description "Shape-based batching speedup measurements"
   :model {:name "linear-regression" :n-obs 5 :n-trace-sites 7}
   :results
   [{:benchmark "dist-sample-n"
     :N N-sample
     :sequential {:mean-ms (:mean-ms b1-sequential) :std-ms (:std-ms b1-sequential)
                  :min-ms (:min-ms b1-sequential) :max-ms (:max-ms b1-sequential)
                  :raw (:raw b1-sequential)}
     :batched    {:mean-ms (:mean-ms b1-batched) :std-ms (:std-ms b1-batched)
                  :min-ms (:min-ms b1-batched) :max-ms (:max-ms b1-batched)
                  :raw (:raw b1-batched)}
     :speedup b1-speedup}
    {:benchmark "vectorized-IS"
     :N N-is
     :sequential {:mean-ms (:mean-ms b2-sequential) :std-ms (:std-ms b2-sequential)
                  :min-ms (:min-ms b2-sequential) :max-ms (:max-ms b2-sequential)
                  :raw (:raw b2-sequential)}
     :batched    {:mean-ms (:mean-ms b2-vectorized) :std-ms (:std-ms b2-vectorized)
                  :min-ms (:min-ms b2-vectorized) :max-ms (:max-ms b2-vectorized)
                  :raw (:raw b2-vectorized)}
     :speedup b2-speedup}
    {:benchmark "batched-SMC-init"
     :N N-smc
     :sequential {:mean-ms (:mean-ms b3-sequential) :std-ms (:std-ms b3-sequential)
                  :min-ms (:min-ms b3-sequential) :max-ms (:max-ms b3-sequential)
                  :raw (:raw b3-sequential)}
     :batched    {:mean-ms (:mean-ms b3-batched) :std-ms (:std-ms b3-batched)
                  :min-ms (:min-ms b3-batched) :max-ms (:max-ms b3-batched)
                  :raw (:raw b3-batched)}
     :speedup b3-speedup}]
   :speedups
   {:dist-sample-n b1-speedup
    :vectorized-IS b2-speedup
    :batched-SMC-init b3-speedup}})

(println "\nVectorization benchmark complete.")
