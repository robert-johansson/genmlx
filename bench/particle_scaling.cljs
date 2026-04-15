(ns bench.particle-scaling
  "Particle Scaling — sublinear cost on Metal GPU.

   Demonstrates that going from 100 to 10000 particles costs almost nothing
   extra because the GPU absorbs the parallelism. Vectorized importance
   sampling runs the model body ONCE regardless of particle count; MLX
   broadcasting handles all [N]-shaped arithmetic on Metal.

   Output: results/particle-scaling/data.json

   Usage: bun run --bun nbb bench/particle_scaling.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.importance :as is]
            [genmlx.vectorized :as vec])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/particle-scaling")))

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
              :or {warmup-n 2 outer-n 3 inner-n 3}}]
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
;; Model: dynamic linear regression
;; ---------------------------------------------------------------------------

(def n-obs 5)
(def xs-raw [1.0 2.0 3.0 4.0 5.0])

(def model
  (dyn/auto-key
    (gen [xs]
      (let [slope     (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 10))]
        (doseq [[j x] (map-indexed vector xs)]
          (trace (keyword (str "y" j))
                 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x)) intercept) 1)))
        slope))))

(def obs
  (-> cm/EMPTY
      (cm/set-choice [:y0] (mx/scalar 2.3))
      (cm/set-choice [:y1] (mx/scalar 4.7))
      (cm/set-choice [:y2] (mx/scalar 6.1))
      (cm/set-choice [:y3] (mx/scalar 8.9))
      (cm/set-choice [:y4] (mx/scalar 10.2))))

;; ---------------------------------------------------------------------------
;; Particle counts to sweep
;; ---------------------------------------------------------------------------

(def particle-counts [100 250 500 1000 2500 5000 10000])

;; ---------------------------------------------------------------------------
;; Run scaling experiment
;; ---------------------------------------------------------------------------

(println "\n=== Particle Scaling on Metal GPU ===")
(println (str "Model: dynamic-linreg, " n-obs " observations"))
(println (str "Particle counts: " (pr-str particle-counts)))

(def scaling-results
  (vec
    (for [n particle-counts]
      (let [label (str "VIS-N=" n)
            ;; Benchmark timing
            timing (benchmark label
                     #(let [{:keys [vtrace log-ml-estimate]}
                            (is/vectorized-importance-sampling {:samples n}
                                                                model [xs-raw] obs)]
                        (mx/eval! log-ml-estimate))
                     :warmup-n 2 :outer-n 3 :inner-n 3)
            ;; Extract log-ML and ESS from a fresh run
            {:keys [vtrace log-ml-estimate]}
            (is/vectorized-importance-sampling {:samples n} model [xs-raw] obs)
            _ (mx/materialize! log-ml-estimate)
            log-ml (mx/item log-ml-estimate)
            ess (vec/vtrace-ess vtrace)]
        (println (str "    log-ML: " (.toFixed log-ml 4)
                      "  ESS: " (.toFixed ess 1) "/" n))
        {:particles n
         :mean-ms (:mean-ms timing)
         :std-ms (:std-ms timing)
         :log-ml log-ml
         :ess ess
         :raw (:raw timing)}))))

;; ---------------------------------------------------------------------------
;; Compute scaling ratios
;; ---------------------------------------------------------------------------

(defn find-result [n]
  (first (filter #(= (:particles %) n) scaling-results)))

(def r100  (find-result 100))
(def r1k   (find-result 1000))
(def r10k  (find-result 10000))

(def ratio-1k-vs-100
  (when (and r1k r100 (pos? (:mean-ms r100)))
    (/ (:mean-ms r1k) (:mean-ms r100))))

(def ratio-10k-vs-100
  (when (and r10k r100 (pos? (:mean-ms r100)))
    (/ (:mean-ms r10k) (:mean-ms r100))))

(def ratio-10k-vs-1k
  (when (and r10k r1k (pos? (:mean-ms r1k)))
    (/ (:mean-ms r10k) (:mean-ms r1k))))

;; ---------------------------------------------------------------------------
;; Summary table
;; ---------------------------------------------------------------------------

(println "\n\n========================================")
(println "      PARTICLE SCALING RESULTS")
(println "========================================\n")

(println "| Particles | Time (ms)         | log-ML  | ESS     |")
(println "|-----------|-------------------|---------|---------|")

(doseq [{:keys [particles mean-ms std-ms log-ml ess]} scaling-results]
  (println (str "| " (.padStart (str particles) 9 " ")
                " | " (.padStart (.toFixed mean-ms 3) 8 " ")
                " +/- " (.padStart (.toFixed std-ms 3) 6 " ")
                " | " (.padStart (.toFixed log-ml 2) 7 " ")
                " | " (.padStart (.toFixed ess 1) 7 " ") " |")))

(println "\nScaling ratios (time_N / time_100):")
(when ratio-1k-vs-100
  (println (str "  1000  vs 100:   " (.toFixed ratio-1k-vs-100 2) "x  (10x more particles)")))
(when ratio-10k-vs-100
  (println (str "  10000 vs 100:   " (.toFixed ratio-10k-vs-100 2) "x  (100x more particles)")))
(when ratio-10k-vs-1k
  (println (str "  10000 vs 1000:  " (.toFixed ratio-10k-vs-1k 2) "x  (10x more particles)")))

(println "\nSublinear = ratio << particle increase factor.")
(println "GPU absorbs parallelism: 100x more particles costs far less than 100x time.")

;; ---------------------------------------------------------------------------
;; Write data.json
;; ---------------------------------------------------------------------------

(write-json "data.json"
  {:experiment "particle-scaling"
   :model {:name "dynamic-linreg" :n-obs n-obs}
   :scaling
   (mapv (fn [{:keys [particles mean-ms std-ms log-ml ess raw]}]
           {:particles particles
            :mean-ms mean-ms
            :std-ms std-ms
            :log-ml log-ml
            :ess ess
            :raw raw})
         scaling-results)
   :ratios {:1k-vs-100  ratio-1k-vs-100
            :10k-vs-100 ratio-10k-vs-100
            :10k-vs-1k  ratio-10k-vs-1k}})

(println "\nParticle scaling benchmark complete.")
