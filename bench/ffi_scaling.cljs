(ns bench.ffi-scaling
  "FFI overhead vs dimensionality benchmark.

   Compares two approaches for the same model:
     A) Per-site gaussians — D independent trace sites (forces handler path)
     B) gaussian-vec — single vector-valued trace site

   For each D in [10, 25, 50, 100, 200], measures both p/simulate and
   vectorized importance sampling (N=1000 particles).

   Output: results/ffi-scaling/data.json

   Usage: bun run --bun nbb bench/ffi_scaling.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.importance :as is])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure (same pattern as compilation_ladder.cljs)
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/ffi-scaling")))

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
              :or {warmup-n 5 outer-n 5 inner-n 5}}]
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
;; Model constructors
;; ---------------------------------------------------------------------------

(defn make-per-site-model
  "D independent gaussian trace sites with dynamic keyword addresses.
   Forces handler path (no compilation) — measures FFI overhead."
  [D]
  (dyn/auto-key
    (gen []
      (dotimes [i D]
        (trace (keyword (str "x" i)) (dist/gaussian 0 1))))))

(defn make-vec-model
  "Single gaussian-vec trace site of dimension D.
   One FFI call for sampling, one for log-prob."
  [D]
  (dyn/auto-key
    (gen []
      (trace :params (dist/gaussian-vec (mx/zeros [D]) (mx/ones [D]))))))

;; ---------------------------------------------------------------------------
;; Benchmark parameters
;; ---------------------------------------------------------------------------

(def dimensions [10 25 50 100 200])
(def is-particles 1000)

;; ---------------------------------------------------------------------------
;; Run benchmarks
;; ---------------------------------------------------------------------------

(println "\n=== FFI Scaling: Per-site vs gaussian-vec ===")
(println (str "Dimensions: " dimensions))
(println (str "IS particles: " is-particles))

(def results
  (vec
    (for [D dimensions]
      (do
        (println (str "\n\n--- D = " D " ---"))

        ;; Build models
        (let [per-site-model (make-per-site-model D)
              vec-model      (make-vec-model D)]

          ;; Simulate benchmarks
          (let [per-site-sim (benchmark (str "per-site-simulate-D" D)
                               #(p/simulate per-site-model [])
                               :warmup-n 5 :outer-n 5 :inner-n 5)

                _ (mx/clear-cache!)

                vec-sim (benchmark (str "vec-simulate-D" D)
                          #(p/simulate vec-model [])
                          :warmup-n 5 :outer-n 5 :inner-n 5)

                _ (mx/clear-cache!)

                ;; Importance sampling benchmarks
                per-site-is (benchmark (str "per-site-IS-D" D)
                              #(let [{:keys [log-ml-estimate]}
                                     (is/vectorized-importance-sampling
                                       {:samples is-particles}
                                       per-site-model [] cm/EMPTY)]
                                 (mx/eval! log-ml-estimate))
                              :warmup-n 5 :outer-n 5 :inner-n 5)

                _ (mx/clear-cache!)

                vec-is (benchmark (str "vec-IS-D" D)
                         #(let [{:keys [log-ml-estimate]}
                                (is/vectorized-importance-sampling
                                  {:samples is-particles}
                                  vec-model [] cm/EMPTY)]
                            (mx/eval! log-ml-estimate))
                         :warmup-n 5 :outer-n 5 :inner-n 5)

                sim-speedup (/ (:mean-ms per-site-sim) (:mean-ms vec-sim))
                is-speedup  (/ (:mean-ms per-site-is) (:mean-ms vec-is))]

            (mx/clear-cache!)

            {:D D
             :per-site-simulate {:mean-ms (:mean-ms per-site-sim)
                                 :std-ms  (:std-ms per-site-sim)}
             :vec-simulate      {:mean-ms (:mean-ms vec-sim)
                                 :std-ms  (:std-ms vec-sim)}
             :per-site-is       {:mean-ms (:mean-ms per-site-is)
                                 :std-ms  (:std-ms per-site-is)}
             :vec-is            {:mean-ms (:mean-ms vec-is)
                                 :std-ms  (:std-ms vec-is)}
             :simulate-speedup  sim-speedup
             :is-speedup        is-speedup}))))))

;; ---------------------------------------------------------------------------
;; Summary table
;; ---------------------------------------------------------------------------

(println "\n\n========================================")
(println "       FFI SCALING RESULTS")
(println "========================================\n")

(println "| D | Per-site sim (ms) | Vec sim (ms) | Sim speedup | Per-site IS (ms) | Vec IS (ms) | IS speedup |")
(println "|---|-------------------|--------------|-------------|------------------|-------------|------------|")
(doseq [{:keys [D per-site-simulate vec-simulate simulate-speedup
                per-site-is vec-is is-speedup]} results]
  (println (str "| " D
                " | " (.toFixed (:mean-ms per-site-simulate) 2) " +/- " (.toFixed (:std-ms per-site-simulate) 2)
                " | " (.toFixed (:mean-ms vec-simulate) 2) " +/- " (.toFixed (:std-ms vec-simulate) 2)
                " | " (.toFixed simulate-speedup 1) "x"
                " | " (.toFixed (:mean-ms per-site-is) 2) " +/- " (.toFixed (:std-ms per-site-is) 2)
                " | " (.toFixed (:mean-ms vec-is) 2) " +/- " (.toFixed (:std-ms vec-is) 2)
                " | " (.toFixed is-speedup 1) "x |")))

;; ---------------------------------------------------------------------------
;; Write data.json
;; ---------------------------------------------------------------------------

(write-json "data.json"
  {:experiment "ffi-scaling"
   :dimensions dimensions
   :is-particles is-particles
   :results results})

(println "\nFFI scaling benchmark complete.")
