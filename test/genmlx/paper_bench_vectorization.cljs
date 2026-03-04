(ns genmlx.paper-bench-vectorization
  "Paper Experiment 1: Vectorization speedup curves.

   Measures sequential vs batched inference across N = 1..10000 particles.
   Uses proper timing protocol (performance.now, warmup, nested loops).

   Usage: bun run --bun nbb test/genmlx/paper_bench_vectorization.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.vectorized :as vec]
            [genmlx.inference.importance :as is]
            [genmlx.inference.smc :as smc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Timing infrastructure
;; ---------------------------------------------------------------------------

(defn perf-now [] (js/performance.now))

(defn timing [f repeats inner-repeats]
  (let [times (loop [i 0 acc (transient [])]
                (if (>= i repeats)
                  (persistent! acc)
                  (let [inner-min
                        (loop [j 0 best js/Infinity]
                          (if (>= j inner-repeats)
                            best
                            (let [start (perf-now)
                                  _ (f)
                                  elapsed (- (perf-now) start)]
                              (mx/clear-cache!)
                              (recur (inc j) (min best elapsed)))))]
                    (recur (inc i) (conj! acc inner-min)))))
        n (count times)
        mean (/ (reduce + times) n)
        variance (/ (reduce + (map #(let [d (- % mean)] (* d d)) times)) n)
        std (js/Math.sqrt variance)]
    {:times times :mean mean :std std}))

(defn benchmark-with-warmup [f opts]
  (let [{:keys [warmup-runs repeats inner-repeats]
         :or {warmup-runs 5 repeats 10 inner-repeats 10}} opts]
    (dotimes [_ warmup-runs] (f) (mx/clear-cache!))
    (timing f repeats inner-repeats)))

;; ---------------------------------------------------------------------------
;; JSON output
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))
(def results-dir
  (.resolve path-mod (js/process.cwd) "results/exp1_vectorization"))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir results-dir)
  (let [filepath (str results-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  Wrote: " filepath))))

;; ---------------------------------------------------------------------------
;; Model: 7-site linear regression (slope, intercept, 5 obs)
;; ---------------------------------------------------------------------------

(def linreg-model
  (dyn/auto-key
    (gen [xs]
      (let [slope     (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 10))]
        (doseq [[j x] (map-indexed vector xs)]
          (trace (keyword (str "y" j))
                 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                        intercept) 1)))
        slope))))

(def linreg-xs [1.0 2.0 3.0 4.0 5.0])
(def linreg-obs
  (cm/choicemap :y0 (mx/scalar 2.1) :y1 (mx/scalar 3.9)
                :y2 (mx/scalar 6.2) :y3 (mx/scalar 7.8) :y4 (mx/scalar 10.1)))

;; ---------------------------------------------------------------------------
;; Benchmark 1a: Particle scaling
;; ---------------------------------------------------------------------------

(println "\n=== Paper Experiment 1: Vectorization Speedup ===")
(println "Model: 7-site linear regression, proper timing protocol")

;; Cap at N=1000 for sequential benchmark — N=5000+ sequential generates
;; exceed Metal resource limits due to cumulative array allocation
(def N-values [1 10 100 500 1000])

(defn run-particle-scaling []
  (println "\n-- Benchmark 1a: Particle scaling --")
  (let [results
        (doall
          (for [n N-values]
            (do
              (println (str "\n  N=" n ":"))
              ;; Adaptive reps: fewer for large N
              (let [reps (cond (>= n 1000) {:warmup-runs 2 :repeats 5 :inner-repeats 3}
                               (>= n 500)  {:warmup-runs 3 :repeats 5 :inner-repeats 3}
                               :else        {:warmup-runs 3 :repeats 7 :inner-repeats 5})

                    ;; Sequential: N separate generate calls
                    seq-result
                    (benchmark-with-warmup
                      (fn []
                        (mx/tidy
                          #(dotimes [_ n]
                             (let [{:keys [trace weight]} (p/generate linreg-model [linreg-xs] linreg-obs)]
                               (mx/eval! (:score trace) weight)))))
                      reps)
                    _ (println (str "    Sequential: " (.toFixed (:mean seq-result) 2)
                                    " +/- " (.toFixed (:std seq-result) 2) " ms"))

                    ;; Batched: single vgenerate
                    batch-result
                    (benchmark-with-warmup
                      (fn []
                        (mx/tidy
                          #(let [vt (dyn/vgenerate linreg-model [linreg-xs] linreg-obs n nil)]
                             (mx/eval! (:score vt) (:weight vt)))))
                      reps)
                    _ (println (str "    Batched:    " (.toFixed (:mean batch-result) 2)
                                    " +/- " (.toFixed (:std batch-result) 2) " ms"))

                    speedup (/ (:mean seq-result) (:mean batch-result))]
                (println (str "    Speedup: " (.toFixed speedup 1) "x"))
                {:n n
                 :sequential {:mean (:mean seq-result) :std (:std seq-result)}
                 :batched {:mean (:mean batch-result) :std (:std batch-result)}
                 :speedup speedup}))))]
    (write-json "particle_scaling.json"
                {:benchmark "particle_scaling"
                 :model "7-site linear regression"
                 :methodology "performance.now, warmup, min-of-inner, mean+std-of-outer"
                 :results results})
    results))

;; ---------------------------------------------------------------------------
;; Benchmark 1b: Speedup by inference method (at N=1000)
;; ---------------------------------------------------------------------------

(defn run-method-speedup []
  (println "\n-- Benchmark 1b: Speedup by inference method (N=1000) --")
  (let [n 1000
        reps {:warmup-runs 2 :repeats 5 :inner-repeats 3}

        ;; 1. dist-sample-n
        _ (println "\n  dist-sample-n:")
        d (dist/gaussian 0 1)
        key (rng/fresh-key)
        seq-dist
        (benchmark-with-warmup
          (fn []
            (mx/tidy
              #(let [keys (rng/split-n key n)]
                 (doseq [k keys]
                   (let [v (dc/dist-sample d k)]
                     (mx/eval! v))))))
          reps)
        _ (println (str "    Sequential: " (.toFixed (:mean seq-dist) 2) " ms"))
        batch-dist
        (benchmark-with-warmup
          (fn []
            (mx/tidy
              #(let [v (dc/dist-sample-n d key n)]
                 (mx/eval! v))))
          reps)
        _ (println (str "    Batched:    " (.toFixed (:mean batch-dist) 2) " ms"))
        dist-speedup (/ (:mean seq-dist) (:mean batch-dist))
        _ (println (str "    Speedup: " (.toFixed dist-speedup 1) "x"))

        ;; 2. IS (importance sampling)
        _ (println "\n  Importance sampling:")
        seq-is
        (benchmark-with-warmup
          (fn []
            (mx/tidy
              #(let [r (is/importance-sampling {:samples n} linreg-model [linreg-xs] linreg-obs)]
                 (mx/eval! (:log-ml-estimate r)))))
          reps)
        _ (println (str "    Sequential: " (.toFixed (:mean seq-is) 2) " ms"))
        batch-is
        (benchmark-with-warmup
          (fn []
            (mx/tidy
              #(let [r (is/vectorized-importance-sampling {:samples n} linreg-model [linreg-xs] linreg-obs)]
                 (mx/eval! (:log-ml-estimate r)))))
          reps)
        _ (println (str "    Batched:    " (.toFixed (:mean batch-is) 2) " ms"))
        is-speedup (/ (:mean seq-is) (:mean batch-is))
        _ (println (str "    Speedup: " (.toFixed is-speedup 1) "x"))

        ;; 3. SMC-init
        _ (println "\n  SMC init:")
        seq-smc
        (benchmark-with-warmup
          (fn []
            (mx/tidy
              #(let [results (mapv (fn [_] (p/generate linreg-model [linreg-xs] linreg-obs)) (range n))
                     weights (mapv :weight results)]
                 (mx/eval! (mx/array (mapv mx/realize weights))))))
          reps)
        _ (println (str "    Sequential: " (.toFixed (:mean seq-smc) 2) " ms"))
        batch-smc
        (benchmark-with-warmup
          (fn []
            (mx/tidy
              #(let [{:keys [vtrace]} (smc/vsmc-init linreg-model [linreg-xs] linreg-obs n nil)]
                 (mx/eval! (:weight vtrace)))))
          reps)
        _ (println (str "    Batched:    " (.toFixed (:mean batch-smc) 2) " ms"))
        smc-speedup (/ (:mean seq-smc) (:mean batch-smc))
        _ (println (str "    Speedup: " (.toFixed smc-speedup 1) "x"))

        results [{:method "dist-sample-n"
                  :sequential {:mean (:mean seq-dist) :std (:std seq-dist)}
                  :batched {:mean (:mean batch-dist) :std (:std batch-dist)}
                  :speedup dist-speedup}
                 {:method "importance_sampling"
                  :sequential {:mean (:mean seq-is) :std (:std seq-is)}
                  :batched {:mean (:mean batch-is) :std (:std batch-is)}
                  :speedup is-speedup}
                 {:method "smc_init"
                  :sequential {:mean (:mean seq-smc) :std (:std seq-smc)}
                  :batched {:mean (:mean batch-smc) :std (:std batch-smc)}
                  :speedup smc-speedup}]]
    (write-json "method_speedup.json"
                {:benchmark "method_speedup"
                 :n_particles n
                 :methodology "performance.now, warmup, min-of-inner, mean+std-of-outer"
                 :results results})
    results))

;; ---------------------------------------------------------------------------
;; Main
;; ---------------------------------------------------------------------------

(let [scaling-results (run-particle-scaling)
      method-results (run-method-speedup)]

  ;; Write SUMMARY.md
  (let [summary
        (str "# Experiment 1: Vectorized Inference Speedup\n\n"
             "**Date:** 2026-03-03\n"
             "**Platform:** macOS, Apple Silicon, MLX GPU via @frost-beta/mlx, Bun + nbb\n"
             "**Benchmark file:** `test/genmlx/paper_bench_vectorization.cljs`\n"
             "**Methodology:** performance.now(), warmup, min-of-inner, mean+std-of-outer\n\n"
             "## Benchmark 1a: Particle Scaling\n\n"
             "| N | Sequential (ms) | Batched (ms) | Speedup |\n"
             "|---|----------------|--------------|--------|\n"
             (apply str
               (for [{:keys [n sequential batched speedup]} scaling-results]
                 (str "| " n " | " (.toFixed (:mean sequential) 1) " +/- " (.toFixed (:std sequential) 1)
                      " | " (.toFixed (:mean batched) 2) " +/- " (.toFixed (:std batched) 2)
                      " | " (.toFixed speedup 1) "x |\n")))
             "\n## Benchmark 1b: Method Speedup (N=1000)\n\n"
             "| Method | Sequential (ms) | Batched (ms) | Speedup |\n"
             "|--------|----------------|--------------|--------|\n"
             (apply str
               (for [{:keys [method sequential batched speedup]} method-results]
                 (str "| " method " | " (.toFixed (:mean sequential) 1) " +/- " (.toFixed (:std sequential) 1)
                      " | " (.toFixed (:mean batched) 2) " +/- " (.toFixed (:std batched) 2)
                      " | " (.toFixed speedup 1) "x |\n"))))]
    (.writeFileSync fs (str results-dir "/SUMMARY.md") summary)
    (println (str "\n  Wrote: " results-dir "/SUMMARY.md"))))

(println "\nAll benchmarks complete.")
