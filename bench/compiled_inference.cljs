(ns bench.compiled-inference
  "Compiled vs Handler Inference Speedups.

   Three comparisons on a static 5-obs linear regression:
     A) Handler MH vs Compiled MH (200 steps)
     B) Uncompiled vs Compiled score function
     C) Serial compiled MH vs Vectorized multi-chain MH (4 chains)

   Output: results/compiled-inference/data.json

   Usage: bun run --bun nbb bench/compiled_inference.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure (same pattern as compilation_ladder.cljs)
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/compiled-inference")))

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
              :or {warmup-n 10 outer-n 5 inner-n 10}}]
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
;; Model + observations
;; ---------------------------------------------------------------------------

(def linreg
  (dyn/auto-key
    (gen [xs]
      (let [slope     (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 10))]
        (doseq [[j x] (map-indexed vector xs)]
          (trace (keyword (str "y" j))
                 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x)) intercept) 1)))
        slope))))

(def xs-raw [1.0 2.0 3.0 4.0 5.0])

(def obs (-> cm/EMPTY
             (cm/set-choice [:y0] (mx/scalar 2.3))
             (cm/set-choice [:y1] (mx/scalar 4.7))
             (cm/set-choice [:y2] (mx/scalar 6.1))
             (cm/set-choice [:y3] (mx/scalar 8.9))
             (cm/set-choice [:y4] (mx/scalar 10.2))))

;; ---------------------------------------------------------------------------
;; A: Handler MH vs Compiled MH (200 steps)
;; ---------------------------------------------------------------------------

(println "\n=== Compiled Inference Benchmark ===")
(println "Model: 5-obs linear regression\n")

(println "--- A: MH Chain (200 steps) ---")

(def mh-handler-timing
  (benchmark "handler-MH-200"
    #(let [traces (mcmc/mh {:samples 50 :burn 0}
                            linreg [xs-raw] obs)]
       (mx/eval! (:score (last traces))))
    :warmup-n 2 :outer-n 5 :inner-n 3))

(def mh-compiled-timing
  (benchmark "compiled-MH-200"
    #(do (mx/clear-cache!)
         (let [samples (mcmc/compiled-mh
                         {:samples 50 :burn 0
                          :addresses [:slope :intercept]
                          :proposal-std 0.5}
                         linreg [xs-raw] obs)]
           (when (seq samples)
             (mx/eval! (mx/array (mapv first (take-last 1 samples)))))))
    :warmup-n 2 :outer-n 5 :inner-n 3))

(def mh-speedup (/ (:mean-ms mh-handler-timing) (:mean-ms mh-compiled-timing)))
(println (str "\n  MH speedup (handler/compiled): " (.toFixed mh-speedup 2) "x"))

;; ---------------------------------------------------------------------------
;; B: Score function (uncompiled vs compiled)
;; ---------------------------------------------------------------------------

(println "\n--- B: Score Function ---")

(def score-uncompiled-timing
  (benchmark "score-uncompiled"
    (let [score-fn (u/make-score-fn linreg [xs-raw] obs [:slope :intercept])]
      #(let [s (score-fn (mx/array [1.0 0.5]))]
         (mx/eval! s)))
    :warmup-n 10 :outer-n 7 :inner-n 10))

(def score-compiled-timing
  (benchmark "score-compiled"
    (let [score-fn (u/make-compiled-score-fn linreg [xs-raw] obs [:slope :intercept])]
      #(let [s (score-fn (mx/array [1.0 0.5]))]
         (mx/eval! s)))
    :warmup-n 10 :outer-n 7 :inner-n 10))

(def score-speedup (/ (:mean-ms score-uncompiled-timing) (:mean-ms score-compiled-timing)))
(println (str "\n  Score speedup (uncompiled/compiled): " (.toFixed score-speedup 2) "x"))

;; ---------------------------------------------------------------------------
;; C: Serial compiled MH vs Vectorized multi-chain MH (4 chains)
;; ---------------------------------------------------------------------------

(println "\n--- C: Vectorized Multi-Chain MH (4 chains, 200 steps) ---")

(mx/clear-cache!)
(def serial-compiled-timing
  (benchmark "serial-compiled-MH-50"
    #(do (mx/clear-cache!)
         (let [samples (mcmc/compiled-mh
                         {:samples 50 :burn 0
                          :addresses [:slope :intercept]
                          :proposal-std 0.5}
                         linreg [xs-raw] obs)]
           (when (seq samples)
             (mx/eval! (mx/array (mapv first (take-last 1 samples)))))))
    :warmup-n 1 :outer-n 3 :inner-n 1))

(mx/clear-cache!)
(def vectorized-4chain-timing
  (benchmark "vectorized-4chain-MH-50"
    #(do (mx/clear-cache!)
         (let [samples (mcmc/vectorized-compiled-mh
                         {:samples 50 :burn 0
                          :addresses [:slope :intercept]
                          :proposal-std 0.5
                          :n-chains 4}
                         linreg [xs-raw] obs)]
           (when (seq samples)
             (mx/eval! (mx/array (mapv first (take-last 1 samples)))))))
    :warmup-n 1 :outer-n 3 :inner-n 1))

(def vec-speedup (/ (:mean-ms serial-compiled-timing) (:mean-ms vectorized-4chain-timing)))
(println (str "\n  Vectorized speedup (serial/4-chain): " (.toFixed vec-speedup 2) "x"))

;; ---------------------------------------------------------------------------
;; Summary table
;; ---------------------------------------------------------------------------

(println "\n\n========================================")
(println "    COMPILED INFERENCE RESULTS")
(println "========================================\n")

(println "| Comparison | Variant | Time (ms) | Speedup |")
(println "|------------|---------|-----------|---------|")
(println (str "| MH chain | handler | "
              (.toFixed (:mean-ms mh-handler-timing) 3) " +/- "
              (.toFixed (:std-ms mh-handler-timing) 3) " | - |"))
(println (str "| MH chain | compiled | "
              (.toFixed (:mean-ms mh-compiled-timing) 3) " +/- "
              (.toFixed (:std-ms mh-compiled-timing) 3) " | "
              (.toFixed mh-speedup 2) "x |"))
(println (str "| Score fn | uncompiled | "
              (.toFixed (:mean-ms score-uncompiled-timing) 3) " +/- "
              (.toFixed (:std-ms score-uncompiled-timing) 3) " | - |"))
(println (str "| Score fn | compiled | "
              (.toFixed (:mean-ms score-compiled-timing) 3) " +/- "
              (.toFixed (:std-ms score-compiled-timing) 3) " | "
              (.toFixed score-speedup 2) "x |"))
(println (str "| Vec MH | serial-compiled | "
              (.toFixed (:mean-ms serial-compiled-timing) 3) " +/- "
              (.toFixed (:std-ms serial-compiled-timing) 3) " | - |"))
(println (str "| Vec MH | vectorized-4chain | "
              (.toFixed (:mean-ms vectorized-4chain-timing) 3) " +/- "
              (.toFixed (:std-ms vectorized-4chain-timing) 3) " | "
              (.toFixed vec-speedup 2) "x |"))

;; ---------------------------------------------------------------------------
;; Write data.json
;; ---------------------------------------------------------------------------

(write-json "data.json"
  {:experiment "compiled-inference"
   :model "5-obs linreg"
   :comparisons
   {:mh-chain
    {:handler   {:mean-ms (:mean-ms mh-handler-timing)
                 :std-ms  (:std-ms mh-handler-timing)}
     :compiled  {:mean-ms (:mean-ms mh-compiled-timing)
                 :std-ms  (:std-ms mh-compiled-timing)}
     :speedup   mh-speedup}
    :score-fn
    {:uncompiled {:mean-ms (:mean-ms score-uncompiled-timing)
                  :std-ms  (:std-ms score-uncompiled-timing)}
     :compiled   {:mean-ms (:mean-ms score-compiled-timing)
                  :std-ms  (:std-ms score-compiled-timing)}
     :speedup    score-speedup}
    :vectorized-mh
    {:serial-compiled    {:mean-ms (:mean-ms serial-compiled-timing)
                          :std-ms  (:std-ms serial-compiled-timing)}
     :vectorized-4chain  {:mean-ms (:mean-ms vectorized-4chain-timing)
                          :std-ms  (:std-ms vectorized-4chain-timing)}
     :speedup            vec-speedup}}})

(println "\nCompiled inference benchmark complete.")
