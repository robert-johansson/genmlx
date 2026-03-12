(ns genmlx.paper.bench-07-system
  "Paper Experiment 7: Cross-System Comparison (GenMLX side).

   Runs the same models/algorithms as Gen.jl and GenJAX benchmarks.
   Results combined with bench/genjl/ and bench/genjax/ for the comparison table.

   Usage: bun run --bun nbb test/genmlx/paper/bench_07_system.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.smc :as smc]
            [genmlx.fit :as fit]
            [genmlx.method-selection :as ms])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(defn perf-now [] (js/performance.now))

(def fs (js/require "fs"))
(def path-mod (js/require "path"))
(def results-dir
  (.resolve path-mod (js/process.cwd) "results/paper/exp07_system"))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir results-dir)
  (let [filepath (str results-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  Wrote: " filepath))))

(defn benchmark [label f & {:keys [warmup-n outer-n inner-n]
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
    (println (str "  [" label "] " (.toFixed mean-ms 3) " ± " (.toFixed std-ms 3) " ms"))
    {:label label :mean-ms mean-ms :std-ms std-ms
     :min-ms (apply min outer-times) :raw outer-times}))

(defn safe-benchmark
  "Like benchmark but catches errors and returns a skip marker."
  [label f & opts]
  (try
    (apply benchmark label f opts)
    (catch :default e
      (println (str "  [" label "] SKIPPED — " (.-message e)))
      {:label label :mean-ms nil :std-ms nil :min-ms nil :raw []
       :skipped true :error (.-message e)})))

;; ---------------------------------------------------------------------------
;; Models (matching Gen.jl and GenJAX versions)
;; ---------------------------------------------------------------------------

;; Dynamic linreg (for fair cross-system comparison)
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

;; Static linreg (for L3/L4 showcase)
(def static-linreg
  (dyn/auto-key
    (gen [x1 x2 x3 x4 x5]
      (let [slope     (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 10))]
        (trace :y1 (dist/gaussian (mx/add (mx/multiply slope x1) intercept) 1))
        (trace :y2 (dist/gaussian (mx/add (mx/multiply slope x2) intercept) 1))
        (trace :y3 (dist/gaussian (mx/add (mx/multiply slope x3) intercept) 1))
        (trace :y4 (dist/gaussian (mx/add (mx/multiply slope x4) intercept) 1))
        (trace :y5 (dist/gaussian (mx/add (mx/multiply slope x5) intercept) 1))
        slope))))

;; HMM step (for unfold)
(def hmm-step
  (dyn/auto-key
    (gen [t prev-state]
      (let [A [[0.9 0.1] [0.1 0.9]]
            mus [-2.0 2.0]
            probs (nth A prev-state)
            state (trace :state (dist/categorical (mx/array probs)))
            state-val (mx/item state)
            mu (nth mus state-val)]
        (trace :y (dist/gaussian (mx/scalar mu) 1.0))
        state-val))))

;; GMM (scalar — uses mx/item, not VIS-compatible)
(def gmm-model
  (dyn/auto-key
    (gen [data]
      (let [mus [-4.0 0.0 4.0]
            sigma 1.0
            K 3]
        (doseq [[i y] (map-indexed vector data)]
          (let [z (trace (keyword (str "z" i))
                         (dist/categorical (mx/array (repeat K (/ 1.0 K)))))
                z-val (mx/item z)
                mu (nth mus z-val)]
            (trace (keyword (str "y" i))
                   (dist/gaussian (mx/scalar mu) sigma))))))))

;; GMM vectorized — no mx/item, shapes flow through for VIS
(def gmm-log-weights (mx/array [(js/Math.log (/ 1.0 3.0))
                                 (js/Math.log (/ 1.0 3.0))
                                 (js/Math.log (/ 1.0 3.0))]))
(def gmm-means-arr (mx/array [-4.0 0.0 4.0]))

(def gmm-vec-model
  (dyn/auto-key
    (gen [data]
      (let [sigma 1.0
            K 3]
        (doseq [[i y] (map-indexed vector data)]
          (let [z (trace (keyword (str "z" i))
                         (dist/categorical gmm-log-weights))
                mu (mx/take-idx gmm-means-arr z)]
            (trace (keyword (str "y" i))
                   (dist/gaussian mu (mx/scalar sigma)))))))))

;; ---------------------------------------------------------------------------
;; Data
;; ---------------------------------------------------------------------------

(def linreg-xs [1.0 2.0 3.0 4.0 5.0])
(def linreg-ys [2.3 4.7 6.1 8.9 10.2])
(def linreg-obs
  (reduce (fn [cm [j y]]
            (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector linreg-ys)))

(def static-args [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)
                  (mx/scalar 4.0) (mx/scalar 5.0)])
(def static-obs
  (-> cm/EMPTY
      (cm/set-choice [:y1] (mx/scalar 2.3))
      (cm/set-choice [:y2] (mx/scalar 4.7))
      (cm/set-choice [:y3] (mx/scalar 6.1))
      (cm/set-choice [:y4] (mx/scalar 8.9))
      (cm/set-choice [:y5] (mx/scalar 10.2))))

(def gmm-data [-3.8 -4.2 0.1 -0.3 3.9 4.1 0.2 3.7])
(def gmm-obs
  (reduce (fn [cm [i y]]
            (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector gmm-data)))

(println "\n=== Experiment 7: Cross-System Comparison (GenMLX) ===")

;; ---------------------------------------------------------------------------
;; 1. LinReg IS (1000 particles) — sequential
;; ---------------------------------------------------------------------------

(def is-linreg-seq
  (benchmark "IS-linreg-seq-1000"
    (fn []
      (let [r (is/importance-sampling {:samples 1000}
                                       linreg-model [linreg-xs] linreg-obs)]
        (mx/eval! (:log-ml-estimate r))))
    :warmup-n 5 :outer-n 5 :inner-n 5))

;; ---------------------------------------------------------------------------
;; 2. LinReg VIS (1000 particles) — vectorized
;; ---------------------------------------------------------------------------

(def vis-linreg
  (benchmark "VIS-linreg-1000"
    (fn []
      (let [{:keys [vtrace log-ml-estimate]}
            (is/vectorized-importance-sampling {:samples 1000}
                                                linreg-model [linreg-xs] linreg-obs)]
        (mx/eval! log-ml-estimate)))
    :warmup-n 5 :outer-n 7 :inner-n 10))

;; ---------------------------------------------------------------------------
;; 3. LinReg VIS (10000 particles) — show sublinear scaling
;; ---------------------------------------------------------------------------

(def vis-linreg-10k
  (benchmark "VIS-linreg-10000"
    (fn []
      (let [{:keys [vtrace log-ml-estimate]}
            (is/vectorized-importance-sampling {:samples 10000}
                                                linreg-model [linreg-xs] linreg-obs)]
        (mx/eval! log-ml-estimate)))
    :warmup-n 3 :outer-n 5 :inner-n 5))

;; ---------------------------------------------------------------------------
;; 4. LinReg MH (5000 steps) — compiled
;; ---------------------------------------------------------------------------

(def mh-linreg
  (benchmark "MH-linreg-5000"
    (fn []
      (let [samples (mcmc/compiled-mh
                      {:samples 5000 :burn 0
                       :addresses [:slope :intercept]
                       :proposal-std 0.5}
                      linreg-model [linreg-xs] linreg-obs)]
        (when (seq samples) (mx/eval! (mx/scalar (count samples))))))
    :warmup-n 2 :outer-n 5 :inner-n 3))

;; ---------------------------------------------------------------------------
;; 5. GMM IS (1000 particles) — sequential
;; ---------------------------------------------------------------------------

(def is-gmm-seq
  (benchmark "IS-gmm-seq-1000"
    (fn []
      (let [r (is/importance-sampling {:samples 1000}
                                       gmm-model [gmm-data] gmm-obs)]
        (mx/eval! (:log-ml-estimate r))))
    :warmup-n 3 :outer-n 5 :inner-n 5))

;; ---------------------------------------------------------------------------
;; 6. GMM VIS (1000 particles)
;; ---------------------------------------------------------------------------

(def vis-gmm
  (benchmark "VIS-gmm-1000"
    (fn []
      (let [{:keys [vtrace log-ml-estimate]}
            (is/vectorized-importance-sampling {:samples 1000}
                                                gmm-vec-model [gmm-data] gmm-obs)]
        (mx/eval! log-ml-estimate)))
    :warmup-n 5 :outer-n 7 :inner-n 10))

;; ---------------------------------------------------------------------------
;; 7. L3 exact (static linreg) — GenMLX only
;; ---------------------------------------------------------------------------

(def l3-exact
  (benchmark "L3-exact-linreg"
    (fn []
      (let [{:keys [trace weight]} (p/generate static-linreg static-args static-obs)]
        (mx/eval! weight)))
    :warmup-n 20 :outer-n 7 :inner-n 15))

;; ---------------------------------------------------------------------------
;; 8. L4 fit (static linreg) — GenMLX only
;; ---------------------------------------------------------------------------

(def l4-fit
  (benchmark "L4-fit-linreg"
    (fn []
      (let [result (fit/fit static-linreg static-args static-obs)]
        (mx/eval! (mx/scalar (:elapsed-ms result)))))
    :warmup-n 5 :outer-n 7 :inner-n 10))

;; ---------------------------------------------------------------------------
;; 9. HMC (500 samples) — GenMLX
;; ---------------------------------------------------------------------------

(def hmc-linreg
  (safe-benchmark "HMC-linreg-500"
    (fn []
      (let [samples (mcmc/hmc {:samples 500 :burn 100
                                :leapfrog-steps 10
                                :addresses [:slope :intercept]
                                :adapt-step-size true}
                               linreg-model [linreg-xs] linreg-obs)]
        (when (seq samples)
          (mx/eval! (mx/scalar (count samples))))))
    :warmup-n 2 :outer-n 5 :inner-n 2))

;; ---------------------------------------------------------------------------
;; 10. NUTS (500 samples) — GenMLX
;; ---------------------------------------------------------------------------

(def nuts-linreg
  (safe-benchmark "NUTS-linreg-500"
    (fn []
      (let [samples (mcmc/nuts {:samples 500 :burn 200
                                 :addresses [:slope :intercept]
                                 :adapt-step-size true
                                 :adapt-metric true}
                                linreg-model [linreg-xs] linreg-obs)]
        (when (seq samples)
          (mx/eval! (mx/scalar (count samples))))))
    :warmup-n 2 :outer-n 3 :inner-n 2))

;; ---------------------------------------------------------------------------
;; Collect & write results
;; ---------------------------------------------------------------------------

(println "\n\n========================================")
(println "    CROSS-SYSTEM COMPARISON (GenMLX)")
(println "========================================\n")

(def all-results
  [{:config "IS-linreg-seq" :particles 1000 :timing is-linreg-seq}
   {:config "VIS-linreg" :particles 1000 :timing vis-linreg}
   {:config "VIS-linreg" :particles 10000 :timing vis-linreg-10k}
   {:config "MH-linreg" :samples 5000 :timing mh-linreg}
   {:config "IS-gmm-seq" :particles 1000 :timing is-gmm-seq}
   {:config "VIS-gmm" :particles 1000 :timing vis-gmm}
   {:config "L3-exact" :timing l3-exact}
   {:config "L4-fit" :timing l4-fit}
   {:config "HMC-linreg" :samples 500 :timing hmc-linreg}
   {:config "NUTS-linreg" :samples 500 :timing nuts-linreg}])

(println "| Config | N | Time (ms) |")
(println "|--------|---|-----------|")
(doseq [{:keys [config particles samples timing]} all-results]
  (if (:skipped timing)
    (println (str "| " config " | " (or particles samples "—") " | SKIPPED |"))
    (println (str "| " config " | " (or particles samples "—")
                  " | " (.toFixed (:mean-ms timing) 3) " ± "
                  (.toFixed (:std-ms timing) 3) " |"))))

(write-json "genmlx_results.json"
  {:experiment "exp07_cross_system"
   :system "GenMLX"
   :timestamp (.toISOString (js/Date.))
   :hardware {:platform "macOS" :chip "Apple Silicon" :gpu "Metal"}
   :runtime "bun+nbb (ClojureScript interpreter)"
   :results
   (mapv (fn [{:keys [config particles samples timing]}]
           {:config config
            :particles (or particles samples)
            :mean_ms (:mean-ms timing)
            :std_ms (:std-ms timing)
            :min_ms (:min-ms timing)
            :raw_times (:raw timing)})
         all-results)})

(println "\nExperiment 7 (GenMLX) complete.")
