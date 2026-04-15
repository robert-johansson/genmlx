(ns bench.funnel
  "Neal's Funnel -- challenging posterior geometry for gradient-based samplers.

   Model: v ~ N(0, 3), x_i ~ N(0, exp(v/2)) for i=1..10. D=11 total dimensions.
   Observe all x_i = 0 (funnel center).

   The funnel's correlated geometry (narrow neck at large v, wide base at
   small v) makes it a standard stress test for HMC, NUTS, and MH.

   Ground truth: E[v | x=0] = 0 (by symmetry).

   Algorithms:
   1. HMC (200 samples, burn 100, 10 leapfrog steps)
   2. NUTS (100 samples, burn 50)
   3. Compiled MH (500 samples, burn 200)

   Output: results/funnel/data.json

   Usage: bun run --bun nbb bench/funnel.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.diagnostics :as diag])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/funnel")))

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
  "Run f once, return timing and result."
  [label f]
  (println (str "\n  [" label "] running..."))
  (mx/clear-cache!)
  (let [t0 (perf-now)
        result (f)
        elapsed (- (perf-now) t0)]
    (println (str "  [" label "] " (.toFixed elapsed 1) " ms"))
    {:label label :elapsed-ms elapsed :result result}))

(defn safe-benchmark
  "Run benchmark with try/catch. Returns nil result on failure."
  [label f]
  (try
    (benchmark label f)
    (catch :default e
      (println (str "  [" label "] FAILED: " (.-message e)))
      {:label label :elapsed-ms nil :result nil :error (str (.-message e))})))

;; ---------------------------------------------------------------------------
;; Statistics helpers
;; ---------------------------------------------------------------------------

(defn mean [xs]
  (when (seq xs)
    (/ (reduce + xs) (count xs))))

(defn std [xs]
  (when (and (seq xs) (> (count xs) 1))
    (let [m (mean xs)
          n (count xs)]
      (js/Math.sqrt (/ (reduce + (map #(let [d (- % m)] (* d d)) xs))
                       (dec n))))))

;; ---------------------------------------------------------------------------
;; Model: Neal's funnel
;; ---------------------------------------------------------------------------

(def funnel-model
  (dyn/auto-key
    (gen []
      (let [v (trace :v (dist/gaussian 0 3))]
        (dotimes [i 10]
          (trace (keyword (str "x" i))
                 (dist/gaussian 0 (mx/exp (mx/divide v (mx/scalar 2))))))
        v))))

;; Observations: all x_i = 0
(def funnel-obs
  (reduce (fn [cm i]
            (cm/set-choice cm [(keyword (str "x" i))] (mx/scalar 0.0)))
          cm/EMPTY
          (range 10)))

(def all-addresses [:v :x0 :x1 :x2 :x3 :x4 :x5 :x6 :x7 :x8 :x9])

;; ---------------------------------------------------------------------------
;; Extract posterior v from samples
;; ---------------------------------------------------------------------------

(defn extract-v-from-params
  "HMC/NUTS/compiled-mh return flat parameter arrays. The first element is v
   (addresses are ordered [:v :x0 :x1 ... :x9])."
  [samples]
  (mapv (fn [s]
          (if (number? s)
            s
            (if (sequential? s) (first s) s)))
        samples))

(defn extract-v-from-traces
  "MH returns traces. Extract :v from each trace's choices."
  [traces]
  (mapv (fn [tr]
          (let [v (cm/get-choice (:choices tr) [:v])]
            (mx/item v)))
        traces))

;; ---------------------------------------------------------------------------
;; Run experiments
;; ---------------------------------------------------------------------------

(println "\n=== Neal's Funnel ===")
(println "Model: v ~ N(0,3), x_i ~ N(0, exp(v/2)) for i=1..10")
(println "Observations: all x_i = 0")
(println "Ground truth: E[v | x=0] ~ 0 (by symmetry)")

;; --- 1. HMC (100 samples, burn 50) ---

(println "\n--- 1. HMC (100 samples, burn 50, 10 leapfrog steps) ---")
(mx/clear-cache!)

(def hmc-result
  (safe-benchmark "HMC-100"
    (fn []
      (mx/clear-cache!)
      (mcmc/hmc {:samples 100 :burn 50
                 :leapfrog-steps 10
                 :step-size 0.01
                 :addresses all-addresses
                 :adapt-step-size true
                 :target-accept 0.65}
                funnel-model [] funnel-obs))))

(def hmc-v-samples
  (when-let [samples (:result hmc-result)]
    (extract-v-from-params samples)))

(when hmc-v-samples
  (let [m (mean hmc-v-samples)
        s (std hmc-v-samples)]
    (println (str "  posterior v: mean=" (.toFixed m 4) " std=" (.toFixed s 4)))
    (println (str "  n-samples: " (count hmc-v-samples)))
    (when-let [ar (:acceptance-rate (meta (:result hmc-result)))]
      (println (str "  acceptance rate: " (.toFixed (* 100 ar) 1) "%")))))

(mx/clear-cache!)

;; --- 2. NUTS (100 samples, burn 50) ---

(println "\n--- 2. NUTS (100 samples, burn 50) ---")
(mx/clear-cache!)

(def nuts-result
  (safe-benchmark "NUTS-100"
    (fn []
      (mx/clear-cache!)
      (mcmc/nuts {:samples 100 :burn 50
                  :step-size 0.01
                  :max-depth 8
                  :addresses all-addresses
                  :adapt-step-size true
                  :target-accept 0.8}
                 funnel-model [] funnel-obs))))

(def nuts-v-samples
  (when-let [samples (:result nuts-result)]
    (extract-v-from-params samples)))

(when nuts-v-samples
  (let [m (mean nuts-v-samples)
        s (std nuts-v-samples)]
    (println (str "  posterior v: mean=" (.toFixed m 4) " std=" (.toFixed s 4)))
    (println (str "  n-samples: " (count nuts-v-samples)))
    (when-let [ar (:acceptance-rate (meta (:result nuts-result)))]
      (println (str "  acceptance rate: " (.toFixed (* 100 ar) 1) "%")))))

(mx/clear-cache!)

;; --- 3. Compiled MH (100 samples, burn 50) ---

(println "\n--- 3. Compiled MH (100 samples, burn 50) ---")
(mx/clear-cache!)

(def cmh-result
  (safe-benchmark "compiled-MH-100"
    (fn []
      (mx/clear-cache!)
      (mcmc/compiled-mh {:samples 100 :burn 50
                         :addresses all-addresses
                         :proposal-std 0.3}
                        funnel-model [] funnel-obs))))

(def cmh-v-samples
  (when-let [samples (:result cmh-result)]
    (extract-v-from-params samples)))

(when cmh-v-samples
  (let [m (mean cmh-v-samples)
        s (std cmh-v-samples)]
    (println (str "  posterior v: mean=" (.toFixed m 4) " std=" (.toFixed s 4)))
    (println (str "  n-samples: " (count cmh-v-samples)))
    (when-let [ar (:acceptance-rate (meta (:result cmh-result)))]
      (println (str "  acceptance rate: " (.toFixed (* 100 ar) 1) "%")))))

;; ---------------------------------------------------------------------------
;; Summary table
;; ---------------------------------------------------------------------------

(println "\n\n========================================")
(println "         FUNNEL RESULTS")
(println "========================================\n")

(println "| Algorithm     | v mean  | v std   | Time (ms) | Samples |")
(println "|---------------|---------|---------|-----------|---------|")

(doseq [[label v-samples result]
        [["HMC-100"        hmc-v-samples  hmc-result]
         ["NUTS-100"       nuts-v-samples nuts-result]
         ["compiled-MH"   cmh-v-samples  cmh-result]]]
  (if v-samples
    (let [m (mean v-samples)
          s (std v-samples)]
      (println (str "| " (.padEnd label 13 " ")
                    " | " (.padStart (.toFixed m 4) 7 " ")
                    " | " (.padStart (.toFixed s 4) 7 " ")
                    " | " (.padStart (.toFixed (or (:elapsed-ms result) 0) 1) 9 " ")
                    " | " (.padStart (str (count v-samples)) 7 " ") " |")))
    (println (str "| " (.padEnd label 13 " ")
                  " |    N/A  |    N/A  |       N/A |     N/A |"
                  (when-let [e (:error result)] (str "  " e))))))

(println "\nGround truth: E[v | x=0] ~ 0")

;; ---------------------------------------------------------------------------
;; Write data.json
;; ---------------------------------------------------------------------------

(defn summarize [label v-samples result]
  (let [base {:algorithm label
              :elapsed-ms (:elapsed-ms result)
              :error (:error result)}]
    (if v-samples
      (merge base {:n-samples (count v-samples)
                   :v-mean (mean v-samples)
                   :v-std (std v-samples)
                   :acceptance-rate (:acceptance-rate (meta (:result result)))})
      base)))

(write-json "data.json"
  {:experiment "neals-funnel"
   :model {:name "funnel" :dimensions 11
           :description "v ~ N(0,3), x_i ~ N(0, exp(v/2)), all x_i observed at 0"}
   :ground-truth {:v-mean 0.0
                  :note "E[v|x=0] ~ 0 by symmetry"}
   :results
   [(summarize "HMC-100" hmc-v-samples hmc-result)
    (summarize "NUTS-100" nuts-v-samples nuts-result)
    (summarize "compiled-MH-100" cmh-v-samples cmh-result)]})

(println "\nFunnel benchmark complete.")
