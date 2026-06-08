(ns bench.funnel
  "Neal's Funnel -- challenging posterior geometry for gradient-based samplers.

   Model: v ~ N(0, 3), x_i ~ N(0, exp(v/2)) for i=1..10. D=11 total dimensions.
   No observations -- the target IS the joint funnel distribution (the classic
   Neal funnel benchmark; the narrow neck at small v stresses HMC/NUTS/MH).

   Ground truth: v ~ N(0, 3) marginally -- mean=0, std=3.

   Algorithms (4 chains each -> multi-chain R-hat + bulk/tail-ESS):
   1. HMC  (600 samples, burn 300, 25 leapfrog, adapt step+metric)
   2. NUTS (600 samples, burn 300, max-depth 8, adapt step+metric)
   3. Compiled MH (2000 samples, burn 1000)

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

;; No observations: the target is the joint funnel (v + 10 x_i). We sample the
;; joint and report diagnostics on the v marginal (ground truth v ~ N(0,3)).
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
;; Multi-chain runner: 4 chains -> R-hat + ESS + bulk/tail-ESS on the v marginal
;; ---------------------------------------------------------------------------
;; All three kernels return flat parameter arrays (v is index 0). We run 4
;; chains per kernel (seeds 42-45) and report honest multi-chain diagnostics:
;;   - R-hat  (Gelman-Rubin, multi-chain; target <= 1.01, Vehtari et al. 2021)
;;   - ESS    (sum of per-chain Geyer ESS)
;;   - bulk-ESS / tail-ESS  (rank-normalized, Vehtari et al. 2021)
;; Aggressive cache/GC clearing between chains keeps NUTS within the Metal
;; buffer budget (same approach as paper_bench_funnel).

(defn run-chain
  "Run one chain with a given seed; return {:v-samples [...] :meta {...}}."
  [algo-fn opts seed]
  (let [full-opts (assoc opts :addresses all-addresses :key (rng/fresh-key seed))
        samples (algo-fn full-opts funnel-model [] cm/EMPTY)
        v-samples (extract-v-from-params samples)]
    (mx/clear-cache!) (mx/force-gc!) (mx/clear-cache!)
    {:v-samples v-samples :meta (meta samples)}))

(defn run-kernel
  "Run 4 chains and compute multi-chain diagnostics on v. Returns a result map
   (or an :error map on failure). Timing is chain-1 wall time."
  [label algo-fn opts]
  (println (str "\n-- " label " (4 chains) --"))
  (try
    (let [t0 (perf-now)
          r1 (run-chain algo-fn opts 42)
          t1 (- (perf-now) t0)
          _  (do (mx/clear-cache!) (mx/force-gc!) (println "  chain 2..."))
          r2 (run-chain algo-fn opts 43)
          _  (do (mx/clear-cache!) (mx/force-gc!) (println "  chain 3..."))
          r3 (run-chain algo-fn opts 44)
          _  (do (mx/clear-cache!) (mx/force-gc!) (println "  chain 4..."))
          r4 (run-chain algo-fn opts 45)
          chains    [(:v-samples r1) (:v-samples r2) (:v-samples r3) (:v-samples r4)]
          mx-chains (mapv (fn [c] (mapv #(mx/scalar %) c)) chains)
          all       (vec (apply concat chains))
          rhat (diag/r-hat mx-chains)
          ess  (reduce + (map diag/ess mx-chains))
          be   (diag/bulk-ess chains)
          te   (diag/tail-ess chains)]
      (mx/clear-cache!) (mx/force-gc!)
      (let [res {:algorithm label
                 :n-chains 4
                 :n-samples-per-chain (count (:v-samples r1))
                 :elapsed-ms t1
                 :v-mean (mean all)
                 :v-std (std all)
                 :acceptance-rate (:acceptance-rate (:meta r1))
                 :r-hat rhat
                 :ess ess
                 :bulk-ess be
                 :tail-ess te}]
        (println (str "  v: mean=" (.toFixed (:v-mean res) 4)
                      " std=" (.toFixed (:v-std res) 4) " (truth: 0)"))
        (println (str "  R-hat=" (.toFixed rhat 3)
                      "  ESS=" (.toFixed ess 1)
                      "  bulk-ESS=" (.toFixed be 1)
                      "  tail-ESS=" (.toFixed te 1)
                      "  time=" (.toFixed t1 0) "ms"))
        res))
    (catch :default e
      (println (str "  FAILED: " (.-message e)))
      {:algorithm label :error (str (.-message e))})))

;; ---------------------------------------------------------------------------
;; Run experiments
;; ---------------------------------------------------------------------------

(println "\n=== Neal's Funnel (joint, no observations) ===")
(println "Model: v ~ N(0,3), x_i ~ N(0, exp(v/2)) for i=1..10")
(println "Ground truth: v ~ N(0,3) marginally  ->  mean=0, std=3")
(mx/clear-cache!)

(def hmc-summary
  (run-kernel "HMC"
    mcmc/hmc {:samples 600 :burn 300 :leapfrog-steps 25
              :adapt-step-size true :adapt-metric true :target-accept 0.65}))

(mx/clear-cache!) (mx/force-gc!)

(def nuts-summary
  (run-kernel "NUTS"
    mcmc/nuts {:samples 600 :burn 300 :max-depth 8
               :adapt-step-size true :adapt-metric true :target-accept 0.8}))

(mx/clear-cache!) (mx/force-gc!)

(def cmh-summary
  (run-kernel "compiled-MH"
    mcmc/compiled-mh {:samples 2000 :burn 1000 :proposal-std 0.1}))

;; ---------------------------------------------------------------------------
;; Summary table + write data.json
;; ---------------------------------------------------------------------------

(def all-summaries [hmc-summary nuts-summary cmh-summary])

(println "\n\n========================================")
(println "            FUNNEL RESULTS")
(println "========================================\n")
(println "| Algorithm    | v mean  | v std   | R-hat | ESS    | bulk | tail | Time(ms) |")
(println "|--------------|---------|---------|-------|--------|------|------|----------|")
(doseq [{:keys [algorithm v-mean v-std r-hat ess bulk-ess tail-ess elapsed-ms error]}
        all-summaries]
  (if error
    (println (str "| " (.padEnd algorithm 12 " ") " | ERROR: " error))
    (println (str "| " (.padEnd algorithm 12 " ")
                  " | " (.padStart (.toFixed v-mean 4) 7 " ")
                  " | " (.padStart (.toFixed v-std 4) 7 " ")
                  " | " (.padStart (.toFixed r-hat 3) 5 " ")
                  " | " (.padStart (.toFixed ess 1) 6 " ")
                  " | " (.padStart (.toFixed bulk-ess 0) 4 " ")
                  " | " (.padStart (.toFixed tail-ess 0) 4 " ")
                  " | " (.padStart (.toFixed (or elapsed-ms 0) 0) 8 " ") " |"))))

(println "\nGround truth: v ~ N(0,3) (mean=0, std=3). R-hat target <= 1.01 (Vehtari et al. 2021).")
(println "bulk/tail-ESS are rank-normalized (Vehtari); R-hat + ESS are multi-chain on v.")

(write-json "data.json"
  {:experiment "neals-funnel"
   :model {:name "funnel" :dimensions 11
           :description "joint funnel: v ~ N(0,3), x_i ~ N(0, exp(v/2)), no observations"}
   :diagnostics {:r-hat-target 1.01
                 :n-chains 4
                 :ess-note (str "ess = sum of per-chain Geyer ESS; "
                                "bulk/tail-ESS rank-normalized (Vehtari et al. 2021)")}
   :ground-truth {:v-mean 0.0 :v-std 3.0
                  :note "joint funnel: v ~ N(0,3) marginally (no observations)"}
   :results all-summaries})

(println "\nFunnel benchmark complete.")
