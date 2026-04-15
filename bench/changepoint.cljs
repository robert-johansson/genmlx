(ns bench.changepoint
  "Changepoint Detection -- sequential model where IS weight collapse is severe.

   Model: T=50 timesteps. At each step, the mean can change with probability
   p_change=0.1. Observations are noisy measurements of the current mean.

   This is a classic model where importance sampling fails because the
   discrete change indicators create an exponentially large latent space.
   SMC (if available) succeeds by resampling incrementally.

   Algorithms:
   1. Sequential IS (N=100)
   2. Sequential IS (N=500)
   3. SMC (N=100, if available)

   Output: results/changepoint/data.json

   Usage: bun run --bun nbb bench/changepoint.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.importance :as is]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/changepoint")))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir out-dir)
  (let [filepath (str out-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  wrote: " filepath))))

(defn perf-now [] (js/performance.now))

(defn safe-benchmark
  "Run a benchmark with try/catch. Returns result map or error map."
  [label f]
  (println (str "\n  [" label "] running..."))
  (try
    (mx/clear-cache!)
    (let [t0 (perf-now)
          result (f)
          elapsed (- (perf-now) t0)]
      (println (str "  [" label "] " (.toFixed elapsed 1) " ms"))
      {:label label :elapsed-ms elapsed :result result})
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

(defn ess-from-log-weights
  "Effective sample size from unnormalized log-weights."
  [log-ws]
  (let [ws (if (every? number? log-ws)
             log-ws
             (mapv mx/item log-ws))
        max-w (apply max ws)
        exp-w (mapv #(js/Math.exp (- % max-w)) ws)
        sum-w (reduce + exp-w)
        norm  (mapv #(/ % sum-w) exp-w)]
    (/ 1.0 (reduce + (map #(* % %) norm)))))

(defn log-ml-from-log-weights
  "Log marginal likelihood from unnormalized log-weights."
  [log-ws]
  (let [ws (if (every? number? log-ws)
             log-ws
             (mapv mx/item log-ws))
        n (count ws)
        max-w (apply max ws)]
    (+ max-w (- (js/Math.log (reduce + (map #(js/Math.exp (- % max-w)) ws)))
                (js/Math.log n)))))

;; ---------------------------------------------------------------------------
;; Model: changepoint detection
;; ---------------------------------------------------------------------------

(def T 50)
(def p-change 0.1)

(def changepoint-model
  (dyn/auto-key
    (gen [T]
      (loop [t 0 mu 0.0]
        (when (< t T)
          (let [change? (trace (keyword (str "c" t)) (dist/bernoulli p-change))
                new-mu (if (pos? (mx/item change?))
                         (mx/item (trace (keyword (str "m" t)) (dist/gaussian 0 5)))
                         mu)]
            (trace (keyword (str "y" t)) (dist/gaussian (mx/scalar new-mu) 1))
            (recur (inc t) new-mu)))))))

;; ---------------------------------------------------------------------------
;; Generate synthetic data from known changepoints
;; ---------------------------------------------------------------------------

(println "\n=== Changepoint Detection ===")
(println (str "Model: T=" T " timesteps, p_change=" p-change))

;; True changepoints at t=12 (mu: 0 -> 3) and t=30 (mu: 3 -> -2)
(def true-changepoints [12 30])
(def true-means [0.0 3.0 -2.0])

(defn generate-data
  "Generate synthetic observations from known changepoints."
  [T changepoints means]
  (let [key (rng/fresh-key)
        keys (rng/split-n key T)]
    (loop [t 0 seg 0 data []]
      (if (>= t T)
        data
        (let [seg' (if (and (< seg (count changepoints))
                            (>= t (nth changepoints seg)))
                     (inc seg) seg)
              mu (nth means seg')
              ;; Sample noise from N(0, 1)
              noise (mx/item (rng/normal (nth keys t) []))
              y (+ mu noise)]
          (recur (inc t) seg' (conj data y)))))))

(def synthetic-data (generate-data T true-changepoints true-means))

(println (str "True changepoints at t=" (pr-str true-changepoints)))
(println (str "True means: " (pr-str true-means)))
(println (str "Data range: [" (.toFixed (apply min synthetic-data) 2)
              ", " (.toFixed (apply max synthetic-data) 2) "]"))

;; Build observation choicemap
(def obs
  (reduce (fn [cm [t y]]
            (cm/set-choice cm [(keyword (str "y" t))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector synthetic-data)))

;; ---------------------------------------------------------------------------
;; Algorithm 1: Sequential IS (N=100)
;; ---------------------------------------------------------------------------

(println "\n--- 1. Sequential IS (N=100) ---")

(def is100-result
  (safe-benchmark "seq-IS-100"
    (fn []
      (let [r (is/importance-sampling {:samples 100}
                                       changepoint-model [T] obs)]
        (mx/materialize! (:log-ml-estimate r))
        r))))

(def is100-ess
  (when-let [r (:result is100-result)]
    (ess-from-log-weights (:log-weights r))))

(def is100-log-ml
  (when-let [r (:result is100-result)]
    (mx/item (:log-ml-estimate r))))

(when is100-ess
  (println (str "  ESS: " (.toFixed is100-ess 1) " / 100"
                " (" (.toFixed (* 100 (/ is100-ess 100)) 1) "%)"))
  (println (str "  log-ML: " (.toFixed is100-log-ml 2))))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; Algorithm 2: Sequential IS (N=200)
;; ---------------------------------------------------------------------------

(println "\n--- 2. Sequential IS (N=200) ---")

(def is500-result
  (safe-benchmark "seq-IS-200"
    (fn []
      (let [r (is/importance-sampling {:samples 200}
                                       changepoint-model [T] obs)]
        (mx/materialize! (:log-ml-estimate r))
        r))))

(def is500-ess
  (when-let [r (:result is500-result)]
    (ess-from-log-weights (:log-weights r))))

(def is500-log-ml
  (when-let [r (:result is500-result)]
    (mx/item (:log-ml-estimate r))))

(when is500-ess
  (println (str "  ESS: " (.toFixed is500-ess 1) " / 200"
                " (" (.toFixed (* 100 (/ is500-ess 200)) 1) "%)"))
  (println (str "  log-ML: " (.toFixed is500-log-ml 2))))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; Algorithm 3: SMC (N=100, if available)
;; ---------------------------------------------------------------------------

(println "\n--- 3. SMC (N=100) ---")

;; SMC needs incremental observations -- one choicemap per timestep.
;; Each step adds a new y_t observation.
(def obs-seq
  (mapv (fn [t]
          (cm/set-choice cm/EMPTY [(keyword (str "y" t))]
                         (mx/scalar (nth synthetic-data t))))
        (range T)))

(def smc-result
  (safe-benchmark "SMC-100"
    (fn []
      (let [r (smc/smc {:particles 100
                         :ess-threshold 0.5}
                        changepoint-model [T] obs-seq)]
        (mx/materialize! (:log-ml-estimate r))
        r))))

(def smc-ess
  (when-let [r (:result smc-result)]
    (ess-from-log-weights (:log-weights r))))

(def smc-log-ml
  (when-let [r (:result smc-result)]
    (mx/item (:log-ml-estimate r))))

(when smc-ess
  (println (str "  ESS: " (.toFixed smc-ess 1) " / 100"
                " (" (.toFixed (* 100 (/ smc-ess 100)) 1) "%)"))
  (println (str "  log-ML: " (.toFixed smc-log-ml 2))))

;; ---------------------------------------------------------------------------
;; Summary table
;; ---------------------------------------------------------------------------

(println "\n\n========================================")
(println "      CHANGEPOINT DETECTION RESULTS")
(println "========================================\n")

(println (str "T=" T ", true changepoints at " (pr-str true-changepoints)))
(println (str "True means: " (pr-str true-means)))
(println "")

(println "| Algorithm      | ESS / N       | log-ML   | Time (ms) |")
(println "|----------------|---------------|----------|-----------|")

(doseq [[label ess n-particles log-ml result]
        [["seq-IS-100"   is100-ess 100 is100-log-ml is100-result]
         ["seq-IS-200"   is500-ess 200 is500-log-ml is500-result]
         ["SMC-100"      smc-ess   100 smc-log-ml   smc-result]]]
  (if ess
    (println (str "| " (.padEnd label 14 " ")
                  " | " (.padStart (.toFixed ess 1) 5 " ") " / "
                  (.padStart (str n-particles) 3 " ")
                  " (" (.padStart (.toFixed (* 100 (/ ess n-particles)) 1) 4 " ") "%)"
                  " | " (.padStart (.toFixed log-ml 2) 8 " ")
                  " | " (.padStart (.toFixed (or (:elapsed-ms result) 0) 1) 9 " ") " |"))
    (println (str "| " (.padEnd label 14 " ")
                  " |           N/A |      N/A |       N/A |"
                  (when-let [e (:error result)] (str "  " e))))))

(println "\nExpected: IS ESS near 1 (weight collapse on 2^50 latent space).")
(println "SMC should maintain higher ESS by resampling incrementally.")

;; ---------------------------------------------------------------------------
;; Write data.json
;; ---------------------------------------------------------------------------

(defn summarize [label ess n-particles log-ml result]
  (let [base {:algorithm label
              :elapsed-ms (:elapsed-ms result)
              :error (:error result)}]
    (if ess
      (merge base {:ess ess
                   :n-particles n-particles
                   :ess-ratio (/ ess n-particles)
                   :log-ml log-ml})
      base)))

(write-json "data.json"
  {:experiment "changepoint-detection"
   :model {:name "changepoint"
           :T T
           :p-change p-change
           :true-changepoints true-changepoints
           :true-means true-means
           :description "Sequential changepoint model with Bernoulli switches"}
   :data {:n-obs T
          :data-range [(apply min synthetic-data) (apply max synthetic-data)]}
   :results
   [(summarize "seq-IS-100" is100-ess 100 is100-log-ml is100-result)
    (summarize "seq-IS-200" is500-ess 200 is500-log-ml is500-result)
    (summarize "SMC-100"    smc-ess   100 smc-log-ml   smc-result)]
   :interpretation
   {:is-collapse "IS ESS near 1: prior proposal misses the posterior in high-dimensional discrete space"
    :smc-advantage "SMC resamples incrementally, maintaining particle diversity across timesteps"}})

(println "\nChangepoint benchmark complete.")
