(ns bench.gmm
  "GMM Inference — IS weight collapse on mixture models.

   Demonstrates that importance sampling fails on multi-modal posteriors:
   with K=3 components and 8 observations, the discrete latent space has
   3^8 = 6561 configurations, and IS concentrates all weight on one particle.

   Algorithms:
   1. Sequential IS  (N=1000)  — expected ESS near 1
   2. Vectorized IS  (N=1000)  — fast but still collapses
   3. Vectorized IS  (N=10000) — more particles, still low ESS

   Output: results/gmm/data.json

   Usage: bun run --bun nbb bench/gmm.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.importance :as is]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/gmm")))

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
              :or {warmup-n 1 outer-n 3 inner-n 1}}]
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
     :min-ms (apply min outer-times) :max-ms (apply max outer-times)
     :raw outer-times}))

;; ---------------------------------------------------------------------------
;; Model
;; ---------------------------------------------------------------------------

;; 3-component GMM with known mixture weights.
;; Scalar model (uses mx/item) — for sequential IS.
(def gmm-scalar
  (dyn/auto-key
    (gen [data]
      (let [mu0 (trace :mu0 (dist/gaussian 0 10))
            mu1 (trace :mu1 (dist/gaussian 0 10))
            mu2 (trace :mu2 (dist/gaussian 0 10))
            mus [mu0 mu1 mu2]
            weights (mx/array [0.3 0.5 0.2])]
        (doseq [[i y] (map-indexed vector data)]
          (let [z (trace (keyword (str "z" i))
                         (dist/categorical (mx/log weights)))
                z-val (mx/item z)
                mu (nth mus z-val)]
            (trace (keyword (str "y" i))
                   (dist/gaussian mu 1))))
        [mu0 mu1 mu2]))))

;; Vectorized model (no mx/item) — shapes flow through for VIS.
(def mix-log-weights (mx/log (mx/array [0.3 0.5 0.2])))

(def gmm-vec
  (dyn/auto-key
    (gen [data]
      (let [mu0 (trace :mu0 (dist/gaussian 0 10))
            mu1 (trace :mu1 (dist/gaussian 0 10))
            mu2 (trace :mu2 (dist/gaussian 0 10))
            means-arr (mx/stack [mu0 mu1 mu2])]
        (doseq [[i y] (map-indexed vector data)]
          (let [z (trace (keyword (str "z" i))
                         (dist/categorical mix-log-weights))
                mu (mx/take-idx means-arr z)]
            (trace (keyword (str "y" i))
                   (dist/gaussian mu 1))))
        [mu0 mu1 mu2]))))

;; ---------------------------------------------------------------------------
;; Data — 8 observations from 3 well-separated clusters
;; ---------------------------------------------------------------------------

(def gmm-data [-3.5 -4.1 0.3 -0.2 3.8 4.2 0.1 3.6])

(def gmm-obs
  (reduce (fn [cm [i y]]
            (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector gmm-data)))

;; ---------------------------------------------------------------------------
;; ESS helpers
;; ---------------------------------------------------------------------------

(defn ess-from-log-weights
  "ESS from a vector of log-weights (MLX scalars or JS numbers).
   ESS = 1 / sum(w_i^2) where w_i are normalized weights."
  [log-weights]
  (u/compute-ess log-weights))

(defn ess-from-vtrace
  "ESS from a VectorizedTrace's [N]-shaped weight array."
  [vtrace]
  (let [w (:weight vtrace)
        _ (mx/materialize! w)
        log-w (mx/->clj w)]
    ;; normalize in JS for stability
    (let [max-w (apply max log-w)
          exp-w (mapv #(js/Math.exp (- % max-w)) log-w)
          sum-w (reduce + exp-w)
          norm  (mapv #(/ % sum-w) exp-w)]
      (/ 1.0 (reduce + (map #(* % %) norm))))))

;; ---------------------------------------------------------------------------
;; Run experiments
;; ---------------------------------------------------------------------------

(println "\n=== GMM Inference: IS Weight Collapse ===")
(println (str "Model: 3-component GMM, " (count gmm-data) " observations"))
(println (str "Latent space: 3^" (count gmm-data) " = " (js/Math.pow 3 (count gmm-data)) " configurations"))

;; --- 1. Sequential IS (N=200) ---

(println "\n--- 1. Sequential IS (N=200) ---")

(def seq-is-result (atom nil))
(mx/clear-cache!)
(def seq-is-timing
  (benchmark "seq-IS-200"
    (fn []
      (let [r (is/importance-sampling {:samples 200}
                                       gmm-scalar [gmm-data] gmm-obs)]
        (reset! seq-is-result r)
        (mx/eval! (:log-ml-estimate r))))
    :warmup-n 1 :outer-n 3 :inner-n 1))

(def seq-is-ess (ess-from-log-weights (:log-weights @seq-is-result)))
(println (str "  ESS: " (.toFixed seq-is-ess 1) " / 200"
              " (" (.toFixed (* 100 (/ seq-is-ess 200)) 1) "%)"))
(println (str "  log-ML: " (.toFixed (mx/item (:log-ml-estimate @seq-is-result)) 4)))

(mx/clear-cache!)

;; --- 2. Vectorized IS (N=1000) ---

(println "\n--- 2. Vectorized IS (N=1000) ---")

(def vis-1k-result (atom nil))
(mx/clear-cache!)
(def vis-1k-timing
  (benchmark "VIS-1000"
    (fn []
      (let [r (is/vectorized-importance-sampling {:samples 1000}
                                                   gmm-vec [gmm-data] gmm-obs)]
        (reset! vis-1k-result r)
        (mx/eval! (:log-ml-estimate r))))
    :warmup-n 1 :outer-n 3 :inner-n 1))

(def vis-1k-ess (ess-from-vtrace (:vtrace @vis-1k-result)))
(println (str "  ESS: " (.toFixed vis-1k-ess 1) " / 1000"
              " (" (.toFixed (* 100 (/ vis-1k-ess 1000)) 1) "%)"))
(println (str "  log-ML: " (.toFixed (mx/item (:log-ml-estimate @vis-1k-result)) 4)))

(mx/clear-cache!)

;; --- 3. Vectorized IS (N=10000) ---

(println "\n--- 3. Vectorized IS (N=10000) ---")

(def vis-10k-result (atom nil))
(mx/clear-cache!)
(def vis-10k-timing
  (benchmark "VIS-10000"
    (fn []
      (let [r (is/vectorized-importance-sampling {:samples 10000}
                                                   gmm-vec [gmm-data] gmm-obs)]
        (reset! vis-10k-result r)
        (mx/eval! (:log-ml-estimate r))))
    :warmup-n 1 :outer-n 3 :inner-n 1))

(def vis-10k-ess (ess-from-vtrace (:vtrace @vis-10k-result)))
(println (str "  ESS: " (.toFixed vis-10k-ess 1) " / 10000"
              " (" (.toFixed (* 100 (/ vis-10k-ess 10000)) 1) "%)"))
(println (str "  log-ML: " (.toFixed (mx/item (:log-ml-estimate @vis-10k-result)) 4)))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println "\n\n========================================")
(println "    GMM IS WEIGHT COLLAPSE RESULTS")
(println "========================================\n")

(def all-results
  [{:config "seq-IS"  :particles 200  :ess seq-is-ess
    :log-ml (mx/item (:log-ml-estimate @seq-is-result))  :timing seq-is-timing}
   {:config "VIS"     :particles 1000  :ess vis-1k-ess
    :log-ml (mx/item (:log-ml-estimate @vis-1k-result))  :timing vis-1k-timing}
   {:config "VIS"     :particles 10000  :ess vis-10k-ess
    :log-ml (mx/item (:log-ml-estimate @vis-10k-result)) :timing vis-10k-timing}])

(println "| Algorithm | N | ESS | ESS% | Time (ms) | log-ML |")
(println "|-----------|---|-----|------|-----------|--------|")
(doseq [{:keys [config particles ess log-ml timing]} all-results]
  (println (str "| " config
                " | " particles
                " | " (.toFixed ess 1)
                " | " (.toFixed (* 100 (/ ess particles)) 1) "%"
                " | " (.toFixed (:mean-ms timing) 1) " ± " (.toFixed (:std-ms timing) 1)
                " | " (.toFixed log-ml 2)
                " |")))

;; ---------------------------------------------------------------------------
;; Write data.json
;; ---------------------------------------------------------------------------

(write-json "data.json"
  {:experiment "gmm-is-weight-collapse"
   :model {:name "3-component-GMM"
           :K 3
           :n-obs (count gmm-data)
           :mixture-weights [0.3 0.5 0.2]
           :obs-sigma 1.0
           :prior-mean 0 :prior-std 10
           :latent-configs (js/Math.pow 3 (count gmm-data))}
   :data gmm-data
   :timestamp (.toISOString (js/Date.))
   :results
   (mapv (fn [{:keys [config particles ess log-ml timing]}]
           {:algorithm config
            :particles particles
            :ess ess
            :ess-pct (/ ess particles)
            :log-ml log-ml
            :mean-ms (:mean-ms timing)
            :std-ms (:std-ms timing)
            :min-ms (:min-ms timing)
            :max-ms (:max-ms timing)
            :raw-times (:raw timing)})
         all-results)})

(println "\nGMM benchmark complete.")
