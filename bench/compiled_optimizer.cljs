(ns bench.compiled-optimizer
  "Compiled Adam vs Handler Loop — optimizer benchmark.

   Compares two optimization paths on the same dynamic linear regression model:
   1. Compiled Adam (co/learn) — auto-selects compilation level, fused gradient + Adam
   2. Handler loop baseline — manual loop with u/make-score-fn + finite-diff Adam

   Reports timing, speedup, final parameters, loss history, and compilation level.

   Output: results/compiled-optimizer/data.json (or GENMLX_RESULTS_DIR env)

   Usage: bun run --bun nbb bench/compiled_optimizer.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.util :as u]
            [genmlx.inference.compiled-optimizer :as co])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure (same pattern as compilation_ladder.cljs)
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/compiled-optimizer")))

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
    (println (str "  [" label "] " (.toFixed mean-ms 3) " +/- " (.toFixed std-ms 3) " ms"))
    {:label label :mean-ms mean-ms :std-ms std-ms
     :min-ms (apply min outer-times) :max-ms (apply max outer-times)
     :raw outer-times}))

;; ---------------------------------------------------------------------------
;; Model: Dynamic linear regression (same as compilation-ladder)
;; ---------------------------------------------------------------------------

(def dynamic-linreg
  (dyn/auto-key
    (gen [xs]
      (let [slope     (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 10))]
        (doseq [[j x] (map-indexed vector xs)]
          (trace (keyword (str "y" j))
                 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                         intercept)
                                 1)))
        slope))))

(def xs-raw [1.0 2.0 3.0 4.0 5.0])

(def observations
  (-> cm/EMPTY
      (cm/set-choice [:y0] (mx/scalar 2.3))
      (cm/set-choice [:y1] (mx/scalar 4.7))
      (cm/set-choice [:y2] (mx/scalar 6.1))
      (cm/set-choice [:y3] (mx/scalar 8.9))
      (cm/set-choice [:y4] (mx/scalar 10.2))))

(def latent-addresses [:slope :intercept])
(def n-iters 200)
(def learning-rate 0.01)

;; ---------------------------------------------------------------------------
;; Benchmark
;; ---------------------------------------------------------------------------

(println "\n=== Compiled Optimizer Benchmark ===")
(println (str "Model: dynamic linear regression, 5 observations"))
(println (str "Task: optimize slope + intercept via Adam (" n-iters " iterations, lr=" learning-rate ")"))

;; --- Path 1: Handler loop baseline ---

(println "\n--- Handler loop baseline (manual Adam, finite-diff gradient) ---")

(defn handler-adam-loop
  "Manual Adam optimization using handler-based score function.
   Each iteration materializes loss and gradient independently."
  [model args obs addresses n-iters lr]
  (let [score-fn (u/make-score-fn model args obs addresses)
        d (count addresses)
        beta1 0.9
        beta2 0.999
        epsilon 1e-8
        beta1-s (mx/scalar beta1)
        beta2-s (mx/scalar beta2)
        one-minus-b1-s (mx/scalar (- 1.0 beta1))
        one-minus-b2-s (mx/scalar (- 1.0 beta2))
        lr-s (mx/scalar lr)
        eps-s (mx/scalar epsilon)]
    (loop [i 0
           params (mx/zeros [d])
           m (mx/zeros [d])
           v (mx/zeros [d])
           final-loss nil]
      (if (>= i n-iters)
        {:params params :final-loss final-loss}
        (let [neg-score-fn (fn [p] (mx/negative (score-fn p)))
              loss (neg-score-fn params)
              ;; Central finite differences for gradient
              h 1e-4
              two-h (mx/scalar (* 2.0 h))
              grad (mx/stack
                     (mapv (fn [j]
                             (let [ej (mx/array (assoc (vec (repeat d 0.0)) j h))
                                   f-plus (neg-score-fn (mx/add params ej))
                                   f-minus (neg-score-fn (mx/subtract params ej))]
                               (mx/divide (mx/subtract f-plus f-minus) two-h)))
                           (range d)))
              _ (mx/materialize! loss grad)

              ;; Adam moment updates
              t (double (inc i))
              new-m (mx/add (mx/multiply beta1-s m)
                            (mx/multiply one-minus-b1-s grad))
              new-v (mx/add (mx/multiply beta2-s v)
                            (mx/multiply one-minus-b2-s (mx/square grad)))
              m-hat (mx/divide new-m (mx/scalar (- 1.0 (js/Math.pow beta1 t))))
              v-hat (mx/divide new-v (mx/scalar (- 1.0 (js/Math.pow beta2 t))))
              update-vec (mx/divide m-hat (mx/add (mx/sqrt v-hat) eps-s))
              new-params (mx/subtract params (mx/multiply lr-s update-vec))
              _ (mx/materialize! new-params new-m new-v)]

          ;; Periodic cleanup
          (when (and (pos? i) (zero? (mod i 50)))
            (mx/clear-cache!)
            (mx/sweep-dead-arrays!))

          (recur (inc i) new-params new-m new-v (mx/item loss)))))))

(mx/clear-cache!)
(def handler-timing
  (benchmark "Handler-Adam-200"
    (fn []
      (handler-adam-loop dynamic-linreg [xs-raw] observations
                         latent-addresses n-iters learning-rate))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; Verify handler loop result
(let [{:keys [params final-loss]}
      (handler-adam-loop dynamic-linreg [xs-raw] observations
                         latent-addresses n-iters learning-rate)]
  (mx/materialize! params)
  (println (str "  Handler final params: " (mx/->clj params)))
  (println (str "  Handler final loss: " (.toFixed final-loss 4))))

;; --- Path 2: Compiled Adam (co/learn) ---

(println "\n--- Compiled Adam (co/learn, fused gradient + update) ---")

(mx/clear-cache!)
(def compiled-timing
  (benchmark "Compiled-Adam-200"
    (fn []
      (co/learn dynamic-linreg [xs-raw] observations latent-addresses
                {:iterations n-iters :lr learning-rate :log-every 1000}))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; Verify compiled Adam result and report details
(let [result (co/learn dynamic-linreg [xs-raw] observations latent-addresses
                       {:iterations n-iters :lr learning-rate :log-every 1})]
  (mx/materialize! (:params result))
  (println (str "  Compiled final params: " (mx/->clj (:params result))))
  (println (str "  Compiled compilation level: " (:compilation-level result)))
  (println (str "  Compiled loss history (last 5): "
                (vec (take-last 5 (:loss-history result))))))

;; ---------------------------------------------------------------------------
;; Speedup
;; ---------------------------------------------------------------------------

(let [speedup (/ (:mean-ms handler-timing) (:mean-ms compiled-timing))]
  (println (str "\n  >>> SPEEDUP: Compiled Adam is "
                (.toFixed speedup 1) "x faster than handler loop <<<")))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println "\n\n========================================")
(println "    COMPILED OPTIMIZER RESULTS")
(println "========================================\n")

(let [speedup (/ (:mean-ms handler-timing) (:mean-ms compiled-timing))]
  (println (str "  Handler Adam (" n-iters " iter):  "
                (.toFixed (:mean-ms handler-timing) 2)
                " +/- " (.toFixed (:std-ms handler-timing) 2) " ms"))
  (println (str "  Compiled Adam (" n-iters " iter): "
                (.toFixed (:mean-ms compiled-timing) 2)
                " +/- " (.toFixed (:std-ms compiled-timing) 2) " ms"))
  (println (str "  Speedup: " (.toFixed speedup 1) "x")))

;; ---------------------------------------------------------------------------
;; Write data.json
;; ---------------------------------------------------------------------------

(let [speedup (/ (:mean-ms handler-timing) (:mean-ms compiled-timing))
      ;; Get detailed results for JSON output
      compiled-result (co/learn dynamic-linreg [xs-raw] observations latent-addresses
                                {:iterations n-iters :lr learning-rate :log-every 1})
      _ (mx/materialize! (:params compiled-result))
      handler-result (handler-adam-loop dynamic-linreg [xs-raw] observations
                                        latent-addresses n-iters learning-rate)
      _ (mx/materialize! (:params handler-result))]
  (write-json "data.json"
    {:experiment "compiled-optimizer"
     :description "Compiled Adam vs handler loop (dynamic linear regression)"
     :timestamp (.toISOString (js/Date.))
     :hardware {:platform "macOS" :chip "Apple Silicon" :gpu "Metal"}

     :config
     {:model "dynamic-linreg"
      :n_observations 5
      :latent_addresses (mapv name latent-addresses)
      :iterations n-iters
      :learning_rate learning-rate}

     :handler_loop
     {:mean_ms (:mean-ms handler-timing)
      :std_ms (:std-ms handler-timing)
      :min_ms (:min-ms handler-timing)
      :max_ms (:max-ms handler-timing)
      :raw_times (:raw handler-timing)
      :final_params (mx/->clj (:params handler-result))
      :final_loss (:final-loss handler-result)}

     :compiled_adam
     {:mean_ms (:mean-ms compiled-timing)
      :std_ms (:std-ms compiled-timing)
      :min_ms (:min-ms compiled-timing)
      :max_ms (:max-ms compiled-timing)
      :raw_times (:raw compiled-timing)
      :final_params (mx/->clj (:params compiled-result))
      :compilation_level (name (:compilation-level compiled-result))
      :loss_history_last_5 (vec (take-last 5 (:loss-history compiled-result)))}

     :speedup speedup}))

(println "\nCompiled optimizer benchmark complete.")
