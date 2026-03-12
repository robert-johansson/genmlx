(ns genmlx.paper.bench-10-optimization
  "Paper Experiment 10: L4 Fused Optimization

   Sub-experiments:
   10A: Compiled Adam vs handler loop (speedup benchmark)
   10B: Method selection accuracy (6 models)
   10C: fit API end-to-end (3 models, auto-select verification)

   Usage: bun run --bun nbb test/genmlx/paper/bench_10_optimization.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.util :as u]
            [genmlx.inference.compiled-optimizer :as co]
            [genmlx.method-selection :as ms]
            [genmlx.fit :as fit])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(defn perf-now [] (js/performance.now))

(def fs (js/require "fs"))
(def path-mod (js/require "path"))
(def results-dir
  (.resolve path-mod (js/process.cwd) "results/paper/exp10_optimization"))

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
        (vec (for [rep (range outer-n)]
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
    (println (str "  [" label "] " (.toFixed mean-ms 3) " +/- "
                  (.toFixed std-ms 3) " ms"))
    {:label label :mean-ms mean-ms :std-ms std-ms
     :min-ms (apply min outer-times) :max-ms (apply max outer-times)
     :raw outer-times}))

(println "\n============================================================")
(println "  Experiment 10: L4 Fused Optimization")
(println "============================================================")

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

;; Static linear regression (5 obs, fully static addresses)
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

;; Dynamic linear regression (loop, computed addresses)
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

;; Mixed model (partial conjugate: slope/intercept conjugate, sigma non-conjugate)
(def mixed-model
  (dyn/auto-key
    (gen [x1 x2 x3 x4 x5]
      (let [slope     (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 10))
            sigma     (trace :sigma (dist/gamma-dist 2 1))]
        (trace :y1 (dist/gaussian (mx/add (mx/multiply slope x1) intercept) sigma))
        (trace :y2 (dist/gaussian (mx/add (mx/multiply slope x2) intercept) sigma))
        (trace :y3 (dist/gaussian (mx/add (mx/multiply slope x3) intercept) sigma))
        (trace :y4 (dist/gaussian (mx/add (mx/multiply slope x4) intercept) sigma))
        (trace :y5 (dist/gaussian (mx/add (mx/multiply slope x5) intercept) sigma))
        slope))))

;; Empty model (no trace sites)
(def empty-model
  (dyn/auto-key
    (gen [] (mx/scalar 1))))

;; Large model (15+ latents -> should trigger :vi)
(def large-model
  (dyn/auto-key
    (gen [x1 x2 x3]
      (let [a1  (trace :a1  (dist/gaussian 0 1))
            a2  (trace :a2  (dist/gaussian 0 1))
            a3  (trace :a3  (dist/gaussian 0 1))
            a4  (trace :a4  (dist/gaussian 0 1))
            a5  (trace :a5  (dist/gaussian 0 1))
            a6  (trace :a6  (dist/gaussian 0 1))
            a7  (trace :a7  (dist/gaussian 0 1))
            a8  (trace :a8  (dist/gaussian 0 1))
            a9  (trace :a9  (dist/gaussian 0 1))
            a10 (trace :a10 (dist/gaussian 0 1))
            a11 (trace :a11 (dist/gaussian 0 1))
            a12 (trace :a12 (dist/gaussian 0 1))
            a13 (trace :a13 (dist/gaussian 0 1))
            a14 (trace :a14 (dist/gaussian 0 1))
            a15 (trace :a15 (dist/gaussian 0 1))
            s   (mx/add a1 (mx/add a2 (mx/add a3 (mx/add a4 (mx/add a5
                  (mx/add a6 (mx/add a7 (mx/add a8 (mx/add a9 (mx/add a10
                    (mx/add a11 (mx/add a12 (mx/add a13 (mx/add a14 a15))))))))))))))]
        (trace :obs1 (dist/gaussian s 1))
        (trace :obs2 (dist/gaussian s 1))
        (trace :obs3 (dist/gaussian s 1))
        s))))

;; Static model with splice (uses a sub-model call)
(def sub-model
  (dyn/auto-key
    (gen [mu]
      (let [x (trace :x (dist/gaussian mu 1))]
        x))))

(def splice-model
  (dyn/auto-key
    (gen [a]
      (let [z (trace :z (dist/gaussian 0 1))
            w (splice :sub (sub-model z))]
        (trace :obs (dist/gaussian w 1))
        w))))

;; ---------------------------------------------------------------------------
;; Data setup
;; ---------------------------------------------------------------------------

(def xs-mlx [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)
             (mx/scalar 4.0) (mx/scalar 5.0)])

(def static-obs
  (-> cm/EMPTY
      (cm/set-choice [:y1] (mx/scalar 2.3))
      (cm/set-choice [:y2] (mx/scalar 4.7))
      (cm/set-choice [:y3] (mx/scalar 6.1))
      (cm/set-choice [:y4] (mx/scalar 8.9))
      (cm/set-choice [:y5] (mx/scalar 10.2))))

(def dynamic-obs
  (-> cm/EMPTY
      (cm/set-choice [:y0] (mx/scalar 2.3))
      (cm/set-choice [:y1] (mx/scalar 4.7))
      (cm/set-choice [:y2] (mx/scalar 6.1))
      (cm/set-choice [:y3] (mx/scalar 8.9))
      (cm/set-choice [:y4] (mx/scalar 10.2))))

(def mixed-obs static-obs)

(def large-obs
  (-> cm/EMPTY
      (cm/set-choice [:obs1] (mx/scalar 5.0))
      (cm/set-choice [:obs2] (mx/scalar 4.5))
      (cm/set-choice [:obs3] (mx/scalar 5.5))))

(def splice-obs
  (-> cm/EMPTY
      (cm/set-choice [:obs] (mx/scalar 1.0))))

;; =========================================================================
;; 10A: Compiled Adam vs Handler Loop
;; =========================================================================

(println "\n------------------------------------------------------------")
(println "  10A: Compiled Adam vs Handler Loop")
(println "------------------------------------------------------------")
(println "  Model: Static linear regression (5 observations)")
(println "  Task: Optimize slope + intercept via Adam (200 iterations)")

;; --- Path 1: Handler loop (manual score-fn + Adam step + per-iter eval) ---

(println "\n-- Handler loop (manual Adam, per-iteration materialize) --")

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
        (let [;; Evaluate score and compute finite-diff gradient
              neg-score-fn (fn [p] (mx/negative (score-fn p)))
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

(def handler-timing
  (benchmark "Handler-Adam-200"
    (fn []
      (handler-adam-loop static-linreg xs-mlx static-obs
                         [:slope :intercept] 200 0.01))
    :warmup-n 2 :outer-n 5 :inner-n 3))

;; Verify handler loop result
(let [{:keys [params final-loss]}
      (handler-adam-loop static-linreg xs-mlx static-obs
                         [:slope :intercept] 200 0.01)]
  (mx/materialize! params)
  (println (str "  Handler final params: " (mx/->clj params)))
  (println (str "  Handler final loss: " (.toFixed final-loss 4))))

;; --- Path 2: Compiled Adam (co/learn, fused graph) ---

(println "\n-- Compiled Adam (co/learn, fused gradient + update) --")

(def compiled-timing
  (benchmark "Compiled-Adam-200"
    (fn []
      (co/learn static-linreg xs-mlx static-obs [:slope :intercept]
                {:iterations 200 :lr 0.01 :log-every 1000}))
    :warmup-n 2 :outer-n 5 :inner-n 3))

;; Verify compiled Adam result
(let [result (co/learn static-linreg xs-mlx static-obs [:slope :intercept]
                       {:iterations 200 :lr 0.01 :log-every 50})]
  (mx/materialize! (:params result))
  (println (str "  Compiled final params: " (mx/->clj (:params result))))
  (println (str "  Compiled compilation level: " (:compilation-level result)))
  (println (str "  Compiled loss history (last 3): "
                (take-last 3 (:loss-history result)))))

;; --- Speedup ---

(let [speedup (/ (:mean-ms handler-timing) (:mean-ms compiled-timing))]
  (println (str "\n  >>> SPEEDUP: Compiled Adam is "
                (.toFixed speedup 1) "x faster than handler loop <<<"))
  (println (str "  Handler: " (.toFixed (:mean-ms handler-timing) 2) " ms"))
  (println (str "  Compiled: " (.toFixed (:mean-ms compiled-timing) 2) " ms")))

;; =========================================================================
;; 10B: Method Selection Accuracy
;; =========================================================================

(println "\n\n------------------------------------------------------------")
(println "  10B: Method Selection Accuracy")
(println "------------------------------------------------------------")

(defn check-method [label model observations expected-method]
  (let [result (ms/select-method model observations)
        actual (:method result)
        pass? (= actual expected-method)]
    (println (str "  [" (if pass? "PASS" "FAIL") "] " label
                  ": expected=" (name expected-method)
                  ", actual=" (name actual)
                  " (" (:reason result) ")"
                  ", residual=" (:n-residual result)
                  ", eliminated=" (count (:eliminated result))))
    {:label label
     :expected (name expected-method)
     :actual (name actual)
     :pass pass?
     :reason (:reason result)
     :n_residual (:n-residual result)
     :n_eliminated (count (:eliminated result))}))

;; 1. Static LinReg (fully conjugate) -> expect :exact
(def ms-1
  (check-method "Static LinReg (fully conjugate)"
                static-linreg static-obs :exact))

;; 2. Mixed Model (partial conjugate, sigma non-conjugate) -> expect :hmc
(def ms-2
  (check-method "Mixed Model (partial conjugate)"
                mixed-model mixed-obs :hmc))

;; 3. Dynamic LinReg (dynamic addresses) -> expect :handler-is
(def ms-3
  (check-method "Dynamic LinReg (dynamic addresses)"
                dynamic-linreg dynamic-obs :handler-is))

;; 4. Empty model (no trace sites) -> expect :exact
(def ms-4
  (check-method "Empty model (no trace sites)"
                empty-model nil :exact))

;; 5. Large model with 15+ latents -> expect :vi (>10 residual)
(def ms-5
  (check-method "Large model (15+ latents)"
                large-model large-obs :vi))

;; 6. Static model with splice -> expect :smc
(def ms-6
  (check-method "Splice model (sub-model call)"
                splice-model splice-obs :smc))

(let [all-ms [ms-1 ms-2 ms-3 ms-4 ms-5 ms-6]
      pass-count (count (filter :pass all-ms))
      total (count all-ms)]
  (println (str "\n  Method selection: " pass-count "/" total " correct")))

;; =========================================================================
;; 10C: fit API End-to-End
;; =========================================================================

(println "\n\n------------------------------------------------------------")
(println "  10C: fit API End-to-End")
(println "------------------------------------------------------------")

;; --- 10C-1: Static LinReg via fit (should auto-select :exact) ---

(println "\n-- 10C-1: fit(static-linreg) --")

(let [t0 (perf-now)
      result (fit/fit static-linreg xs-mlx static-obs {:verbose? true})
      elapsed (- (perf-now) t0)]
  (println (str "  Method: " (:method result)))
  (println (str "  Log-ML: " (:log-ml result)))
  (println (str "  Posterior: " (:posterior result)))
  (println (str "  Elapsed: " (.toFixed elapsed 1) " ms"))
  (println (str "  [" (if (= :exact (:method result)) "PASS" "FAIL")
                "] Expected :exact, got " (name (:method result))))
  (def fit-static-result
    {:model "static-linreg"
     :method (name (:method result))
     :expected_method "exact"
     :pass (= :exact (:method result))
     :log_ml (:log-ml result)
     :posterior (:posterior result)
     :elapsed_ms elapsed}))

;; --- 10C-2: Dynamic LinReg via fit (should auto-select :handler-is) ---

(println "\n-- 10C-2: fit(dynamic-linreg) --")

(let [t0 (perf-now)
      result (fit/fit dynamic-linreg [(vec [1.0 2.0 3.0 4.0 5.0])]
                       dynamic-obs {:verbose? true})
      elapsed (- (perf-now) t0)]
  (println (str "  Method: " (:method result)))
  (println (str "  Log-ML: " (:log-ml result)))
  (println (str "  Has posterior: " (some? (:posterior result))))
  (println (str "  Elapsed: " (.toFixed elapsed 1) " ms"))
  (println (str "  [" (if (= :handler-is (:method result)) "PASS" "FAIL")
                "] Expected :handler-is, got " (name (:method result))))
  (def fit-dynamic-result
    {:model "dynamic-linreg"
     :method (name (:method result))
     :expected_method "handler-is"
     :pass (= :handler-is (:method result))
     :log_ml (:log-ml result)
     :has_posterior (some? (:posterior result))
     :elapsed_ms elapsed}))

;; --- 10C-3: Mixed model via fit (should auto-select :hmc) ---

(println "\n-- 10C-3: fit(mixed-model) --")

(let [t0 (perf-now)
      result (fit/fit mixed-model xs-mlx mixed-obs {:verbose? true})
      elapsed (- (perf-now) t0)]
  (println (str "  Method: " (:method result)))
  (println (str "  Has posterior: " (some? (:posterior result))))
  (println (str "  Elapsed: " (.toFixed elapsed 1) " ms"))
  (println (str "  [" (if (= :hmc (:method result)) "PASS" "FAIL")
                "] Expected :hmc, got " (name (:method result))))
  (def fit-mixed-result
    {:model "mixed-model"
     :method (name (:method result))
     :expected_method "hmc"
     :pass (= :hmc (:method result))
     :log_ml (:log-ml result)
     :has_posterior (some? (:posterior result))
     :elapsed_ms elapsed}))

;; =========================================================================
;; Summary
;; =========================================================================

(println "\n\n============================================================")
(println "         EXPERIMENT 10 SUMMARY")
(println "============================================================")

;; 10A summary
(let [speedup (/ (:mean-ms handler-timing) (:mean-ms compiled-timing))]
  (println "\n-- 10A: Compiled Adam vs Handler Loop --")
  (println (str "  Handler Adam (200 iter):  " (.toFixed (:mean-ms handler-timing) 2)
                " +/- " (.toFixed (:std-ms handler-timing) 2) " ms"))
  (println (str "  Compiled Adam (200 iter): " (.toFixed (:mean-ms compiled-timing) 2)
                " +/- " (.toFixed (:std-ms compiled-timing) 2) " ms"))
  (println (str "  Speedup: " (.toFixed speedup 1) "x")))

;; 10B summary
(let [all-ms [ms-1 ms-2 ms-3 ms-4 ms-5 ms-6]
      pass-count (count (filter :pass all-ms))]
  (println "\n-- 10B: Method Selection --")
  (println (str "  " pass-count "/6 methods correctly selected")))

;; 10C summary
(let [all-fit [fit-static-result fit-dynamic-result fit-mixed-result]
      pass-count (count (filter :pass all-fit))]
  (println "\n-- 10C: fit API --")
  (println (str "  " pass-count "/3 fit calls auto-selected correct method")))

;; =========================================================================
;; Write JSON Results
;; =========================================================================

(let [speedup (/ (:mean-ms handler-timing) (:mean-ms compiled-timing))]
  (write-json "optimization_results.json"
    {:experiment "exp10_optimization"
     :timestamp (.toISOString (js/Date.))
     :hardware {:platform "macOS" :chip "Apple Silicon" :gpu "Metal"}

     :exp10a_compiled_adam
     {:description "Compiled Adam vs handler loop (200 iterations, static linreg)"
      :handler {:mean_ms (:mean-ms handler-timing)
                :std_ms (:std-ms handler-timing)
                :min_ms (:min-ms handler-timing)
                :max_ms (:max-ms handler-timing)
                :raw_times (:raw handler-timing)}
      :compiled {:mean_ms (:mean-ms compiled-timing)
                 :std_ms (:std-ms compiled-timing)
                 :min_ms (:min-ms compiled-timing)
                 :max_ms (:max-ms compiled-timing)
                 :raw_times (:raw compiled-timing)}
      :speedup speedup
      :iterations 200
      :learning_rate 0.01}

     :exp10b_method_selection
     {:description "Method selection accuracy across 6 model types"
      :results [ms-1 ms-2 ms-3 ms-4 ms-5 ms-6]
      :total 6
      :correct (count (filter :pass [ms-1 ms-2 ms-3 ms-4 ms-5 ms-6]))}

     :exp10c_fit_api
     {:description "fit API end-to-end auto-select verification"
      :results [fit-static-result fit-dynamic-result fit-mixed-result]
      :total 3
      :correct (count (filter :pass [fit-static-result fit-dynamic-result
                                      fit-mixed-result]))}}))

(println "\nExperiment 10 complete.")
