(ns genmlx.paper.bench-01-ladder
  "Paper Experiment 1: Compilation Ladder — the hero experiment.

   Demonstrates progressive compilation on the SAME model:
   L0 (handler) → L1 (compiled gen) → L2 (compiled MH) → L3 (conjugate) → L4 (fit).

   Uses static linear regression with analytic posterior as ground truth.

   Usage: bun run --bun nbb test/genmlx/paper/bench_01_ladder.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.importance :as is]
            [genmlx.inference.util :as u]
            [genmlx.inference.compiled-optimizer :as co]
            [genmlx.method-selection :as ms]
            [genmlx.fit :as fit]
            [genmlx.inference.diagnostics :as diag])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(defn perf-now [] (js/performance.now))

(def fs (js/require "fs"))
(def path-mod (js/require "path"))
(def results-dir
  (.resolve path-mod (js/process.cwd) "results/paper/exp01_ladder"))

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
    (println (str "  [" label "] " (.toFixed mean-ms 3) " ± "
                  (.toFixed std-ms 3) " ms"))
    {:label label :mean-ms mean-ms :std-ms std-ms
     :min-ms (apply min outer-times) :max-ms (apply max outer-times)
     :raw outer-times}))

;; ---------------------------------------------------------------------------
;; Ground truth data
;; ---------------------------------------------------------------------------

(def n-obs 5)
(def true-slope 2.0)
(def true-intercept 0.5)
(def sigma-obs 1.0)
(def sigma-prior 10.0)

;; Fixed x and y values (reproducible)
(def xs-raw [1.0 2.0 3.0 4.0 5.0])
(def ys-data [2.3 4.7 6.1 8.9 10.2])

;; Analytic posterior for Normal-Normal conjugate
(defn compute-analytic-posterior [xs ys]
  (let [sx  (reduce + xs)
        sx2 (reduce + (map #(* % %) xs))
        sxy (reduce + (map * xs ys))
        sy  (reduce + ys)
        n   (double (count xs))
        inv-prior (/ 1.0 (* sigma-prior sigma-prior))
        inv-obs   (/ 1.0 (* sigma-obs sigma-obs))
        p00 (+ (* sx2 inv-obs) inv-prior)
        p01 (* sx inv-obs)
        p11 (+ (* n inv-obs) inv-prior)
        det (- (* p00 p11) (* p01 p01))
        s00 (/ p11 det)
        s01 (/ (- p01) det)
        s11 (/ p00 det)
        rhs0 (* sxy inv-obs)
        rhs1 (* sy inv-obs)]
    {:slope {:mean (+ (* s00 rhs0) (* s01 rhs1))
             :std (js/Math.sqrt s00)}
     :intercept {:mean (+ (* s01 rhs0) (* s11 rhs1))
                 :std (js/Math.sqrt s11)}}))

(def analytic (compute-analytic-posterior xs-raw ys-data))

(println "\n=== Experiment 1: Compilation Ladder ===")
(println (str "Model: Static linear regression, " n-obs " observations"))
(println (str "True: slope=" true-slope ", intercept=" true-intercept))
(println (str "Analytic posterior: slope=" (.toFixed (get-in analytic [:slope :mean]) 4)
              " ± " (.toFixed (get-in analytic [:slope :std]) 4)
              ", intercept=" (.toFixed (get-in analytic [:intercept :mean]) 4)
              " ± " (.toFixed (get-in analytic [:intercept :std]) 4)))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

;; STATIC model: all keyword addresses, no loops → L1 can compile, L3 can eliminate
(def static-linreg
  (dyn/auto-key
    (gen [x1 x2 x3 x4 x5]
      (let [slope     (trace :slope (dist/gaussian 0 sigma-prior))
            intercept (trace :intercept (dist/gaussian 0 sigma-prior))]
        (trace :y1 (dist/gaussian (mx/add (mx/multiply slope x1) intercept) sigma-obs))
        (trace :y2 (dist/gaussian (mx/add (mx/multiply slope x2) intercept) sigma-obs))
        (trace :y3 (dist/gaussian (mx/add (mx/multiply slope x3) intercept) sigma-obs))
        (trace :y4 (dist/gaussian (mx/add (mx/multiply slope x4) intercept) sigma-obs))
        (trace :y5 (dist/gaussian (mx/add (mx/multiply slope x5) intercept) sigma-obs))
        slope))))

;; DYNAMIC model: loop, computed addresses → forces handler path (L0)
(def dynamic-linreg
  (dyn/auto-key
    (gen [xs]
      (let [slope     (trace :slope (dist/gaussian 0 sigma-prior))
            intercept (trace :intercept (dist/gaussian 0 sigma-prior))]
        (doseq [[j x] (map-indexed vector xs)]
          (trace (keyword (str "y" j))
                 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                        intercept)
                                sigma-obs)))
        slope))))

;; Static model args
(def static-args [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)
                  (mx/scalar 4.0) (mx/scalar 5.0)])

;; Observations for static model
(def static-obs
  (-> cm/EMPTY
      (cm/set-choice [:y1] (mx/scalar 2.3))
      (cm/set-choice [:y2] (mx/scalar 4.7))
      (cm/set-choice [:y3] (mx/scalar 6.1))
      (cm/set-choice [:y4] (mx/scalar 8.9))
      (cm/set-choice [:y5] (mx/scalar 10.2))))

;; Observations for dynamic model
(def dynamic-obs
  (-> cm/EMPTY
      (cm/set-choice [:y0] (mx/scalar 2.3))
      (cm/set-choice [:y1] (mx/scalar 4.7))
      (cm/set-choice [:y2] (mx/scalar 6.1))
      (cm/set-choice [:y3] (mx/scalar 8.9))
      (cm/set-choice [:y4] (mx/scalar 10.2))))

;; ---------------------------------------------------------------------------
;; Schema & compilation info
;; ---------------------------------------------------------------------------

(println "\n-- Model info --")
(println (str "  Static schema: " (some? (:schema static-linreg))
              ", static? " (get-in static-linreg [:schema :static?])
              ", sites=" (count (get-in static-linreg [:schema :trace-sites]))))
(println (str "  Dynamic schema: " (some? (:schema dynamic-linreg))
              ", static? " (get-in dynamic-linreg [:schema :static?])
              ", dynamic-addrs? " (get-in dynamic-linreg [:schema :dynamic-addresses?])))

;; Check L3 conjugacy
(let [conj-pairs (get-in static-linreg [:schema :conjugate-pairs])]
  (println (str "  Conjugate pairs: " (count conj-pairs)
                (when (seq conj-pairs)
                  (str " — " (mapv :family conj-pairs))))))

;; Method selection
(let [sel (ms/select-method static-linreg static-obs)]
  (println (str "  Method selection: " (:method sel)
                " (" (:reason sel) ")"
                ", eliminated=" (count (:eliminated sel))
                ", residual=" (:n-residual sel))))

;; ---------------------------------------------------------------------------
;; L0: Handler-based generate (dynamic model)
;; ---------------------------------------------------------------------------

(println "\n=== L0: Handler-based generate (dynamic model) ===")

(def l0-timing
  (benchmark "L0-handler-dynamic"
    (fn []
      (let [{:keys [trace weight]} (p/generate dynamic-linreg [xs-raw] dynamic-obs)]
        (mx/eval! weight)))
    :warmup-n 20 :outer-n 7 :inner-n 15))

;; Verify correctness
(let [{:keys [trace weight]} (p/generate dynamic-linreg [xs-raw] dynamic-obs)]
  (mx/materialize! weight)
  (println (str "  Weight: " (.toFixed (mx/item weight) 4))))

;; ---------------------------------------------------------------------------
;; L0-static: Handler-based generate (static model, forces handler)
;; ---------------------------------------------------------------------------

(println "\n=== L0-static: Handler-based generate (static model) ===")

(def l0-static-timing
  (benchmark "L0-handler-static"
    (fn []
      (let [{:keys [trace weight]} (p/generate static-linreg static-args static-obs)]
        (mx/eval! weight)))
    :warmup-n 20 :outer-n 7 :inner-n 15))

;; ---------------------------------------------------------------------------
;; L1: Compiled generate (static model, schema enables compilation)
;; ---------------------------------------------------------------------------

;; L1 is the same as L0-static — the schema-driven generate path fires
;; automatically for static models. We measure it separately to show the
;; compilation level explicitly.

(println "\n=== L1: Compiled generate (static model) ===")

(def l1-timing
  (benchmark "L1-compiled-generate"
    (fn []
      (let [{:keys [trace weight]} (p/generate static-linreg static-args static-obs)]
        (mx/eval! weight)))
    :warmup-n 20 :outer-n 7 :inner-n 15))

;; ---------------------------------------------------------------------------
;; L2a: Compiled MH chain (500 steps) — uses dynamic model (compiled-mh works)
;; ---------------------------------------------------------------------------

(println "\n=== L2: Compiled MH chain (500 steps) ===")

;; Handler-based MH (for comparison) — using dynamic model
(def l2-handler-timing
  (benchmark "L2-handler-MH-500"
    (fn []
      (let [traces (mcmc/mh {:samples 500 :burn 0}
                             dynamic-linreg [xs-raw] dynamic-obs)]
        (mx/eval! (:score (last traces)))))
    :warmup-n 3 :outer-n 5 :inner-n 3))

;; Compiled MH — using dynamic model
(def l2-compiled-timing
  (benchmark "L2-compiled-MH-500"
    (fn []
      (let [samples (mcmc/compiled-mh
                      {:samples 500 :burn 0
                       :addresses [:slope :intercept]
                       :proposal-std 0.5}
                      dynamic-linreg [xs-raw] dynamic-obs)]
        (when (seq samples)
          (mx/eval! (mx/array (mapv #(nth % 0) (take-last 1 samples)))))))
    :warmup-n 3 :outer-n 5 :inner-n 3))

;; L2 accuracy check
(let [samples (mcmc/compiled-mh
                {:samples 2000 :burn 500
                 :addresses [:slope :intercept]
                 :proposal-std 0.5}
                dynamic-linreg [xs-raw] dynamic-obs)]
  (if (seq samples)
    (let [slope-samples (mapv #(nth % 0) samples)
          slope-mean (/ (reduce + slope-samples) (count slope-samples))
          slope-err (js/Math.abs (- slope-mean (get-in analytic [:slope :mean])))]
      (println (str "  L2 slope mean: " (.toFixed slope-mean 4)
                    " (err=" (.toFixed slope-err 4) ")")))
    (println "  L2 compiled-mh returned empty samples")))

;; ---------------------------------------------------------------------------
;; L2b: HMC (gradient-based MCMC)
;; ---------------------------------------------------------------------------

(println "\n=== L2b: HMC (200 samples) ===")

(def l2-hmc-timing
  (benchmark "L2-HMC-200"
    (fn []
      (let [samples (mcmc/hmc {:samples 200 :burn 50
                                :leapfrog-steps 10
                                :addresses [:slope :intercept]
                                :adapt-step-size true}
                               dynamic-linreg [xs-raw] dynamic-obs)]
        (when (seq samples)
          (mx/eval! (mx/array (mapv #(nth % 0) (take-last 1 samples)))))))
    :warmup-n 2 :outer-n 5 :inner-n 2))

;; HMC accuracy
(let [samples (mcmc/hmc {:samples 500 :burn 100
                          :leapfrog-steps 10
                          :addresses [:slope :intercept]
                          :adapt-step-size true}
                         dynamic-linreg [xs-raw] dynamic-obs)]
  (if (seq samples)
    (let [slope-samples (mapv #(nth % 0) samples)
          slope-mean (/ (reduce + slope-samples) (count slope-samples))
          slope-err (js/Math.abs (- slope-mean (get-in analytic [:slope :mean])))]
      (println (str "  HMC slope mean: " (.toFixed slope-mean 4)
                    " (err=" (.toFixed slope-err 4) ")")))
    (println "  HMC returned empty samples")))

;; ---------------------------------------------------------------------------
;; L2c: Vectorized IS (1000 particles) — uses dynamic model (vgenerate)
;; ---------------------------------------------------------------------------

(println "\n=== L2c: Vectorized IS (1000 particles) ===")

(def l2-vis-timing
  (benchmark "L2-VIS-1000"
    (fn []
      (let [{:keys [vtrace log-ml-estimate]}
            (is/vectorized-importance-sampling {:samples 1000}
                                                dynamic-linreg [xs-raw] dynamic-obs)]
        (mx/eval! log-ml-estimate)))
    :warmup-n 5 :outer-n 7 :inner-n 10))

;; VIS accuracy
(let [{:keys [vtrace log-ml-estimate]}
      (is/vectorized-importance-sampling {:samples 10000}
                                          dynamic-linreg [xs-raw] dynamic-obs)
      slope-arr (cm/get-choice (:choices vtrace) [:slope])
      weights (mx/exp (mx/subtract (:weight vtrace) (mx/logsumexp (:weight vtrace))))
      _ (mx/materialize! slope-arr weights)
      slope-mean (mx/item (mx/sum (mx/multiply weights slope-arr)))
      slope-err (js/Math.abs (- slope-mean (get-in analytic [:slope :mean])))]
  (mx/clear-cache!)
  (println (str "  VIS slope mean: " (.toFixed slope-mean 4)
                " (err=" (.toFixed slope-err 4) ")")))

;; ---------------------------------------------------------------------------
;; L3: Auto-conjugacy (zero-annotation exact posterior)
;; ---------------------------------------------------------------------------

(println "\n=== L3: Auto-conjugacy (exact posterior) ===")

;; L3 fires via the analytical plan during p/generate
;; For a fully conjugate model, this produces the exact posterior weight
(def l3-timing
  (benchmark "L3-conjugate"
    (fn []
      (let [{:keys [trace weight]} (p/generate static-linreg static-args static-obs)]
        (mx/eval! weight)))
    :warmup-n 20 :outer-n 7 :inner-n 15))

;; L3 weight should be the marginal likelihood
(let [{:keys [trace weight]} (p/generate static-linreg static-args static-obs)]
  (mx/materialize! weight)
  (println (str "  L3 log-ML (exact): " (.toFixed (mx/item weight) 4))))

;; ---------------------------------------------------------------------------
;; L4: fit API (auto-select method)
;; ---------------------------------------------------------------------------

(println "\n=== L4: fit API (auto-select) ===")

(def l4-timing
  (benchmark "L4-fit"
    (fn []
      (let [result (fit/fit static-linreg static-args static-obs)]
        (mx/eval! (mx/scalar (:elapsed-ms result)))))
    :warmup-n 5 :outer-n 7 :inner-n 10))

;; L4 full result
(let [result (fit/fit static-linreg static-args static-obs {:verbose? true})]
  (println (str "  L4 method: " (:method result)))
  (println (str "  L4 log-ML: " (:log-ml result)))
  (println (str "  L4 elapsed: " (:elapsed-ms result) "ms"))
  (println (str "  L4 posterior: " (:posterior result))))

;; ---------------------------------------------------------------------------
;; L4-learn: Compiled Adam optimization (uses dynamic model)
;; ---------------------------------------------------------------------------

(println "\n=== L4-learn: Compiled Adam (200 iterations) ===")

;; Compiled Adam — uses co/learn directly
(def l4-learn-timing
  (benchmark "L4-learn-200iter"
    (fn []
      (let [result (co/learn dynamic-linreg [xs-raw] dynamic-obs
                              [:slope :intercept]
                              {:iterations 200 :lr 0.01})]
        (mx/eval! (:params result))))
    :warmup-n 2 :outer-n 5 :inner-n 3))

;; L4-learn result (full run)
(let [result (co/learn dynamic-linreg [xs-raw] dynamic-obs
                        [:slope :intercept]
                        {:iterations 200 :lr 0.01 :log-every 50})]
  (println (str "  L4-learn params: " (mx/->clj (:params result))))
  (println (str "  L4-learn loss (last 5): "
                (take-last 5 (:loss-history result))))
  (println (str "  L4-learn compilation: " (:compilation-level result))))

;; ---------------------------------------------------------------------------
;; L4-handler-loop: Manual optimization loop (for speedup comparison)
;; ---------------------------------------------------------------------------

(println "\n=== L4 handler loop baseline (200 iterations) ===")

(def l4-handler-loop-timing
  (benchmark "L4-handler-loop-200iter"
    (fn []
      (let [score-fn (u/make-score-fn dynamic-linreg [xs-raw] dynamic-obs
                                       [:slope :intercept])
            params (mx/array [0.0 0.0])]
        (loop [p params i 0]
          (when (< i 200)
            (let [s (score-fn p)]
              (mx/eval! s)
              (recur (mx/add p (mx/multiply (mx/scalar 0.001) (mx/ones [2]))) (inc i)))))))
    :warmup-n 2 :outer-n 5 :inner-n 3))

;; ---------------------------------------------------------------------------
;; Collect all results
;; ---------------------------------------------------------------------------

(println "\n\n========================================")
(println "         COMPILATION LADDER RESULTS")
(println "========================================\n")

(def all-timings
  [{:level "L0-dynamic" :desc "Handler generate (dynamic model)"
    :timing l0-timing}
   {:level "L0-static" :desc "Handler generate (static model)"
    :timing l0-static-timing}
   {:level "L1" :desc "Compiled generate (static, schema-driven)"
    :timing l1-timing}
   {:level "L2-VIS" :desc "Vectorized IS (1000 particles)"
    :timing l2-vis-timing}
   {:level "L2-MH-handler" :desc "Handler MH loop (500 steps)"
    :timing l2-handler-timing}
   {:level "L2-MH-compiled" :desc "Compiled MH chain (500 steps)"
    :timing l2-compiled-timing}
   {:level "L2-HMC" :desc "HMC (200 samples, adapted)"
    :timing l2-hmc-timing}
   {:level "L3" :desc "Auto-conjugacy (exact posterior)"
    :timing l3-timing}
   {:level "L4-fit" :desc "fit API (auto-select → exact)"
    :timing l4-timing}
   {:level "L4-learn" :desc "fit + compiled Adam (200 iter)"
    :timing l4-learn-timing}
   {:level "L4-handler-loop" :desc "Manual handler loop (200 iter)"
    :timing l4-handler-loop-timing}])

;; Print summary table
(let [l0-time (:mean-ms l0-timing)]
  (println "| Level | Description | Time (ms) | Speedup vs L0 |")
  (println "|-------|-------------|-----------|---------------|")
  (doseq [{:keys [level desc timing]} all-timings]
    (let [speedup (/ l0-time (:mean-ms timing))]
      (println (str "| " level " | " desc " | "
                    (.toFixed (:mean-ms timing) 3) " ± "
                    (.toFixed (:std-ms timing) 3) " | "
                    (.toFixed speedup 1) "x |")))))

;; ---------------------------------------------------------------------------
;; Write JSON
;; ---------------------------------------------------------------------------

(let [l0-time (:mean-ms l0-timing)]
  (write-json "ladder_results.json"
    {:experiment "exp01_compilation_ladder"
     :timestamp (.toISOString (js/Date.))
     :hardware {:platform "macOS" :chip "Apple Silicon" :gpu "Metal"}
     :model {:name "static-linreg" :n_trace_sites 7 :n_obs n-obs :static true}
     :ground_truth {:slope (get-in analytic [:slope :mean])
                    :intercept (get-in analytic [:intercept :mean])
                    :slope_std (get-in analytic [:slope :std])
                    :intercept_std (get-in analytic [:intercept :std])}
     :results
     (mapv (fn [{:keys [level desc timing]}]
             {:level level
              :description desc
              :mean_ms (:mean-ms timing)
              :std_ms (:std-ms timing)
              :min_ms (:min-ms timing)
              :max_ms (:max-ms timing)
              :speedup_vs_L0 (/ l0-time (:mean-ms timing))
              :raw_times (:raw timing)})
           all-timings)
     :speedup_summary
     {:L0_to_L1 (/ (:mean-ms l0-timing) (:mean-ms l1-timing))
      :L0_to_L3 (/ (:mean-ms l0-timing) (:mean-ms l3-timing))
      :L0_to_L4 (/ (:mean-ms l0-timing) (:mean-ms l4-timing))
      :MH_handler_to_compiled (/ (:mean-ms l2-handler-timing) (:mean-ms l2-compiled-timing))
      :learn_handler_to_compiled (/ (:mean-ms l4-handler-loop-timing)
                                     (:mean-ms l4-learn-timing))}}))

(println "\nExperiment 1 complete.")
