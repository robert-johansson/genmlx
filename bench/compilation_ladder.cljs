(ns bench.compilation-ladder
  "Compilation Ladder — the hero experiment.

   Demonstrates progressive compilation on the SAME model:
   L0 (handler) → L1 (compiled gen) → L2 (compiled MH, HMC, VIS)
   → L3 (auto-conjugate) → L4 (fit, compiled Adam).

   Uses static linear regression with analytic posterior as ground truth.

   Output: results/compilation-ladder/data.json

   Usage: bun run --bun nbb bench/compilation_ladder.cljs"
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

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

;; Output dir: from env (orchestrator) or default
(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/compilation-ladder")))

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
    (println (str "  [" label "] " (.toFixed mean-ms 3) " ± " (.toFixed std-ms 3) " ms"))
    {:label label :mean-ms mean-ms :std-ms std-ms
     :min-ms (apply min outer-times) :max-ms (apply max outer-times)
     :raw outer-times}))

;; ---------------------------------------------------------------------------
;; Ground truth
;; ---------------------------------------------------------------------------

(def n-obs 5)
(def true-slope 2.0)
(def true-intercept 0.5)
(def sigma-obs 1.0)
(def sigma-prior 10.0)

(def xs-raw [1.0 2.0 3.0 4.0 5.0])
(def ys-data [2.3 4.7 6.1 8.9 10.2])

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

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

;; STATIC: all keyword addresses, no loops → L1 compiles, L3 eliminates
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

;; DYNAMIC: loop, computed addresses → forces handler path
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

(def static-args [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)
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

;; ---------------------------------------------------------------------------
;; Run all levels
;; ---------------------------------------------------------------------------

(println "\n=== Compilation Ladder ===")
(println (str "Model: static linreg, " n-obs " obs"))
(println (str "Analytic posterior: slope=" (.toFixed (get-in analytic [:slope :mean]) 4)
              " ± " (.toFixed (get-in analytic [:slope :std]) 4)))

;; L0: handler generate (dynamic)
(println "\n--- L0: Handler generate (dynamic model) ---")
(def l0-timing
  (benchmark "L0-handler-dynamic"
    #(let [{:keys [weight]} (p/generate dynamic-linreg [xs-raw] dynamic-obs)]
       (mx/eval! weight))
    :warmup-n 20 :outer-n 7 :inner-n 15))

;; L0-static: handler generate (static, for baseline)
(println "\n--- L0-static: Handler generate (static model) ---")
(def l0s-timing
  (benchmark "L0-handler-static"
    #(let [{:keys [weight]} (p/generate static-linreg static-args static-obs)]
       (mx/eval! weight))
    :warmup-n 20 :outer-n 7 :inner-n 15))

;; L1: compiled generate (same call, schema-driven path fires automatically)
(println "\n--- L1: Compiled generate (static model) ---")
(def l1-timing
  (benchmark "L1-compiled-generate"
    #(let [{:keys [weight]} (p/generate static-linreg static-args static-obs)]
       (mx/eval! weight))
    :warmup-n 20 :outer-n 7 :inner-n 15))

;; L2a: Handler MH (200 steps)
(println "\n--- L2: Handler MH (200 steps) ---")
(def l2h-timing
  (benchmark "L2-handler-MH-200"
    #(let [traces (mcmc/mh {:samples 200 :burn 0}
                            dynamic-linreg [xs-raw] dynamic-obs)]
       (mx/eval! (:score (last traces))))
    :warmup-n 2 :outer-n 5 :inner-n 2))

;; L2b: Compiled MH (200 steps)
(println "\n--- L2: Compiled MH (200 steps) ---")
(def l2c-timing
  (benchmark "L2-compiled-MH-200"
    #(do (mx/clear-cache!)
         (let [samples (mcmc/compiled-mh
                         {:samples 200 :burn 0
                          :addresses [:slope :intercept]
                          :proposal-std 0.5}
                         dynamic-linreg [xs-raw] dynamic-obs)]
           (when (seq samples)
             (mx/eval! (mx/array (mapv first (take-last 1 samples)))))))
    :warmup-n 1 :outer-n 5 :inner-n 2))

;; L2c: HMC (100 samples)
(println "\n--- L2: HMC (100 samples) ---")
(def l2hmc-timing
  (benchmark "L2-HMC-100"
    #(do (mx/clear-cache!)
         (let [samples (mcmc/hmc {:samples 100 :burn 20
                                   :leapfrog-steps 10
                                   :addresses [:slope :intercept]
                                   :adapt-step-size true}
                                  dynamic-linreg [xs-raw] dynamic-obs)]
           (when (seq samples)
             (mx/eval! (mx/array (mapv first (take-last 1 samples)))))))
    :warmup-n 1 :outer-n 3 :inner-n 2))

;; L2d: Vectorized IS (1000 particles)
(println "\n--- L2: Vectorized IS (1000 particles) ---")
(def l2vis-timing
  (benchmark "L2-VIS-1000"
    #(let [{:keys [log-ml-estimate]}
           (is/vectorized-importance-sampling {:samples 1000}
                                               dynamic-linreg [xs-raw] dynamic-obs)]
       (mx/eval! log-ml-estimate))
    :warmup-n 5 :outer-n 7 :inner-n 10))

;; L3: Auto-conjugacy (exact posterior)
(println "\n--- L3: Auto-conjugacy (exact posterior) ---")
(def l3-timing
  (benchmark "L3-conjugate"
    #(let [{:keys [weight]} (p/generate static-linreg static-args static-obs)]
       (mx/eval! weight))
    :warmup-n 20 :outer-n 7 :inner-n 15))

(let [{:keys [weight]} (p/generate static-linreg static-args static-obs)]
  (mx/materialize! weight)
  (println (str "  L3 log-ML (exact): " (.toFixed (mx/item weight) 4))))

;; L4a: fit API (auto-select method)
(println "\n--- L4: fit API (auto-select) ---")
(def l4fit-timing
  (benchmark "L4-fit"
    #(let [result (fit/fit static-linreg static-args static-obs)]
       (mx/eval! (mx/scalar (:elapsed-ms result))))
    :warmup-n 5 :outer-n 7 :inner-n 10))

(let [result (fit/fit static-linreg static-args static-obs {:verbose? true})]
  (println (str "  L4 method: " (:method result)))
  (println (str "  L4 log-ML: " (:log-ml result))))

;; L4b: Compiled Adam (200 iterations)
(println "\n--- L4: Compiled Adam (200 iter) ---")
(def l4learn-timing
  (benchmark "L4-learn-200iter"
    #(let [result (co/learn dynamic-linreg [xs-raw] dynamic-obs
                             [:slope :intercept]
                             {:iterations 200 :lr 0.01})]
       (mx/eval! (:params result)))
    :warmup-n 2 :outer-n 5 :inner-n 3))

;; L4c: Handler loop baseline (200 iterations)
(println "\n--- L4: Handler loop baseline (200 iter) ---")
(def l4loop-timing
  (benchmark "L4-handler-loop-200iter"
    #(let [score-fn (u/make-score-fn dynamic-linreg [xs-raw] dynamic-obs
                                      [:slope :intercept])
           params (mx/array [0.0 0.0])]
       (loop [p params i 0]
         (when (< i 200)
           (let [s (score-fn p)]
             (mx/eval! s)
             (recur (mx/add p (mx/multiply (mx/scalar 0.001) (mx/ones [2]))) (inc i))))))
    :warmup-n 2 :outer-n 5 :inner-n 3))

;; ---------------------------------------------------------------------------
;; Collect and write results
;; ---------------------------------------------------------------------------

(def all-results
  [{:level "L0-dynamic"      :desc "Handler generate (dynamic model)"      :timing l0-timing}
   {:level "L0-static"       :desc "Handler generate (static model)"       :timing l0s-timing}
   {:level "L1"              :desc "Compiled generate (schema-driven)"     :timing l1-timing}
   {:level "L2-VIS"          :desc "Vectorized IS (1000 particles)"        :timing l2vis-timing}
   {:level "L2-MH-handler"   :desc "Handler MH loop (200 steps)"           :timing l2h-timing}
   {:level "L2-MH-compiled"  :desc "Compiled MH chain (200 steps)"         :timing l2c-timing}
   {:level "L2-HMC"          :desc "HMC (100 samples)"                     :timing l2hmc-timing}
   {:level "L3"              :desc "Auto-conjugacy (exact posterior)"       :timing l3-timing}
   {:level "L4-fit"          :desc "fit API (auto-select → exact)"         :timing l4fit-timing}
   {:level "L4-learn"        :desc "Compiled Adam (200 iter)"              :timing l4learn-timing}
   {:level "L4-handler-loop" :desc "Handler loop baseline (200 iter)"      :timing l4loop-timing}])

(println "\n\n========================================")
(println "         COMPILATION LADDER RESULTS")
(println "========================================\n")

(let [l0-time (:mean-ms l0-timing)]
  (println "| Level | Description | Time (ms) | vs L0 |")
  (println "|-------|-------------|-----------|-------|")
  (doseq [{:keys [level desc timing]} all-results]
    (println (str "| " level " | " desc " | "
                  (.toFixed (:mean-ms timing) 3) " ± " (.toFixed (:std-ms timing) 3)
                  " | " (.toFixed (/ l0-time (:mean-ms timing)) 1) "x |"))))

;; Write data.json
(let [l0-time (:mean-ms l0-timing)]
  (write-json "data.json"
    {:experiment "compilation-ladder"
     :model {:name "static-linreg" :n-trace-sites 7 :n-obs n-obs :static true}
     :ground-truth {:slope-mean     (get-in analytic [:slope :mean])
                    :slope-std      (get-in analytic [:slope :std])
                    :intercept-mean (get-in analytic [:intercept :mean])
                    :intercept-std  (get-in analytic [:intercept :std])}
     :results
     (mapv (fn [{:keys [level desc timing]}]
             {:level       level
              :description desc
              :mean-ms     (:mean-ms timing)
              :std-ms      (:std-ms timing)
              :min-ms      (:min-ms timing)
              :max-ms      (:max-ms timing)
              :speedup     (/ l0-time (:mean-ms timing))
              :raw-times   (:raw timing)})
           all-results)
     :speedups
     {:mh-handler-to-compiled   (/ (:mean-ms l2h-timing)    (:mean-ms l2c-timing))
      :learn-handler-to-compiled (/ (:mean-ms l4loop-timing) (:mean-ms l4learn-timing))
      :l0-to-l3                  (/ l0-time                  (:mean-ms l3-timing))
      :l0-to-l4-fit              (/ l0-time                  (:mean-ms l4fit-timing))}}))

(println "\nCompilation ladder complete.")
