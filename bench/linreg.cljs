(ns bench.linreg
  "Inference algorithm comparison on linear regression.

   Runs 7 algorithms on a normal-normal conjugate linear regression model
   (20 observations) and compares posterior estimates against the analytic
   posterior (ground truth).

   Algorithms: Compiled MH, HMC, NUTS, Vectorized IS (1K), Vectorized IS (10K).

   Output: results/linreg/data.json

   Usage: bun run --bun nbb bench/linreg.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.importance :as is]
            [genmlx.inference.diagnostics :as diag])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/linreg")))

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
;; Ground truth
;; ---------------------------------------------------------------------------

(def n-obs 20)
(def true-slope 2.0)
(def true-intercept 0.5)
(def sigma-obs 1.0)
(def sigma-prior 10.0)

;; xs = [0, 1, ..., 19]
(def xs-raw (vec (range n-obs)))

;; Generate ys from true params + noise (fixed seed for reproducibility)
(def ys-data
  (let [key (rng/fresh-key 42)
        noise (mx/->clj (rng/normal key [(count xs-raw)]))]
    (mapv (fn [x n]
            (+ (* true-slope x) true-intercept (* sigma-obs n)))
          xs-raw noise)))

(defn compute-analytic-posterior
  "Normal-normal conjugate posterior for linear regression.
   Returns {:slope {:mean _ :std _} :intercept {:mean _ :std _}}."
  [xs ys]
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
;; Model
;; ---------------------------------------------------------------------------

(def linreg
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

;; Observations choice map
(def observations
  (reduce (fn [cm [j y]]
            (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector ys-data)))

;; ---------------------------------------------------------------------------
;; Helper: extract posterior slope mean from MCMC samples
;; ---------------------------------------------------------------------------

(defn mcmc-slope-mean
  "Extract slope posterior mean from compiled-mh / hmc / nuts samples.
   All three return vectors where each element is a JS array of parameter values.
   slope is the first parameter (index 0)."
  [samples]
  (let [vals (mapv #(nth % 0) samples)]
    (/ (reduce + vals) (count vals))))

(defn vis-slope-mean
  "Extract weighted slope posterior mean from vectorized IS result."
  [{:keys [vtrace]}]
  (let [slope-arr (cm/get-choice (:choices vtrace) [:slope])
        weights (mx/exp (mx/subtract (:weight vtrace) (mx/logsumexp (:weight vtrace))))
        _ (mx/materialize! slope-arr weights)]
    (mx/item (mx/sum (mx/multiply weights slope-arr)))))

;; ---------------------------------------------------------------------------
;; Run experiments
;; ---------------------------------------------------------------------------

(println "\n=== Inference Algorithm Comparison: Linear Regression ===")
(println (str "Model: dynamic linreg, " n-obs " obs, sigma_obs=" sigma-obs ", sigma_prior=" sigma-prior))
(println (str "True params: slope=" true-slope ", intercept=" true-intercept))
(println (str "Analytic posterior: slope=" (.toFixed (get-in analytic [:slope :mean]) 4)
              " +/- " (.toFixed (get-in analytic [:slope :std]) 4)
              ", intercept=" (.toFixed (get-in analytic [:intercept :mean]) 4)
              " +/- " (.toFixed (get-in analytic [:intercept :std]) 4)))

;; --- 1. Compiled MH (200 samples, burn 100) ---
(println "\n--- 1. Compiled MH (200 samples, burn 100) ---")
(mx/clear-cache!)
(def cmh-samples
  (mcmc/compiled-mh {:samples 200 :burn 100
                     :addresses [:slope :intercept]
                     :proposal-std 0.5}
                    linreg [xs-raw] observations))
(def cmh-slope-mean (mcmc-slope-mean cmh-samples))
(def cmh-slope-err (js/Math.abs (- cmh-slope-mean (get-in analytic [:slope :mean]))))
(println (str "  slope mean=" (.toFixed cmh-slope-mean 4)
              ", error=" (.toFixed cmh-slope-err 4)))

(mx/clear-cache!)
(def cmh-timing
  (benchmark "compiled-MH-200"
    #(do (mx/clear-cache!)
         (let [samples (mcmc/compiled-mh
                         {:samples 200 :burn 100
                          :addresses [:slope :intercept]
                          :proposal-std 0.5}
                         linreg [xs-raw] observations)]
           (when (seq samples)
             (mx/eval! (mx/array (mapv first (take-last 1 samples)))))))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; --- 2. HMC (100 samples, burn 50, 10 leapfrog steps) ---
(println "\n--- 2. HMC (100 samples, burn 50, L=10) ---")
(mx/clear-cache!)
(def hmc-samples
  (mcmc/hmc {:samples 100 :burn 50
             :leapfrog-steps 10
             :addresses [:slope :intercept]
             :adapt-step-size true}
            linreg [xs-raw] observations))
(def hmc-slope-mean (mcmc-slope-mean hmc-samples))
(def hmc-slope-err (js/Math.abs (- hmc-slope-mean (get-in analytic [:slope :mean]))))
(println (str "  slope mean=" (.toFixed hmc-slope-mean 4)
              ", error=" (.toFixed hmc-slope-err 4)))

(mx/clear-cache!)
(def hmc-timing
  (benchmark "HMC-100"
    #(do (mx/clear-cache!)
         (let [samples (mcmc/hmc {:samples 100 :burn 50
                                  :leapfrog-steps 10
                                  :addresses [:slope :intercept]
                                  :adapt-step-size true}
                                 linreg [xs-raw] observations)]
           (when (seq samples)
             (mx/eval! (mx/array (mapv first (take-last 1 samples)))))))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; --- 3. NUTS (100 samples, burn 50) ---
(println "\n--- 3. NUTS (100 samples, burn 50) ---")
(mx/clear-cache!)
(def nuts-samples
  (mcmc/nuts {:samples 100 :burn 50
              :addresses [:slope :intercept]
              :adapt-step-size true
              :adapt-metric true}
             linreg [xs-raw] observations))
(def nuts-slope-mean (mcmc-slope-mean nuts-samples))
(def nuts-slope-err (js/Math.abs (- nuts-slope-mean (get-in analytic [:slope :mean]))))
(println (str "  slope mean=" (.toFixed nuts-slope-mean 4)
              ", error=" (.toFixed nuts-slope-err 4)))

(mx/clear-cache!)
(def nuts-timing
  (benchmark "NUTS-100"
    #(do (mx/clear-cache!)
         (let [samples (mcmc/nuts {:samples 100 :burn 50
                                   :addresses [:slope :intercept]
                                   :adapt-step-size true
                                   :adapt-metric true}
                                  linreg [xs-raw] observations)]
           (when (seq samples)
             (mx/eval! (mx/array (mapv first (take-last 1 samples)))))))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; --- 4. Vectorized IS (1000 particles) ---
(println "\n--- 4. Vectorized IS (1000 particles) ---")
(mx/clear-cache!)
(def vis1k-result
  (is/vectorized-importance-sampling {:samples 1000}
                                      linreg [xs-raw] observations))
(def vis1k-slope-mean (vis-slope-mean vis1k-result))
(def vis1k-slope-err (js/Math.abs (- vis1k-slope-mean (get-in analytic [:slope :mean]))))
(println (str "  slope mean=" (.toFixed vis1k-slope-mean 4)
              ", error=" (.toFixed vis1k-slope-err 4)))

(mx/clear-cache!)
(def vis1k-timing
  (benchmark "VIS-1000"
    #(let [{:keys [log-ml-estimate]}
           (is/vectorized-importance-sampling {:samples 1000}
                                               linreg [xs-raw] observations)]
       (mx/eval! log-ml-estimate))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; --- 5. Vectorized IS (10000 particles) ---
(println "\n--- 5. Vectorized IS (10000 particles) ---")
(mx/clear-cache!)
(def vis10k-result
  (is/vectorized-importance-sampling {:samples 10000}
                                      linreg [xs-raw] observations))
(def vis10k-slope-mean (vis-slope-mean vis10k-result))
(def vis10k-slope-err (js/Math.abs (- vis10k-slope-mean (get-in analytic [:slope :mean]))))
(println (str "  slope mean=" (.toFixed vis10k-slope-mean 4)
              ", error=" (.toFixed vis10k-slope-err 4)))

(mx/clear-cache!)
(def vis10k-timing
  (benchmark "VIS-10000"
    #(let [{:keys [log-ml-estimate]}
           (is/vectorized-importance-sampling {:samples 10000}
                                               linreg [xs-raw] observations)]
       (mx/eval! log-ml-estimate))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; ---------------------------------------------------------------------------
;; Summary table
;; ---------------------------------------------------------------------------

(def all-results
  [{:algorithm "Compiled MH"   :config "200 samples, burn 100"
    :timing cmh-timing   :slope-mean cmh-slope-mean   :slope-err cmh-slope-err}
   {:algorithm "HMC"           :config "100 samples, burn 50, L=10"
    :timing hmc-timing   :slope-mean hmc-slope-mean   :slope-err hmc-slope-err}
   {:algorithm "NUTS"          :config "100 samples, burn 50"
    :timing nuts-timing  :slope-mean nuts-slope-mean  :slope-err nuts-slope-err}
   {:algorithm "VIS-1K"        :config "1000 particles"
    :timing vis1k-timing :slope-mean vis1k-slope-mean :slope-err vis1k-slope-err}
   {:algorithm "VIS-10K"       :config "10000 particles"
    :timing vis10k-timing :slope-mean vis10k-slope-mean :slope-err vis10k-slope-err}])

(println "\n\n========================================")
(println "    INFERENCE COMPARISON RESULTS")
(println "========================================\n")
(println (str "Analytic posterior slope: "
              (.toFixed (get-in analytic [:slope :mean]) 4)
              " +/- " (.toFixed (get-in analytic [:slope :std]) 4)))
(println "")
(println "| Algorithm | Config | Time (ms) | Slope Mean | Slope Error |")
(println "|-----------|--------|-----------|------------|-------------|")
(doseq [{:keys [algorithm config timing slope-mean slope-err]} all-results]
  (println (str "| " algorithm
                " | " config
                " | " (.toFixed (:mean-ms timing) 1) " +/- " (.toFixed (:std-ms timing) 1)
                " | " (.toFixed slope-mean 4)
                " | " (.toFixed slope-err 4) " |")))

;; ---------------------------------------------------------------------------
;; Write data.json
;; ---------------------------------------------------------------------------

(write-json "data.json"
  {:experiment "linreg-inference-comparison"
   :model {:name "dynamic-linreg" :n-obs n-obs :n-trace-sites (+ 2 n-obs)
           :true-slope true-slope :true-intercept true-intercept
           :sigma-obs sigma-obs :sigma-prior sigma-prior}
   :ground-truth {:slope-mean     (get-in analytic [:slope :mean])
                  :slope-std      (get-in analytic [:slope :std])
                  :intercept-mean (get-in analytic [:intercept :mean])
                  :intercept-std  (get-in analytic [:intercept :std])}
   :results
   (mapv (fn [{:keys [algorithm config timing slope-mean slope-err]}]
           {:algorithm   algorithm
            :config      config
            :mean-ms     (:mean-ms timing)
            :std-ms      (:std-ms timing)
            :min-ms      (:min-ms timing)
            :max-ms      (:max-ms timing)
            :raw-times   (:raw timing)
            :slope-mean  slope-mean
            :slope-error slope-err})
         all-results)})

(println "\nLinear regression inference comparison complete.")
