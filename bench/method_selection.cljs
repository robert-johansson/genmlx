(ns bench.method-selection
  "Method Selection -- automatic inference strategy dispatch.

   Sub-experiments:
   A: Method selection accuracy (6 model types)
   B: fit API end-to-end (3 models, auto-select verification)

   Demonstrates that select-method and fit correctly identify the best
   inference strategy for each model structure: exact (fully conjugate),
   hmc (partial conjugate), handler-is (dynamic addresses), vi (large),
   smc (splice), and the fit API's end-to-end auto-dispatch.

   Output: results/method-selection/data.json

   Usage: bun run --bun nbb bench/method_selection.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.method-selection :as ms]
            [genmlx.fit :as fit])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/method-selection")))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir out-dir)
  (let [filepath (str out-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  wrote: " filepath))))

(defn perf-now [] (js/performance.now))

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
;; A: Method Selection Accuracy
;; =========================================================================

(println "\n=== Method Selection ===")
(println "\n--- A: Method Selection Accuracy (6 models) ---")

(defn check-method [label model observations expected-method]
  (let [t0     (perf-now)
        result (ms/select-method model observations)
        elapsed (- (perf-now) t0)
        actual (:method result)
        pass?  (= actual expected-method)]
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
     :n_eliminated (count (:eliminated result))
     :elapsed_ms elapsed}))

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

(def all-ms [ms-1 ms-2 ms-3 ms-4 ms-5 ms-6])

(let [pass-count (count (filter :pass all-ms))
      total (count all-ms)]
  (println (str "\n  Method selection: " pass-count "/" total " correct")))

;; =========================================================================
;; B: fit API End-to-End
;; =========================================================================

(println "\n--- B: fit API End-to-End (3 models) ---")

;; B-1: Static LinReg via fit (should auto-select :exact)
(println "\n-- B-1: fit(static-linreg) --")

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

;; B-2: Dynamic LinReg via fit (should auto-select :handler-is)
(println "\n-- B-2: fit(dynamic-linreg) --")

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

;; B-3: Mixed model via fit (should auto-select :hmc)
(println "\n-- B-3: fit(mixed-model) --")

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

(def all-fit [fit-static-result fit-dynamic-result fit-mixed-result])

(println "\n\n========================================")
(println "       METHOD SELECTION RESULTS")
(println "========================================")

(let [ms-pass (count (filter :pass all-ms))
      fit-pass (count (filter :pass all-fit))]
  (println (str "\n-- A: Method Selection: " ms-pass "/6 correct --"))
  (doseq [{:keys [label expected actual pass]} all-ms]
    (println (str "  " (if pass "PASS" "FAIL") "  " label
                  "  (expected=" expected ", actual=" actual ")")))
  (println (str "\n-- B: fit API: " fit-pass "/3 correct --"))
  (doseq [{:keys [model expected_method method pass]} all-fit]
    (println (str "  " (if pass "PASS" "FAIL") "  " model
                  "  (expected=" expected_method ", actual=" method ")"))))

;; =========================================================================
;; Write data.json
;; =========================================================================

(let [ms-pass  (count (filter :pass all-ms))
      fit-pass (count (filter :pass all-fit))]
  (write-json "data.json"
    {:experiment "method-selection"
     :timestamp (.toISOString (js/Date.))

     :method_selection
     {:description "Method selection accuracy across 6 model types"
      :results all-ms
      :total 6
      :correct ms-pass}

     :fit_api
     {:description "fit API end-to-end auto-select verification"
      :results all-fit
      :total 3
      :correct fit-pass}}))

(println "\nMethod selection benchmark complete.")
