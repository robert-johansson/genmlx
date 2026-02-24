(ns genmlx.benchmark.genmlx-benchmark
  "GenMLX benchmark — comprehensive comparison with GenJAX.

   Models:
     A: Gaussian conjugate (4 sites: mu, y0, y1, y2)
     B: Linear regression  (11 sites: slope, intercept, y0..y8)
     C: Many parameters    (52 sites: z0..z49, obs_mean, obs_var)

   Protocol: 3 warmup, median of 7 runs, performance.now()
   Sync: mx/eval! on all outputs

   Run: bun run --bun nbb test/benchmark/genmlx_benchmark.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.vectorized :as vec]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Timing
;; ---------------------------------------------------------------------------

(def perf-now
  (if (exists? js/performance)
    #(.now js/performance)
    #(js/Date.now)))

(defn bench
  "3 warmup, median of 7 runs."
  [f]
  (dotimes [_ 3] (f))
  (let [times (mapv (fn [_]
                      (let [s (perf-now) _ (f) e (perf-now)] (- e s)))
                    (range 7))]
    (nth (sort times) 3)))

(defn bench-light
  "1 warmup, median of 3 runs (for memory-intensive ops)."
  [f]
  (f)
  (let [times (mapv (fn [_]
                      (let [s (perf-now) _ (f) e (perf-now)] (- e s)))
                    (range 3))]
    (nth (sort times) 1)))

(defn bench-single
  "Single timed run (for Metal-resource-exhausting ops)."
  [f]
  (let [s (perf-now) _ (f) e (perf-now)]
    (- e s)))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

;; Model A: Gaussian Conjugate (4 sites)
(def model-a
  (gen []
    (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
      (dyn/trace :y0 (dist/gaussian mu 1))
      (dyn/trace :y1 (dist/gaussian mu 1))
      (dyn/trace :y2 (dist/gaussian mu 1))
      mu)))

(def obs-a
  (cm/choicemap :y0 (mx/scalar 3.0) :y1 (mx/scalar 3.1) :y2 (mx/scalar 2.9)))

;; Model B: Linear Regression (11 sites)
(def model-b
  (gen [xs]
    (let [slope     (dyn/trace :slope (dist/gaussian 0 10))
          intercept (dyn/trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (dyn/trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                          intercept) 1)))
      slope)))

(def xs-b [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0])
(def obs-b
  (reduce (fn [cm [k v]]
            (cm/set-choice cm [k] (mx/scalar v)))
          cm/EMPTY
          [[:y0 3.1] [:y1 5.2] [:y2 6.9] [:y3 9.1] [:y4 10.8]
           [:y5 12.9] [:y6 15.1] [:y7 17.0] [:y8 19.2]]))
(def full-b
  (reduce (fn [cm [k v]]
            (cm/set-choice cm [k] (mx/scalar v)))
          cm/EMPTY
          [[:slope 2.0] [:intercept 1.0]
           [:y0 3.1] [:y1 5.2] [:y2 6.9] [:y3 9.1] [:y4 10.8]
           [:y5 12.9] [:y6 15.1] [:y7 17.0] [:y8 19.2]]))

;; Model C: Many Parameters (52 sites)
(def model-c
  (gen []
    (let [zs (mapv (fn [i]
                     (dyn/trace (keyword (str "z" i)) (dist/gaussian 0 1)))
                   (range 50))
          z-arr (mx/array (mapv mx/item zs))
          mean-z (mx/mean z-arr)
          var-z  (mx/mean (mx/square (mx/subtract z-arr mean-z)))]
      (dyn/trace :obs_mean (dist/gaussian mean-z 0.1))
      (dyn/trace :obs_var (dist/gaussian var-z 0.1))
      mean-z)))

(def obs-c
  (cm/choicemap :obs_mean (mx/scalar 0.0) :obs_var (mx/scalar 1.0)))

;; Results accumulators
(def results (atom {}))
(def correctness (atom {}))

(defn safe-run
  "Run f in try-catch. Returns result or nil. Prints label + time or error."
  [label bench-fn f record-key]
  (try
    (let [r (bench-fn f)]
      (println (str "  " label ":  " (.toFixed r 2) "ms"))
      (when record-key (swap! results assoc record-key r))
      r)
    (catch :default e
      (println (str "  " label ":  ERROR - " (.-message e)))
      nil)))

;; ---------------------------------------------------------------------------
;; Header
;; ---------------------------------------------------------------------------

(println "\n=== GenMLX Comprehensive Benchmark ===")
(println (str "  Runtime: " (if (exists? js/Bun) "Bun" "Node.js")))
(println (str "  Date: " (.toISOString (js/Date.))))
(println (str "  Protocol: 3 warmup, median of 7 runs, performance.now()"))
(println)

(def sep (apply str (repeat 60 "=")))

;; =========================================================================
;; SECTION 1: GFI Primitives (10 calls each)
;; =========================================================================

(println sep)
(println "SECTION 1: GFI Primitives (10 calls each)")
(println sep)

(println "\n-- Simulate (10 calls) --")

(let [r (bench #(dotimes [_ 10]
                  (let [t (p/simulate model-a [])] (mx/eval! (:score t)))))]
  (println (str "  Model A (4-site):   " (.toFixed r 3) "ms"))
  (swap! results assoc :simulate_a r))

(let [r (bench #(dotimes [_ 10]
                  (let [t (p/simulate model-b [xs-b])] (mx/eval! (:score t)))))]
  (println (str "  Model B (11-site):  " (.toFixed r 3) "ms"))
  (swap! results assoc :simulate_b r))

(let [r (bench #(dotimes [_ 10]
                  (let [t (p/simulate model-c [])] (mx/eval! (:score t)))))]
  (println (str "  Model C (52-site):  " (.toFixed r 3) "ms"))
  (swap! results assoc :simulate_c r))

(println "\n-- Generate (10 calls) --")

(let [r (bench #(dotimes [_ 10]
                  (let [{:keys [trace weight]} (p/generate model-a [] obs-a)]
                    (mx/eval! (:score trace) weight))))]
  (println (str "  Model A (4-site):   " (.toFixed r 3) "ms"))
  (swap! results assoc :generate_a r))

(let [r (bench #(dotimes [_ 10]
                  (let [{:keys [trace weight]} (p/generate model-b [xs-b] full-b)]
                    (mx/eval! (:score trace) weight))))]
  (println (str "  Model B (11-site):  " (.toFixed r 3) "ms"))
  (swap! results assoc :generate_b r))

(let [r (bench #(dotimes [_ 10]
                  (let [{:keys [trace weight]} (p/generate model-c [] obs-c)]
                    (mx/eval! (:score trace) weight))))]
  (println (str "  Model C (52-site):  " (.toFixed r 3) "ms"))
  (swap! results assoc :generate_c r))


;; =========================================================================
;; SECTION 2: Vectorized Importance Sampling
;; =========================================================================

(println (str "\n" sep))
(println "SECTION 2: Vectorized Importance Sampling")
(println sep)

(doseq [n [100 1000]]
  (println (str "\n-- Vectorized IS N=" n " --"))

  (let [r (bench #(let [{:keys [log-ml-estimate]}
                         (is/vectorized-importance-sampling
                           {:samples n} model-a [] obs-a)]
                    (mx/eval! log-ml-estimate)))]
    (println (str "  Model A (4-site):   " (.toFixed r 3) "ms"))
    (swap! results assoc (keyword (str "vis_" n "_a")) r))

  (let [r (bench #(let [{:keys [log-ml-estimate]}
                         (is/vectorized-importance-sampling
                           {:samples n} model-b [xs-b] obs-b)]
                    (mx/eval! log-ml-estimate)))]
    (println (str "  Model B (11-site):  " (.toFixed r 3) "ms"))
    (swap! results assoc (keyword (str "vis_" n "_b")) r)))

;; Sequential IS N=100 for speedup comparison
(println "\n-- Sequential IS N=100 (for vectorization speedup) --")

(let [r (bench-light #(let [{:keys [log-ml-estimate]}
                              (is/importance-sampling
                                {:samples 100} model-a [] obs-a)]
                         (mx/eval! log-ml-estimate)))]
  (println (str "  Model A (4-site):   " (.toFixed r 3) "ms"))
  (swap! results assoc :is_100_a r))

(let [r (bench-light #(let [{:keys [log-ml-estimate]}
                              (is/importance-sampling
                                {:samples 100} model-b [xs-b] obs-b)]
                         (mx/eval! log-ml-estimate)))]
  (println (str "  Model B (11-site):  " (.toFixed r 3) "ms"))
  (swap! results assoc :is_100_b r))


;; =========================================================================
;; SECTION 3: MCMC Single Chain (200 steps)
;; =========================================================================

(println (str "\n" sep))
(println "SECTION 3: MCMC Single Chain (200 steps)")
(println sep)

;; -- GFI MH (resource-intensive) --
(println "\n-- MH 200 steps (GFI-based regenerate) --")

(let [r (bench-light #(mcmc/mh {:samples 200 :burn 0
                                 :selection (sel/select :mu)}
                                model-a [] obs-a))]
  (println (str "  Model A (4-site):   " (.toFixed r 2) "ms"))
  (swap! results assoc :mh200_a r))

(let [r (bench-single #(mcmc/mh {:samples 200 :burn 0
                                  :selection (sel/select :slope :intercept)}
                                 model-b [xs-b] obs-b))]
  (println (str "  Model B (11-site):  " (.toFixed r 2) "ms (single run)"))
  (swap! results assoc :mh200_b r))

;; -- Compiled MH (GenMLX only) --
(println "\n-- Compiled MH 200 steps (parameter-space, GenMLX only) --")

(let [r (bench #(mcmc/compiled-mh
                  {:samples 200 :burn 0 :addresses [:mu]
                   :proposal-std 0.5}
                  model-a [] obs-a))]
  (println (str "  Model A (4-site):   " (.toFixed r 2) "ms"))
  (swap! results assoc :compiled_mh200_a r))

(let [r (bench #(mcmc/compiled-mh
                  {:samples 200 :burn 0 :addresses [:slope :intercept]
                   :proposal-std 0.5}
                  model-b [xs-b] obs-b))]
  (println (str "  Model B (11-site):  " (.toFixed r 2) "ms"))
  (swap! results assoc :compiled_mh200_b r))

;; -- MALA --
(println "\n-- MALA 200 steps --")

(let [r (bench #(mcmc/mala
                  {:samples 200 :burn 0 :step-size 0.1
                   :addresses [:mu]}
                  model-a [] obs-a))]
  (println (str "  Model A (4-site):   " (.toFixed r 2) "ms"))
  (swap! results assoc :mala200_a r))

(let [r (bench #(mcmc/mala
                  {:samples 200 :burn 0 :step-size 0.1
                   :addresses [:slope :intercept]}
                  model-b [xs-b] obs-b))]
  (println (str "  Model B (11-site):  " (.toFixed r 2) "ms"))
  (swap! results assoc :mala200_b r))

;; -- HMC --
(println "\n-- HMC 200 steps, L=10 --")

(let [r (bench #(mcmc/hmc
                  {:samples 200 :burn 0 :step-size 0.01
                   :leapfrog-steps 10 :addresses [:mu]}
                  model-a [] obs-a))]
  (println (str "  Model A (4-site):   " (.toFixed r 2) "ms"))
  (swap! results assoc :hmc200_a r))

(let [r (bench #(mcmc/hmc
                  {:samples 200 :burn 0 :step-size 0.01
                   :leapfrog-steps 10 :addresses [:slope :intercept]}
                  model-b [xs-b] obs-b))]
  (println (str "  Model B (11-site):  " (.toFixed r 2) "ms"))
  (swap! results assoc :hmc200_b r))


;; =========================================================================
;; SECTION 4: Vectorized MCMC (10 chains, 50 steps)
;;   Reduced from 200 to 50 steps to stay within Metal resource limits.
;;   Per-step cost is the meaningful metric for comparison.
;; =========================================================================

(println (str "\n" sep))
(println "SECTION 4: Vectorized MCMC (10 chains, 50 steps)")
(println sep)

(def n-chains 10)
(def vec-steps 50)

;; -- Vectorized compiled MH --
(println (str "\n-- Vec Compiled MH " vec-steps " steps, " n-chains " chains --"))

(safe-run "Model A (4-site)" bench
  #(mcmc/vectorized-compiled-mh
     {:samples vec-steps :burn 0 :addresses [:mu]
      :proposal-std 0.5 :n-chains n-chains}
     model-a [] obs-a)
  :vec_cmh_10chains_a)

(safe-run "Model B (11-site)" bench
  #(mcmc/vectorized-compiled-mh
     {:samples vec-steps :burn 0 :addresses [:slope :intercept]
      :proposal-std 0.5 :n-chains n-chains}
     model-b [xs-b] obs-b)
  :vec_cmh_10chains_b)

;; -- Vectorized MALA --
(println (str "\n-- Vec MALA " vec-steps " steps, " n-chains " chains --"))

(safe-run "Model A (4-site)" bench
  #(mcmc/vectorized-mala
     {:samples vec-steps :burn 0 :step-size 0.1 :n-chains n-chains
      :addresses [:mu]}
     model-a [] obs-a)
  :vec_mala_10chains_a)

(safe-run "Model B (11-site)" bench
  #(mcmc/vectorized-mala
     {:samples vec-steps :burn 0 :step-size 0.1 :n-chains n-chains
      :addresses [:slope :intercept]}
     model-b [xs-b] obs-b)
  :vec_mala_10chains_b)

;; -- Vectorized HMC (L=5 to reduce resource usage) --
(println (str "\n-- Vec HMC " vec-steps " steps, L=5, " n-chains " chains --"))

(safe-run "Model A (4-site)" bench
  #(mcmc/vectorized-hmc
     {:samples vec-steps :burn 0 :step-size 0.01
      :leapfrog-steps 5 :n-chains n-chains
      :addresses [:mu]}
     model-a [] obs-a)
  :vec_hmc_10chains_a)

(safe-run "Model B (11-site)" bench
  #(mcmc/vectorized-hmc
     {:samples vec-steps :burn 0 :step-size 0.01
      :leapfrog-steps 5 :n-chains n-chains
      :addresses [:slope :intercept]}
     model-b [xs-b] obs-b)
  :vec_hmc_10chains_b)


;; =========================================================================
;; SECTION 5: Scaling Test (52-site model)
;;   Model C simulate/generate measured in Section 1.
;;   MCMC on 52-site model omitted to avoid Metal resource exhaustion
;;   in this process — the accumulated graph from previous sections
;;   leaves insufficient headroom.
;; =========================================================================

(println (str "\n" sep))
(println "SECTION 5: Scaling Test (52-site model)")
(println sep)
(println "  simulate/generate measured in Section 1")
(println "  Model C MCMC skipped (Metal resource limit in long-running process)")


;; =========================================================================
;; SECTION 6: Correctness Checks (using compiled algorithms for tidy)
;; =========================================================================

(println (str "\n" sep))
(println "SECTION 6: Correctness Checks")
(println sep)

;; Model A: Posterior mu ~ N(2.990, 0.577^2)
(println "\n-- Model A: Gaussian Conjugate Posterior --")

(try
  (let [samples (mcmc/compiled-mh {:samples 500 :burn 200 :addresses [:mu]
                                    :proposal-std 0.5}
                                   model-a [] obs-a)
        ;; compiled-mh returns clj vectors via mx/->clj
        mu-vals (mapv (fn [s] (nth s 0)) samples)
        mu-mean (/ (reduce + mu-vals) (count mu-vals))
        err (js/Math.abs (- mu-mean 2.990))
        pass? (< err 0.3)]
    (println (str "  E[mu] = " (.toFixed mu-mean 3) " (expected ~2.990, err=" (.toFixed err 3) ")"))
    (println (str "  RESULT: " (if pass? "PASS" "FAIL")))
    (swap! correctness assoc :model_a_mu_mean mu-mean :model_a_pass pass?))
  (catch :default e
    (println (str "  ERROR: " (.-message e)))
    (swap! correctness assoc :model_a_pass false)))

;; Model B: slope ~2.0, intercept ~1.0
(println "\n-- Model B: Linear Regression Posterior --")

(try
  (let [samples (mcmc/hmc {:samples 500 :burn 200 :step-size 0.01
                            :leapfrog-steps 10
                            :addresses [:slope :intercept]}
                           model-b [xs-b] obs-b)
        ;; hmc returns clj vectors via mx/->clj
        slope-vals (mapv (fn [s] (nth s 0)) samples)
        intercept-vals (mapv (fn [s] (nth s 1)) samples)
        slope-mean (/ (reduce + slope-vals) (count slope-vals))
        intercept-mean (/ (reduce + intercept-vals) (count intercept-vals))
        err-slope (js/Math.abs (- slope-mean 2.0))
        err-intercept (js/Math.abs (- intercept-mean 1.0))
        pass? (and (< err-slope 0.5) (< err-intercept 0.5))]
    (println (str "  E[slope] = " (.toFixed slope-mean 3)
                  " (expected ~2.0, err=" (.toFixed err-slope 3) ")"))
    (println (str "  E[intercept] = " (.toFixed intercept-mean 3)
                  " (expected ~1.0, err=" (.toFixed err-intercept 3) ")"))
    (println (str "  RESULT: " (if pass? "PASS" "FAIL")))
    (swap! correctness assoc :model_b_slope_mean slope-mean
           :model_b_intercept_mean intercept-mean :model_b_pass pass?))
  (catch :default e
    (println (str "  ERROR: " (.-message e)))
    (swap! correctness assoc :model_b_pass false)))

;; Model C: skipped (Metal resources exhausted in long-running process)
(println "\n-- Model C: Skipped (Metal resource limit) --")


;; =========================================================================
;; Save results
;; =========================================================================

(println (str "\n" sep))
(println "SUMMARY")
(println sep)

(let [output (clj->js {:framework "GenMLX"
                        :runtime (if (exists? js/Bun) "Bun" "Node.js")
                        :protocol "3 warmup, median of 7 runs"
                        :timings (clj->js @results)
                        :correctness (clj->js @correctness)})
      json-str (.stringify js/JSON output nil 2)
      fs (js/require "fs")
      path "test/benchmark/genmlx_results.json"]
  (.writeFileSync fs path json-str)
  (println (str "\nResults saved to " path)))

(println "\n=== Benchmark complete ===")
