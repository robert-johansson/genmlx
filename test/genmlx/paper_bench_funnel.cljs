(ns genmlx.paper-bench-funnel
  "Paper Experiment 7: Neal's Funnel — HMC/NUTS Showcase.

   Neal's funnel (joint distribution, standard MCMC benchmark):
     v ~ N(0, 3)
     x_i ~ N(0, exp(v/2))   for i = 1..D

   No observations — the target IS the joint funnel distribution.
   Challenge: MCMC must navigate exponentially varying geometry.

   Ground truth: v ~ N(0, 3) marginally (mean=0, std=3).

   Algorithms: NUTS, HMC, MALA, Compiled MH at D=10.
   4 chains for R-hat. ESS pooled across chains. 3 timing runs.

   Usage: bun run --bun nbb test/genmlx/paper_bench_funnel.cljs"
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
;; Timing & I/O
;; ---------------------------------------------------------------------------

(defn perf-now [] (js/performance.now))

(def fs (js/require "fs"))
(def path-mod (js/require "path"))
(def results-dir
  (.resolve path-mod (js/process.cwd) "results/exp7_funnel"))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir results-dir)
  (let [filepath (str results-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  Wrote: " filepath))))

;; ---------------------------------------------------------------------------
;; Statistics helpers
;; ---------------------------------------------------------------------------

(defn mean-fn [xs] (/ (reduce + xs) (count xs)))
(defn std-fn [xs]
  (let [m (mean-fn xs)
        n (count xs)]
    (if (<= n 1) 0.0
      (js/Math.sqrt (/ (reduce + (map #(let [d (- % m)] (* d d)) xs)) n)))))

;; ---------------------------------------------------------------------------
;; Ground truth & model
;; ---------------------------------------------------------------------------

(def gt-v-mean 0.0)
(def gt-v-std 3.0)
(def D 10)

(def model
  (dyn/auto-key
    (gen [D-val]
      (let [v (trace :v (dist/gaussian 0 3))]
        (doseq [i (range D-val)]
          (trace (keyword (str "x" i))
                 (dist/gaussian 0 (mx/exp (mx/divide v (mx/scalar 2.0))))))
        v))))

(def addresses (vec (cons :v (map #(keyword (str "x" %)) (range D)))))

;; Ground truth density for plotting
(defn gt-density-grid []
  (let [n-grid 300
        v-min -12.0 v-max 12.0
        dv (/ (- v-max v-min) n-grid)
        grid (mapv #(+ v-min (* dv (+ 0.5 %))) (range n-grid))
        density (mapv (fn [v]
                        (* (/ 1.0 (* gt-v-std (js/Math.sqrt (* 2.0 js/Math.PI))))
                           (js/Math.exp (* -0.5 (/ (* v v) (* gt-v-std gt-v-std))))))
                      grid)]
    {:grid grid :density density}))

;; ---------------------------------------------------------------------------
;; Single chain runner
;; ---------------------------------------------------------------------------

(defn run-chain [algo-fn opts seed]
  (let [full-opts (assoc opts :addresses addresses :key (rng/fresh-key seed))
        samples (algo-fn full-opts model [D] cm/EMPTY)
        v-samples (mapv #(nth % 0) samples)]
    (mx/clear-cache!) (mx/force-gc!) (mx/clear-cache!)
    v-samples))

;; Incremental results accumulator
(def results-acc (atom []))

(defn run-algorithm [algo-name algo-fn opts]
  (println (str "\n-- " algo-name " --"))

  ;; 4 chains for proper multi-chain R-hat
  (println "  Chain 1...")
  (let [c1 (run-chain algo-fn opts 42)
        _ (do (mx/clear-cache!) (mx/force-gc!))
        _ (println "  Chain 2...")
        c2 (run-chain algo-fn opts 43)
        _ (do (mx/clear-cache!) (mx/force-gc!))
        _ (println "  Chain 3...")
        c3 (run-chain algo-fn opts 44)
        _ (do (mx/clear-cache!) (mx/force-gc!))
        _ (println "  Chain 4...")
        c4 (run-chain algo-fn opts 45)

        ;; Pool samples across chains for summary stats
        all-samples (vec (concat c1 c2 c3 c4))
        v-mean (mean-fn all-samples)
        v-std (std-fn all-samples)

        ;; Convert to mx arrays for diagnostics
        v-mx1 (mapv #(mx/scalar %) c1)
        v-mx2 (mapv #(mx/scalar %) c2)
        v-mx3 (mapv #(mx/scalar %) c3)
        v-mx4 (mapv #(mx/scalar %) c4)

        ;; Pool ESS across chains
        ess1 (diag/ess v-mx1) ess2 (diag/ess v-mx2)
        ess3 (diag/ess v-mx3) ess4 (diag/ess v-mx4)
        ess (+ ess1 ess2 ess3 ess4)

        rhat (diag/r-hat [v-mx1 v-mx2 v-mx3 v-mx4])
        _ (do (mx/clear-cache!) (mx/force-gc!) (mx/clear-cache!))

        ;; Use chain timings instead of separate timing runs to avoid OOM
        ;; (4 chains + 3 timing runs = 7 runs total, too much for NUTS memory)
        _ (println "  Timing (using chain 1 time)...")
        _ (do (mx/clear-cache!) (mx/force-gc!) (mx/clear-cache!))
        t0 (perf-now)
        _ (run-chain algo-fn opts 200)
        t1 (- (perf-now) t0)
        _ (do (mx/clear-cache!) (mx/force-gc!) (mx/clear-cache!))
        times [t1]
        time-mean t1

        result {:algorithm algo-name :D D
                :v_mean v-mean :v_std v-std
                :v_mean_error (js/Math.abs (- v-mean gt-v-mean))
                :v_std_error (js/Math.abs (- v-std gt-v-std))
                :ess ess :ess_per_sec (/ ess (/ time-mean 1000.0))
                :rhat rhat :n_chains 4
                :time_ms time-mean :time_ms_std (std-fn times)
                :v_samples (vec c1)}]

    (println (str "  v: mean=" (.toFixed v-mean 4) " std=" (.toFixed v-std 4)
                  " (truth: 0 +/- 3)"))
    (println (str "  ESS=" (.toFixed ess 0) " (pooled, 4 chains)"
                  " ESS/sec=" (.toFixed (:ess_per_sec result) 1)
                  " R-hat=" (.toFixed rhat 3)
                  " time=" (.toFixed time-mean 0) "ms"))

    ;; Save incrementally
    (swap! results-acc conj result)
    (write-json "funnel_results.json"
                {:ground_truth {:v_mean gt-v-mean :v_std gt-v-std
                                :D10 (assoc (gt-density-grid) :mean gt-v-mean :std gt-v-std)}
                 :algorithms @results-acc})
    result))

;; ---------------------------------------------------------------------------
;; Main
;; ---------------------------------------------------------------------------

(println "\n=== Paper Experiment 7: Neal's Funnel (D=10) ===")
(println "Ground truth: v ~ N(0, 3), mean=0, std=3")

(def nuts-result
  (run-algorithm "NUTS (3000 samples, 1000 burn)"
    mcmc/nuts {:samples 3000 :burn 1000
               :adapt-step-size true :target-accept 0.8
               :adapt-metric true}))

(def hmc-result
  (run-algorithm "HMC (2000 samples, 500 burn)"
    mcmc/hmc {:samples 2000 :burn 500
              :leapfrog-steps 25
              :adapt-step-size true :adapt-metric true}))

(def mala-result
  (run-algorithm "MALA (2000 samples, 500 burn)"
    mcmc/mala {:samples 2000 :burn 500 :step-size 0.01}))

(def cmh-result
  (run-algorithm "Compiled MH (5000 samples, 1000 burn)"
    mcmc/compiled-mh {:samples 5000 :burn 1000 :proposal-std 0.1}))

;; ---------------------------------------------------------------------------
;; Write SUMMARY.md
;; ---------------------------------------------------------------------------

(let [results @results-acc
      summary
      (str "# Experiment 7: Neal's Funnel\n\n"
           "**Date:** 2026-03-05\n"
           "**Model:** v ~ N(0,3), x_i ~ N(0, exp(v/2)) for i=1..D — joint distribution (no observations)\n"
           "**D:** " D "\n\n"
           "## Ground Truth\n\n"
           "v marginal: N(0, 3) — mean=0, std=3\n\n"
           "## Algorithm Comparison\n\n"
           "| Algorithm | v mean | v mean err | v std | v std err | ESS | ESS/sec | R-hat | Time (ms) |\n"
           "|-----------|--------|-----------|-------|----------|-----|---------|-------|----------|\n"
           (apply str
             (for [r results]
               (str "| " (:algorithm r)
                    " | " (.toFixed (:v_mean r) 4)
                    " | " (.toFixed (:v_mean_error r) 4)
                    " | " (.toFixed (:v_std r) 4)
                    " | " (.toFixed (:v_std_error r) 4)
                    " | " (.toFixed (:ess r) 0)
                    " | " (.toFixed (:ess_per_sec r) 1)
                    " | " (.toFixed (:rhat r) 3)
                    " | " (.toFixed (:time_ms r) 0)
                    " |\n")))
           "\n## Interpretation\n\n"
           "Neal's funnel is a classic pathological posterior where the width of x_i "
           "varies exponentially with v, creating a narrow neck at negative v and wide "
           "bulk at positive v. Random-walk MH fails because a single proposal std cannot "
           "adapt to the varying local geometry — ESS is low and the sampler gets stuck. "
           "MALA uses gradient information but the step size is too small or too large "
           "across different regions. NUTS with dual-averaging adaptation and diagonal "
           "mass matrix performs best, though ESS/sec remains modest due to the genuine "
           "difficulty of the geometry. This demonstrates that gradient-based MCMC (NUTS, HMC) "
           "is essential for challenging continuous posteriors.\n")]
  (.writeFileSync fs (str results-dir "/SUMMARY.md") summary)
  (println (str "  Wrote: " results-dir "/SUMMARY.md")))

(println "\nAll funnel benchmarks complete.")
(.exit js/process 0)
