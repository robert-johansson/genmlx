(ns bench.l3-5-regenerate-analytical
  "L3.5 Regenerate with Analytical Elimination benchmark.

   Demonstrates MCMC (Metropolis-Hastings) with analytical elimination:
   conjugate parameters are analytically marginalized during regenerate,
   reducing the effective dimension of the MH proposal.

   Model: 3 conjugate (normal-normal) + 2 non-conjugate (uniform) parameters.

   Two conditions:
     1. Full MH (5D): strip analytical paths, Gibbs cycles on all 5 params.
        Conjugate params proposed from prior, rarely accepted.
     2. Analytical MH (2D): auto-detected conjugacy eliminates 3 params.
        Conjugate params sampled exactly from posterior during regenerate.
        Only scale + offset remain under standard MH.

   Metrics: wall-clock time, acceptance rate, ESS for each parameter.

   Output: results/l3.5-regenerate-analytical/data.json

   Usage: bun run --bun nbb bench/l3_5_regenerate_analytical.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.trace :as tr]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.kernel :as kern]
            [genmlx.inference.util :as u]
            [genmlx.compiled :as compiled])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/l3.5-regenerate-analytical")))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir out-dir)
  (let [filepath (str out-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  Wrote: " filepath))))

(defn perf-now [] (js/performance.now))

;; ---------------------------------------------------------------------------
;; Statistics helpers
;; ---------------------------------------------------------------------------

(defn mean [xs] (/ (reduce + xs) (count xs)))

(defn variance [xs]
  (let [m (mean xs)
        n (count xs)]
    (/ (reduce + (map #(let [d (- % m)] (* d d)) xs)) n)))

(defn std [xs] (js/Math.sqrt (variance xs)))

(defn ess
  "Effective sample size from a vector of scalar samples.
   Uses lag-1 autocorrelation: ESS ~ N / (1 + 2*rho1)."
  [xs]
  (let [n (count xs)]
    (if (< n 4)
      (double n)
      (let [m (mean xs)
            centered (mapv #(- % m) xs)
            c0 (/ (reduce + (map #(* % %) centered)) n)
            c1 (/ (reduce + (map * (butlast centered) (rest centered))) n)]
        (if (< (js/Math.abs c0) 1e-15)
          (double n)
          (let [rho1 (/ c1 c0)
                denom (+ 1.0 (* 2.0 (max 0.0 rho1)))]
            (/ (double n) denom)))))))

;; ---------------------------------------------------------------------------
;; Strip analytical paths (forces standard handler for all operations)
;; ---------------------------------------------------------------------------

(defn strip-analytical
  "Remove auto-handlers from a gen-fn schema, forcing the standard handler
   path for generate and regenerate. This disables analytical elimination."
  [gf]
  (assoc gf :schema (dissoc (:schema gf)
                            :auto-handlers :auto-regenerate-handlers
                            :auto-regenerate-transition
                            :conjugate-pairs :has-conjugate? :analytical-plan)))

;; ---------------------------------------------------------------------------
;; Model: 3 conjugate + 2 non-conjugate parameters
;; ---------------------------------------------------------------------------

(def xs-raw [1.0 2.0 3.0 4.0 5.0])

;; True parameter values for data generation
(def true-scale 2.0)
(def true-offset 1.0)

(def model
  (dyn/auto-key
    (gen [xs]
      (let [;; Conjugate (normal-normal -- will be eliminated analytically)
            mu1 (trace :mu1 (dist/gaussian 0 10))
            mu2 (trace :mu2 (dist/gaussian 0 10))
            mu3 (trace :mu3 (dist/gaussian 0 10))
            ;; Non-conjugate (uniform prior -- remain under MH)
            scale (trace :scale (dist/uniform 0.1 5))
            offset (trace :offset (dist/uniform -5 5))]
        ;; Observations linked to conjugate params
        (trace :y1 (dist/gaussian mu1 1))
        (trace :y2 (dist/gaussian mu2 1))
        (trace :y3 (dist/gaussian mu3 1))
        ;; Observations linked to non-conjugate params
        (doseq [[j x] (map-indexed vector xs)]
          (trace (keyword (str "z" j))
                 (dist/gaussian (mx/add (mx/multiply scale x) offset) 1)))
        {:mu1 mu1 :mu2 mu2 :mu3 mu3 :scale scale :offset offset}))))

;; Observations: conjugate obs near known values, regression obs from true params
(def obs
  (-> cm/EMPTY
      (cm/set-value :y1 (mx/scalar 2.5))
      (cm/set-value :y2 (mx/scalar -1.0))
      (cm/set-value :y3 (mx/scalar 0.5))
      (cm/set-value :z0 (mx/scalar 3.1))   ;; scale*1 + offset + noise
      (cm/set-value :z1 (mx/scalar 5.2))   ;; scale*2 + offset + noise
      (cm/set-value :z2 (mx/scalar 6.8))   ;; scale*3 + offset + noise
      (cm/set-value :z3 (mx/scalar 9.1))   ;; scale*4 + offset + noise
      (cm/set-value :z4 (mx/scalar 10.9)))) ;; scale*5 + offset + noise

;; ---------------------------------------------------------------------------
;; Extract parameter values from a trace
;; ---------------------------------------------------------------------------

(defn extract-param [trace addr]
  (let [v (cm/get-choice (:choices trace) [addr])]
    (mx/item v)))

;; ---------------------------------------------------------------------------
;; Run Gibbs MCMC and collect diagnostics
;; ---------------------------------------------------------------------------

(defn run-gibbs-condition
  "Run Gibbs-style MH (cycling through one param at a time) and return diagnostics.
   The kernel cycles: mu1 -> mu2 -> mu3 -> scale -> offset -> repeat.
   With analytical: conjugate params are sampled from posterior (always accepted).
   Without analytical: conjugate params proposed from prior (often rejected)."
  [label model-fn n-samples n-burn]
  (println (str "\n  [" label "] Running Gibbs MH: " n-samples " samples, "
                n-burn " burn-in..."))
  (let [;; Gibbs kernel: cycle through each latent one at a time
        gibbs-kernel (kern/gibbs :mu1 :mu2 :mu3 :scale :offset)
        ;; Initialize from the constrained model
        {:keys [trace]} (p/generate model-fn [(mapv mx/scalar xs-raw)] obs)
        t0 (perf-now)
        traces (kern/run-kernel {:samples n-samples :burn n-burn}
                                gibbs-kernel trace)
        t1 (perf-now)
        elapsed-ms (- t1 t0)
        acceptance-rate (:acceptance-rate (meta traces))

        ;; Extract parameter chains
        scale-chain  (mapv #(extract-param % :scale) traces)
        offset-chain (mapv #(extract-param % :offset) traces)
        mu1-chain    (mapv #(extract-param % :mu1) traces)
        mu2-chain    (mapv #(extract-param % :mu2) traces)
        mu3-chain    (mapv #(extract-param % :mu3) traces)

        ;; ESS for each parameter
        ess-scale  (ess scale-chain)
        ess-offset (ess offset-chain)
        ess-mu1    (ess mu1-chain)
        ess-mu2    (ess mu2-chain)
        ess-mu3    (ess mu3-chain)

        ;; Posterior means
        mean-scale  (mean scale-chain)
        mean-offset (mean offset-chain)
        mean-mu1    (mean mu1-chain)
        mean-mu2    (mean mu2-chain)
        mean-mu3    (mean mu3-chain)]

    (println (str "  [" label "] Time:       " (.toFixed elapsed-ms 1) " ms"))
    (println (str "  [" label "] Acceptance: " (.toFixed (* 100 acceptance-rate) 1) "%"))
    (println (str "  [" label "] ESS scale:  " (.toFixed ess-scale 1)
                  ", offset: " (.toFixed ess-offset 1)))
    (println (str "  [" label "] ESS mu1:    " (.toFixed ess-mu1 1)
                  ", mu2: " (.toFixed ess-mu2 1)
                  ", mu3: " (.toFixed ess-mu3 1)))
    (println (str "  [" label "] Mean scale: " (.toFixed mean-scale 3)
                  " (true: " true-scale ")"))
    (println (str "  [" label "] Mean offset:" (.toFixed mean-offset 3)
                  " (true: " true-offset ")"))

    {:label label
     :timing-ms elapsed-ms
     :samples n-samples
     :burn n-burn
     :acceptance-rate acceptance-rate
     :ess {:scale ess-scale :offset ess-offset
           :mu1 ess-mu1 :mu2 ess-mu2 :mu3 ess-mu3}
     :posterior-means {:scale mean-scale :offset mean-offset
                       :mu1 mean-mu1 :mu2 mean-mu2 :mu3 mean-mu3}
     :posterior-stds {:scale (std scale-chain) :offset (std offset-chain)
                      :mu1 (std mu1-chain) :mu2 (std mu2-chain)
                      :mu3 (std mu3-chain)}}))

;; =========================================================================
;; Main execution
;; =========================================================================

(println "\n" (apply str (repeat 70 "=")))
(println "  L3.5 REGENERATE ANALYTICAL ELIMINATION BENCHMARK")
(println (apply str (repeat 70 "=")))
(println "\n  Model: 3 conjugate (mu1,mu2,mu3) + 2 non-conjugate (scale,offset)")
(println "  Without analytical: Gibbs MH on all 5 params (prior proposals)")
(println "  With analytical:    Gibbs MH, conjugate params eliminated analytically")
(println "                      (mu1,mu2,mu3 sampled from posterior, scale+offset under MH)")
(println)

(def n-samples 500)
(def n-burn 100)

;; ---------------------------------------------------------------------------
;; Check that model has analytical regenerate handlers
;; ---------------------------------------------------------------------------

(let [schema (:schema model)
      has-regen? (boolean (:auto-regenerate-transition schema))
      pairs (:conjugate-pairs schema)]
  (println (str "  Analytical regenerate available: " has-regen?))
  (println (str "  Conjugate pairs detected: " (count pairs)))
  (when (seq pairs)
    (doseq [pair pairs]
      (println (str "    " (:prior-addr pair) " -> " (:obs-addr pair)
                    " [" (:family pair) "]")))))

;; ---------------------------------------------------------------------------
;; Condition 1: Full MH (no analytical) -- 5D Gibbs
;; ---------------------------------------------------------------------------

(println "\n" (apply str (repeat 60 "-")))
(println "  Condition 1: Full Gibbs MH (5D, no analytical elimination)")
(println (apply str (repeat 60 "-")))

(def stripped-model (dyn/auto-key (strip-analytical model)))

(def result-full
  (run-gibbs-condition "full-mh-5d" stripped-model n-samples n-burn))

;; ---------------------------------------------------------------------------
;; Condition 2: MH + analytical elimination -- effective 2D
;; ---------------------------------------------------------------------------

(println "\n" (apply str (repeat 60 "-")))
(println "  Condition 2: Analytical Gibbs MH (2D effective, 3 params eliminated)")
(println (apply str (repeat 60 "-")))

(def result-analytical
  (run-gibbs-condition "analytical-mh-2d" model n-samples n-burn))

;; ---------------------------------------------------------------------------
;; Comparison
;; ---------------------------------------------------------------------------

(println "\n" (apply str (repeat 70 "=")))
(println "  COMPARISON")
(println (apply str (repeat 70 "=")))

(let [speedup (/ (:timing-ms result-full) (max (:timing-ms result-analytical) 0.01))
      ess-scale-ratio (/ (get-in result-analytical [:ess :scale])
                         (max (get-in result-full [:ess :scale]) 0.01))
      ess-offset-ratio (/ (get-in result-analytical [:ess :offset])
                          (max (get-in result-full [:ess :offset]) 0.01))
      ess-mu1-ratio (/ (get-in result-analytical [:ess :mu1])
                       (max (get-in result-full [:ess :mu1]) 0.01))
      acc-full (:acceptance-rate result-full)
      acc-anal (:acceptance-rate result-analytical)]
  (println (str "\n  Timing:      full=" (.toFixed (:timing-ms result-full) 1)
                " ms, analytical=" (.toFixed (:timing-ms result-analytical) 1) " ms"
                " (speedup: " (.toFixed speedup 2) "x)"))
  (println (str "  Acceptance:  full=" (.toFixed (* 100 acc-full) 1)
                "%, analytical=" (.toFixed (* 100 acc-anal) 1) "%"))
  (println (str "  ESS scale:   full=" (.toFixed (get-in result-full [:ess :scale]) 1)
                ", analytical=" (.toFixed (get-in result-analytical [:ess :scale]) 1)
                " (" (.toFixed ess-scale-ratio 1) "x)"))
  (println (str "  ESS offset:  full=" (.toFixed (get-in result-full [:ess :offset]) 1)
                ", analytical=" (.toFixed (get-in result-analytical [:ess :offset]) 1)
                " (" (.toFixed ess-offset-ratio 1) "x)"))
  (println (str "  ESS mu1:     full=" (.toFixed (get-in result-full [:ess :mu1]) 1)
                ", analytical=" (.toFixed (get-in result-analytical [:ess :mu1]) 1)
                " (" (.toFixed ess-mu1-ratio 1) "x)"))
  (println (str "  Dimension:   5D -> 2D (3 conjugate params eliminated)")))

;; ---------------------------------------------------------------------------
;; Write JSON results
;; ---------------------------------------------------------------------------

(let [speedup (/ (:timing-ms result-full) (max (:timing-ms result-analytical) 0.01))]
  (write-json "data.json"
    {:experiment "l3.5-regenerate-analytical"
     :description "MCMC with analytical elimination: conjugate params marginalized during regenerate"
     :timestamp (.toISOString (js/Date.))
     :hardware {:platform "macOS" :chip "Apple Silicon" :gpu "Metal"}
     :model "3 conjugate + 2 non-conjugate params"
     :true_values {:scale true-scale :offset true-offset}
     :observations {:y1 2.5 :y2 -1.0 :y3 0.5
                    :z0 3.1 :z1 5.2 :z2 6.8 :z3 9.1 :z4 10.9}
     :conditions
     {:full-mh-5d
      {:samples n-samples
       :burn n-burn
       :timing-ms (:timing-ms result-full)
       :acceptance-rate (:acceptance-rate result-full)
       :ess (:ess result-full)
       :posterior-means (:posterior-means result-full)
       :posterior-stds (:posterior-stds result-full)}
      :analytical-mh-2d
      {:samples n-samples
       :burn n-burn
       :timing-ms (:timing-ms result-analytical)
       :acceptance-rate (:acceptance-rate result-analytical)
       :ess (:ess result-analytical)
       :posterior-means (:posterior-means result-analytical)
       :posterior-stds (:posterior-stds result-analytical)
       :eliminated-dims 3}}
     :speedup speedup
     :dimension-reduction "5D -> 2D"}))

(println "\nBenchmark complete.")
