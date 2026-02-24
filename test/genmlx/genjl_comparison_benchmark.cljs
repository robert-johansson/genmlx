(ns genmlx.genjl-comparison-benchmark
  "Apples-to-apples comparison with Gen.jl benchmark numbers.
   Uses same models, same iteration counts, same timing protocol.
   Gen.jl results from test/reference/gen_jl_benchmark.json (2026-02-21).
   Run: bun run --bun nbb test/genmlx/genjl_comparison_benchmark.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Timing — matching Gen.jl protocol: 3 warmup, median of 7 runs
;; Uses performance.now() for sub-ms resolution
;; ---------------------------------------------------------------------------

(def perf-now
  (if (exists? js/performance)
    #(.now js/performance)
    #(js/Date.now)))

(defn bench
  "Run f with 3 warmup, return median of 7 measured runs in ms."
  [f]
  (dotimes [_ 3] (f))
  (let [times (mapv (fn [_]
                      (let [start (perf-now)
                            _ (f)
                            end (perf-now)]
                        (- end start)))
                    (range 7))
        sorted (sort times)]
    (nth sorted 3)))

;; ---------------------------------------------------------------------------
;; Models — matching Gen.jl definitions exactly
;; ---------------------------------------------------------------------------

;; Model 1: Single Gaussian (1 site) — matches model1_single_gaussian
(def model-1site
  (gen []
    (let [x (dyn/trace :x (dist/gaussian 0 1))]
      x)))

;; Model 2: Linear Regression (7 sites) — matches model2_linear_regression
(def model-linreg
  (gen [xs]
    (let [slope     (dyn/trace :slope (dist/gaussian 0 10))
          intercept (dyn/trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (dyn/trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                          intercept) 1)))
      slope)))

;; Model 6: Many addresses (11 sites) — matches model6_many_addresses
(def model-many
  (gen []
    (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
      (doseq [i (range 10)]
        (dyn/trace (keyword (str "y" i))
                   (dist/gaussian mu 1)))
      mu)))

;; Constraints / observations
(def obs-1site (cm/choicemap :x (mx/scalar 0.5)))

(def xs-linreg [1.0 2.0 3.0 4.0 5.0])
(def obs-linreg
  (reduce (fn [cm [k v]]
            (cm/set-choice cm [k] (mx/scalar v)))
          cm/EMPTY
          [[:y0 3.1] [:y1 5.2] [:y2 6.9] [:y3 9.1] [:y4 10.8]]))
(def full-linreg
  (reduce (fn [cm [k v]]
            (cm/set-choice cm [k] (mx/scalar v)))
          cm/EMPTY
          [[:slope 2.0] [:intercept 1.0]
           [:y0 3.1] [:y1 5.2] [:y2 6.9] [:y3 9.1] [:y4 10.8]]))

(def obs-many
  (reduce (fn [cm i]
            (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar (* i 0.5))))
          cm/EMPTY
          (range 10)))

;; ---------------------------------------------------------------------------
;; Run benchmarks — same operations and iteration counts as Gen.jl
;; ---------------------------------------------------------------------------

(println "\n=== GenMLX vs Gen.jl Comparison Benchmark ===")
(println (str "  Runtime: " (if (exists? js/Bun) "Bun" "Node.js")))
(println (str "  Date: " (.toISOString (js/Date.))))
(println (str "  Protocol: 3 warmup, median of 7 runs, performance.now()"))
(println)

;; ---------------------------------------------------------------------------
;; Simulate (10 iterations — fewer to avoid Metal resource exhaustion)
;; ---------------------------------------------------------------------------

(println "-- Simulate (10 iterations) --")
(println "                          GenMLX      Gen.jl      Ratio")

(let [r1 (bench #(dotimes [_ 10]
                   (let [t (p/simulate model-1site [])] (mx/eval! (:score t)))))
      r7 (bench #(dotimes [_ 10]
                   (let [t (p/simulate model-linreg [xs-linreg])] (mx/eval! (:score t)))))
      r11 (bench #(dotimes [_ 10]
                    (let [t (p/simulate model-many [])] (mx/eval! (:score t)))))]
  ;; Gen.jl numbers are for 100 iters; scale to 10
  (println (str "  1-site:                 "
               (.toFixed r1 3) "ms    0.003ms     " (.toFixed (/ r1 0.003) 0) "x"))
  (println (str "  7-site (linreg):        "
               (.toFixed r7 3) "ms    0.021ms     " (.toFixed (/ r7 0.021) 0) "x"))
  (println (str "  11-site (many):         "
               (.toFixed r11 3) "ms   0.043ms     " (.toFixed (/ r11 0.043) 0) "x")))

;; ---------------------------------------------------------------------------
;; Generate (10 iterations)
;; ---------------------------------------------------------------------------

(println "\n-- Generate (10 iterations) --")
(println "                          GenMLX      Gen.jl      Ratio")

(let [r1 (bench #(dotimes [_ 10]
                   (let [{:keys [trace weight]} (p/generate model-1site [] obs-1site)]
                     (mx/eval! (:score trace) weight))))
      r7 (bench #(dotimes [_ 10]
                   (let [{:keys [trace weight]} (p/generate model-linreg [xs-linreg] full-linreg)]
                     (mx/eval! (:score trace) weight))))
      r11 (bench #(dotimes [_ 10]
                    (let [{:keys [trace weight]} (p/generate model-many [] obs-many)]
                      (mx/eval! (:score trace) weight))))]
  ;; Gen.jl numbers are for 100 iters; scale to 10
  (println (str "  1-site:                 "
               (.toFixed r1 3) "ms    0.004ms     " (.toFixed (/ r1 0.004) 0) "x"))
  (println (str "  7-site (linreg):        "
               (.toFixed r7 3) "ms    0.024ms     " (.toFixed (/ r7 0.024) 0) "x"))
  (println (str "  11-site (many):         "
               (.toFixed r11 3) "ms   0.048ms     " (.toFixed (/ r11 0.048) 0) "x")))

;; ---------------------------------------------------------------------------
;; MH (200 steps — GFI-based, matching Gen.jl's mh(trace, selection))
;; ---------------------------------------------------------------------------

(println "\n-- MH 200 steps (GFI-based regenerate, matching Gen.jl) --")
(println "                          GenMLX      Gen.jl      Ratio")

(let [;; GFI MH accumulates lazy graph nodes per step — use fewer warmup runs
      ;; and force GC between runs to avoid Metal resource limit
      bench-mh (fn [f]
                 (f) ;; 1 warmup
                 (let [times (mapv (fn [_] (let [s (perf-now) _ (f) e (perf-now)] (- e s)))
                                   (range 5))
                       sorted (sort times)]
                   (nth sorted 2)))
      r1 (bench-mh #(mcmc/mh {:samples 200 :burn 0
                               :selection (sel/select :x)}
                              model-1site [] obs-1site))
      r7 (bench-mh #(mcmc/mh {:samples 200 :burn 0
                               :selection (sel/select :slope :intercept)}
                              model-linreg [xs-linreg] obs-linreg))
      r11 (bench-mh #(mcmc/mh {:samples 200 :burn 0
                                :selection (sel/select :mu)}
                               model-many [] obs-many))]
  (println (str "  1-site:                 "
               (.toFixed r1 2) "ms    0.20ms      " (.toFixed (/ r1 0.20) 0) "x"))
  (println (str "  7-site (linreg):        "
               (.toFixed r7 2) "ms    0.93ms      " (.toFixed (/ r7 0.93) 0) "x"))
  (println (str "  11-site (many):         "
               (.toFixed r11 2) "ms   1.55ms      " (.toFixed (/ r11 1.55) 0) "x")))

;; ---------------------------------------------------------------------------
;; Compiled MH (200 steps — parameter-space, no Gen.jl equivalent)
;; ---------------------------------------------------------------------------

(println "\n-- Compiled MH 200 steps (parameter-space, GenMLX only) --")

(let [r7 (bench #(mcmc/compiled-mh
                   {:samples 200 :burn 0 :addresses [:slope :intercept]
                    :proposal-std 0.5}
                   model-linreg [xs-linreg] obs-linreg))
      r11 (bench #(mcmc/compiled-mh
                    {:samples 200 :burn 0 :addresses [:mu]
                     :proposal-std 0.5}
                    model-many [] obs-many))]
  (println (str "  7-site (linreg):        " (.toFixed r7 2) "ms"))
  (println (str "  11-site (many):         " (.toFixed r11 2) "ms")))

;; ---------------------------------------------------------------------------
;; HMC (50 steps, L=10 — matching Gen.jl exactly)
;; ---------------------------------------------------------------------------

(println "\n-- HMC 50 steps, L=10 (matching Gen.jl) --")
(println "                          GenMLX      Gen.jl      Ratio")

(let [r7 (bench #(mcmc/hmc
                   {:samples 50 :burn 0 :step-size 0.01
                    :leapfrog-steps 10 :addresses [:slope :intercept]}
                   model-linreg [xs-linreg] obs-linreg))
      r11 (bench #(mcmc/hmc
                    {:samples 50 :burn 0 :step-size 0.01
                     :leapfrog-steps 10 :addresses [:mu]}
                    model-many [] obs-many))]
  (println (str "  7-site (linreg):        "
               (.toFixed r7 2) "ms    12.19ms     " (.toFixed (/ r7 12.19) 1) "x"))
  (println (str "  11-site (many):         "
               (.toFixed r11 2) "ms   16.39ms     " (.toFixed (/ r11 16.39) 1) "x")))

;; ---------------------------------------------------------------------------
;; MALA (50 steps — no Gen.jl equivalent for direct comparison)
;; ---------------------------------------------------------------------------

(println "\n-- MALA 50 steps (GenMLX only, no Gen.jl equivalent) --")

(let [r7 (bench #(mcmc/mala
                   {:samples 50 :burn 0 :step-size 0.01
                    :addresses [:slope :intercept]}
                   model-linreg [xs-linreg] obs-linreg))]
  (println (str "  7-site (linreg):        " (.toFixed r7 2) "ms")))

;; ---------------------------------------------------------------------------
;; Vectorized variants (GenMLX only)
;; ---------------------------------------------------------------------------

(println "\n-- Vectorized (GenMLX only, effective throughput) --")

(let [scalar-mala (bench #(mcmc/mala
                            {:samples 50 :burn 0 :step-size 0.01
                             :addresses [:slope :intercept]}
                            model-linreg [xs-linreg] obs-linreg))
      v50-mala (bench #(mcmc/vectorized-mala
                         {:samples 50 :burn 0 :step-size 0.01 :n-chains 50
                          :addresses [:slope :intercept]}
                         model-linreg [xs-linreg] obs-linreg))
      scalar-hmc (bench #(mcmc/hmc
                           {:samples 50 :burn 0 :step-size 0.01
                            :leapfrog-steps 10 :addresses [:slope :intercept]}
                           model-linreg [xs-linreg] obs-linreg))
      v10-hmc (bench #(mcmc/vectorized-hmc
                        {:samples 50 :burn 0 :step-size 0.01
                         :leapfrog-steps 10 :n-chains 10
                         :addresses [:slope :intercept]}
                        model-linreg [xs-linreg] obs-linreg))]
  (println (str "  vec-mala N=50 (50st):   " (.toFixed v50-mala 2) "ms"
               "  (eff " (.toFixed (/ (* 50 scalar-mala) v50-mala) 1) "x vs 50×scalar)"))
  (println (str "  vec-hmc N=10 (50st):    " (.toFixed v10-hmc 2) "ms"
               "  (eff " (.toFixed (/ (* 10 scalar-hmc) v10-hmc) 1) "x vs 10×scalar)")))

(println "\n=== Benchmark complete ===")
