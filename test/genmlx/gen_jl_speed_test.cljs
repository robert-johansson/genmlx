(ns genmlx.gen-jl-speed-test
  "Speed benchmark: GenMLX vs Gen.jl.

   Runs the same models and operations as test/reference/gen_jl_benchmark.jl,
   then loads the Gen.jl JSON results (if available) and prints a comparison
   table with speedup ratios.

   Protocol: 3 warmup runs, median of 7 measured runs (wall-clock ms).

   Usage:
     npx nbb test/genmlx/gen_jl_speed_test.cljs

   To generate Gen.jl reference:
     julia test/reference/gen_jl_benchmark.jl"
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.combinators :as comb]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            ["fs" :as fs]
            ["path" :as path])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Timing infrastructure
;; ---------------------------------------------------------------------------

(defn- bench
  "Run f with warmup, then measure `runs` executions and return median ms.
   Each run is wrapped in mx/tidy with nil return to free MLX arrays."
  [f {:keys [warmup runs] :or {warmup 3 runs 7}}]
  (let [;; Wrap f so it returns nil — prevents tidy from traversing deep structures
        tidy-f (fn [] (mx/tidy (fn [] (f) nil)))]
    (dotimes [_ warmup] (tidy-f))
    (let [times (mapv (fn [_]
                        (let [start (js/Date.now)
                              _ (tidy-f)
                              end (js/Date.now)]
                          (- end start)))
                      (range runs))
          sorted (sort times)
          median (nth sorted (quot runs 2))]
      {:times-ms (vec sorted)
       :median-ms median})))

;; ---------------------------------------------------------------------------
;; Load Gen.jl reference data (graceful fallback if missing)
;; ---------------------------------------------------------------------------

(def ^:private ref-path
  (let [script-file (aget js/process.argv 2)]
    (path/resolve (path/dirname script-file) "../reference/gen_jl_benchmark.json")))

(def ^:private gen-jl-data
  (try
    (let [raw (.readFileSync fs ref-path "utf8")
          parsed (js->clj (js/JSON.parse raw) :keywordize-keys true)]
      (println (str "Loaded Gen.jl benchmarks from: " ref-path))
      (println (str "  Gen.jl version: " (get-in parsed [:metadata :gen_version])))
      (println (str "  Julia version: " (get-in parsed [:metadata :julia_version])))
      (println (str "  Date: " (get-in parsed [:metadata :date])))
      parsed)
    (catch :default _
      (println "No Gen.jl benchmark data found (run: julia test/reference/gen_jl_benchmark.jl)")
      (println "Running GenMLX-only benchmarks.")
      nil)))

(defn- lookup-gen-jl
  "Look up Gen.jl median_ms for a given model + operation."
  [model operation]
  (when gen-jl-data
    (some (fn [entry]
            (when (and (= (:model entry) model)
                       (= (:operation entry) operation))
              (:median_ms entry)))
          (:benchmarks gen-jl-data))))

;; ---------------------------------------------------------------------------
;; Models (matching gen_jl_benchmark.jl exactly)
;; ---------------------------------------------------------------------------

;; Model 1: Single Gaussian (1 site)
(def single-gaussian
  (gen []
    (dyn/trace :x (dist/gaussian 0 1))))

;; Model 2: Linear Regression (5 obs + 2 latents = 7 sites)
(def linear-regression
  (gen [xs]
    (let [slope     (dyn/trace :slope (dist/gaussian 0 10))
          intercept (dyn/trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (dyn/trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                          intercept) 1)))
      slope)))

(def linreg-xs [1.0 2.0 3.0 4.0 5.0])

;; Model 3: Mixed discrete/continuous (2 sites)
(def mixed-model
  (gen []
    (let [coin (dyn/trace :coin (dist/bernoulli 0.5))
          _ (mx/eval! coin)
          coin-val (mx/item coin)]
      (if (> coin-val 0.5)
        (dyn/trace :x (dist/gaussian 10 1))
        (dyn/trace :x (dist/gaussian 0 1))))))

;; Model 4: Map combinator (3 elements)
(def map-kernel
  (gen [x]
    (dyn/trace :y (dist/gaussian (mx/scalar x) 1))))

(def map-model (comb/map-combinator map-kernel))

;; Model 5: Unfold combinator (3 steps)
(def unfold-kernel
  (gen [t state]
    (dyn/trace :x (dist/gaussian state 1))))

(def unfold-model (comb/unfold-combinator unfold-kernel))

;; Model 6: Many addresses (11 sites)
(def many-addresses
  (gen []
    (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
      (doseq [i (range 10)]
        (dyn/trace (keyword (str "y" i))
                   (dist/gaussian mu 1)))
      mu)))

;; ---------------------------------------------------------------------------
;; Constraint builders
;; ---------------------------------------------------------------------------

(defn- constraints-single-gaussian []
  (cm/choicemap :x (mx/scalar 0.5)))

(defn- constraints-linear-regression []
  (-> cm/EMPTY
      (cm/set-choice [:slope] (mx/scalar 2.0))
      (cm/set-choice [:intercept] (mx/scalar 1.0))
      (cm/set-choice [:y0] (mx/scalar 3.1))
      (cm/set-choice [:y1] (mx/scalar 5.2))
      (cm/set-choice [:y2] (mx/scalar 6.9))
      (cm/set-choice [:y3] (mx/scalar 9.1))
      (cm/set-choice [:y4] (mx/scalar 10.8))))

(defn- obs-linear-regression []
  (-> cm/EMPTY
      (cm/set-choice [:y0] (mx/scalar 3.1))
      (cm/set-choice [:y1] (mx/scalar 5.2))
      (cm/set-choice [:y2] (mx/scalar 6.9))
      (cm/set-choice [:y3] (mx/scalar 9.1))
      (cm/set-choice [:y4] (mx/scalar 10.8))))

(defn- constraints-mixed []
  (cm/choicemap :coin (mx/scalar 1.0) :x (mx/scalar 10.5)))

(defn- constraints-map []
  (-> cm/EMPTY
      (cm/set-choice [0] (cm/choicemap :y (mx/scalar 1.5)))
      (cm/set-choice [1] (cm/choicemap :y (mx/scalar 2.5)))
      (cm/set-choice [2] (cm/choicemap :y (mx/scalar 3.5)))))

(defn- constraints-unfold []
  (-> cm/EMPTY
      (cm/set-choice [0] (cm/choicemap :x (mx/scalar 0.5)))
      (cm/set-choice [1] (cm/choicemap :x (mx/scalar 1.0)))
      (cm/set-choice [2] (cm/choicemap :x (mx/scalar 1.5)))))

(defn- constraints-many-addresses []
  (reduce (fn [cm i]
            (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar (* i 0.5))))
          (cm/set-choice cm/EMPTY [:mu] (mx/scalar 2.0))
          (range 10)))

(defn- obs-many-addresses []
  (reduce (fn [cm i]
            (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar (* i 0.5))))
          cm/EMPTY
          (range 10)))

;; ---------------------------------------------------------------------------
;; Benchmark collection
;; ---------------------------------------------------------------------------

(def ^:private benchmark-results (volatile! []))

(defn- run-bench!
  "Run a benchmark and store the result."
  [model-name operation iterations f]
  (let [{:keys [median-ms times-ms]} (bench f {:warmup 3 :runs 7})]
    (vswap! benchmark-results conj
            {:model model-name
             :operation operation
             :iterations iterations
             :median-ms median-ms
             :times-ms times-ms
             :gen-jl-ms (lookup-gen-jl model-name operation)})
    (println (str "  " model-name " / " operation " [" iterations "x]: " median-ms " ms"))))

;; ---------------------------------------------------------------------------
;; Simulate benchmarks (100 iterations per timed run)
;; ---------------------------------------------------------------------------

(println "\n=== GenMLX Speed Benchmarks ===")
(println "\n-- Simulate --")

(run-bench! "single_gaussian" "simulate" 100
  (fn [] (dotimes [_ 100]
           (p/simulate single-gaussian []))))

(run-bench! "linear_regression" "simulate" 100
  (fn [] (dotimes [_ 100]
           (p/simulate linear-regression [linreg-xs]))))

(run-bench! "mixed" "simulate" 100
  (fn [] (dotimes [_ 100]
           (p/simulate mixed-model []))))

(run-bench! "map_combinator" "simulate" 100
  (fn [] (dotimes [_ 100]
           (p/simulate map-model [[1.0 2.0 3.0]]))))

(run-bench! "unfold_combinator" "simulate" 100
  (fn [] (dotimes [_ 100]
           (p/simulate unfold-model [3 (mx/scalar 0.0)]))))

(run-bench! "many_addresses" "simulate" 100
  (fn [] (dotimes [_ 100]
           (p/simulate many-addresses []))))

;; ---------------------------------------------------------------------------
;; Generate benchmarks (fully constrained, 100 iterations)
;; ---------------------------------------------------------------------------

(println "\n-- Generate --")

(run-bench! "single_gaussian" "generate" 100
  (fn [] (let [c (constraints-single-gaussian)]
           (dotimes [_ 100]
             (p/generate single-gaussian [] c)))))

(run-bench! "linear_regression" "generate" 100
  (fn [] (let [c (constraints-linear-regression)]
           (dotimes [_ 100]
             (p/generate linear-regression [linreg-xs] c)))))

(run-bench! "mixed" "generate" 100
  (fn [] (let [c (constraints-mixed)]
           (dotimes [_ 100]
             (p/generate mixed-model [] c)))))

(run-bench! "map_combinator" "generate" 100
  (fn [] (let [c (constraints-map)]
           (dotimes [_ 100]
             (p/generate map-model [[1.0 2.0 3.0]] c)))))

(run-bench! "unfold_combinator" "generate" 100
  (fn [] (let [c (constraints-unfold)]
           (dotimes [_ 100]
             (p/generate unfold-model [3 (mx/scalar 0.0)] c)))))

(run-bench! "many_addresses" "generate" 100
  (fn [] (let [c (constraints-many-addresses)]
           (dotimes [_ 100]
             (p/generate many-addresses [] c)))))

;; ---------------------------------------------------------------------------
;; Update benchmarks (models 1, 2, 6 — 100 iterations)
;; ---------------------------------------------------------------------------

(println "\n-- Update --")

(run-bench! "single_gaussian" "update" 100
  (fn [] (let [c (constraints-single-gaussian)
               {:keys [trace]} (p/generate single-gaussian [] c)
               new-c (cm/choicemap :x (mx/scalar -0.5))]
           (dotimes [_ 100]
             (p/update single-gaussian trace new-c)))))

(run-bench! "linear_regression" "update" 100
  (fn [] (let [c (constraints-linear-regression)
               {:keys [trace]} (p/generate linear-regression [linreg-xs] c)
               new-c (cm/set-choice cm/EMPTY [:slope] (mx/scalar 3.0))]
           (dotimes [_ 100]
             (p/update linear-regression trace new-c)))))

(run-bench! "many_addresses" "update" 100
  (fn [] (let [c (constraints-many-addresses)
               {:keys [trace]} (p/generate many-addresses [] c)
               new-c (cm/set-choice cm/EMPTY [:mu] (mx/scalar 3.0))]
           (dotimes [_ 100]
             (p/update many-addresses trace new-c)))))

;; ---------------------------------------------------------------------------
;; Importance sampling benchmarks (all 6 models, 100 particles, 10 calls)
;; No tidy — IS already manages arrays internally
;; ---------------------------------------------------------------------------

(println "\n-- Importance Sampling (100 particles x 10 calls) --")

(run-bench! "single_gaussian" "importance_sampling" 10
  (fn [] (dotimes [_ 10]
           (is/importance-sampling {:samples 100}
             single-gaussian [] (constraints-single-gaussian)))))

(run-bench! "linear_regression" "importance_sampling" 10
  (fn [] (dotimes [_ 10]
           (is/importance-sampling {:samples 100}
             linear-regression [linreg-xs] (obs-linear-regression)))))

(run-bench! "mixed" "importance_sampling" 10
  (fn [] (dotimes [_ 10]
           (is/importance-sampling {:samples 100}
             mixed-model [] (cm/choicemap :x (mx/scalar 10.5))))))

(run-bench! "map_combinator" "importance_sampling" 10
  (fn [] (dotimes [_ 10]
           (is/importance-sampling {:samples 100}
             map-model [[1.0 2.0 3.0]] (constraints-map)))))

(run-bench! "unfold_combinator" "importance_sampling" 10
  (fn [] (dotimes [_ 10]
           (is/importance-sampling {:samples 100}
             unfold-model [3 (mx/scalar 0.0)] (constraints-unfold)))))

(run-bench! "many_addresses" "importance_sampling" 10
  (fn [] (dotimes [_ 10]
           (is/importance-sampling {:samples 100}
             many-addresses [] (obs-many-addresses)))))

;; ---------------------------------------------------------------------------
;; MH benchmarks (models 1, 2, 6 — 200 steps)
;; No tidy — MH manages its own trace chain
;; ---------------------------------------------------------------------------

(println "\n-- MH (200 steps) --")

(run-bench! "single_gaussian" "mh" 200
  (fn [] (mcmc/mh {:samples 200
                   :selection (sel/select :x)}
                  single-gaussian [] (constraints-single-gaussian))))

(run-bench! "linear_regression" "mh" 200
  (fn [] (mcmc/mh {:samples 200
                   :selection (sel/select :slope :intercept)}
                  linear-regression [linreg-xs] (obs-linear-regression))))

(run-bench! "many_addresses" "mh" 200
  (fn [] (mcmc/mh {:samples 200
                   :selection (sel/select :mu)}
                  many-addresses [] (obs-many-addresses))))

;; ---------------------------------------------------------------------------
;; Compiled MH benchmarks (200 steps, eager — for comparison with lazy)
;; ---------------------------------------------------------------------------

(println "\n-- Compiled MH (200 steps) --")

(run-bench! "single_gaussian" "compiled_mh" 200
  (fn [] (mcmc/compiled-mh
           {:samples 200 :addresses [:x]}
           single-gaussian [] (constraints-single-gaussian))))

(run-bench! "linear_regression" "compiled_mh" 200
  (fn [] (mcmc/compiled-mh
           {:samples 200 :addresses [:slope :intercept]}
           linear-regression [linreg-xs] (obs-linear-regression))))

(run-bench! "many_addresses" "compiled_mh" 200
  (fn [] (mcmc/compiled-mh
           {:samples 200 :addresses [:mu]}
           many-addresses [] (obs-many-addresses))))

;; ---------------------------------------------------------------------------
;; Lazy Compiled MH benchmarks (200 steps, no per-step GPU sync)
;; ---------------------------------------------------------------------------

(println "\n-- Lazy Compiled MH (200 steps) --")

(run-bench! "single_gaussian" "compiled_mh_lazy" 200
  (fn [] (mcmc/compiled-mh-lazy
           {:samples 200 :addresses [:x]}
           single-gaussian [] (constraints-single-gaussian))))

(run-bench! "linear_regression" "compiled_mh_lazy" 200
  (fn [] (mcmc/compiled-mh-lazy
           {:samples 200 :addresses [:slope :intercept]}
           linear-regression [linreg-xs] (obs-linear-regression))))

(run-bench! "many_addresses" "compiled_mh_lazy" 200
  (fn [] (mcmc/compiled-mh-lazy
           {:samples 200 :addresses [:mu]}
           many-addresses [] (obs-many-addresses))))

;; ---------------------------------------------------------------------------
;; HMC benchmarks (models 2, 6 — 50 steps, L=10)
;; No tidy — HMC manages its own parameter chain
;; ---------------------------------------------------------------------------

(println "\n-- HMC (50 steps, L=10) --")

(run-bench! "linear_regression" "hmc" 50
  (fn [] (mcmc/hmc {:samples 50 :burn 0 :step-size 0.01
                    :leapfrog-steps 10
                    :addresses [:slope :intercept]}
                   linear-regression [linreg-xs] (obs-linear-regression))))

(run-bench! "many_addresses" "hmc" 50
  (fn [] (mcmc/hmc {:samples 50 :burn 0 :step-size 0.01
                    :leapfrog-steps 10
                    :addresses [:mu]}
                   many-addresses [] (obs-many-addresses))))

;; ---------------------------------------------------------------------------
;; Lazy HMC benchmarks (50 steps, L=10, no per-step GPU sync)
;; ---------------------------------------------------------------------------

(println "\n-- Lazy HMC (50 steps, L=10) --")

(run-bench! "linear_regression" "hmc_lazy" 50
  (fn [] (mcmc/hmc-lazy
           {:samples 50 :burn 0 :step-size 0.01
            :leapfrog-steps 10 :addresses [:slope :intercept]}
           linear-regression [linreg-xs] (obs-linear-regression))))

(run-bench! "many_addresses" "hmc_lazy" 50
  (fn [] (mcmc/hmc-lazy
           {:samples 50 :burn 0 :step-size 0.01
            :leapfrog-steps 10 :addresses [:mu]}
           many-addresses [] (obs-many-addresses))))

;; ---------------------------------------------------------------------------
;; Comparison table
;; ---------------------------------------------------------------------------

(println "\n=== Speed Comparison: Gen.jl vs GenMLX ===\n")

(defn- pad-right [s n]
  (let [s (str s)]
    (if (>= (count s) n) s
        (str s (apply str (repeat (- n (count s)) " "))))))

(defn- pad-left [s n]
  (let [s (str s)]
    (if (>= (count s) n) s
        (str (apply str (repeat (- n (count s)) " ")) s))))

(defn- format-ms [ms]
  (if ms
    (pad-left (.toFixed ms 1) 8)
    (pad-left "n/a" 8)))

(defn- format-ratio [gen-jl-ms genmlx-ms]
  (if (and gen-jl-ms genmlx-ms (pos? genmlx-ms))
    (let [ratio (/ gen-jl-ms genmlx-ms)]
      (pad-left (str (.toFixed ratio 2) "x") 9))
    (pad-left "n/a" 9)))

(let [;; Abbreviate labels for table readability
      abbrev {"single_gaussian" "gauss1"
              "linear_regression" "linreg"
              "mixed" "mixed"
              "map_combinator" "map"
              "unfold_combinator" "unfold"
              "many_addresses" "many11"
              "importance_sampling" "IS(100p)"
              "simulate" "simulate"
              "generate" "generate"
              "update" "update"
              "mh" "MH(200)"
              "compiled_mh" "cMH(200)"
              "compiled_mh_lazy" "lazyMH(200)"
              "hmc" "HMC(50)"
              "hmc_lazy" "lazyHMC(50)"}
      results @benchmark-results
      w 28  ;; label column width
      header (str (pad-right "Benchmark" w) " "
                  (pad-left "Iter" 5) " "
                  (pad-left "Gen.jl" 8) " "
                  (pad-left "GenMLX" 8) " "
                  (pad-left "Ratio" 9))
      sep-line (apply str (repeat (count header) "-"))]
  (println sep-line)
  (println header)
  (println (str (pad-right "" w) " "
                (pad-left "" 5) " "
                (pad-left "(ms)" 8) " "
                (pad-left "(ms)" 8) " "
                (pad-left "(Jl/MLX)" 9)))
  (println sep-line)

  (doseq [{:keys [model operation iterations median-ms gen-jl-ms]} results]
    (let [label (str (get abbrev model model) " / " (get abbrev operation operation))]
      (println (str (pad-right label w) " "
                    (pad-left (str iterations) 5) " "
                    (format-ms gen-jl-ms) " "
                    (format-ms median-ms) " "
                    (format-ratio gen-jl-ms median-ms)))))

  (println sep-line)

  ;; Geometric mean of ratios (only where both values exist)
  (let [ratios (keep (fn [{:keys [median-ms gen-jl-ms]}]
                       (when (and gen-jl-ms median-ms (pos? median-ms) (pos? gen-jl-ms))
                         (/ gen-jl-ms median-ms)))
                     results)]
    (when (seq ratios)
      (let [geo-mean (js/Math.pow (reduce * ratios) (/ 1.0 (count ratios)))]
        (println (str "\nGeometric mean ratio (Gen.jl / GenMLX): "
                      (.toFixed geo-mean 2) "x"
                      " (across " (count ratios) " benchmarks)"))
        (println (str "  >1 = GenMLX faster, <1 = Gen.jl faster"))))))

(println "\nBenchmarks complete.")
