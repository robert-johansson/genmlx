(ns genmlx.compiled-benchmark
  "Benchmark suite: compiled inference vs GFI-based inference.
   Measures speedup from compiled score functions and parameter-space iteration.

   Timing protocol: performance.now(), warmup, nested loop timing.
   Memory-aware: mx/clear-cache! between benchmarks, reduced reps for MCMC."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.vi :as vi]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Timing infrastructure
;; ---------------------------------------------------------------------------

(defn perf-now [] (js/performance.now))

(defn timing [f repeats inner-repeats]
  (let [times (loop [i 0 acc (transient [])]
                (if (>= i repeats)
                  (persistent! acc)
                  (let [inner-min
                        (loop [j 0 best js/Infinity]
                          (if (>= j inner-repeats)
                            best
                            (let [start (perf-now)
                                  _ (f)
                                  elapsed (- (perf-now) start)]
                              (mx/clear-cache!)
                              (recur (inc j) (min best elapsed)))))]
                    (recur (inc i) (conj! acc inner-min)))))
        n (count times)
        mean (/ (reduce + times) n)
        variance (/ (reduce + (map #(let [d (- % mean)] (* d d)) times)) n)
        std (js/Math.sqrt variance)]
    {:times times :mean mean :std std}))

(defn bench
  "Run f with warmup, then timing. Report mean +/- std in ms."
  [label f {:keys [warmup-runs repeats inner-repeats]
            :or {warmup-runs 3 repeats 7 inner-repeats 3}}]
  (dotimes [_ warmup-runs]
    (f) (mx/clear-cache!))
  (let [{:keys [mean std]} (timing f repeats inner-repeats)]
    (println (str "  " label ": " (.toFixed mean 1) " +/- " (.toFixed std 1) " ms"))
    {:mean mean :std std}))

;; ---------------------------------------------------------------------------
;; JSON output
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))
(def results-dir
  (.resolve path-mod (js/process.cwd) "results/exp6_compilation"))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir results-dir)
  (let [filepath (str results-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  Wrote: " filepath))))

;; ---------------------------------------------------------------------------
;; Benchmark model: linear regression (2 latents, 5 observations)
;; ---------------------------------------------------------------------------

(def linreg-model
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                          intercept) 1)))
      slope)))

(def linreg-xs [1.0 2.0 3.0 4.0 5.0])
(def linreg-obs
  (reduce (fn [cm [j y]]
            (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector [2.1 3.9 6.2 7.8 10.1])))

;; ---------------------------------------------------------------------------
;; Run benchmarks
;; ---------------------------------------------------------------------------

(println "\n=== GenMLX Compiled Inference Benchmarks ===")
(println "Timing: performance.now(), warmup + nested loop timing")

(def all-results (atom {}))

;; ---------------------------------------------------------------------------
;; Benchmark 1: GFI MH vs Compiled MH (500 samples)
;; ---------------------------------------------------------------------------

(println "\n-- 1. MH: GFI vs Compiled (linear regression, 500 samples) --")

(let [gfi (bench "GFI MH"
            (fn [] (mcmc/mh {:samples 500 :burn 50
                              :selection (sel/select :slope :intercept)}
                             linreg-model [linreg-xs] linreg-obs))
            {:warmup-runs 2 :repeats 5 :inner-repeats 3})
      _ (mx/clear-cache!)
      compiled (bench "Compiled MH"
                 (fn [] (mcmc/compiled-mh
                          {:samples 500 :burn 50
                           :addresses [:slope :intercept]
                           :proposal-std 0.5}
                          linreg-model [linreg-xs] linreg-obs))
                 {})
      speedup (/ (:mean gfi) (:mean compiled))]
  (println (str "  Speedup: " (.toFixed speedup 1) "x"))
  (swap! all-results assoc :bench1_gfi_mh_vs_compiled_mh
         {:gfi_mh gfi :compiled_mh compiled :speedup speedup}))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; Benchmark 2: Compiled vs Uncompiled score-fn (50 evaluations)
;; ---------------------------------------------------------------------------

(println "\n-- 2. Compiled vs Uncompiled score-fn (50 evaluations) --")

(let [score-fn (u/make-score-fn linreg-model [linreg-xs] linreg-obs
                                [:slope :intercept])
      compiled-score (mx/compile-fn score-fn)
      test-params (mx/array [2.0 0.5])

      raw (bench "Uncompiled score-fn"
            (fn [] (mx/tidy #(dotimes [_ 50]
                               (let [s (score-fn test-params)]
                                 (mx/eval! s)))))
            {:warmup-runs 3 :repeats 10 :inner-repeats 10})
      comp (bench "Compiled score-fn"
             (fn [] (mx/tidy #(dotimes [_ 50]
                                (let [s (compiled-score test-params)]
                                  (mx/eval! s)))))
             {:warmup-runs 3 :repeats 10 :inner-repeats 10})
      speedup (/ (:mean raw) (:mean comp))]
  (println (str "  Speedup: " (.toFixed speedup 1) "x"))
  (swap! all-results assoc :bench2_score_fn_compilation
         {:uncompiled raw :compiled comp :speedup speedup}))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; Benchmark 3: HMC GenMLX vs Handcoded (200 samples)
;; ---------------------------------------------------------------------------

(println "\n-- 3. HMC: GenMLX vs Handcoded (linear regression, 200 samples) --")

(let [genmlx (bench "GenMLX HMC"
               (fn [] (mcmc/hmc
                        {:samples 200 :burn 50 :step-size 0.005
                         :leapfrog-steps 10
                         :addresses [:slope :intercept]}
                        linreg-model [linreg-xs] linreg-obs))
               {})

      _ (mx/clear-cache!)

      ;; Handcoded HMC — direct MLX, no GFI
      xs-arr (mx/array linreg-xs)
      ys-arr (mx/array [2.1 3.9 6.2 7.8 10.1])

      log-density (fn [params]
                    (let [slope (mx/index params 0)
                          intercept (mx/index params 1)
                          pred (mx/add (mx/multiply slope xs-arr) intercept)
                          resid (mx/subtract ys-arr pred)
                          ll (mx/negative (mx/divide (mx/sum (mx/square resid))
                                                     (mx/scalar 2.0)))
                          lp (mx/negative (mx/divide (mx/sum (mx/square params))
                                                     (mx/scalar 200.0)))]
                      (mx/add ll lp)))

      grad-ld (mx/compile-fn (mx/grad log-density))
      log-density-compiled (mx/compile-fn log-density)
      eps-val 0.005
      L 10
      eps (mx/scalar eps-val)
      half-eps (mx/scalar (* 0.5 eps-val))
      half (mx/scalar 0.5)

      handcoded-hmc
      (fn []
        (let [init-q (mx/array [0.0 0.0])
              q-shape [2]]
          (loop [i 0, q init-q, samples (transient [])]
            (if (>= i 250) ;; 200 samples + 50 burn
              (persistent! samples)
              (let [p0 (doto (rng/normal (rng/fresh-key) q-shape) mx/eval!)
                    neg-U (log-density-compiled q)
                    K0 (mx/multiply half (mx/sum (mx/square p0)))
                    _ (mx/eval! neg-U K0)
                    current-H (+ (mx/item neg-U) (mx/item K0))
                    ;; Fused leapfrog: L+1 gradient evals (matches GenMLX)
                    [q' p'] (let [r (mx/tidy
                                      (fn []
                                        (let [g (grad-ld q)
                                              pi (mx/subtract p0 (mx/multiply half-eps g))
                                              qi (mx/add q (mx/multiply eps pi))]
                                          (loop [step 1, qi qi, pi pi]
                                            (if (>= step L)
                                              (let [g (grad-ld qi)
                                                    pi (mx/subtract pi (mx/multiply half-eps g))]
                                                (mx/eval! qi pi)
                                                #js [qi pi])
                                              (let [g (grad-ld qi)
                                                    pi (mx/subtract pi (mx/multiply eps g))
                                                    qi (mx/add qi (mx/multiply eps pi))]
                                                (recur (inc step) qi pi)))))))]
                              [(aget r 0) (aget r 1)])
                    neg-U' (log-density-compiled q')
                    K1 (mx/multiply half (mx/sum (mx/square p')))
                    _ (mx/eval! neg-U' K1)
                    proposed-H (+ (mx/item neg-U') (mx/item K1))
                    log-alpha (- current-H proposed-H)
                    accept? (or (>= log-alpha 0) (< (js/Math.log (js/Math.random)) log-alpha))
                    q-next (if accept? q' q)]
                (recur (inc i) q-next
                       (if (>= i 50) (conj! samples (mx/->clj q-next)) samples)))))))

      handcoded (bench "Handcoded HMC" handcoded-hmc {})
      overhead (/ (:mean genmlx) (:mean handcoded))]
  (println (str "  GenMLX overhead: " (.toFixed overhead 2) "x"))
  (swap! all-results assoc :bench3_hmc_genmlx_vs_handcoded
         {:genmlx_hmc genmlx :handcoded_hmc handcoded :overhead overhead}))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; Benchmark 4: Serial vs Vectorized MH (N parallel chains, 100 samples)
;; ---------------------------------------------------------------------------

(println "\n-- 4. Serial vs Vectorized MH (10 chains, 100 samples) --")

(let [n-chains 10
      n-samples 100
      one (bench "1x Compiled MH"
            (fn [] (mcmc/compiled-mh
                     {:samples n-samples :burn 20
                      :addresses [:slope :intercept]
                      :proposal-std 0.5}
                     linreg-model [linreg-xs] linreg-obs))
            {})
      serial-mean (* n-chains (:mean one))
      serial-std (* n-chains (:std one))
      _ (println (str "  Serial " n-chains "x (extrapolated): "
                      (.toFixed serial-mean 1) " +/- " (.toFixed serial-std 1) " ms"))
      _ (mx/clear-cache!)
      vec-result (bench (str "Vectorized MH (" n-chains " chains)")
                   (fn [] (mcmc/vectorized-compiled-mh
                            {:samples n-samples :burn 20
                             :addresses [:slope :intercept]
                             :proposal-std 0.5
                             :n-chains n-chains}
                            linreg-model [linreg-xs] linreg-obs))
                   {})
      speedup (/ serial-mean (:mean vec-result))]
  (println (str "  Speedup: " (.toFixed speedup 1) "x"))
  (swap! all-results assoc :bench4_serial_vs_vectorized_mh
         {:one_chain one :serial_extrapolated {:mean serial-mean :std serial-std}
          :vectorized vec-result :speedup speedup :n_chains n-chains}))

;; ---------------------------------------------------------------------------
;; Write results
;; ---------------------------------------------------------------------------

(println "\n-- Writing results --")

(write-json "compiled_speedup.json" @all-results)

(let [r @all-results
      b1 (:bench1_gfi_mh_vs_compiled_mh r)
      b2 (:bench2_score_fn_compilation r)
      b3 (:bench3_hmc_genmlx_vs_handcoded r)
      b4 (:bench4_serial_vs_vectorized_mh r)
      fmt (fn [x] (if x (.toFixed x 1) "—"))
      summary
      (str "# Experiment 6: Loop Compilation Speedup\n\n"
           "**Date:** 2026-03-03\n"
           "**Platform:** macOS, Apple Silicon, MLX GPU via @mlx-node/core, Bun + nbb\n"
           "**Benchmark file:** `test/genmlx/compiled_benchmark.cljs`\n"
           "**Methodology:** performance.now(), warmup + nested loop (min-of-inner, mean+std-of-outer)\n\n"
           "## Results\n\n"
           "| Benchmark | Baseline (ms) | Compiled/Batched (ms) | Speedup |\n"
           "|-----------|--------------|----------------------|--------|\n"
           "| GFI MH vs Compiled MH (500 samples) | "
           (fmt (get-in b1 [:gfi_mh :mean])) " +/- " (fmt (get-in b1 [:gfi_mh :std]))
           " | " (fmt (get-in b1 [:compiled_mh :mean])) " +/- " (fmt (get-in b1 [:compiled_mh :std]))
           " | " (fmt (:speedup b1)) "x |\n"
           "| Uncompiled vs Compiled score-fn (50 evals) | "
           (fmt (get-in b2 [:uncompiled :mean])) " +/- " (fmt (get-in b2 [:uncompiled :std]))
           " | " (fmt (get-in b2 [:compiled :mean])) " +/- " (fmt (get-in b2 [:compiled :std]))
           " | " (fmt (:speedup b2)) "x |\n"
           "| GenMLX HMC vs Handcoded HMC (200 samples) | "
           (fmt (get-in b3 [:genmlx_hmc :mean])) " +/- " (fmt (get-in b3 [:genmlx_hmc :std]))
           " | " (fmt (get-in b3 [:handcoded_hmc :mean])) " +/- " (fmt (get-in b3 [:handcoded_hmc :std]))
           " | " (.toFixed (:overhead b3) 2) "x overhead |\n"
           "| Serial 10x MH vs Vectorized 10-chain MH | "
           (fmt (get-in b4 [:serial_extrapolated :mean]))
           " | " (fmt (get-in b4 [:vectorized :mean])) " +/- " (fmt (get-in b4 [:vectorized :std]))
           " | " (fmt (:speedup b4)) "x |\n\n"
           "## Key Findings\n\n"
           "1. **Compiled MH achieves " (fmt (:speedup b1)) "x speedup over GFI MH.**\n\n"
           "2. **Score function compilation gives " (fmt (:speedup b2)) "x speedup.**\n\n"
           "3. **GenMLX HMC overhead vs handcoded: " (.toFixed (:overhead b3) 2) "x.**\n"
           "   Both use fused leapfrog (L+1 gradient evals for L steps).\n\n"
           "4. **Vectorized 10-chain MH gives " (fmt (:speedup b4)) "x speedup.**\n")]
  (.writeFileSync fs (str results-dir "/SUMMARY.md") summary)
  (println (str "  Wrote: " results-dir "/SUMMARY.md")))

(println "\nAll benchmarks complete.")
