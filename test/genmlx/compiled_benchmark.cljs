(ns genmlx.compiled-benchmark
  "Benchmark suite: compiled inference vs GFI-based inference.
   Measures speedup from compiled score functions and parameter-space iteration."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Timing infrastructure
;; ---------------------------------------------------------------------------

(defn bench
  "Run f with warmup, then measure `runs` executions and report median ms."
  [label f {:keys [warmup runs] :or {warmup 2 runs 5}}]
  (dotimes [_ warmup] (f))
  (let [times (mapv (fn [_]
                      (let [start (js/Date.now)
                            _ (f)
                            end (js/Date.now)]
                        (- end start)))
                    (range runs))
        sorted (sort times)
        median (nth sorted (quot runs 2))]
    (println (str "  " label ": " median "ms (median of " runs ")"))
    median))

;; ---------------------------------------------------------------------------
;; Benchmark models
;; ---------------------------------------------------------------------------

;; Model A: simple Gaussian (1 latent, 1 observation)
(def simple-model
  (gen []
    (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
      (dyn/trace :obs (dist/gaussian mu 1))
      mu)))

(def simple-obs (cm/choicemap :obs (mx/scalar 3.0)))

;; Model B: linear regression (2 latents, 5 observations)
(def linreg-model
  (gen [xs]
    (let [slope     (dyn/trace :slope (dist/gaussian 0 10))
          intercept (dyn/trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (dyn/trace (keyword (str "y" j))
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

;; ---------------------------------------------------------------------------
;; Benchmark 1: GFI MH vs Compiled MH (linear regression, 500 samples)
;; ---------------------------------------------------------------------------

(println "\n-- 1. MH: GFI vs Compiled (linear regression, 500 samples) --")

(let [gfi-ms (bench "GFI MH"
               (fn [] (mcmc/mh {:samples 500 :burn 50
                                :selection (sel/select :slope :intercept)}
                               linreg-model [linreg-xs] linreg-obs))
               {:warmup 1 :runs 3})
      compiled-ms (bench "Compiled MH"
                    (fn [] (mcmc/compiled-mh
                             {:samples 500 :burn 50
                              :addresses [:slope :intercept]
                              :proposal-std 0.5}
                             linreg-model [linreg-xs] linreg-obs))
                    {:warmup 1 :runs 3})
      speedup (if (pos? compiled-ms) (/ gfi-ms compiled-ms) ##Inf)]
  (println (str "  Speedup: " (.toFixed speedup 1) "x")))

;; ---------------------------------------------------------------------------
;; Benchmark 2: Compiled vs Uncompiled score-fn (run early, lightweight)
;; ---------------------------------------------------------------------------

(println "\n-- 2. Compiled vs Uncompiled score-fn (200 evaluations) --")

(let [score-fn (u/make-score-fn linreg-model [linreg-xs] linreg-obs
                                [:slope :intercept])
      compiled (mx/compile-fn score-fn)
      test-params (mx/array [2.0 0.5])

      raw-ms (bench "Uncompiled score-fn"
               (fn [] (dotimes [_ 200]
                        (let [s (score-fn test-params)]
                          (mx/eval! s))))
               {:warmup 1 :runs 3})
      comp-ms (bench "Compiled score-fn"
                (fn [] (dotimes [_ 200]
                         (let [s (compiled test-params)]
                           (mx/eval! s))))
                {:warmup 1 :runs 3})
      speedup (if (pos? comp-ms) (/ raw-ms comp-ms) ##Inf)]
  (println (str "  Speedup: " (.toFixed speedup 1) "x")))

;; ---------------------------------------------------------------------------
;; Benchmark 3: Compiled MH vs MALA (linear regression, 500 samples)
;; ---------------------------------------------------------------------------

(println "\n-- 3. Compiled MH vs MALA (linear regression, 500 samples) --")

(let [cmh-ms (bench "Compiled MH"
               (fn [] (mcmc/compiled-mh
                        {:samples 500 :burn 50
                         :addresses [:slope :intercept]
                         :proposal-std 0.5}
                        linreg-model [linreg-xs] linreg-obs))
               {:warmup 1 :runs 3})
      mala-ms (bench "MALA"
                (fn [] (mcmc/mala
                         {:samples 500 :burn 50 :step-size 0.01
                          :addresses [:slope :intercept]}
                         linreg-model [linreg-xs] linreg-obs))
                {:warmup 1 :runs 3})
      ratio (if (pos? mala-ms) (/ mala-ms cmh-ms) ##Inf)]
  (println (str "  MALA/Compiled-MH ratio: " (.toFixed ratio 1) "x")))

;; ---------------------------------------------------------------------------
;; Benchmark 4: HMC GenMLX vs Handcoded (linear regression, 200 samples)
;; ---------------------------------------------------------------------------

(println "\n-- 4. HMC: GenMLX vs Handcoded (linear regression, 200 samples) --")

(let [;; GenMLX HMC
      genmlx-ms (bench "GenMLX HMC"
                  (fn [] (mcmc/hmc
                           {:samples 200 :burn 50 :step-size 0.005
                            :leapfrog-steps 10
                            :addresses [:slope :intercept]}
                           linreg-model [linreg-xs] linreg-obs))
                  {:warmup 1 :runs 3})

      ;; Handcoded HMC — direct MLX, no GFI
      xs-arr (mx/array linreg-xs)
      ys-arr (mx/array [2.1 3.9 6.2 7.8 10.1])

      ;; Hand-written log-density: Gaussian prior + Gaussian likelihood
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
              (let [p0 (doto (mx/random-normal q-shape) mx/eval!)
                    neg-U (log-density-compiled q)
                    K0 (mx/multiply half (mx/sum (mx/square p0)))
                    _ (mx/eval! neg-U K0)
                    current-H (+ (mx/item neg-U) (mx/item K0))
                    [q' p'] (let [r (mx/tidy
                                      (fn []
                                        (loop [step 0, qi q, pi p0]
                                          (if (>= step L)
                                            (do (mx/eval! qi pi) #js [qi pi])
                                            (let [g (grad-ld qi)
                                                  pi (mx/subtract pi (mx/multiply half-eps g))
                                                  qi (mx/add qi (mx/multiply eps pi))
                                                  g (grad-ld qi)
                                                  pi (mx/subtract pi (mx/multiply half-eps g))]
                                              (recur (inc step) qi pi))))))]
                              [(aget r 0) (aget r 1)])
                    neg-U' (log-density-compiled q')
                    K1 (mx/multiply half (mx/sum (mx/square p')))
                    _ (mx/eval! neg-U' K1)
                    proposed-H (+ (mx/item neg-U') (mx/item K1))
                    log-alpha (- current-H proposed-H)
                    accept? (or (>= log-alpha 0) (< (js/Math.log (js/Math.random)) log-alpha))
                    q-next (if accept? q' q)]
                (recur (inc i)
                       q-next
                       (if (>= i 50) (conj! samples (mx/->clj q-next)) samples)))))))

      handcoded-ms (bench "Handcoded HMC"
                     handcoded-hmc
                     {:warmup 1 :runs 3})
      overhead (if (pos? handcoded-ms) (/ genmlx-ms handcoded-ms) ##Inf)]
  (println (str "  GenMLX overhead: " (.toFixed overhead 1) "x")))

;; ---------------------------------------------------------------------------
;; Benchmark 5: Compiled MH vs Vectorized MH (N parallel chains)
;; ---------------------------------------------------------------------------

(println "\n-- 5. Compiled MH vs Vectorized MH (linear regression, 200 samples) --")

(let [n-chains 10
      n-samples 200
      ;; Measure 1 compiled-MH chain, extrapolate serial cost for N chains
      one-ms (bench "1x Compiled MH"
               (fn [] (mcmc/compiled-mh
                        {:samples n-samples :burn 20
                         :addresses [:slope :intercept]
                         :proposal-std 0.5}
                        linreg-model [linreg-xs] linreg-obs))
               {:warmup 1 :runs 3})
      serial-ms (* n-chains one-ms)
      _ (println (str "  Serial " n-chains "x (extrapolated): " serial-ms "ms"))
      ;; Vectorized MH: N chains in parallel via broadcasting
      vec-ms (bench (str "Vectorized MH (" n-chains " chains)")
               (fn [] (mcmc/vectorized-compiled-mh
                        {:samples n-samples :burn 20
                         :addresses [:slope :intercept]
                         :proposal-std 0.5
                         :n-chains n-chains}
                        linreg-model [linreg-xs] linreg-obs))
               {:warmup 1 :runs 3})
      speedup (if (pos? vec-ms) (/ serial-ms vec-ms) ##Inf)]
  (println (str "  Speedup: " (.toFixed speedup 1) "x")))

;; ---------------------------------------------------------------------------
;; Benchmark 6: Sequential SMC vs Vectorized SMC (time-series, 5 steps)
;; ---------------------------------------------------------------------------

(println "\nAll benchmarks complete.")
(println "(See vsmc_benchmark.cljs for Sequential SMC vs Vectorized SMC)")
