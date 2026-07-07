(ns bench.cross-system
  "Cross-System Comparison benchmark (GenMLX side).

   Runs the same models/algorithms as Gen.jl and GenJAX benchmarks.
   Results combined with bench/genjl/ and bench/genjax/ for the comparison table.

   Usage: bun run --bun nbb bench/cross_system.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.smc :as smc]
            [genmlx.fit :as fit]
            [genmlx.method-selection :as ms])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(defn perf-now [] (js/performance.now))

(def fs (js/require "fs"))
(def path-mod (js/require "path"))
(def results-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/cross-system")))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir results-dir)
  (let [filepath (str results-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  Wrote: " filepath))))

(defn benchmark [label f & {:keys [warmup-n outer-n inner-n]
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
     :min-ms (apply min outer-times) :raw outer-times}))

(defn safe-benchmark
  "Like benchmark but catches errors and returns a skip marker."
  [label f & opts]
  (try
    (apply benchmark label f opts)
    (catch :default e
      (println (str "  [" label "] SKIPPED -- " (.-message e)))
      {:label label :mean-ms nil :std-ms nil :min-ms nil :raw []
       :skipped true :error (.-message e)})))

;; ---------------------------------------------------------------------------
;; Models (matching Gen.jl and GenJAX versions)
;; ---------------------------------------------------------------------------

;; Dynamic linreg (for fair cross-system comparison)
(def linreg-model
  (dyn/auto-key
    (gen [xs]
      (let [slope     (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 10))]
        (doseq [[j x] (map-indexed vector xs)]
          (trace (keyword (str "y" j))
                 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                        intercept) 1)))
        slope))))

;; Static linreg (for L3/L4 showcase)
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

;; HMM (shared cross-system spec, matching scripts/exp4_genjl_benchmarks.jl and
;; scripts/exp4_genmlx_is1000.cljs: T=50, 2 states, sticky A=[[0.9,0.1],[0.1,0.9]],
;; init [0.5,0.5], emission y_t | z_t=k ~ N(mu_k, 1) with mu=[-2,2]; same ys data
;; as Gen.jl's HMM_YS, so the hmm rows here are matched-spec vs genjl.json)
(def hmm-T 50)
(def hmm-init-logits (mx/log (mx/array #js [0.5 0.5])))
(def hmm-transition-logits
  (mx/log (mx/array #js [#js [0.9 0.1] #js [0.1 0.9]])))
(def hmm-emission-means (mx/array #js [-2.0 2.0]))
(def hmm-sigma-arr (mx/scalar 1.0))

;; Flat unrolled HMM for vectorized IS — no mx/item, shapes flow through
;; (same model as bench/hmm.cljs's hmm-vec-model)
(def hmm-vec-model
  (dyn/auto-key
    (gen [T-steps]
      (loop [t 0, z nil]
        (if (>= t T-steps)
          z
          (let [logits (if (nil? z)
                         hmm-init-logits
                         (mx/take-idx hmm-transition-logits z 0))
                z-new (trace (keyword (str "z" t)) (dist/categorical logits))
                mu (mx/take-idx hmm-emission-means z-new)
                _ (trace (keyword (str "y" t)) (dist/gaussian mu hmm-sigma-arr))]
            (recur (inc t) z-new)))))))

;; Per-step HMM kernel for batched SMC (bootstrap particle filter via
;; smc/batched-smc-unfold — same kernel as bench/hmm.cljs's hmm-kernel-vec)
(def hmm-kernel-vec
  (gen [t z-prev]
    (let [logits (if (nil? z-prev)
                   hmm-init-logits
                   (mx/take-idx hmm-transition-logits z-prev 0))
          z (trace :z (dist/categorical logits))
          mu (mx/take-idx hmm-emission-means z)
          _ (trace :y (dist/gaussian mu hmm-sigma-arr))]
      z)))

;; GMM (scalar -- uses mx/item, not VIS-compatible)
(def gmm-model
  (dyn/auto-key
    (gen [data]
      (let [mus [-4.0 0.0 4.0]
            sigma 1.0
            K 3]
        (doseq [[i y] (map-indexed vector data)]
          (let [z (trace (keyword (str "z" i))
                         (dist/categorical (mx/array (repeat K (/ 1.0 K)))))
                z-val (mx/item z)
                mu (nth mus z-val)]
            (trace (keyword (str "y" i))
                   (dist/gaussian (mx/scalar mu) sigma))))))))

;; GMM vectorized -- no mx/item, shapes flow through for VIS
(def gmm-log-weights (mx/array [(js/Math.log (/ 1.0 3.0))
                                 (js/Math.log (/ 1.0 3.0))
                                 (js/Math.log (/ 1.0 3.0))]))
(def gmm-means-arr (mx/array [-4.0 0.0 4.0]))

(def gmm-vec-model
  (dyn/auto-key
    (gen [data]
      (let [sigma 1.0
            K 3]
        (doseq [[i y] (map-indexed vector data)]
          (let [z (trace (keyword (str "z" i))
                         (dist/categorical gmm-log-weights))
                mu (mx/take-idx gmm-means-arr z)]
            (trace (keyword (str "y" i))
                   (dist/gaussian mu (mx/scalar sigma)))))))))

;; ---------------------------------------------------------------------------
;; Data
;; ---------------------------------------------------------------------------

(def linreg-xs [1.0 2.0 3.0 4.0 5.0])
(def linreg-ys [2.3 4.7 6.1 8.9 10.2])
(def linreg-obs
  (reduce (fn [cm [j y]]
            (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector linreg-ys)))

(def static-args [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)
                  (mx/scalar 4.0) (mx/scalar 5.0)])
(def static-obs
  (-> cm/EMPTY
      (cm/set-choice [:y1] (mx/scalar 2.3))
      (cm/set-choice [:y2] (mx/scalar 4.7))
      (cm/set-choice [:y3] (mx/scalar 6.1))
      (cm/set-choice [:y4] (mx/scalar 8.9))
      (cm/set-choice [:y5] (mx/scalar 10.2))))

(def gmm-data [-3.8 -4.2 0.1 -0.3 3.9 4.1 0.2 3.7])
(def gmm-obs
  (reduce (fn [cm [i y]]
            (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector gmm-data)))

;; HMM observations — identical to Gen.jl's HMM_YS (exp4 shared spec)
(def hmm-ys [3.069579601287842 1.29532390832901 2.305454909801483
             -3.5862162113189697 -3.4873690605163574 -2.0841751396656036
             2.0168902575969696 0.928423285484314 3.504786491394043
             1.6985026597976685 4.161322116851807 -1.769323617219925
             -1.535953313112259 -0.8051328659057617 -3.148682117462158
             -1.7772709429264069 -2.746698319911957 -1.2596661448478699
             -1.9912031944841146 -4.088978290557861 -2.291228175163269
             -1.472139060497284 -1.9415821731090546 -2.348399966955185
             -2.660956621170044 -2.9283525347709656 -2.5985397696495056
             -1.1935933232307434 -3.5146095752716064 -1.9819734804332256
             -2.570974826812744 -2.3379217088222504 -2.9130019545555115
             -1.7724827527999878 0.14748835563659668 -1.9908120324835181
             2.650322377681732 2.6598562598228455 1.8102920204401016
             2.1211230158805847 2.97222638130188 5.103949785232544
             1.4137526154518127 3.213983416557312 2.1396096646785736
             1.7250688076019287 2.607529580593109 1.4608569145202637
             2.41530442237854 2.0015695926267654])

;; Flat observations for vectorized IS (flat keys :y0, :y1, ...)
(def hmm-vec-obs
  (reduce (fn [cm t]
            (cm/set-choice cm [(keyword (str "y" t))] (mx/scalar (nth hmm-ys t))))
          cm/EMPTY (range hmm-T)))

;; Per-step observations for batched SMC (kernel-level :y per timestep)
(def hmm-smc-obs-seq
  (mapv (fn [t] (cm/choicemap :y (mx/scalar (nth hmm-ys t)))) (range hmm-T)))

(println "\n=== Cross-System Comparison (GenMLX) ===")

;; ---------------------------------------------------------------------------
;; 1. LinReg IS (200 particles) -- sequential
;; ---------------------------------------------------------------------------

(mx/clear-cache!)
(def is-linreg-seq
  (benchmark "IS-linreg-seq-200"
    (fn []
      (let [r (is/importance-sampling {:samples 200}
                                       linreg-model [linreg-xs] linreg-obs)]
        (mx/eval! (:log-ml-estimate r))))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; ---------------------------------------------------------------------------
;; 2. LinReg VIS (1000 particles) -- vectorized
;; ---------------------------------------------------------------------------

(mx/clear-cache!)
(def vis-linreg
  (benchmark "VIS-linreg-1000"
    (fn []
      (let [{:keys [vtrace log-ml-estimate]}
            (is/vectorized-importance-sampling {:samples 1000}
                                                linreg-model [linreg-xs] linreg-obs)]
        (mx/eval! log-ml-estimate)))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; ---------------------------------------------------------------------------
;; 3. LinReg VIS (10000 particles) -- show sublinear scaling
;; ---------------------------------------------------------------------------

(mx/clear-cache!)
(def vis-linreg-10k
  (benchmark "VIS-linreg-10000"
    (fn []
      (let [{:keys [vtrace log-ml-estimate]}
            (is/vectorized-importance-sampling {:samples 10000}
                                                linreg-model [linreg-xs] linreg-obs)]
        (mx/eval! log-ml-estimate)))
    :warmup-n 1 :outer-n 3 :inner-n 1))
;; ---------------------------------------------------------------------------
;; 4. LinReg MH (200 steps) -- compiled
;; ---------------------------------------------------------------------------

(mx/clear-cache!)
(def mh-linreg
  (benchmark "MH-linreg-200"
    (fn []
      (mx/clear-cache!)
      (let [samples (mcmc/compiled-mh
                      {:samples 200 :burn 0
                       :addresses [:slope :intercept]
                       :proposal-std 0.5}
                      linreg-model [linreg-xs] linreg-obs)]
        (when (seq samples) (mx/eval! (mx/scalar (count samples))))))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; ---------------------------------------------------------------------------
;; 5. GMM IS (200 particles) -- sequential
;; ---------------------------------------------------------------------------

(mx/clear-cache!)
(def is-gmm-seq
  (benchmark "IS-gmm-seq-200"
    (fn []
      (let [r (is/importance-sampling {:samples 200}
                                       gmm-model [gmm-data] gmm-obs)]
        (mx/eval! (:log-ml-estimate r))))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; ---------------------------------------------------------------------------
;; 6. GMM VIS (1000 particles)
;; ---------------------------------------------------------------------------

(mx/clear-cache!)
(def vis-gmm
  (benchmark "VIS-gmm-1000"
    (fn []
      (let [{:keys [vtrace log-ml-estimate]}
            (is/vectorized-importance-sampling {:samples 1000}
                                                gmm-vec-model [gmm-data] gmm-obs)]
        (mx/eval! log-ml-estimate)))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; ---------------------------------------------------------------------------
;; 7. HMM VIS (1000 particles) -- matches Gen.jl hmm IS(1000)
;; ---------------------------------------------------------------------------

(mx/clear-cache!)
(def vis-hmm
  (benchmark "VIS-hmm-1000"
    (fn []
      (let [{:keys [vtrace log-ml-estimate]}
            (is/vectorized-importance-sampling {:samples 1000}
                                                hmm-vec-model [hmm-T] hmm-vec-obs)]
        (mx/eval! log-ml-estimate)))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; ---------------------------------------------------------------------------
;; 8. HMM SMC (100 particles, T=50) -- batched bootstrap PF
;;    (Gen.jl's counterpart resamples adaptively at ESS<N/2; batched-smc-unfold
;;     resamples every step -- timing-comparable, not step-for-step identical)
;; ---------------------------------------------------------------------------

(mx/clear-cache!)
(def smc-hmm
  (safe-benchmark "SMC-hmm-100"
    (fn []
      (let [result (smc/batched-smc-unfold {:particles 100}
                                            hmm-kernel-vec nil hmm-smc-obs-seq)]
        (mx/eval! (:log-ml result))))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; ---------------------------------------------------------------------------
;; 9. L3 exact (static linreg) -- GenMLX only
;; ---------------------------------------------------------------------------

(mx/clear-cache!)
(def l3-exact
  (benchmark "L3-exact-linreg"
    (fn []
      (let [{:keys [trace weight]} (p/generate static-linreg static-args static-obs)]
        (mx/eval! weight)))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; ---------------------------------------------------------------------------
;; 10. L4 fit (static linreg) -- GenMLX only
;; ---------------------------------------------------------------------------

(mx/clear-cache!)
(def l4-fit
  (benchmark "L4-fit-linreg"
    (fn []
      (let [result (fit/fit static-linreg static-args static-obs)]
        (mx/eval! (mx/scalar (:elapsed-ms result)))))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; ---------------------------------------------------------------------------
;; 11. HMC (100 samples) -- GenMLX
;; ---------------------------------------------------------------------------

(mx/clear-cache!)
(def hmc-linreg
  (safe-benchmark "HMC-linreg-100"
    (fn []
      (mx/clear-cache!)
      (let [samples (mcmc/hmc {:samples 100 :burn 50
                                :leapfrog-steps 10
                                :addresses [:slope :intercept]
                                :adapt-step-size true}
                               linreg-model [linreg-xs] linreg-obs)]
        (when (seq samples)
          (mx/eval! (mx/scalar (count samples))))))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; ---------------------------------------------------------------------------
;; 12. NUTS (100 samples) -- GenMLX
;; ---------------------------------------------------------------------------

(mx/clear-cache!)
(def nuts-linreg
  (safe-benchmark "NUTS-linreg-100"
    (fn []
      (mx/clear-cache!)
      (let [samples (mcmc/nuts {:samples 100 :burn 50
                                 :addresses [:slope :intercept]
                                 :adapt-step-size true
                                 :adapt-metric true}
                                linreg-model [linreg-xs] linreg-obs)]
        (when (seq samples)
          (mx/eval! (mx/scalar (count samples))))))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; ---------------------------------------------------------------------------
;; 13. Fused MH (static linreg, cached) -- GenMLX only
;; ---------------------------------------------------------------------------

;; Pre-compile fused chain (one-time cost, not benchmarked)
(println "\n  [fused-mh] compiling chain (one-time)...")
(def fused-warmup
  (mcmc/fused-mh {:samples 10 :burn 10 :addresses [:slope :intercept]
                   :proposal-std 0.5 :device :cpu}
                 static-linreg static-args static-obs))

(mx/clear-cache!)
(def fused-mh-linreg
  (benchmark "fused-MH-linreg-200"
    (fn []
      (let [result (mcmc/fused-mh
                     {:samples 200 :burn 0 :addresses [:slope :intercept]
                      :proposal-std 0.5 :device :cpu
                      :chain-fn (:chain-fn fused-warmup)}
                     static-linreg static-args static-obs)]
        (mx/eval! (mx/scalar (first (mx/shape (:samples result)))))))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; ---------------------------------------------------------------------------
;; 14. Fused Vectorized MH (N=8 chains, static linreg) -- GenMLX only
;; ---------------------------------------------------------------------------

(println "\n  [fused-vec-mh] compiling vectorized chain (one-time)...")
(def fused-vec-warmup
  (mcmc/fused-vectorized-mh
    {:samples 10 :burn 10 :n-chains 8 :addresses [:slope :intercept]
     :proposal-std 0.5 :device :cpu}
    static-linreg static-args static-obs))

(mx/clear-cache!)
(def fused-vec-mh-linreg
  (benchmark "fused-vec-MH-linreg-N8-200"
    (fn []
      (let [result (mcmc/fused-vectorized-mh
                     {:samples 200 :burn 0 :n-chains 8
                      :addresses [:slope :intercept]
                      :proposal-std 0.5 :device :cpu
                      :chain-fn (:chain-fn fused-vec-warmup)}
                     static-linreg static-args static-obs)]
        (mx/eval! (mx/scalar (first (mx/shape (:samples result)))))))
    :warmup-n 1 :outer-n 3 :inner-n 1))

;; ---------------------------------------------------------------------------
;; Collect & write results
;; ---------------------------------------------------------------------------

(println "\n\n========================================")
(println "    CROSS-SYSTEM COMPARISON (GenMLX)")
(println "========================================\n")

(def all-results
  [{:config "IS-linreg-seq" :particles 200 :timing is-linreg-seq}
   {:config "VIS-linreg" :particles 1000 :timing vis-linreg}
   {:config "VIS-linreg" :particles 10000 :timing vis-linreg-10k}
   {:config "MH-linreg" :samples 200 :timing mh-linreg}
   {:config "fused-MH-linreg" :samples 200 :timing fused-mh-linreg}
   {:config "fused-vec-MH-linreg-N8" :samples 200 :timing fused-vec-mh-linreg}
   {:config "IS-gmm-seq" :particles 200 :timing is-gmm-seq}
   {:config "VIS-gmm" :particles 1000 :timing vis-gmm}
   {:config "VIS-hmm" :particles 1000 :timing vis-hmm}
   {:config "SMC-hmm" :particles 100 :timing smc-hmm}
   {:config "L3-exact" :timing l3-exact}
   {:config "L4-fit" :timing l4-fit}
   {:config "HMC-linreg" :samples 100 :timing hmc-linreg}
   {:config "NUTS-linreg" :samples 100 :timing nuts-linreg}])

(println "| Config | N | Time (ms) |")
(println "|--------|---|-----------|")
(doseq [{:keys [config particles samples timing]} all-results]
  (if (:skipped timing)
    (println (str "| " config " | " (or particles samples "---") " | SKIPPED |"))
    (println (str "| " config " | " (or particles samples "---")
                  " | " (.toFixed (:mean-ms timing) 3) " +/- "
                  (.toFixed (:std-ms timing) 3) " |"))))

(write-json "data.json"
  {:experiment "cross_system"
   :system "GenMLX"
   :timestamp (.toISOString (js/Date.))
   ;; Honest provenance: detect the host instead of hardcoding macOS/Metal
   ;; (this bench also runs on the Thor CUDA box).
   :hardware (let [os-mod (js/require "os")]
               {:platform (.-platform js/process)
                :chip (.arch os-mod)
                :gpu (if (mx/metal-is-available?) "Metal" "CUDA")})
   :runtime "bun+nbb (ClojureScript interpreter)"
   :results
   (mapv (fn [{:keys [config particles samples timing]}]
           {:config config
            :particles (or particles samples)
            :mean_ms (:mean-ms timing)
            :std_ms (:std-ms timing)
            :min_ms (:min-ms timing)
            :raw_times (:raw timing)})
         all-results)})

(println "\nCross-system comparison (GenMLX) complete.")
