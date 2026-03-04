(ns genmlx.exp4-genmlx-is1000
  "Experiment 4: GenMLX IS(1000) benchmarks for system comparison.

   Runs IS with N=1000 particles on all 3 canonical models (LinReg, HMM, GMM)
   to produce comparable timings against Gen.jl and GenJAX IS(1000).

   All three models run vectorized IS (shape-based batching).
   Sequential p/generate loops are also included for comparison.

   Protocol: 5 warmup, 20 timed runs.
   Output: results/exp4_system_comparison/genmlx_is1000.json

   Usage: bun run --bun nbb scripts/exp4_genmlx_is1000.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.importance :as is]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

(def fs (js/require "fs"))
(def path-mod (js/require "path"))
(def results-dir
  (.resolve path-mod (js/process.cwd) "results/exp4_system_comparison"))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn perf-now [] (js/performance.now))

(defn write-json [filename data]
  (ensure-dir results-dir)
  (let [fpath (.resolve path-mod results-dir filename)]
    (.writeFileSync fs fpath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "Wrote: " fpath))))

;; ---------------------------------------------------------------------------
;; Data (same as exp3)
;; ---------------------------------------------------------------------------

;; LinReg
(def sigma-prior 2.0)
(def sigma-obs 1.0)

(def linreg-xs [-2.5 -2.236842105263158 -1.973684210526316 -1.7105263157894737
                -1.4473684210526316 -1.1842105263157896 -0.9210526315789473
                -0.6578947368421053 -0.3947368421052633 -0.13157894736842124
                0.1315789473684208 0.3947368421052633 0.6578947368421053
                0.9210526315789473 1.1842105263157894 1.4473684210526319
                1.7105263157894735 1.973684210526316 2.2368421052631575 2.5])

(def linreg-ys [-4.645086392760277 -2.232466120468943 -4.743588723634419
                -1.3603573410134566 -3.226216689536446 -3.294092987713061
                -1.7810833391390348 -1.0100273317412327 0.3002335711529378
                -0.06383435977132734 0.1810731197658333 1.4528234914729472
                2.0677273100928257 2.639107370062878 2.240536426243029
                3.3727551091854515 4.502427813253904 4.419489033912358
                7.114258427368966 5.38109839707613])

;; HMM
(def T 50)
(def hmm-sigma 1.0)
(def init-logits (mx/log (mx/array #js [0.5 0.5])))
(def transition-logits (mx/log (mx/array #js [#js [0.9 0.1] #js [0.1 0.9]])))
(def emission-means (mx/array #js [-2.0 2.0]))

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

;; GMM
(def gmm-K 3)
(def gmm-N 8)
(def gmm-means (mx/array #js [-4.0 0.0 4.0]))
(def gmm-sigma 1.0)
(def gmm-log-weights (mx/log (mx/array #js [0.333 0.333 0.333])))

(def gmm-ys [0.8350437879562378 4.188530474901199 -2.888857841491699
             -4.104988642036915 5.638246417045593 0.10598962754011154
             -2.8335559368133545 -4.310329109430313])

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

;; LinReg model (same as paper_bench_linreg.cljs)
(def linreg-model
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 sigma-prior))
          intercept (trace :intercept (dist/gaussian 0 sigma-prior))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept)
                              sigma-obs)))
      slope)))

(def linreg-observations
  (reduce (fn [cm [j y]]
            (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector linreg-ys)))

;; HMM kernel + unfold (same as paper_bench_hmm.cljs)
(def hmm-kernel
  (gen [t z-prev]
    (let [logits (if (nil? z-prev)
                   init-logits
                   (mx/take-idx transition-logits (mx/scalar (int z-prev) mx/int32) 0))
          z (trace :z (dist/categorical logits))
          _ (mx/eval! z)
          z-val (mx/item z)
          mu (mx/take-idx emission-means (mx/scalar (int z-val) mx/int32))
          _ (trace :y (dist/gaussian mu (mx/scalar hmm-sigma)))]
      z-val)))

(def hmm-unfold (comb/unfold-combinator (dyn/auto-key hmm-kernel)))

(def hmm-observations
  (reduce (fn [cm t]
            (cm/set-choice cm [t :y] (mx/scalar (nth hmm-ys t))))
          cm/EMPTY (range T)))

;; GMM model (same as paper_bench_gmm.cljs)
(def gmm-model
  (gen [ys]
    (let [n (count ys)]
      (doseq [i (range n)]
        (let [z (trace (keyword (str "z" i)) (dist/categorical gmm-log-weights))
              _ (mx/eval! z)
              z-val (mx/item z)
              mu (mx/take-idx gmm-means (mx/scalar (int z-val) mx/int32))
              _ (trace (keyword (str "y" i)) (dist/gaussian mu (mx/scalar gmm-sigma)))])))))

(def gmm-observations
  (reduce (fn [cm i]
            (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar (nth gmm-ys i))))
          cm/EMPTY (range gmm-N)))

;; ---------------------------------------------------------------------------
;; Vectorized models (no mx/eval!, no mx/item — shapes flow through)
;; ---------------------------------------------------------------------------

(def gmm-sigma-arr (mx/scalar gmm-sigma))
(def hmm-sigma-arr (mx/scalar hmm-sigma))

;; GMM vectorized: logits are [K] (global), z is [N], mu is [N]
(def gmm-vec-model
  (gen [ys]
    (let [n (count ys)]
      (doseq [i (range n)]
        (let [z (trace (keyword (str "z" i)) (dist/categorical gmm-log-weights))
              mu (mx/take-idx gmm-means z)
              _ (trace (keyword (str "y" i)) (dist/gaussian mu gmm-sigma-arr))])))))

;; HMM flat vectorized: unrolled loop with flat address keys
;; t=0: logits [K] → z [N]; t>0: logits [N,K] → z [N]
(def hmm-vec-model
  (gen [T-steps]
    (loop [t 0, z nil]
      (if (>= t T-steps)
        z
        (let [logits (if (nil? z)
                       init-logits
                       (mx/take-idx transition-logits z 0))
              z-new (trace (keyword (str "z" t)) (dist/categorical logits))
              mu (mx/take-idx emission-means z-new)
              _ (trace (keyword (str "y" t)) (dist/gaussian mu hmm-sigma-arr))]
          (recur (inc t) z-new))))))

(def hmm-vec-observations
  (reduce (fn [cm t]
            (cm/set-choice cm [(keyword (str "y" t))] (mx/scalar (nth hmm-ys t))))
          cm/EMPTY (range T)))

;; ---------------------------------------------------------------------------
;; Benchmark helpers
;; ---------------------------------------------------------------------------

(println "\n=== Experiment 4: GenMLX IS(1000) System Comparison ===\n")

;; Verify models
(println "-- Verifying models --")
(let [tr (p/simulate (dyn/auto-key linreg-model) [linreg-xs])
      score (mx/realize (:score tr))]
  (println (str "  LinReg simulate score: " (.toFixed score 2))))

(let [r (p/generate (dyn/auto-key hmm-unfold) [T nil] hmm-observations)
      w (mx/realize (:weight r))]
  (println (str "  HMM generate weight: " (.toFixed w 2))))

(let [r (p/generate (dyn/auto-key gmm-model) [gmm-ys] gmm-observations)
      w (mx/realize (:weight r))]
  (println (str "  GMM generate weight: " (.toFixed w 2))))

;; Verify vectorized models
(println "-- Verifying vectorized models --")
(let [r (is/vectorized-importance-sampling
          {:samples 10 :key (rng/fresh-key 42)}
          gmm-vec-model [gmm-ys] gmm-observations)
      lml (mx/realize (:log-ml-estimate r))]
  (println (str "  GMM vec IS(10) log-ML: " (.toFixed lml 2))))

(let [r (is/vectorized-importance-sampling
          {:samples 10 :key (rng/fresh-key 43)}
          hmm-vec-model [T] hmm-vec-observations)
      lml (mx/realize (:log-ml-estimate r))]
  (println (str "  HMM vec IS(10) log-ML: " (.toFixed lml 2))))

(mx/clear-cache!)
(mx/force-gc!)

;; ---------------------------------------------------------------------------
;; LinReg IS(1000) — Vectorized
;; ---------------------------------------------------------------------------

(println "\n-- LinReg Vectorized IS (N=1000), 5 warmup + 20 runs --")

;; Warmup
(dotimes [_ 3]
  (is/vectorized-importance-sampling
    {:samples 1000 :key (rng/fresh-key (rand-int 10000))}
    linreg-model [linreg-xs] linreg-observations)
  (mx/clear-cache!)
  (mx/force-gc!))

(def linreg-times
  (vec (for [i (range 20)]
         (let [start (perf-now)
               _ (is/vectorized-importance-sampling
                   {:samples 1000 :key (rng/fresh-key (+ 100 i))}
                   linreg-model [linreg-xs] linreg-observations)
               elapsed (- (perf-now) start)]
           (mx/clear-cache!)
           (mx/force-gc!)
           elapsed))))

(let [mean-t (/ (reduce + linreg-times) (count linreg-times))
      sorted (sort linreg-times)
      min-t (first sorted)]
  (println (str "  Mean: " (.toFixed mean-t 2) " ms"))
  (println (str "  Min:  " (.toFixed min-t 2) " ms")))

;; ---------------------------------------------------------------------------
;; HMM IS(1000) — Sequential p/generate loop
;; ---------------------------------------------------------------------------

(println "\n-- HMM IS (N=1000), 5 warmup + 20 runs --")

(defn run-hmm-is [n-particles seed]
  (loop [i 0, rk (rng/ensure-key (rng/fresh-key seed)), acc (transient [])]
    (if (>= i n-particles)
      (persistent! acc)
      (let [[ki next-rk] (rng/split rk)
            w (mx/tidy
                #(let [r (p/generate (dyn/with-key hmm-unfold ki)
                                     [T nil] hmm-observations)]
                   (mx/realize (:weight r))))]
        (when (zero? (mod i 50))
          (mx/clear-cache!)
          (mx/force-gc!))
        (recur (inc i) next-rk (conj! acc w))))))

;; Warmup
(dotimes [i 3]
  (run-hmm-is 1000 (+ 200 i))
  (mx/clear-cache!)
  (mx/force-gc!))

(def hmm-times
  (vec (for [i (range 20)]
         (let [start (perf-now)
               _ (run-hmm-is 1000 (+ 300 i))
               elapsed (- (perf-now) start)]
           (mx/clear-cache!)
           (mx/force-gc!)
           elapsed))))

(let [mean-t (/ (reduce + hmm-times) (count hmm-times))
      min-t (apply min hmm-times)]
  (println (str "  Mean: " (.toFixed mean-t 2) " ms"))
  (println (str "  Min:  " (.toFixed min-t 2) " ms")))

;; ---------------------------------------------------------------------------
;; GMM IS(1000) — Sequential p/generate loop
;; ---------------------------------------------------------------------------

(println "\n-- GMM IS (N=1000), 5 warmup + 20 runs --")

(defn run-gmm-is [n-particles seed]
  (loop [i 0, rk (rng/ensure-key (rng/fresh-key seed)), acc (transient [])]
    (if (>= i n-particles)
      (persistent! acc)
      (let [[ki next-rk] (rng/split rk)
            w (mx/tidy
                #(let [r (p/generate (dyn/with-key gmm-model ki)
                                     [gmm-ys] gmm-observations)]
                   (mx/realize (:weight r))))]
        (when (zero? (mod i 200))
          (mx/clear-cache!)
          (mx/force-gc!))
        (recur (inc i) next-rk (conj! acc w))))))

;; Warmup
(dotimes [i 3]
  (run-gmm-is 1000 (+ 400 i))
  (mx/clear-cache!)
  (mx/force-gc!))

(def gmm-times
  (vec (for [i (range 20)]
         (let [start (perf-now)
               _ (run-gmm-is 1000 (+ 500 i))
               elapsed (- (perf-now) start)]
           (mx/clear-cache!)
           (mx/force-gc!)
           elapsed))))

(let [mean-t (/ (reduce + gmm-times) (count gmm-times))
      min-t (apply min gmm-times)]
  (println (str "  Mean: " (.toFixed mean-t 2) " ms"))
  (println (str "  Min:  " (.toFixed min-t 2) " ms")))

;; ---------------------------------------------------------------------------
;; GMM IS(1000) — Vectorized
;; ---------------------------------------------------------------------------

(println "\n-- GMM Vectorized IS (N=1000), 5 warmup + 20 runs --")

(dotimes [_ 5]
  (is/vectorized-importance-sampling
    {:samples 1000 :key (rng/fresh-key (rand-int 10000))}
    gmm-vec-model [gmm-ys] gmm-observations)
  (mx/clear-cache!)
  (mx/force-gc!))

(def gmm-vec-times
  (vec (for [i (range 20)]
         (let [start (perf-now)
               _ (is/vectorized-importance-sampling
                   {:samples 1000 :key (rng/fresh-key (+ 600 i))}
                   gmm-vec-model [gmm-ys] gmm-observations)
               elapsed (- (perf-now) start)]
           (mx/clear-cache!)
           (mx/force-gc!)
           elapsed))))

(let [mean-t (/ (reduce + gmm-vec-times) (count gmm-vec-times))
      min-t (apply min gmm-vec-times)]
  (println (str "  Mean: " (.toFixed mean-t 2) " ms"))
  (println (str "  Min:  " (.toFixed min-t 2) " ms")))

;; ---------------------------------------------------------------------------
;; HMM IS(1000) — Vectorized
;; ---------------------------------------------------------------------------

(println "\n-- HMM Vectorized IS (N=1000), 5 warmup + 20 runs --")

(dotimes [_ 5]
  (is/vectorized-importance-sampling
    {:samples 1000 :key (rng/fresh-key (rand-int 10000))}
    hmm-vec-model [T] hmm-vec-observations)
  (mx/clear-cache!)
  (mx/force-gc!))

(def hmm-vec-times
  (vec (for [i (range 20)]
         (let [start (perf-now)
               _ (is/vectorized-importance-sampling
                   {:samples 1000 :key (rng/fresh-key (+ 700 i))}
                   hmm-vec-model [T] hmm-vec-observations)
               elapsed (- (perf-now) start)]
           (mx/clear-cache!)
           (mx/force-gc!)
           elapsed))))

(let [mean-t (/ (reduce + hmm-vec-times) (count hmm-vec-times))
      min-t (apply min hmm-vec-times)]
  (println (str "  Mean: " (.toFixed mean-t 2) " ms"))
  (println (str "  Min:  " (.toFixed min-t 2) " ms")))

;; ---------------------------------------------------------------------------
;; LinReg Vectorized IS (N=10,000)
;; ---------------------------------------------------------------------------

(println "\n-- LinReg Vectorized IS (N=10000), 5 warmup + 20 runs --")

(dotimes [_ 5]
  (is/vectorized-importance-sampling
    {:samples 10000 :key (rng/fresh-key (rand-int 10000))}
    linreg-model [linreg-xs] linreg-observations)
  (mx/clear-cache!)
  (mx/force-gc!))

(def linreg-10k-times
  (vec (for [i (range 20)]
         (let [start (perf-now)
               _ (is/vectorized-importance-sampling
                   {:samples 10000 :key (rng/fresh-key (+ 1000 i))}
                   linreg-model [linreg-xs] linreg-observations)
               elapsed (- (perf-now) start)]
           (mx/clear-cache!)
           (mx/force-gc!)
           elapsed))))

(let [mean-t (/ (reduce + linreg-10k-times) (count linreg-10k-times))
      min-t (apply min linreg-10k-times)]
  (println (str "  Mean: " (.toFixed mean-t 2) " ms"))
  (println (str "  Min:  " (.toFixed min-t 2) " ms")))

;; ---------------------------------------------------------------------------
;; GMM Vectorized IS (N=10,000)
;; ---------------------------------------------------------------------------

(println "\n-- GMM Vectorized IS (N=10000), 5 warmup + 20 runs --")

(dotimes [_ 5]
  (is/vectorized-importance-sampling
    {:samples 10000 :key (rng/fresh-key (rand-int 10000))}
    gmm-vec-model [gmm-ys] gmm-observations)
  (mx/clear-cache!)
  (mx/force-gc!))

(def gmm-10k-times
  (vec (for [i (range 20)]
         (let [start (perf-now)
               _ (is/vectorized-importance-sampling
                   {:samples 10000 :key (rng/fresh-key (+ 1100 i))}
                   gmm-vec-model [gmm-ys] gmm-observations)
               elapsed (- (perf-now) start)]
           (mx/clear-cache!)
           (mx/force-gc!)
           elapsed))))

(let [mean-t (/ (reduce + gmm-10k-times) (count gmm-10k-times))
      min-t (apply min gmm-10k-times)]
  (println (str "  Mean: " (.toFixed mean-t 2) " ms"))
  (println (str "  Min:  " (.toFixed min-t 2) " ms")))

;; ---------------------------------------------------------------------------
;; HMM Vectorized IS (N=10,000)
;; ---------------------------------------------------------------------------

(println "\n-- HMM Vectorized IS (N=10000), 5 warmup + 20 runs --")

(dotimes [_ 5]
  (is/vectorized-importance-sampling
    {:samples 10000 :key (rng/fresh-key (rand-int 10000))}
    hmm-vec-model [T] hmm-vec-observations)
  (mx/clear-cache!)
  (mx/force-gc!))

(def hmm-10k-times
  (vec (for [i (range 20)]
         (let [start (perf-now)
               _ (is/vectorized-importance-sampling
                   {:samples 10000 :key (rng/fresh-key (+ 1200 i))}
                   hmm-vec-model [T] hmm-vec-observations)
               elapsed (- (perf-now) start)]
           (mx/clear-cache!)
           (mx/force-gc!)
           elapsed))))

(let [mean-t (/ (reduce + hmm-10k-times) (count hmm-10k-times))
      min-t (apply min hmm-10k-times)]
  (println (str "  Mean: " (.toFixed mean-t 2) " ms"))
  (println (str "  Min:  " (.toFixed min-t 2) " ms")))

;; ---------------------------------------------------------------------------
;; Write JSON
;; ---------------------------------------------------------------------------

(println "\n-- Writing results --")

(let [mean-fn (fn [xs] (/ (reduce + xs) (count xs)))
      std-fn (fn [xs]
               (let [mu (mean-fn xs)
                     n (count xs)
                     ss (reduce + (map #(* (- % mu) (- % mu)) xs))]
                 (js/Math.sqrt (/ ss (dec n)))))
      output {:system "genmlx"
              :version "0.1.0"
              :hardware "Apple M2"
              :backend "Metal GPU"
              :timing_protocol "5 warmup, 20 runs, performance.now"
              :comparisons
              [{:model "linreg"
                :algorithm "IS"
                :n_particles 1000
                :method "vectorized"
                :time_ms (mean-fn linreg-times)
                :time_ms_std (std-fn linreg-times)
                :time_ms_min (apply min linreg-times)
                :times_ms linreg-times}
               {:model "hmm"
                :algorithm "IS"
                :n_particles 1000
                :method "sequential"
                :time_ms (mean-fn hmm-times)
                :time_ms_std (std-fn hmm-times)
                :time_ms_min (apply min hmm-times)
                :times_ms hmm-times}
               {:model "gmm"
                :algorithm "IS"
                :n_particles 1000
                :method "sequential"
                :time_ms (mean-fn gmm-times)
                :time_ms_std (std-fn gmm-times)
                :time_ms_min (apply min gmm-times)
                :times_ms gmm-times}
               {:model "gmm"
                :algorithm "IS"
                :n_particles 1000
                :method "vectorized"
                :time_ms (mean-fn gmm-vec-times)
                :time_ms_std (std-fn gmm-vec-times)
                :time_ms_min (apply min gmm-vec-times)
                :times_ms gmm-vec-times}
               {:model "hmm"
                :algorithm "IS"
                :n_particles 1000
                :method "vectorized"
                :time_ms (mean-fn hmm-vec-times)
                :time_ms_std (std-fn hmm-vec-times)
                :time_ms_min (apply min hmm-vec-times)
                :times_ms hmm-vec-times}
               {:model "linreg"
                :algorithm "IS"
                :n_particles 10000
                :method "vectorized"
                :time_ms (mean-fn linreg-10k-times)
                :time_ms_std (std-fn linreg-10k-times)
                :time_ms_min (apply min linreg-10k-times)
                :times_ms linreg-10k-times}
               {:model "gmm"
                :algorithm "IS"
                :n_particles 10000
                :method "vectorized"
                :time_ms (mean-fn gmm-10k-times)
                :time_ms_std (std-fn gmm-10k-times)
                :time_ms_min (apply min gmm-10k-times)
                :times_ms gmm-10k-times}
               {:model "hmm"
                :algorithm "IS"
                :n_particles 10000
                :method "vectorized"
                :time_ms (mean-fn hmm-10k-times)
                :time_ms_std (std-fn hmm-10k-times)
                :time_ms_min (apply min hmm-10k-times)
                :times_ms hmm-10k-times}]}]
  (write-json "genmlx_is1000.json" output))

(println "\nDone.")
