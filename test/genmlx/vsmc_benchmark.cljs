(ns genmlx.vsmc-benchmark
  "Benchmark: Sequential SMC vs Vectorized SMC."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.smc :as smc]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

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

(println "\n=== Vectorized SMC Benchmark ===")

;; Time-series model: mu ~ N(0,10), y_t ~ N(mu, 1) at each timestep
(def ts-model
  (gen [t]
    (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
      (dyn/trace :y (dist/gaussian mu 1))
      mu)))

;; ---------------------------------------------------------------------------
;; Benchmark 1: Sequential SMC vs Vectorized SMC (50 particles, 5 steps)
;; ---------------------------------------------------------------------------

(println "\n-- 1. Sequential SMC vs Vectorized SMC (50 particles, 5 steps) --")

(let [n-particles 50
      obs-seq (mapv (fn [y] (cm/choicemap :y (mx/scalar y)))
                    [3.0 3.5 2.8 3.2 3.1])
      key (rng/fresh-key)
      seq-ms (bench "Sequential SMC"
               (fn [] (smc/smc {:particles n-particles :ess-threshold 0.5}
                               ts-model [0] obs-seq))
               {:warmup 1 :runs 3})
      _ (.clearCache mx/core)  ;; free buffers between measurements
      vec-ms (bench "Vectorized SMC"
               (fn [] (smc/vsmc {:particles n-particles :ess-threshold 0.5
                                 :key key}
                                ts-model [0] obs-seq))
               {:warmup 1 :runs 3})
      speedup (if (pos? vec-ms) (/ seq-ms vec-ms) ##Inf)]
  (println (str "  Speedup: " (.toFixed speedup 1) "x")))

;; ---------------------------------------------------------------------------
;; Benchmark 2: Vectorized SMC with rejuvenation (50 particles, 3 steps, 2 MH)
;; ---------------------------------------------------------------------------

(.clearCache mx/core)

(println "\n-- 2. Vectorized SMC with rejuvenation (50 particles, 3 steps, 2 MH) --")

(let [n-particles 50
      obs-seq (mapv (fn [y] (cm/choicemap :y (mx/scalar y)))
                    [3.0 3.5 2.8])
      key (rng/fresh-key)
      result (smc/vsmc {:particles n-particles :ess-threshold 0.5
                         :rejuvenation-steps 2
                         :rejuvenation-selection (sel/select :mu)
                         :key key
                         :callback (fn [{:keys [step ess resampled?]}]
                                     (println (str "  step=" step
                                                   " ess=" (when ess (.toFixed ess 1))
                                                   (when resampled? " (resampled)"))))}
                        ts-model [0] obs-seq)
      log-ml (:log-ml-estimate result)]
  (println (str "  log-ML estimate: " (.toFixed (mx/item log-ml) 2)))
  (println "  OK - vsmc with rejuvenation completed"))

(println "\nAll vsmc benchmarks complete.")
