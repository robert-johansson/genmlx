(ns genmlx.vectorized-benchmark
  "Benchmark: sequential vs batched inference performance.
   Exit criteria: dist-sample-n >10x for N=100, vgenerate >5x end-to-end."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.vectorized :as vec]
            [genmlx.inference.importance :as is]
            [genmlx.inference.smc :as smc])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn time-ms
  "Execute f and return [result elapsed-ms]."
  [f]
  (let [start (js/Date.now)
        result (f)
        end (js/Date.now)]
    [result (- end start)]))

(println "\n=== Vectorized Inference Benchmarks ===\n")

;; ---------------------------------------------------------------------------
;; Benchmark 1: N x dist-sample vs dist-sample-n
;; ---------------------------------------------------------------------------

(println "-- Benchmark 1: dist-sample vs dist-sample-n (N=100) --")

(let [n 100
      d (dist/gaussian 0 1)
      key (rng/fresh-key)

      ;; Sequential: N separate samples
      [_ seq-ms] (time-ms
                    (fn []
                      (let [keys (rng/split-n key n)]
                        (doseq [k keys]
                          (let [v (dc/dist-sample d k)]
                            (mx/eval! v))))))

      ;; Batched: single dist-sample-n
      [_ batch-ms] (time-ms
                     (fn []
                       (let [v (dc/dist-sample-n d key n)]
                         (mx/eval! v))))

      speedup (if (pos? batch-ms) (/ seq-ms batch-ms) ##Inf)]
  (println (str "  Sequential (100 x dist-sample): " seq-ms "ms"))
  (println (str "  Batched    (dist-sample-n 100):  " batch-ms "ms"))
  (println (str "  Speedup: " (.toFixed speedup 1) "x"))
  (println (str "  EXIT CRITERION (>10x): " (if (> speedup 10) "PASS" "CHECK"))))

;; ---------------------------------------------------------------------------
;; Benchmark 1b: Larger N=1000
;; ---------------------------------------------------------------------------

(println "\n-- Benchmark 1b: dist-sample vs dist-sample-n (N=1000) --")

(let [n 1000
      d (dist/gaussian 0 1)
      key (rng/fresh-key)

      [_ seq-ms] (time-ms
                    (fn []
                      (let [keys (rng/split-n key n)]
                        (doseq [k keys]
                          (let [v (dc/dist-sample d k)]
                            (mx/eval! v))))))

      [_ batch-ms] (time-ms
                     (fn []
                       (let [v (dc/dist-sample-n d key n)]
                         (mx/eval! v))))

      speedup (if (pos? batch-ms) (/ seq-ms batch-ms) ##Inf)]
  (println (str "  Sequential (1000 x dist-sample): " seq-ms "ms"))
  (println (str "  Batched    (dist-sample-n 1000):  " batch-ms "ms"))
  (println (str "  Speedup: " (.toFixed speedup 1) "x")))

;; ---------------------------------------------------------------------------
;; Benchmark 2: N x generate vs vgenerate (5-site model)
;; ---------------------------------------------------------------------------

(println "\n-- Benchmark 2: generate vs vgenerate (5-site model, N=100) --")

(let [model (gen []
              (let [x1 (dyn/trace :x1 (dist/gaussian 0 1))
                    x2 (dyn/trace :x2 (dist/gaussian 0 1))
                    x3 (dyn/trace :x3 (dist/gaussian 0 1))
                    x4 (dyn/trace :x4 (dist/gaussian 0 1))
                    x5 (dyn/trace :x5 (dist/gaussian 0 1))]
                nil))
      n 100
      obs (cm/choicemap :x5 (mx/scalar 2.0))

      ;; Sequential: N separate generates
      [_ seq-ms] (time-ms
                    (fn []
                      (doseq [_ (range n)]
                        (let [{:keys [trace weight]} (p/generate model [] obs)]
                          (mx/eval! (:score trace) weight)))))

      ;; Batched: single vgenerate
      [_ batch-ms] (time-ms
                     (fn []
                       (let [vt (dyn/vgenerate model [] obs n nil)]
                         (mx/eval! (:score vt) (:weight vt)))))

      speedup (if (pos? batch-ms) (/ seq-ms batch-ms) ##Inf)]
  (println (str "  Sequential (100 x generate): " seq-ms "ms"))
  (println (str "  Batched    (vgenerate 100):   " batch-ms "ms"))
  (println (str "  Speedup: " (.toFixed speedup 1) "x"))
  (println (str "  EXIT CRITERION (>5x): " (if (> speedup 5) "PASS" "CHECK"))))

;; ---------------------------------------------------------------------------
;; Benchmark 3: importance-sampling vs vectorized-importance-sampling
;; ---------------------------------------------------------------------------

(println "\n-- Benchmark 3: IS vs vectorized IS (5-site model, N=100) --")

(let [model (gen []
              (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
                (dyn/trace :obs1 (dist/gaussian mu 1))
                (dyn/trace :obs2 (dist/gaussian mu 1))
                (dyn/trace :obs3 (dist/gaussian mu 1))
                mu))
      obs (cm/choicemap :obs1 (mx/scalar 3.0) :obs2 (mx/scalar 3.1) :obs3 (mx/scalar 2.9))
      n 100

      ;; Sequential IS
      [seq-result seq-ms] (time-ms
                             (fn []
                               (let [r (is/importance-sampling
                                         {:samples n} model [] obs)]
                                 (mx/eval! (:log-ml-estimate r))
                                 r)))

      ;; Vectorized IS
      [batch-result batch-ms] (time-ms
                                 (fn []
                                   (let [r (is/vectorized-importance-sampling
                                             {:samples n} model [] obs)]
                                     (mx/eval! (:log-ml-estimate r))
                                     r)))

      speedup (if (pos? batch-ms) (/ seq-ms batch-ms) ##Inf)]
  (println (str "  Sequential IS (100 particles):  " seq-ms "ms"))
  (println (str "  Vectorized IS (100 particles):  " batch-ms "ms"))
  (println (str "  Speedup: " (.toFixed speedup 1) "x"))
  ;; Both should give similar log-ml estimates
  (let [seq-ml (mx/realize (:log-ml-estimate seq-result))
        batch-ml (mx/realize (:log-ml-estimate batch-result))]
    (println (str "  Sequential log-ml: " (.toFixed seq-ml 2)))
    (println (str "  Vectorized log-ml: " (.toFixed batch-ml 2)))))

;; ---------------------------------------------------------------------------
;; Benchmark 4: SMC init — sequential vs vectorized
;; ---------------------------------------------------------------------------

(println "\n-- Benchmark 4: SMC init vs vsmc-init (5-site, N=100) --")

(let [model (gen []
              (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
                (dyn/trace :obs1 (dist/gaussian mu 1))
                (dyn/trace :obs2 (dist/gaussian mu 1))
                mu))
      obs (cm/choicemap :obs1 (mx/scalar 3.0) :obs2 (mx/scalar 3.1))
      n 100

      ;; Sequential SMC init (extracted from smc)
      [_ seq-ms] (time-ms
                    (fn []
                      (let [results (mapv (fn [_] (p/generate model [] obs)) (range n))
                            weights (mapv :weight results)]
                        (mx/eval! (mx/array (mapv mx/realize weights))))))

      ;; Vectorized SMC init
      [_ batch-ms] (time-ms
                     (fn []
                       (let [{:keys [vtrace]} (smc/vsmc-init
                                                model [] obs n nil)]
                         (mx/eval! (:weight vtrace)))))

      speedup (if (pos? batch-ms) (/ seq-ms batch-ms) ##Inf)]
  (println (str "  Sequential SMC init (100 particles): " seq-ms "ms"))
  (println (str "  Vectorized SMC init (100 particles): " batch-ms "ms"))
  (println (str "  Speedup: " (.toFixed speedup 1) "x")))

(println "\nAll benchmarks complete.")
