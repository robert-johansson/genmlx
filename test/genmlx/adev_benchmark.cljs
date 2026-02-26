(ns genmlx.adev-benchmark
  "Benchmark: sequential adev-gradient vs vectorized vadev-gradient.
   Gate: ≥10x speedup at N=100."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.handler :as h]
            [genmlx.inference.adev :as adev])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn bench
  "Run f with warmup runs then timed runs, return median ms."
  [f {:keys [warmup runs] :or {warmup 2 runs 5}}]
  (dotimes [_ warmup] (f))
  (let [times (mapv (fn [_]
                      (let [start (js/Date.now)]
                        (f)
                        (- (js/Date.now) start)))
                    (range runs))]
    (nth (sort times) (quot runs 2))))

(println "\n=== ADEV Benchmark: Sequential vs Vectorized ===\n")

;; ---------------------------------------------------------------------------
;; Model definitions
;; ---------------------------------------------------------------------------

(def model-5g
  "5 gaussian trace sites (pure reparam)."
  (gen []
    (let [mu (dyn/param :mu 0.0)]
      (dyn/trace :x0 (dist/gaussian mu (mx/scalar 1.0)))
      (dyn/trace :x1 (dist/gaussian mu (mx/scalar 1.0)))
      (dyn/trace :x2 (dist/gaussian mu (mx/scalar 1.0)))
      (dyn/trace :x3 (dist/gaussian mu (mx/scalar 1.0)))
      (dyn/trace :x4 (dist/gaussian mu (mx/scalar 1.0))))))

(def model-mixed
  "3 gaussian + 2 bernoulli (reparam + REINFORCE)."
  (gen []
    (let [mu (dyn/param :mu 0.0)]
      (dyn/trace :x0 (dist/gaussian mu (mx/scalar 1.0)))
      (dyn/trace :x1 (dist/gaussian mu (mx/scalar 1.0)))
      (dyn/trace :x2 (dist/gaussian mu (mx/scalar 1.0)))
      (dyn/trace :b0 (dist/bernoulli 0.5))
      (dyn/trace :b1 (dist/bernoulli 0.5)))))

(def model-20g
  "20 gaussian trace sites (stress test)."
  (gen []
    (let [mu (dyn/param :mu 0.0)]
      (doseq [i (range 20)]
        (dyn/trace (keyword (str "x" i)) (dist/gaussian mu (mx/scalar 1.0))))
      mu)))

(def cost-fn-scalar (fn [trace] (mx/square (:retval trace))))
(def cost-fn-batch  (fn [result] (mx/square (:retval result))))
(def param-names [:mu])
(def params (mx/array [0.0]))

;; ---------------------------------------------------------------------------
;; Benchmark runner
;; ---------------------------------------------------------------------------

(defn run-bench [label model]
  (println (str "\n-- " label " (N=100) --"))
  (let [n 100
        seq-ms (bench
                 (fn []
                   (let [{:keys [loss grad]}
                         (adev/adev-gradient {:n-samples n}
                                             model [] cost-fn-scalar
                                             param-names params)]
                     (mx/eval! loss grad)))
                 {:warmup 1 :runs 3})
        vec-ms (bench
                 (fn []
                   (let [{:keys [loss grad]}
                         (adev/vadev-gradient {:n-samples n}
                                              model [] cost-fn-batch
                                              param-names params)]
                     (mx/eval! loss grad)))
                 {:warmup 1 :runs 3})
        speedup (if (pos? vec-ms) (/ seq-ms vec-ms) ##Inf)]
    (println (str "  Sequential (adev-gradient):  " seq-ms "ms"))
    (println (str "  Vectorized (vadev-gradient): " vec-ms "ms"))
    (println (str "  Speedup: " (.toFixed speedup 1) "x"))
    (println (str "  GATE (≥10x): " (if (>= speedup 10) "PASS" "CHECK")))
    speedup))

(let [s1 (run-bench "5-gaussian (pure reparam)" model-5g)
      s2 (run-bench "mixed (3 gauss + 2 bernoulli)" model-mixed)
      s3 (run-bench "20-site gaussian (stress)" model-20g)]
  (println "\n-- Summary --")
  (println (str "  5-gaussian: " (.toFixed s1 1) "x"))
  (println (str "  mixed:      " (.toFixed s2 1) "x"))
  (println (str "  20-site:    " (.toFixed s3 1) "x"))
  (println (str "  ALL ≥10x:   " (if (and (>= s1 10) (>= s2 10) (>= s3 10)) "PASS" "CHECK"))))

(println "\n=== ADEV Benchmark Complete ===")
