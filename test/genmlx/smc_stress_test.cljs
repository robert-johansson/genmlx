(ns genmlx.smc-stress-test
  "Stress test for SMC on HMM: verifies P0-4 Metal pipeline leak is resolved.
   Tests batched-smc-unfold and smc-unfold at various particle counts on T=50 HMM."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.inference.smc :as smc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; HMM parameters — transition-logits MUST be [K,K] via reshape (mx/array
;; doesn't support nested JS arrays — they produce NaN)
(def K 2)
(def transition-logits
  (mx/reshape (mx/array (clj->js [(js/Math.log 0.9) (js/Math.log 0.1)
                                   (js/Math.log 0.1) (js/Math.log 0.9)]))
              [K K]))
(def init-logits (mx/array (clj->js [(js/Math.log 0.5) (js/Math.log 0.5)])))
(def emission-means (mx/array (clj->js [-2.0 2.0])))
(def sigma-obs (mx/scalar 1.0))

(def ys-data [3.070 -2.705 -1.695 -3.586 -3.487 -2.084 -1.983 -3.072
              -0.495 -2.301 0.161 -1.769 -1.536 -0.805 0.851 2.223
              1.253 2.740 2.009 -0.089 1.709 2.528 -1.942 -2.348
              -2.661 -2.928 -2.599 -1.194 -3.515 -1.982 -2.571 -2.338
              -2.913 -1.772 0.147 -1.991 -1.350 2.660 1.810 2.121
              2.972 5.104 1.414 3.214 2.140 1.725 2.608 1.461
              2.415 -1.998])
(def T (count ys-data))

(def hmm-kernel
  (gen [t z-prev]
    (let [logits (if (nil? z-prev) init-logits
                   (mx/take-idx transition-logits z-prev 0))
          z (trace :z (dist/categorical logits))
          mu (mx/take-idx emission-means z)
          _ (trace :y (dist/gaussian mu sigma-obs))]
      z)))

(def obs-seq (mapv (fn [t] (cm/choicemap :y (mx/scalar (nth ys-data t)))) (range T)))

;; Exact log-ML from forward algorithm (precomputed)
(def exact-log-ml -86.8644)

(println "\n=== SMC Stress Test: P0-4 Metal Pipeline Leak ===")
(println (str "HMM: K=" K ", T=" T ", exact log-ML=" exact-log-ml))

;; Test 1: batched-smc-unfold N=100, T=50
(println "\n-- Test 1: batched-smc-unfold N=100 T=50 --")
(let [t0 (js/performance.now)
      result (smc/batched-smc-unfold {:particles 100 :key (rng/fresh-key 42)}
                                      (dyn/auto-key hmm-kernel) nil obs-seq)
      elapsed (- (js/performance.now) t0)
      log-ml (mx/item (:log-ml result))
      error (js/Math.abs (- log-ml exact-log-ml))]
  (println "  log-ML:" (.toFixed log-ml 2) "(exact:" (.toFixed exact-log-ml 2) ")")
  (println "  |error|:" (.toFixed error 2) "time:" (.toFixed elapsed 0) "ms")
  (println (if (< error 15) "  PASS" "  FAIL")))

(mx/force-gc!)

;; Test 2: batched-smc-unfold N=500, T=50
(println "\n-- Test 2: batched-smc-unfold N=500 T=50 --")
(let [t0 (js/performance.now)
      result (smc/batched-smc-unfold {:particles 500 :key (rng/fresh-key 99)}
                                      (dyn/auto-key hmm-kernel) nil obs-seq)
      elapsed (- (js/performance.now) t0)
      log-ml (mx/item (:log-ml result))
      error (js/Math.abs (- log-ml exact-log-ml))]
  (println "  log-ML:" (.toFixed log-ml 2) "(exact:" (.toFixed exact-log-ml 2) ")")
  (println "  |error|:" (.toFixed error 2) "time:" (.toFixed elapsed 0) "ms")
  (println (if (< error 5) "  PASS" "  FAIL")))

(mx/force-gc!)

;; Test 3: smc-unfold N=50, T=50 (per-particle extend, moderate)
(println "\n-- Test 3: smc-unfold N=50 T=50 --")
(let [t0 (js/performance.now)
      result (smc/smc-unfold {:particles 50 :key (rng/fresh-key 77)}
                              (dyn/auto-key hmm-kernel) nil obs-seq)
      elapsed (- (js/performance.now) t0)
      log-ml (mx/item (:log-ml result))
      error (js/Math.abs (- log-ml exact-log-ml))]
  (println "  log-ML:" (.toFixed log-ml 2) "(exact:" (.toFixed exact-log-ml 2) ")")
  (println "  |error|:" (.toFixed error 2) "time:" (.toFixed elapsed 0) "ms")
  (println (if (< error 15) "  PASS" "  FAIL")))

(mx/force-gc!)

;; Test 4: smc-unfold N=100, T=50 (was crashing pre-fix)
(println "\n-- Test 4: smc-unfold N=100 T=50 --")
(let [t0 (js/performance.now)
      result (smc/smc-unfold {:particles 100 :key (rng/fresh-key 123)}
                              (dyn/auto-key hmm-kernel) nil obs-seq)
      elapsed (- (js/performance.now) t0)
      log-ml (mx/item (:log-ml result))
      error (js/Math.abs (- log-ml exact-log-ml))]
  (println "  log-ML:" (.toFixed log-ml 2) "(exact:" (.toFixed exact-log-ml 2) ")")
  (println "  |error|:" (.toFixed error 2) "time:" (.toFixed elapsed 0) "ms")
  (println (if (< error 10) "  PASS" "  FAIL")))

(println "\n=== All tests complete ===")
