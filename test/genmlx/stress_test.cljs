(ns genmlx.stress-test
  "Stress tests for Metal GPU resource management (TODO_GPU.md §7.2, 7.3, 7.5).
   Verifies that Phases 2+3 prevent resource exhaustion during long inference."
  (:require [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.importance :as is]
            [genmlx.inference.util :as u]
            [genmlx.gen :refer [gen]]))

(defn assert-true [msg v]
  (if v
    (println "  PASS:" msg)
    (println "  FAIL:" msg)))

;; ---------------------------------------------------------------------------
;; §7.2 — MH doesn't hit resource limit (Beta-Bernoulli, 5000 samples)
;; ---------------------------------------------------------------------------

(println "\n=== Stress Test 7.2: MH on Beta-Bernoulli (5000 samples, 2000 burn) ===")

(let [model (gen [n-obs]
              (let [p (dyn/trace :p (dist/beta-dist 2 2))]
                (mx/eval! p)
                (let [p-val (mx/item p)]
                  (doseq [i (range n-obs)]
                    (dyn/trace (keyword (str "obs" i))
                               (dist/bernoulli p-val)))
                  p-val)))
      n-obs 10
      observations (reduce (fn [cm i]
                             (cm/set-choice cm [(keyword (str "obs" i))] (mx/scalar 1.0)))
                           cm/EMPTY (range n-obs))]

  (mx/clear-cache!)
  (mx/reset-peak-memory!)
  (println "  Before: active=" (mx/get-active-memory) "cache=" (mx/get-cache-memory))

  (let [t0 (js/Date.now)
        traces (mcmc/mh {:samples 5000 :burn 2000 :selection (sel/select :p)}
                         model [n-obs] observations)
        elapsed (- (js/Date.now) t0)]
    (println "  After:  active=" (mx/get-active-memory) "cache=" (mx/get-cache-memory)
             "peak=" (mx/get-peak-memory))
    (println "  Elapsed:" elapsed "ms")
    (assert-true (str "Got " (count traces) " traces (expected 5000)")
                 (= 5000 (count traces)))

    ;; Verify posterior: with 10/10 heads and Beta(2,2) prior, posterior is Beta(12,2)
    ;; E[p] = 12/14 ≈ 0.857
    (let [p-vals (mapv (fn [t]
                         (let [v (cm/get-value (cm/get-submap (:choices t) :p))]
                           (mx/eval! v) (mx/item v)))
                       (take-last 100 traces))
          p-mean (/ (reduce + p-vals) (count p-vals))]
      (println "  Posterior mean p (last 100):" p-mean "(expected ~0.857)")
      (assert-true "Posterior mean near 0.857" (< (js/Math.abs (- p-mean 0.857)) 0.15)))
    ;; Dispose collected traces to free Metal buffers for subsequent tests
    (u/dispose-trace! traces)))

;; Clean up between tests
(mx/clear-cache!)
(u/force-gc!)

;; ---------------------------------------------------------------------------
;; §7.5 — Long inference chains (10000 MH samples on Beta model)
;; Run BEFORE SMC to avoid Metal buffer accumulation from particles.
;; ---------------------------------------------------------------------------

(println "\n=== Stress Test 7.5: Long MH chain (10000 samples, 5000 burn, Beta) ===")

(let [model (gen []
              (let [x (dyn/trace :x (dist/beta-dist 2 2))]
                x))
      obs (cm/choicemap)]

  (mx/clear-cache!)
  (mx/reset-peak-memory!)
  (println "  Before: active=" (mx/get-active-memory) "cache=" (mx/get-cache-memory))

  (let [t0 (js/Date.now)
        traces (mcmc/mh {:samples 10000 :burn 5000 :selection (sel/select :x)}
                         model [] obs)
        elapsed (- (js/Date.now) t0)]
    (println "  After:  active=" (mx/get-active-memory) "cache=" (mx/get-cache-memory)
             "peak=" (mx/get-peak-memory))
    (println "  Elapsed:" elapsed "ms")
    (assert-true (str "Completed " (count traces) " iterations (expected 10000)")
                 (= 10000 (count traces)))

    ;; With Beta(2,2) prior and no observations, posterior is still Beta(2,2)
    ;; E[x] = 0.5
    (let [x-vals (mapv (fn [t]
                         (let [v (cm/get-value (cm/get-submap (:choices t) :x))]
                           (mx/eval! v) (mx/item v)))
                       (take-last 200 traces))
          x-mean (/ (reduce + x-vals) (count x-vals))]
      (println "  Posterior mean x (last 200):" x-mean "(expected ~0.5)")
      (assert-true "Posterior mean near 0.5" (< (js/Math.abs (- x-mean 0.5)) 0.15)))
    ;; Dispose collected traces to free Metal buffers for subsequent tests
    (u/dispose-trace! traces)))

;; Clean up between tests
(mx/clear-cache!)
(u/force-gc!)

;; ---------------------------------------------------------------------------
;; §7.3 — SMC doesn't hit resource limit (many timesteps)
;; Run last since particles accumulate Metal buffers that can't be freed mid-process.
;; ---------------------------------------------------------------------------

(println "\n=== Stress Test 7.3: SMC with 50 timesteps, 100 particles ===")

(let [model (gen [xs]
              (let [mu (dyn/trace :mu (dist/gaussian 0 5))]
                (mx/eval! mu)
                (let [m (mx/item mu)]
                  (doseq [[i x] (map-indexed vector xs)]
                    (dyn/trace (keyword (str "y" i)) (dist/gaussian m 1)))
                  m)))
      ;; Build 20 timesteps of cumulative observations
      n-timesteps 20
      data (vec (repeatedly n-timesteps #(+ 3.0 (* 0.5 (- (js/Math.random) 0.5)))))
      obs-seq (mapv (fn [t]
                      (reduce (fn [cm i]
                                (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar (nth data i))))
                              cm/EMPTY (range (inc t))))
                    (range n-timesteps))]

  (mx/clear-cache!)
  (mx/reset-peak-memory!)
  (println "  Before: active=" (mx/get-active-memory) "cache=" (mx/get-cache-memory))

  (let [t0 (js/Date.now)
        result (smc/smc {:particles 100} model [data] obs-seq)
        elapsed (- (js/Date.now) t0)]
    (println "  After:  active=" (mx/get-active-memory) "cache=" (mx/get-cache-memory)
             "peak=" (mx/get-peak-memory))
    (println "  Elapsed:" elapsed "ms")
    (assert-true (str "Got " (count (:traces result)) " particles (expected 100)")
                 (= 100 (count (:traces result))))
    (let [log-ml (:log-ml-estimate result)]
      (mx/eval! log-ml)
      (println "  Log-ML estimate:" (mx/item log-ml))
      (assert-true "Log-ML is finite" (js/isFinite (mx/item log-ml))))))

(println "\n=== All stress tests complete ===")
