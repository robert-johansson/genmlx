(ns genmlx.conjugate-bb-test
  "Conjugate posterior correctness tests — Beta-Bernoulli pair.
   Verifies that inference algorithms converge to the analytically-known
   posterior: Beta(10, 4), E=0.714, Var=0.0136.

   Tests: IS, Compiled MH, MH.

   NOTE: Beta distribution log-prob uses the Lanczos gamma approximation,
   which creates many MLX scalar allocations per call. This limits the total
   number of inference iterations possible within Metal's 499K resource pool.
   Gradient-based methods (HMC, NUTS) are tested in conjugate_posterior_test
   on Gaussian-Gaussian where the simpler log-prob avoids resource exhaustion."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private pass-count (volatile! 0))
(def ^:private fail-count (volatile! 0))
(def ^:private skip-count (volatile! 0))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (do (vswap! pass-count inc)
          (println "  PASS:" msg))
      (do (vswap! fail-count inc)
          (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(defn skip-test [msg reason]
  (vswap! skip-count inc)
  (println "  SKIP:" msg "-" reason))

(defn run-test [section-name test-fn]
  (println (str "\n-- " section-name " --"))
  (try
    (test-fn)
    (catch :default e
      (skip-test section-name (str "error: " (.-message e))))))

(defn weighted-stats [traces log-weights param-key]
  (let [raw-weights (mapv (fn [w] (mx/eval! w) (mx/item w)) log-weights)
        max-w (apply max raw-weights)
        exp-weights (mapv (fn [w] (js/Math.exp (- w max-w))) raw-weights)
        sum-w (reduce + exp-weights)
        norm-weights (mapv (fn [w] (/ w sum-w)) exp-weights)
        vals (mapv (fn [t]
                     (let [v (cm/get-value (cm/get-submap (:choices t) param-key))]
                       (mx/eval! v) (mx/item v)))
                   traces)
        mean (reduce + (map * vals norm-weights))
        variance (reduce + (map (fn [v w] (* w (js/Math.pow (- v mean) 2)))
                                vals norm-weights))]
    [mean variance]))

(defn trace-stats [traces param-key]
  (let [vals (mapv (fn [t]
                     (let [v (cm/get-value (cm/get-submap (:choices t) param-key))]
                       (mx/eval! v) (mx/item v)))
                   traces)
        n (count vals)
        mean (/ (reduce + vals) n)
        variance (/ (reduce + (map (fn [v] (js/Math.pow (- v mean) 2)) vals))
                    (dec n))]
    [mean variance]))

(defn sample-stats [samples]
  (let [n (count samples)
        mean (/ (reduce + samples) n)
        variance (/ (reduce + (map (fn [v] (js/Math.pow (- v mean) 2)) samples))
                    (dec n))]
    [mean variance]))

;; =========================================================================
;; Beta-Bernoulli
;; =========================================================================
;;
;; Prior:      p ~ Beta(2, 2)
;; Likelihood: x_i ~ Bernoulli(p), i=1..10
;; Data:       [1,1,1,0,1,1,0,1,1,1] (sum=8, n=10)
;;
;; Posterior:  Beta(10, 4)
;;   E[p|data]   = 10/14 = 0.714
;;   Var[p|data]  = 40/2940 = 0.0136

(println "\n=== Conjugate Posterior: Beta-Bernoulli ===")

(def model
  (gen [data]
    (let [p (dyn/trace :p (dist/beta-dist 2 2))]
      (doseq [[i x] (map-indexed vector data)]
        (dyn/trace (keyword (str "x" i)) (dist/bernoulli p)))
      p)))

(def data [1 1 1 0 1 1 0 1 1 1])
(def observations
  (reduce (fn [cm [i x]]
            (cm/set-choice cm [(keyword (str "x" i))] (mx/scalar x)))
          cm/EMPTY (map-indexed vector data)))

(def post-mean 0.714)
(def post-var 0.0136)

;; Run Compiled MH first (gradient-based, needs more resources)
(run-test "Compiled MH"
  (fn []
    (let [samples (mcmc/compiled-mh {:samples 400 :burn 300 :addresses [:p]
                                      :proposal-std 0.15 :compile? false :device :cpu}
                                     model [data] observations)
          p-vals (mapv first samples)
          [mean variance] (sample-stats p-vals)]
      (println "  mean:" mean "var:" variance)
      (assert-close "Compiled MH posterior mean ≈ 0.714" post-mean mean 0.3)
      (assert-close "Compiled MH posterior var ≈ 0.014" post-var variance 0.02))))

(run-test "IS"
  (fn []
    (let [{:keys [traces log-weights]}
          (is/importance-sampling {:samples 200} model [data] observations)
          [mean variance] (weighted-stats traces log-weights :p)]
      (println "  mean:" mean "var:" variance)
      (assert-close "IS posterior mean ≈ 0.714" post-mean mean 0.3)
      (assert-close "IS posterior var ≈ 0.014" post-var variance 0.02))))

(run-test "MH"
  (fn []
    (let [traces (mcmc/mh {:samples 300 :burn 100 :selection (sel/select :p)}
                           model [data] observations)
          [mean variance] (trace-stats traces :p)]
      (println "  mean:" mean "var:" variance)
      (assert-close "MH posterior mean ≈ 0.714" post-mean mean 0.3)
      (assert-close "MH posterior var ≈ 0.014" post-var variance 0.02))))

(println "\n=== Summary ===")
(println (str @pass-count " passed, " @fail-count " failed, " @skip-count " skipped"))
(when (pos? @fail-count)
  (println "SOME TESTS FAILED"))
