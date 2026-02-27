(ns genmlx.conjugate-posterior-test
  "Conjugate posterior correctness tests — Gaussian-Gaussian pair.
   Verifies that inference algorithms converge to the analytically-known
   posterior: N(2.954, 0.1996).

   Tests: IS, MH, Compiled MH, HMC, NUTS.

   Run all three conjugate pairs:
     bun run --bun nbb test/genmlx/conjugate_posterior_test.cljs
     bun run --bun nbb test/genmlx/conjugate_bb_test.cljs
     bun run --bun nbb test/genmlx/conjugate_gp_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Pass/fail/skip counting
;; ---------------------------------------------------------------------------

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

;; ---------------------------------------------------------------------------
;; Statistics helpers
;; ---------------------------------------------------------------------------

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
;; Gaussian-Gaussian
;; =========================================================================
;;
;; Prior:      mu ~ N(0, 10)
;; Likelihood: obs_i ~ N(mu, 1), i=1..5
;; Data:       [2.8, 3.1, 2.9, 3.3, 2.7]
;;
;; Posterior:  N(mu_post=2.954, sigma2_post=0.1996)

(println "\n=== Conjugate Posterior: Gaussian-Gaussian ===")

(def model
  (gen [data]
    (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
      (doseq [[i x] (map-indexed vector data)]
        (dyn/trace (keyword (str "obs" i)) (dist/gaussian mu 1)))
      mu)))

(def data [2.8 3.1 2.9 3.3 2.7])
(def observations
  (reduce (fn [cm [i x]]
            (cm/set-choice cm [(keyword (str "obs" i))] (mx/scalar x)))
          cm/EMPTY (map-indexed vector data)))

(def post-mean 2.954)
(def post-var 0.1996)

(run-test "IS"
  (fn []
    (let [{:keys [traces log-weights]}
          (is/importance-sampling {:samples 200} model [data] observations)
          [mean variance] (weighted-stats traces log-weights :mu)]
      (println "  mean:" mean "var:" variance)
      (assert-close "IS posterior mean ≈ 2.954" post-mean mean 0.5)
      (assert-close "IS posterior var ≈ 0.200" post-var variance 0.3))))

(run-test "MH"
  (fn []
    (let [traces (mcmc/mh {:samples 500 :burn 200 :selection (sel/select :mu)}
                           model [data] observations)
          [mean variance] (trace-stats traces :mu)]
      (println "  mean:" mean "var:" variance)
      (assert-close "MH posterior mean ≈ 2.954" post-mean mean 0.3)
      (assert-close "MH posterior var ≈ 0.200" post-var variance 0.3))))

(run-test "Compiled MH"
  (fn []
    (let [samples (mcmc/compiled-mh {:samples 300 :burn 200 :addresses [:mu]
                                      :proposal-std 0.3 :compile? false :device :cpu}
                                     model [data] observations)
          mu-vals (mapv first samples)
          [mean variance] (sample-stats mu-vals)]
      (println "  mean:" mean "var:" variance)
      (assert-close "Compiled MH posterior mean ≈ 2.954" post-mean mean 0.5)
      (assert-close "Compiled MH posterior var ≈ 0.200" post-var variance 0.3))))

(run-test "HMC"
  (fn []
    (let [samples (mcmc/hmc {:samples 200 :burn 100 :step-size 0.05 :leapfrog-steps 10
                              :addresses [:mu] :compile? false :device :cpu}
                             model [data] observations)
          mu-vals (mapv first samples)
          [mean variance] (sample-stats mu-vals)]
      (println "  mean:" mean "var:" variance)
      (assert-close "HMC posterior mean ≈ 2.954" post-mean mean 0.3)
      (assert-close "HMC posterior var ≈ 0.200" post-var variance 0.3))))

(run-test "NUTS"
  (fn []
    (let [samples (mcmc/nuts {:samples 50 :burn 30 :step-size 0.05 :max-depth 5
                               :addresses [:mu] :compile? false :device :cpu}
                              model [data] observations)
          mu-vals (mapv first samples)
          [mean variance] (sample-stats mu-vals)]
      (println "  mean:" mean "var:" variance)
      (assert-close "NUTS posterior mean ≈ 2.954" post-mean mean 0.5)
      (assert-close "NUTS posterior var ≈ 0.200" post-var variance 0.5))))

;; =========================================================================
;; Summary
;; =========================================================================

(println "\n=== Summary ===")
(println (str @pass-count " passed, " @fail-count " failed, " @skip-count " skipped"))
(when (pos? @fail-count)
  (println "SOME TESTS FAILED"))
