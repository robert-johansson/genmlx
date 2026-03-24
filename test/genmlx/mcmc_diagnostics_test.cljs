(ns genmlx.mcmc-diagnostics-test
  "Phase 5.3 — MCMC diagnostic infrastructure tests. Verifies R-hat,
   ESS, and acceptance rate metadata for correctness and calibration."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.diagnostics :as diag])
  (:require-macros [genmlx.gen :refer [gen]]))

;; --- counters ---
(def pass-count (volatile! 0))
(def fail-count (volatile! 0))

(defn assert-true [msg v]
  (if v
    (do (vswap! pass-count inc) (println "  PASS:" msg))
    (do (vswap! fail-count inc) (println "  FAIL:" msg))))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (do (vswap! pass-count inc) (println "  PASS:" msg))
      (do (vswap! fail-count inc)
          (println "  FAIL:" msg)
          (println "    expected:" expected "+-" tolerance)
          (println "    actual:  " actual "  diff:" diff)))))

;; --- Models ---

;; Model A: single-observation Normal-Normal
;; Posterior: N(1.0, 0.5)
(def model-a
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 1))]
      (trace :y (dist/gaussian mu 1))
      mu)))

(def obs-a (cm/choicemap :y (mx/scalar 2.0)))

;; Model B: multi-observation Normal-Normal
;; Posterior: N(2.994, 0.1996)
(def model-b
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (doseq [j (range 5)]
        (trace (keyword (str "y" j)) (dist/gaussian mu 1)))
      mu)))

(def obs-b
  (let [ys [2.8 3.1 2.9 3.2 3.0]]
    (reduce (fn [cm [j y]]
              (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
            cm/EMPTY
            (map-indexed vector ys))))

;; --- helpers ---

(defn samples->mlx-arrays
  "Convert compiled-mh output (vector of JS arrays) to vector of MLX scalars.
   For a 1-parameter model, each sample is a JS array like [0.75]."
  [samples]
  (mapv #(mx/scalar (first %)) samples))

(defn sample-mean
  "Arithmetic mean of a sequence of numbers."
  [vals]
  (/ (reduce + vals) (count vals)))

(println "\n=== Phase 5.3: MCMC Diagnostics Tests ===")

;; ---------------------------------------------------------------------------
;; 5.3.1 R-hat Convergence
;; ---------------------------------------------------------------------------
;; Run 4 independent MH chains on Model B. Compute R-hat. For well-converged
;; chains targeting the same distribution, R-hat should be close to 1.0.

(println "\n-- 5.3.1 R-hat convergence --")

(try
  (let [seeds      [42 137 271 999]
        run-chain  (fn [seed]
                     (mcmc/compiled-mh
                       {:samples 1000 :burn 500 :addresses [:mu]
                        :proposal-std 0.3 :key (rng/fresh-key seed)}
                       model-b [] obs-b))
        chains     (mapv run-chain seeds)
        ;; Extract JS number vectors for chain means
        chain-vals (mapv (fn [samples] (mapv first samples)) chains)
        ;; Compute per-chain means
        chain-means (mapv sample-mean chain-vals)
        ;; Convert to MLX arrays for r-hat
        mlx-chains  (mapv samples->mlx-arrays chains)
        r           (diag/r-hat mlx-chains)]
    ;; R-hat structural properties
    (assert-true "R-hat >= 1.0 (by construction)" (>= r 1.0))
    (assert-true "R-hat < 1.1 (converged)" (< r 1.1))
    (println "    R-hat:" r)
    ;; All chain means near the analytical posterior mean
    (doseq [[i mu] (map-indexed vector chain-means)]
      (assert-close (str "chain " i " mean near 2.994")
                    2.994 mu 0.15))
    (println "    chain means:" (mapv #(.toFixed % 4) chain-means)))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: 5.3.1 threw:" (.-message e))))

;; ---------------------------------------------------------------------------
;; 5.3.2 Effective Sample Size
;; ---------------------------------------------------------------------------
;; Run compiled-mh on Model B. Compute ESS. It should be a substantial
;; fraction of the nominal sample count, but less than the total.

(println "\n-- 5.3.2 Effective sample size --")

(try
  (let [;; proposal-std 1.0 ≈ 2.38 * σ_post gives acceptance ~0.44 (optimal for 1D)
        ;; and ESS/N ~ 0.2-0.4, so ESS > N/10 = 200 is achievable
        samples   (mcmc/compiled-mh {:samples 2000 :burn 500 :addresses [:mu]
                                      :proposal-std 1.0}
                                     model-b [] obs-b)
        mlx-samps (samples->mlx-arrays samples)
        ess-val   (diag/ess mlx-samps)]
    (assert-true "ESS is finite" (js/isFinite ess-val))
    (assert-true "ESS > 0" (pos? ess-val))
    (assert-true "ESS > N/10 = 200 (well-tuned MH)" (> ess-val 200))
    (assert-true "ESS < 2000 (autocorrelated chain)" (< ess-val 2000))
    (println "    ESS:" ess-val))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: 5.3.2 threw:" (.-message e))))

;; ---------------------------------------------------------------------------
;; 5.3.3 Acceptance Rate Calibration
;; ---------------------------------------------------------------------------
;; Run MH, HMC, MALA each with compile? false to get acceptance metadata.
;; Verify rates are in theoretically justified ranges and all produce
;; correct posterior means.

(println "\n-- 5.3.3 Acceptance rate calibration --")

;; MH
(try
  (let [;; proposal-std 2.0 ≈ 2.38 * σ_post for Model A gives acceptance ~0.44
        ;; which is the Roberts et al. (1997) optimal for d=1 Gaussian target
        samples (mcmc/compiled-mh {:samples 500 :burn 200 :addresses [:mu]
                                    :proposal-std 2.0 :compile? false}
                                   model-a [] obs-a)
        rate    (:acceptance-rate (meta samples))
        mu-vals (mapv first samples)
        mu-mean (sample-mean mu-vals)]
    (assert-true "MH: has acceptance rate" (some? rate))
    ;; Optimal acceptance for d=1 Gaussian target: 0.44 (Roberts et al. 1997)
    ;; With proposal-std=2.0 on σ_post=0.707, we expect acceptance ~0.43
    (assert-true "MH: rate near 0.44 optimal (0.30, 0.55)" (and (> rate 0.30) (< rate 0.55)))
    (assert-close "MH: mean near 1.0" 1.0 mu-mean 0.31)
    (println "    MH acceptance:" rate "  mean:" mu-mean))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: MH acceptance threw:" (.-message e))))

;; HMC
(try
  (let [samples (mcmc/hmc {:samples 200 :burn 100 :step-size 0.1
                             :leapfrog-steps 10 :addresses [:mu]
                             :compile? false :device :cpu}
                            model-a [] obs-a)
        rate    (:acceptance-rate (meta samples))
        mu-vals (mapv first samples)
        mu-mean (sample-mean mu-vals)]
    (assert-true "HMC: has acceptance rate" (some? rate))
    ;; HMC on 1D quadratic: leapfrog nearly conserves energy, so acceptance
    ;; is very high (often >0.95). We verify it is above the theoretical
    ;; floor and below 1.0 (some rejections expected over many samples).
    (assert-true "HMC: rate in (0.40, 1.0)" (and (> rate 0.40) (<= rate 1.0)))
    (assert-close "HMC: mean near 1.0" 1.0 mu-mean 0.31)
    (println "    HMC acceptance:" rate "  mean:" mu-mean))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: HMC acceptance threw:" (.-message e))))

;; MALA
(try
  (let [samples (mcmc/mala {:samples 200 :burn 100 :step-size 0.5
                              :addresses [:mu] :compile? false :device :cpu}
                             model-a [] obs-a)
        rate    (:acceptance-rate (meta samples))
        mu-vals (mapv first samples)
        mu-mean (sample-mean mu-vals)]
    (assert-true "MALA: has acceptance rate" (some? rate))
    ;; MALA on 1D quadratic with step-size 0.5: the gradient-adjusted
    ;; proposal is very accurate, yielding acceptance > 0.90 typically.
    (assert-true "MALA: rate in (0.20, 1.0)" (and (> rate 0.20) (<= rate 1.0)))
    (assert-close "MALA: mean near 1.0" 1.0 mu-mean 0.31)
    (println "    MALA acceptance:" rate "  mean:" mu-mean))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: MALA acceptance threw:" (.-message e))))

;; ---------------------------------------------------------------------------
;; 5.3.4 Adaptive acceptance on multi-dimensional target
;; ---------------------------------------------------------------------------
;; 1D quadratic targets are too easy for HMC/MALA (acceptance ≈ 1.0).
;; A 5D model with dual averaging adaptation tests that:
;; (a) the adaptation mechanism functions (rate is non-trivial)
;; (b) rates are in sensible ranges for gradient-based methods
;; Note: asymptotic optima (HMC ~0.65, MALA ~0.574) require d >> 5.

(println "\n-- 5.3.4 Adaptive acceptance on 5D model --")

(def model-5d
  (gen []
    (let [xs (mapv (fn [j] (trace (keyword (str "x" j)) (dist/gaussian 0 10)))
                   (range 5))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j)) (dist/gaussian x 1)))
      (first xs))))

(def obs-5d
  (reduce (fn [cm [j y]]
            (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector [2.5 3.0 1.5 2.0 2.8])))

(def addrs-5d [:x0 :x1 :x2 :x3 :x4])

;; HMC with dual averaging targeting 0.65
(try
  (let [samples (mcmc/hmc {:samples 200 :burn 200 :step-size 0.5 :leapfrog-steps 10
                             :addresses addrs-5d :compile? false :device :cpu
                             :adapt-step-size true :target-accept 0.65}
                            model-5d [] obs-5d)
        rate    (:acceptance-rate (meta samples))
        x0-mean (sample-mean (mapv first samples))]
    (assert-true "HMC-5D: has acceptance rate" (some? rate))
    (assert-true "HMC-5D: rate non-trivial (0.50, 0.98)"
                 (and (> rate 0.50) (< rate 0.98)))
    ;; x0 posterior: N(2.475, 0.1996) — same conjugate formula
    (assert-close "HMC-5D: x0 mean near 2.475" 2.475 x0-mean 0.30)
    (println "    HMC-5D acceptance:" rate "  x0 mean:" x0-mean))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: HMC-5D threw:" (.-message e))))

;; MALA with dual averaging targeting 0.574
(try
  (let [samples (mcmc/mala {:samples 200 :burn 200 :step-size 0.5
                              :addresses addrs-5d :compile? false :device :cpu
                              :adapt-step-size true :target-accept 0.574}
                             model-5d [] obs-5d)
        rate    (:acceptance-rate (meta samples))
        x0-mean (sample-mean (mapv first samples))]
    (assert-true "MALA-5D: has acceptance rate" (some? rate))
    (assert-true "MALA-5D: rate non-trivial (0.40, 0.95)"
                 (and (> rate 0.40) (< rate 0.95)))
    (assert-close "MALA-5D: x0 mean near 2.475" 2.475 x0-mean 0.30)
    (println "    MALA-5D acceptance:" rate "  x0 mean:" x0-mean))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: MALA-5D threw:" (.-message e))))

;; --- summary ---
(println (str "\n=== " @pass-count " passed, " @fail-count " failed ==="))
(when (pos? @fail-count) (println "SOME TESTS FAILED"))
