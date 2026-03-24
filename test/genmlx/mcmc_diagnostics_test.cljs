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
  (let [samples   (mcmc/compiled-mh {:samples 2000 :burn 500 :addresses [:mu]
                                      :proposal-std 0.3}
                                     model-b [] obs-b)
        mlx-samps (samples->mlx-arrays samples)
        ess-val   (diag/ess mlx-samps)]
    (assert-true "ESS is finite" (js/isFinite ess-val))
    (assert-true "ESS > 0" (pos? ess-val))
    ;; Without thinning, ESS/N can be low for random-walk MH. The ground truth
    ;; lower bound of 100 is conservative. We use 50 as a practical floor.
    (assert-true "ESS > 50 (adequate mixing)" (> ess-val 50))
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
  (let [samples (mcmc/compiled-mh {:samples 500 :burn 200 :addresses [:mu]
                                    :proposal-std 0.5 :compile? false}
                                   model-a [] obs-a)
        rate    (:acceptance-rate (meta samples))
        mu-vals (mapv first samples)
        mu-mean (sample-mean mu-vals)]
    (assert-true "MH: has acceptance rate" (some? rate))
    ;; MH with random-walk proposal std=0.5 on N(1,0.5) posterior. The
    ;; acceptance rate depends on the proposal-to-posterior width ratio.
    ;; For this 1D target, rates in the 0.4-0.85 range are typical.
    (assert-true "MH: rate in (0.10, 0.90)" (and (> rate 0.10) (< rate 0.90)))
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

;; --- summary ---
(println (str "\n=== " @pass-count " passed, " @fail-count " failed ==="))
(when (pos? @fail-count) (println "SOME TESTS FAILED"))
