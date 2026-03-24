(ns genmlx.mcmc-stationary-test
  "Phase 5.2 — MCMC stationary distribution tests. Each test verifies that
   a sampler converges to the analytically known posterior. Tolerances come
   from the ground-truth specification (z=3.5, safety=1.5, N_eff estimates)."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.importance :as is]
            [genmlx.inference.smc :as smc])
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

;; --- Model B: multi-observation Normal-Normal ---
;; Prior: mu ~ N(0,10), Likelihood: y_i ~ N(mu,1) for 5 obs [2.8,3.1,2.9,3.2,3.0]
;; Posterior: N(2.994, 0.1996), sigma=0.4468

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

;; --- Model C: Beta-Bernoulli ---
;; Prior: p ~ Beta(2,2), Likelihood: x_i ~ Bernoulli(p)
;; Data: [1,1,1,0,1,1,0,1,1,1] -> Posterior Beta(10,4)
;; E[p|data] = 10/14 = 0.7143, Var = 0.01361

(def model-c
  (gen []
    (let [p (trace :p (dist/beta-dist 2 2))]
      (doseq [j (range 10)]
        (trace (keyword (str "x" j)) (dist/bernoulli p)))
      p)))

(def obs-c
  (let [xs [1 1 1 0 1 1 0 1 1 1]]
    (reduce (fn [cm [j x]]
              (cm/set-choice cm [(keyword (str "x" j))] (mx/scalar x)))
            cm/EMPTY
            (map-indexed vector xs))))

;; --- helpers ---

(defn sample-mean
  "Arithmetic mean of a sequence of numbers."
  [vals]
  (/ (reduce + vals) (count vals)))

(defn sample-var
  "Unbiased sample variance."
  [vals]
  (let [mu (sample-mean vals)
        n  (count vals)]
    (/ (reduce + (map #(let [d (- % mu)] (* d d)) vals))
       (dec n))))

(defn weighted-mean-from-is
  "Compute the weighted mean of retvals from importance sampling result.
   Normalizes log-weights via logsumexp for numerical stability.
   Retvals may be MLX arrays — extracts JS numbers via eval!/item."
  [{:keys [traces log-weights]}]
  (let [raw-weights  (mapv (fn [w] (mx/eval! w) (mx/item w)) log-weights)
        max-w        (apply max raw-weights)
        exp-weights  (mapv #(js/Math.exp (- % max-w)) raw-weights)
        sum-w        (reduce + exp-weights)
        norm-weights (mapv #(/ % sum-w) exp-weights)
        retvals      (mapv (fn [t]
                             (let [rv (:retval t)]
                               (if (mx/array? rv)
                                 (do (mx/eval! rv) (mx/item rv))
                                 rv)))
                           traces)]
    (reduce + (map * retvals norm-weights))))

(println "\n=== Phase 5.2: MCMC Stationary Distribution Tests ===")

;; ---------------------------------------------------------------------------
;; 5.2.1 MH on Normal-Normal (Model B)
;; ---------------------------------------------------------------------------

(println "\n-- 5.2.1 MH on Normal-Normal (5 obs) --")

(try
  (let [samples  (mcmc/compiled-mh {:samples 2000 :burn 500 :thin 2
                                     :addresses [:mu] :proposal-std 0.3}
                                    model-b [] obs-b)
        mu-vals  (mapv first samples)
        has-nan  (some js/isNaN mu-vals)
        mu-mean  (sample-mean mu-vals)
        mu-var   (sample-var mu-vals)]
    (assert-true "MH-B: 2000 samples returned" (= 2000 (count mu-vals)))
    (assert-true "MH-B: no NaN" (not has-nan))
    (assert-close "MH-B: mean near 2.994" 2.994 mu-mean 0.11)
    (assert-close "MH-B: variance near 0.1996" 0.1996 mu-var 0.067)
    (println "    mean:" mu-mean "  var:" mu-var))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: 5.2.1 threw:" (.-message e))))

;; ---------------------------------------------------------------------------
;; 5.2.2 HMC on Normal-Normal (Model B)
;; ---------------------------------------------------------------------------

(println "\n-- 5.2.2 HMC on Normal-Normal (5 obs) --")

(try
  (let [samples  (mcmc/hmc {:samples 500 :burn 200 :step-size 0.1
                              :leapfrog-steps 10 :addresses [:mu]}
                             model-b [] obs-b)
        mu-vals  (mapv first samples)
        has-nan  (some js/isNaN mu-vals)
        mu-mean  (sample-mean mu-vals)]
    (assert-true "HMC-B: no NaN" (not has-nan))
    (assert-close "HMC-B: mean near 2.994" 2.994 mu-mean 0.15)
    (println "    mean:" mu-mean))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: 5.2.2 threw:" (.-message e))))

;; HMC acceptance rate (separate run with compile? false)
(try
  (let [samples (mcmc/hmc {:samples 200 :burn 100 :step-size 0.1
                             :leapfrog-steps 10 :addresses [:mu]
                             :compile? false :device :cpu}
                            model-b [] obs-b)
        rate    (:acceptance-rate (meta samples))]
    (assert-true "HMC-B: acceptance > 0.5" (> rate 0.5))
    (println "    acceptance:" rate))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: 5.2.2 acceptance threw:" (.-message e))))

;; ---------------------------------------------------------------------------
;; 5.2.3 Multi-Algorithm Agreement (Model B)
;; ---------------------------------------------------------------------------

(println "\n-- 5.2.3 Multi-algorithm agreement --")

(def algorithm-means (volatile! {}))

;; IS: 2000 particles, weighted mean
(try
  (let [result  (is/importance-sampling {:samples 2000} model-b [] obs-b)
        is-mean (weighted-mean-from-is result)]
    (vswap! algorithm-means assoc :IS is-mean)
    (assert-close "IS: mean near 2.994" 2.994 is-mean 0.31)
    (println "    IS mean:" is-mean))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: IS threw:" (.-message e))))

;; compiled-MH: 2000 samples
(try
  (let [samples (mcmc/compiled-mh {:samples 2000 :burn 500 :thin 2
                                    :addresses [:mu] :proposal-std 0.3}
                                   model-b [] obs-b)
        mh-mean (sample-mean (mapv first samples))]
    (vswap! algorithm-means assoc :MH mh-mean)
    (assert-close "MH: mean near 2.994" 2.994 mh-mean 0.11)
    (println "    MH mean:" mh-mean))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: MH threw:" (.-message e))))

;; HMC: 500 samples
(try
  (let [samples  (mcmc/hmc {:samples 500 :burn 200 :step-size 0.1
                              :leapfrog-steps 10 :addresses [:mu]}
                             model-b [] obs-b)
        hmc-mean (sample-mean (mapv first samples))]
    (vswap! algorithm-means assoc :HMC hmc-mean)
    (assert-close "HMC: mean near 2.994" 2.994 hmc-mean 0.15)
    (println "    HMC mean:" hmc-mean))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: HMC threw:" (.-message e))))

;; NUTS: 50 samples, compile? false, device :cpu (Bun safety)
(try
  (let [samples   (mcmc/nuts {:samples 50 :burn 50 :step-size 0.05
                                :addresses [:mu] :compile? false :device :cpu}
                               model-b [] obs-b)
        nuts-mean (sample-mean (mapv first samples))]
    (vswap! algorithm-means assoc :NUTS nuts-mean)
    (assert-close "NUTS: mean near 2.994" 2.994 nuts-mean 0.22)
    (println "    NUTS mean:" nuts-mean))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: NUTS threw:" (.-message e))))

;; SMC: single-step (all observations at once = importance sampling variant)
(try
  (let [result   (smc/smc {:particles 500} model-b [] [obs-b])
        smc-mean (weighted-mean-from-is result)]
    (vswap! algorithm-means assoc :SMC smc-mean)
    (assert-close "SMC: mean near 2.994" 2.994 smc-mean 0.34)
    (println "    SMC mean:" smc-mean))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: SMC threw:" (.-message e))))

;; Pairwise agreement: max |mean_i - mean_j| < 0.45
(try
  (let [means (vals @algorithm-means)
        pairs (for [a means, b means :when (not= a b)]
                (js/Math.abs (- a b)))
        max-diff (if (seq pairs) (apply max pairs) 0)]
    (assert-true "max pairwise difference < 0.45" (< max-diff 0.45))
    (println "    max pairwise diff:" max-diff))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: pairwise check threw:" (.-message e))))

;; ---------------------------------------------------------------------------
;; 5.2.4 MH on Beta-Bernoulli (Model C)
;; ---------------------------------------------------------------------------

(println "\n-- 5.2.4 MH on Beta-Bernoulli --")

(try
  (let [samples  (mcmc/compiled-mh {:samples 3000 :burn 500 :thin 2
                                     :addresses [:p] :proposal-std 0.1}
                                    model-c [] obs-c)
        p-vals   (mapv first samples)
        has-nan  (some js/isNaN p-vals)
        in-range (every? #(and (> % 0) (< % 1)) p-vals)
        p-mean   (sample-mean p-vals)
        p-var    (sample-var p-vals)]
    (assert-true "Beta-Bernoulli: no NaN" (not has-nan))
    (assert-true "Beta-Bernoulli: all in (0,1)" in-range)
    (assert-close "Beta-Bernoulli: mean near 0.7143" 0.7143 p-mean 0.025)
    (assert-close "Beta-Bernoulli: variance near 0.01361" 0.01361 p-var 0.0042)
    (println "    mean:" p-mean "  var:" p-var))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: 5.2.4 threw:" (.-message e))))

;; --- summary ---
(println (str "\n=== " @pass-count " passed, " @fail-count " failed ==="))
(when (pos? @fail-count) (println "SOME TESTS FAILED"))
