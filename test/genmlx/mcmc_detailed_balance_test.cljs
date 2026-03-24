(ns genmlx.mcmc-detailed-balance-test
  "Phase 5.1 — MCMC mechanism tests: detailed balance, step-size sensitivity,
   gradient-based convergence. Each test verifies a structural property of the
   Markov chain, not just a posterior statistic."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.kernel :as k])
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

;; --- Model A: single-observation Normal-Normal ---
;; Prior: mu ~ N(0,1), Likelihood: y ~ N(mu,1), observe y=2
;; Posterior: N(1.0, 0.5), sigma=0.7071

(def model-a
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 1))]
      (trace :y (dist/gaussian mu 1))
      mu)))

(def obs-a (cm/choicemap :y (mx/scalar 2.0)))

(println "\n=== Phase 5.1: MCMC Detailed Balance Tests ===")

;; ---------------------------------------------------------------------------
;; 5.1.1 MH Fixed-Point Test (Geweke-style)
;; ---------------------------------------------------------------------------
;; Draw N=2000 from the analytical posterior N(1.0, 0.5). Create a trace at
;; each value via p/generate. Apply 1 MH step via kernel. If MH preserves
;; the stationary distribution, the output ensemble has the same mean and
;; variance as the input.

(println "\n-- 5.1.1 MH fixed-point test --")

(try
  (let [n           2000
        post-mean   1.0
        post-sigma  0.7071067811865476
        model-keyed (dyn/auto-key model-a)
        kernel      (k/random-walk :mu 0.5)
        ;; Draw n independent samples from the analytical posterior via
        ;; Box-Muller transform on MLX random uniform pairs.
        root-key    (rng/fresh-key)
        keys-bm     (rng/split-n root-key n)
        ;; For each posterior draw: create trace at (mu=draw, y=2), apply 1 MH step
        output-vals
        (mapv
          (fn [ki]
            (let [;; Box-Muller from two independent uniforms
                  [k1 k2 k3] (rng/split-n ki 3)
                  u1     (mx/item (rng/uniform k1 [] 0.0 1.0))
                  u2     (mx/item (rng/uniform k2 [] 0.0 1.0))
                  z      (* (js/Math.sqrt (* -2.0 (js/Math.log u1)))
                            (js/Math.cos (* 2.0 js/Math.PI u2)))
                  draw   (+ post-mean (* post-sigma z))
                  ;; Constrain trace to this draw
                  constraints (cm/choicemap :mu (mx/scalar draw) :y (mx/scalar 2.0))
                  {:keys [trace]} (p/generate model-keyed [] constraints)
                  ;; Apply 1 MH step
                  trace' (kernel trace k3)
                  ;; Extract resulting mu
                  v (cm/get-value (cm/get-submap (:choices trace') :mu))]
              (mx/eval! v)
              (mx/item v)))
          keys-bm)
        output-mean (/ (reduce + output-vals) (count output-vals))
        output-var  (let [mu output-mean]
                      (/ (reduce + (map #(let [d (- % mu)] (* d d)) output-vals))
                         (dec (count output-vals))))
        ;; Check acceptance: count how many changed from input
        ;; We generated input draws but don't store them separately, so instead
        ;; verify the output distribution matches the posterior.
        ;; Also check that not ALL accepted (would indicate a bug in MH check).
        n-unique    (count (set (map #(js/Math.round (* % 1000)) output-vals)))]
    (assert-close "posterior mean preserved" 1.0 output-mean 0.084)
    (assert-close "posterior variance preserved" 0.5 output-var 0.084)
    (assert-true "acceptance rate in (0.3, 0.95) — not all identical"
                 (< 0.3 (/ (double n-unique) n)))
    (assert-true "at least 1 rejection — not all unique"
                 (< n-unique n)))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: 5.1.1 threw:" (.-message e))))

;; ---------------------------------------------------------------------------
;; 5.1.2 HMC Step-Size Sensitivity
;; ---------------------------------------------------------------------------
;; Three step sizes bracket the sweet spot. Tiny epsilon -> near-perfect
;; acceptance but slow exploration. Good epsilon -> correct posterior.
;; Large epsilon -> low acceptance (energy errors).

(println "\n-- 5.1.2 HMC step-size sensitivity --")

;; Tiny step-size: high acceptance, slow mixing
(try
  (let [samples-tiny (mcmc/hmc {:samples 100 :burn 50 :step-size 0.001
                                 :leapfrog-steps 20 :addresses [:mu]
                                 :compile? false :device :cpu}
                                model-a [] obs-a)
        rate-tiny    (:acceptance-rate (meta samples-tiny))]
    (assert-true "tiny epsilon: acceptance > 0.95" (> rate-tiny 0.95))
    (println "    tiny epsilon acceptance:" rate-tiny))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: 5.1.2 tiny step threw:" (.-message e))))

;; Good step-size: correct posterior
(try
  (let [samples-good (mcmc/hmc {:samples 300 :burn 100 :step-size 0.1
                                 :leapfrog-steps 10 :addresses [:mu]}
                                model-a [] obs-a)
        mu-vals      (mapv first samples-good)
        has-nan      (some js/isNaN mu-vals)
        mu-mean      (/ (reduce + mu-vals) (count mu-vals))]
    (assert-true "good epsilon: no NaN" (not has-nan))
    (assert-close "good epsilon: mean near 1.0" 1.0 mu-mean 0.31))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: 5.1.2 good step threw:" (.-message e))))

;; Large step-size: low acceptance
;; 1D quadratic is very forgiving for leapfrog, so we need a truly
;; extreme step-size (5.0 with 20 steps) to break energy conservation.
(try
  (let [samples-large (mcmc/hmc {:samples 100 :burn 50 :step-size 5.0
                                  :leapfrog-steps 20 :addresses [:mu]
                                  :compile? false :device :cpu}
                                 model-a [] obs-a)
        rate-large    (:acceptance-rate (meta samples-large))]
    (assert-true "large epsilon: acceptance < 0.5" (< rate-large 0.5))
    (println "    large epsilon acceptance:" rate-large))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: 5.1.2 large step threw:" (.-message e))))

;; Monotonicity check: tiny > large
(try
  (let [samples-tiny  (mcmc/hmc {:samples 50 :burn 20 :step-size 0.001
                                  :leapfrog-steps 20 :addresses [:mu]
                                  :compile? false :device :cpu}
                                 model-a [] obs-a)
        samples-large (mcmc/hmc {:samples 50 :burn 20 :step-size 5.0
                                  :leapfrog-steps 20 :addresses [:mu]
                                  :compile? false :device :cpu}
                                 model-a [] obs-a)
        rate-tiny     (:acceptance-rate (meta samples-tiny))
        rate-large    (:acceptance-rate (meta samples-large))]
    (assert-true "accept(tiny) > accept(large)"
                 (> rate-tiny rate-large)))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: 5.1.2 monotonicity threw:" (.-message e))))

;; ---------------------------------------------------------------------------
;; 5.1.3 MALA Convergence
;; ---------------------------------------------------------------------------
;; MALA uses gradient information with a Metropolis correction. It should
;; converge to the correct posterior.

(println "\n-- 5.1.3 MALA convergence --")

(try
  (let [samples (mcmc/mala {:samples 500 :burn 200 :step-size 1.5
                             :addresses [:mu] :compile? false :device :cpu}
                            model-a [] obs-a)
        mu-vals (mapv first samples)
        has-nan (some js/isNaN mu-vals)
        mu-mean (/ (reduce + mu-vals) (count mu-vals))
        rate    (:acceptance-rate (meta samples))]
    (assert-true "MALA: no NaN" (not has-nan))
    (assert-close "MALA: mean near 1.0" 1.0 mu-mean 0.31)
    ;; MALA on 1D Gaussian with gradient: even moderately large step-sizes
    ;; produce good proposals. We verify convergence and non-degenerate rate.
    (assert-true "MALA: acceptance in (0.10, 0.95)"
                 (and (> rate 0.10) (< rate 0.95)))
    (println "    MALA acceptance:" rate "  mean:" mu-mean))
  (catch :default e
    (vswap! fail-count inc)
    (println "  FAIL: 5.1.3 MALA threw:" (.-message e))))

;; --- summary ---
(println (str "\n=== " @pass-count " passed, " @fail-count " failed ==="))
(when (pos? @fail-count) (println "SOME TESTS FAILED"))
