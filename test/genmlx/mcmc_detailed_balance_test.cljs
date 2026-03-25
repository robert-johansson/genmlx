(ns genmlx.mcmc-detailed-balance-test
  "Phase 5.1 -- MCMC stationarity tests.
   Verifies that MH preserves the stationary distribution (Geweke fixed-point),
   that HMC acceptance degrades with step-size, and that MALA converges to the
   correct posterior mean. Each test checks a structural property of the Markov
   chain, not just a posterior statistic."
  (:require [cljs.test :refer [deftest is testing run-tests]]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.kernel :as k]
            [genmlx.test-helpers :as th])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Model A: single-observation Normal-Normal
;; Prior: mu ~ N(0,1), Likelihood: y ~ N(mu,1), observe y=2
;; Posterior: N(1.0, 0.5), sigma=0.7071
;; ---------------------------------------------------------------------------

(def model-a
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 1))]
      (trace :y (dist/gaussian mu 1))
      mu)))

(def obs-a (cm/choicemap :y (mx/scalar 2.0)))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- box-muller-draw
  "Draw one sample from N(mu, sigma) via Box-Muller on two MLX uniform keys."
  [k1 k2 mu sigma]
  (let [u1 (mx/item (rng/uniform k1 [] 0.0 1.0))
        u2 (mx/item (rng/uniform k2 [] 0.0 1.0))
        z  (* (js/Math.sqrt (* -2.0 (js/Math.log u1)))
              (js/Math.cos (* 2.0 js/Math.PI u2)))]
    (+ mu (* sigma z))))

(defn- mh-fixed-point-sample
  "Draw from analytical posterior, create trace, apply 1 MH step, return mu."
  [model-keyed kernel ki post-mean post-sigma]
  (let [[k1 k2 k3]  (rng/split-n ki 3)
        draw         (box-muller-draw k1 k2 post-mean post-sigma)
        constraints  (cm/choicemap :mu (mx/scalar draw) :y (mx/scalar 2.0))
        {:keys [trace]} (p/generate model-keyed [] constraints)
        trace'       (kernel trace k3)
        v            (cm/get-value (cm/get-submap (:choices trace') :mu))]
    (mx/eval! v)
    (mx/item v)))

(defn- hmc-mu-samples
  "Run HMC, extract :mu values as JS numbers."
  [opts]
  (let [samples (mcmc/hmc opts model-a [] obs-a)]
    (with-meta (mapv first samples) (meta samples))))

;; ---------------------------------------------------------------------------
;; 5.1.1 MH Fixed-Point Test (Geweke-style stationarity)
;; ---------------------------------------------------------------------------
;; Draw N=2000 from the analytical posterior N(1.0, 0.5). Create a trace at
;; each value via p/generate. Apply 1 MH step via kernel. If MH preserves
;; the stationary distribution, the output ensemble has the same mean and
;; variance as the input.

(deftest mh-fixed-point-test
  (testing "MH preserves posterior mean under Geweke stationarity"
    ;; SE derivation: posterior variance=0.5, N=2000
    ;;   SE(mean) = sqrt(0.5/2000) = 0.0158
    ;;   tolerance = z * safety * SE = 3.5 * 1.5 * 0.0158 = 0.083
    (let [n           2000
          post-mean   1.0
          post-sigma  0.7071067811865476
          model-keyed (dyn/auto-key model-a)
          kernel      (k/random-walk :mu 0.5)
          keys-bm     (rng/split-n (rng/fresh-key) n)
          output-vals (mapv #(mh-fixed-point-sample model-keyed kernel %
                                                    post-mean post-sigma)
                            keys-bm)
          output-mean (th/sample-mean output-vals)]
      (is (th/close? 1.0 output-mean 0.083)
          "posterior mean preserved after 1 MH step")))

  (testing "MH preserves posterior variance under Geweke stationarity"
    ;; SE derivation: Var(s^2) ~ 2*sigma^4/(n-1) for Normal
    ;;   sigma^2 = 0.5, so Var(s^2) = 2*0.25/1999 = 0.00025
    ;;   SE(variance) = sqrt(0.00025) = 0.0158
    ;;   but MH correlation inflates effective variance --
    ;;   conservatively SE ~ 0.0112 (ESS ~ 0.5*N)
    ;;   tolerance = z * safety * SE = 3.5 * 1.5 * 0.0112 = 0.059
    (let [n           2000
          post-mean   1.0
          post-sigma  0.7071067811865476
          model-keyed (dyn/auto-key model-a)
          kernel      (k/random-walk :mu 0.5)
          keys-bm     (rng/split-n (rng/fresh-key) n)
          output-vals (mapv #(mh-fixed-point-sample model-keyed kernel %
                                                    post-mean post-sigma)
                            keys-bm)
          output-var  (th/sample-variance output-vals)]
      (is (th/close? 0.5 output-var 0.059)
          "posterior variance preserved after 1 MH step"))))

;; ---------------------------------------------------------------------------
;; 5.1.2 HMC Step-Size Sensitivity
;; ---------------------------------------------------------------------------
;; Three step sizes bracket the sweet spot. Tiny epsilon -> near-perfect
;; acceptance but slow exploration. Good epsilon -> correct posterior.
;; Large epsilon -> low acceptance (energy errors).

(deftest hmc-step-size-sensitivity-test
  (testing "tiny epsilon: acceptance > 0.95"
    (let [samples  (mcmc/hmc {:samples 100 :burn 50 :step-size 0.001
                              :leapfrog-steps 20 :addresses [:mu]
                              :compile? false :device :cpu}
                             model-a [] obs-a)
          rate     (:acceptance-rate (meta samples))]
      (is (> rate 0.95)
          (str "tiny epsilon acceptance " rate " should exceed 0.95"))))

  (testing "good epsilon: posterior mean near 1.0"
    ;; SE derivation: 300 post-burn HMC samples, posterior var=0.5
    ;;   SE(mean) = sqrt(0.5/300) = 0.041
    ;;   tolerance = z * SE = 3.5 * 0.041 = 0.143
    (let [mu-vals  (hmc-mu-samples {:samples 300 :burn 100 :step-size 0.1
                                    :leapfrog-steps 10 :addresses [:mu]})
          mu-mean  (th/sample-mean mu-vals)]
      (is (not (some js/isNaN mu-vals))
          "good epsilon: no NaN in samples")
      (is (th/close? 1.0 mu-mean 0.143)
          (str "good epsilon: mean " mu-mean " near 1.0"))))

  (testing "large epsilon: acceptance < 0.5"
    ;; 1D quadratic is forgiving for leapfrog, so we need a truly extreme
    ;; step-size (5.0 with 20 steps) to break energy conservation.
    (let [samples  (mcmc/hmc {:samples 100 :burn 50 :step-size 5.0
                              :leapfrog-steps 20 :addresses [:mu]
                              :compile? false :device :cpu}
                             model-a [] obs-a)
          rate     (:acceptance-rate (meta samples))]
      (is (< rate 0.5)
          (str "large epsilon acceptance " rate " should be below 0.5"))))

  (testing "acceptance monotonicity: tiny > large"
    (let [tiny-samples  (mcmc/hmc {:samples 50 :burn 20 :step-size 0.001
                                   :leapfrog-steps 20 :addresses [:mu]
                                   :compile? false :device :cpu}
                                  model-a [] obs-a)
          large-samples (mcmc/hmc {:samples 50 :burn 20 :step-size 5.0
                                   :leapfrog-steps 20 :addresses [:mu]
                                   :compile? false :device :cpu}
                                  model-a [] obs-a)
          rate-tiny     (:acceptance-rate (meta tiny-samples))
          rate-large    (:acceptance-rate (meta large-samples))]
      (is (> rate-tiny rate-large)
          "acceptance rate decreases with step-size"))))

;; ---------------------------------------------------------------------------
;; 5.1.3 MALA Convergence
;; ---------------------------------------------------------------------------
;; MALA uses gradient information with a Metropolis correction. It should
;; converge to the correct posterior.

(deftest mala-convergence-test
  (testing "MALA converges to correct posterior mean"
    ;; SE derivation: 500 samples, ESS ~ 200 (autocorrelation ~0.6)
    ;;   SE(mean) = sqrt(0.5/200) = 0.050
    ;;   tolerance = z * SE = 3.5 * 0.050 = 0.175
    (let [samples  (mcmc/mala {:samples 500 :burn 200 :step-size 1.5
                               :addresses [:mu] :compile? false :device :cpu}
                              model-a [] obs-a)
          mu-vals  (mapv first samples)
          rate     (:acceptance-rate (meta samples))
          mu-mean  (th/sample-mean mu-vals)]
      (is (not (some js/isNaN mu-vals))
          "MALA: no NaN in samples")
      (is (th/close? 1.0 mu-mean 0.175)
          (str "MALA: mean " mu-mean " near 1.0"))
      (is (and (> rate 0.10) (< rate 0.95))
          (str "MALA: acceptance " rate " in (0.10, 0.95)")))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(run-tests)
