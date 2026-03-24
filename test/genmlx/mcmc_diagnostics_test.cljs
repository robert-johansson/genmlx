(ns genmlx.mcmc-diagnostics-test
  "Phase 5.3 -- MCMC diagnostic infrastructure tests.
   Verifies R-hat convergence, effective sample size bounds, and
   acceptance rate calibration for MH, HMC, and MALA on analytically
   tractable Normal-Normal models.

   Models:
     model-a  -- N(0,1) prior, y=2, posterior N(1.0, 0.5)
     model-b  -- N(0,10) prior, 5 obs, posterior N(2.994, 0.1996)
     model-5d -- 5 independent N(0,10) priors, 1 obs each,
                 x_j posterior N(y_j*100/101, 100/101 = 0.9901)"
  (:require [cljs.test :refer [deftest is testing run-tests]]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.diagnostics :as diag]
            [genmlx.test-helpers :as th])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

;; Model A: single-observation Normal-Normal
;; Prior: mu ~ N(0, 1). Likelihood: y ~ N(mu, 1). Observation: y = 2.
;; Posterior: N(1.0, 0.5), sigma_post = sqrt(0.5) = 0.707
(def model-a
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 1))]
      (trace :y (dist/gaussian mu 1))
      mu)))

(def obs-a (cm/choicemap :y (mx/scalar 2.0)))

;; Model B: multi-observation Normal-Normal
;; Prior: mu ~ N(0, 10). Likelihood: y_j ~ N(mu, 1), j=0..4.
;; Posterior precision = 1/100 + 5/1 = 5.01, variance = 1/5.01 = 0.1996
;; Posterior mean = (0/100 + sum(y)/1) / 5.01 = 14.97/5.01 = 2.994
(def model-b
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (doseq [j (range 5)]
        (trace (keyword (str "y" j)) (dist/gaussian mu 1)))
      mu)))

(def obs-b
  (let [ys [2.8 3.1 2.9 3.2 3.0]]
    (reduce (fn [acc [j y]]
              (cm/set-choice acc [(keyword (str "y" j))] (mx/scalar y)))
            cm/EMPTY
            (map-indexed vector ys))))

;; Model 5D: 5 independent Normal-Normal pairs
;; Prior: x_j ~ N(0, 10), j=0..4. Likelihood: y_j ~ N(x_j, 1).
;; Each x_j has ONE observation, so posterior precision = 1/100 + 1/1 = 1.01
;; Posterior variance = 1/1.01 = 0.9901, sigma_post = sqrt(0.9901) = 0.995
;; Posterior mean for x0: (0/100 + 2.5/1) / 1.01 = 2.475
(def model-5d
  (gen []
    (let [xs (mapv (fn [j] (trace (keyword (str "x" j)) (dist/gaussian 0 10)))
                   (range 5))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j)) (dist/gaussian x 1)))
      (first xs))))

(def obs-5d
  (reduce (fn [acc [j y]]
            (cm/set-choice acc [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector [2.5 3.0 1.5 2.0 2.8])))

(def addrs-5d [:x0 :x1 :x2 :x3 :x4])

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- samples->mlx-arrays
  "Convert compiled-mh output (vector of JS arrays) to vector of MLX scalars.
   For a 1-parameter model, each sample is a JS array like [0.75]."
  [samples]
  (mapv #(mx/scalar (first %)) samples))

;; ---------------------------------------------------------------------------
;; 5.3.1 R-hat convergence
;; ---------------------------------------------------------------------------

(deftest r-hat-convergence
  (testing "4 independent MH chains on Model B converge to the same posterior"
    (let [run-chain  (fn [seed]
                       (mcmc/compiled-mh
                         {:samples 1000 :burn 500 :addresses [:mu]
                          :proposal-std 0.3 :key (rng/fresh-key seed)}
                         model-b [] obs-b))
          chains     (mapv run-chain [42 137 271 999])
          chain-vals (mapv (fn [samples] (mapv first samples)) chains)
          chain-means (mapv th/sample-mean chain-vals)
          mlx-chains  (mapv samples->mlx-arrays chains)
          r           (diag/r-hat mlx-chains)]
      (is (>= r 1.0)
          "R-hat >= 1.0 by construction (between-chain variance is non-negative)")
      (is (< r 1.1)
          "R-hat < 1.1 indicates well-converged chains")
      ;; SE derivation: 1000 post-burn samples, ESS > 300 (well-tuned MH),
      ;; sigma_post = sqrt(0.1996) = 0.447, SE = 0.447/sqrt(300) = 0.026.
      ;; At z=3.5: tolerance = 3.5 * 0.026 = 0.091. Use 0.10 for margin.
      (doseq [[i mu] (map-indexed vector chain-means)]
        (is (th/close? 2.994 mu 0.10)
            (str "chain " i " mean near posterior mean 2.994"))))))

;; ---------------------------------------------------------------------------
;; 5.3.2 Effective sample size
;; ---------------------------------------------------------------------------

(deftest effective-sample-size
  (testing "ESS is a substantial fraction of nominal sample count"
    (let [;; proposal-std 1.0 on sigma_post = sqrt(0.1996) = 0.447
          ;; gives ratio 1.0/0.447 = 2.24, near optimal 2.38 for d=1.
          ;; Expect acceptance ~ 0.44 and ESS/N ~ 0.2-0.4, so ESS > N/10 = 200.
          samples   (mcmc/compiled-mh {:samples 2000 :burn 500 :addresses [:mu]
                                        :proposal-std 1.0 :key (rng/fresh-key 77)}
                                       model-b [] obs-b)
          mlx-samps (samples->mlx-arrays samples)
          ess-val   (diag/ess mlx-samps)]
      (is (js/isFinite ess-val)
          "ESS is finite")
      (is (pos? ess-val)
          "ESS is positive")
      (is (> ess-val 200)
          "ESS > N/10 = 200 for well-tuned MH")
      (is (< ess-val 2000)
          "ESS < N because chain is autocorrelated"))))

;; ---------------------------------------------------------------------------
;; 5.3.3 Acceptance rate calibration
;; ---------------------------------------------------------------------------
;; Roberts et al. (1997): optimal acceptance rate for d=1 Gaussian target
;; is 0.44. HMC and MALA on a 1D quadratic are too easy -- leapfrog nearly
;; conserves energy and the gradient-adjusted proposal is very accurate --
;; so acceptance is often > 0.90. We use wide ranges here because the
;; asymptotic optima (HMC ~ 0.65, MALA ~ 0.574) require d >> 5.

(deftest mh-acceptance-calibration
  (testing "MH acceptance rate and posterior mean on Model A"
    ;; proposal-std=2.0 on sigma_post=sqrt(0.5)=0.707 gives ratio
    ;; 2.0/0.707 = 2.83 (above the optimal 2.38 for d=1)
    (let [samples (mcmc/compiled-mh {:samples 500 :burn 200 :addresses [:mu]
                                      :proposal-std 2.0 :compile? false
                                      :key (rng/fresh-key 101)}
                                     model-a [] obs-a)
          rate    (:acceptance-rate (meta samples))
          mu-vals (mapv first samples)]
      (is (some? rate)
          "acceptance rate is present in metadata")
      ;; Optimal acceptance for d=1 Gaussian: 0.44 (Roberts et al. 1997).
      ;; With ratio 2.83 we expect slightly lower, range (0.30, 0.55).
      (is (and (> rate 0.30) (< rate 0.55))
          "acceptance rate in (0.30, 0.55) near 0.44 optimal for d=1")
      ;; SE derivation: 500 samples, ESS ~ 150 (autocorrelated),
      ;; sigma_post = 0.707, SE = 0.707/sqrt(150) = 0.058.
      ;; At z=3.5: tolerance = 3.5 * 0.058 = 0.20.
      (is (th/close? 1.0 (th/sample-mean mu-vals) 0.20)
          "posterior mean near analytical 1.0"))))

(deftest hmc-acceptance-calibration
  (testing "HMC acceptance rate and posterior mean on Model A"
    ;; 1D quadratic is too easy for HMC: leapfrog nearly conserves
    ;; Hamiltonian energy, so acceptance is very high (often > 0.95).
    ;; Asymptotic optimal ~ 0.65 only manifests at d >> 5.
    (let [samples (mcmc/hmc {:samples 200 :burn 100 :step-size 0.1
                               :leapfrog-steps 10 :addresses [:mu]
                               :compile? false :device :cpu
                               :key (rng/fresh-key 202)}
                              model-a [] obs-a)
          rate    (:acceptance-rate (meta samples))
          mu-vals (mapv first samples)]
      (is (some? rate)
          "acceptance rate is present in metadata")
      (is (and (> rate 0.40) (<= rate 1.0))
          "acceptance rate in (0.40, 1.0) -- wide because d=1 is too easy")
      ;; SE derivation: 200 samples, near-independent with HMC,
      ;; sigma_post = 0.707, SE = 0.707/sqrt(200) = 0.050.
      ;; At z=3.5: tolerance = 3.5 * 0.050 = 0.175.
      (is (th/close? 1.0 (th/sample-mean mu-vals) 0.175)
          "posterior mean near analytical 1.0"))))

(deftest mala-acceptance-calibration
  (testing "MALA acceptance rate and posterior mean on Model A"
    ;; 1D quadratic with step-size 0.5: the gradient-adjusted proposal is
    ;; very accurate, yielding acceptance > 0.90 typically. Asymptotic
    ;; optimal ~ 0.574 only manifests at d >> 5.
    (let [samples (mcmc/mala {:samples 200 :burn 100 :step-size 0.5
                                :addresses [:mu] :compile? false :device :cpu
                                :key (rng/fresh-key 303)}
                               model-a [] obs-a)
          rate    (:acceptance-rate (meta samples))
          mu-vals (mapv first samples)]
      (is (some? rate)
          "acceptance rate is present in metadata")
      (is (and (> rate 0.20) (<= rate 1.0))
          "acceptance rate in (0.20, 1.0) -- wide because d=1 is too easy")
      ;; SE derivation: 200 samples, ESS ~ 150 (MALA is correlated),
      ;; sigma_post = 0.707, SE = 0.707/sqrt(150) = 0.058.
      ;; At z=3.5: tolerance = 3.5 * 0.058 = 0.20.
      (is (th/close? 1.0 (th/sample-mean mu-vals) 0.20)
          "posterior mean near analytical 1.0"))))

;; ---------------------------------------------------------------------------
;; 5.3.4 Adaptive acceptance on 5D model
;; ---------------------------------------------------------------------------
;; 1D quadratic targets are too easy for HMC/MALA (acceptance ~ 1.0).
;; A 5D model with dual averaging adaptation tests that:
;;   (a) the adaptation mechanism functions (rate is non-trivial)
;;   (b) rates are in sensible ranges for gradient-based methods
;; Note: asymptotic optima (HMC ~ 0.65, MALA ~ 0.574) require d >> 5,
;; so at d=5 we still see higher acceptance than the limiting values.

(deftest hmc-5d-adaptive-acceptance
  (testing "HMC with dual averaging on 5D model"
    (let [samples (mcmc/hmc {:samples 200 :burn 200 :step-size 0.5 :leapfrog-steps 10
                               :addresses addrs-5d :compile? false :device :cpu
                               :adapt-step-size true :target-accept 0.65
                               :key (rng/fresh-key 555)}
                              model-5d [] obs-5d)
          rate    (:acceptance-rate (meta samples))
          x0-vals (mapv first samples)]
      (is (some? rate)
          "acceptance rate is present in metadata")
      ;; Non-trivial range: adaptation should bring rate into (0.50, 0.98).
      ;; At d=5 the rate is typically higher than the d->inf asymptote of 0.65.
      (is (and (> rate 0.50) (< rate 0.98))
          "acceptance rate in (0.50, 0.98) -- non-trivial with adaptation")
      ;; x0 posterior: N(2.475, 0.9901), sigma_post = 0.995
      ;; (Each x_j has a SINGLE observation, so posterior variance = 1/1.01 = 0.9901,
      ;;  NOT 0.1996 which is Model B's variance from 5 observations on a shared mean.)
      ;; SE derivation: 200 samples, ESS ~ 180 (HMC is efficient),
      ;; sigma_post = 0.995, SE = 0.995/sqrt(180) = 0.074.
      ;; At z=3.5: tolerance = 3.5 * 0.074 = 0.26.
      (is (th/close? 2.475 (th/sample-mean x0-vals) 0.26)
          "x0 mean near posterior mean 2.475"))))

(deftest mala-5d-adaptive-acceptance
  (testing "MALA with dual averaging on 5D model"
    (let [samples (mcmc/mala {:samples 200 :burn 200 :step-size 0.5
                                :addresses addrs-5d :compile? false :device :cpu
                                :adapt-step-size true :target-accept 0.574
                                :key (rng/fresh-key 505)}
                               model-5d [] obs-5d)
          rate    (:acceptance-rate (meta samples))
          x0-vals (mapv first samples)]
      (is (some? rate)
          "acceptance rate is present in metadata")
      ;; MALA at d=5: typically higher acceptance than the d->inf limit of 0.574.
      (is (and (> rate 0.40) (< rate 0.95))
          "acceptance rate in (0.40, 0.95) -- non-trivial with adaptation")
      ;; x0 posterior: N(2.475, 0.9901), sigma_post = 0.995
      ;; SE derivation: 200 samples, ESS ~ 150 (MALA less efficient than HMC),
      ;; sigma_post = 0.995, SE = 0.995/sqrt(150) = 0.081.
      ;; At z=3.5: tolerance = 3.5 * 0.081 = 0.28. Use 0.26 conservatively.
      (is (th/close? 2.475 (th/sample-mean x0-vals) 0.26)
          "x0 mean near posterior mean 2.475"))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(run-tests)
