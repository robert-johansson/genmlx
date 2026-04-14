(ns genmlx.inference-convergence-test
  "Inference convergence tests on conjugate models.
   Normal-Normal conjugate: prior N(0,1), likelihood N(x,1), obs y=2.
   Posterior: N(1.0, 0.5). Log-ML: log N(2; 0, sqrt(2)) = -2.2655.
   Gamma-Poisson conjugate: prior Gamma(3,1), likelihood Poisson(lambda).
   HMC/NUTS acceptance rate tests."
  (:require [cljs.test :refer [deftest is are testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.kernel :as kern]
            [genmlx.inference.vi :as vi])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Gamma-Poisson conjugate
;; Prior:      lambda ~ Gamma(alpha=3, rate=1)
;; Likelihood: x_i ~ Poisson(lambda), i=1..5
;; Data:       [2, 4, 3, 5, 1] (sum=15)
;; Posterior:  Gamma(alpha + sum = 18, rate + n = 6)
;; E[lambda|data] = 18/6 = 3.0
;; ---------------------------------------------------------------------------

(def gamma-poisson-model
  (gen [data]
       (let [lam (trace :lambda (dist/gamma-dist 3 1))]
         (mx/eval! lam)
         (let [lam-val (mx/item lam)]
           (doseq [[i x] (map-indexed vector data)]
             (trace (keyword (str "x" i)) (dist/poisson lam-val)))
           lam-val))))

(def gp-data [2 4 3 5 1])
(def gp-observations
  (reduce (fn [cm [i x]]
            (cm/set-choice cm [(keyword (str "x" i))] (mx/scalar x)))
          cm/EMPTY (map-indexed vector gp-data)))

(deftest gamma-poisson-is-test
  (testing "IS (100 particles)"
    (let [{:keys [traces log-weights]} (is/importance-sampling
                                        {:samples 100} gamma-poisson-model [gp-data] gp-observations)
          raw-weights (mapv (fn [w] (mx/eval! w) (mx/item w)) log-weights)
          max-w (apply max raw-weights)
          exp-weights (mapv (fn [w] (js/Math.exp (- w max-w))) raw-weights)
          sum-w (reduce + exp-weights)
          norm-weights (mapv (fn [w] (/ w sum-w)) exp-weights)
          lambda-vals (mapv :retval traces)
          weighted-mean (reduce + (map * lambda-vals norm-weights))]
      (is (h/close? 3.0 weighted-mean 0.5) "IS posterior mean(lambda) ~ 3.0")
      (is (= 100 (count traces)) "IS returns 100 traces"))))

(deftest gamma-poisson-mh-test
  (testing "MH (500 samples)"
    (let [model-mh (gen [data]
                        (let [lam (trace :lambda (dist/gamma-dist 3 1))]
                          (mx/eval! lam)
                          (let [lam-val (mx/item lam)]
                            (doseq [[i x] (map-indexed vector data)]
                              (trace (keyword (str "x" i)) (dist/poisson lam-val)))
                            lam-val)))
          traces (mcmc/mh {:samples 500 :burn 200 :selection (sel/select :lambda)}
                          model-mh [gp-data] gp-observations)
          lambda-vals (mapv (fn [t]
                              (let [v (cm/get-value (cm/get-submap (:choices t) :lambda))]
                                (mx/eval! v) (mx/item v)))
                            traces)
          lambda-mean (/ (reduce + lambda-vals) (count lambda-vals))]
      (is (h/close? 3.0 lambda-mean 0.5) "MH posterior mean(lambda) ~ 3.0")
      (is (= 500 (count traces)) "MH returns 500 traces"))))

;; ---------------------------------------------------------------------------
;; HMC/NUTS acceptance rate
;; ---------------------------------------------------------------------------

(def normal-normal
  (gen []
       (let [mu (trace :mu (dist/gaussian 0 10))]
         (doseq [i (range 5)]
           (trace (keyword (str "obs" i)) (dist/gaussian mu 1)))
         mu)))

(def nn-observations
  (reduce (fn [cm i]
            (cm/set-choice cm [(keyword (str "obs" i))]
                           (mx/scalar (+ 3.0 (* 0.1 (- i 2))))))
          cm/EMPTY (range 5)))

(deftest hmc-acceptance-test
  (testing "HMC acceptance rate"
    (let [samples (mcmc/hmc {:samples 100 :burn 50 :step-size 0.05 :leapfrog-steps 10
                             :addresses [:mu] :compile? false :device :cpu}
                            normal-normal [] nn-observations)
          mu-vals (mapv first samples)
          has-nan (some js/isNaN mu-vals)
          n-unique (count (set mu-vals))
          acceptance-rate (/ (double n-unique) (count mu-vals))]
      (is (not has-nan) "HMC: no NaN in samples")
      (is (> acceptance-rate 0.3) "HMC: acceptance rate > 0.3")
      (is (h/close? 3.0 (/ (reduce + mu-vals) (count mu-vals)) 1.0)
          "HMC: posterior mean ~ 3"))))

(deftest nuts-acceptance-test
  (testing "NUTS acceptance rate"
    (let [samples (mcmc/nuts {:samples 50 :burn 50 :step-size 0.05
                              :addresses [:mu] :compile? false :device :cpu}
                             normal-normal [] nn-observations)
          mu-vals (mapv first samples)
          has-nan (some js/isNaN mu-vals)
          n-unique (count (set mu-vals))
          acceptance-rate (/ (double n-unique) (count mu-vals))]
      (is (not has-nan) "NUTS: no NaN in samples")
      (is (> acceptance-rate 0.3) "NUTS: acceptance rate > 0.3")
      (is (h/close? 3.0 (/ (reduce + mu-vals) (count mu-vals)) 1.0)
          "NUTS: posterior mean ~ 3"))))

;; ===========================================================================
;; Normal-Normal conjugate model
;; Prior:      x ~ N(0, 1)
;; Likelihood: y ~ N(x, 1)
;; Observed:   y = 2.0
;; Posterior:  x | y=2 ~ N(1.0, 0.5)
;;   posterior mean = 1.0, posterior variance = 0.5
;; Log marginal likelihood:
;;   log p(y=2) = log N(2; 0, sqrt(2)) = -2.2655
;; ===========================================================================

(def conjugate-model
  (gen []
       (let [x (trace :x (dist/gaussian 0 1))]
         (trace :y (dist/gaussian x 1))
         x)))

(def conjugate-obs (cm/choicemap :y (mx/scalar 2.0)))

;; Analytical posterior values
(def ^:private posterior-mean 1.0)
(def ^:private posterior-variance 0.5)
(def ^:private posterior-sigma (js/Math.sqrt posterior-variance))
(def ^:private expected-log-ml (h/gaussian-lp 2.0 0.0 (js/Math.sqrt 2.0)))

;; Shared tolerance: 0.3 accommodates Monte Carlo noise at ~500 effective samples
(def ^:private mean-tol 0.3)
(def ^:private log-ml-tol 0.3)

(defn- extract-x
  "Extract the scalar x-value from a conjugate model trace."
  [trace]
  (-> (:choices trace) (cm/get-choice [:x]) mx/item))

;; ---------------------------------------------------------------------------
;; 1. importance-resampling
;; ---------------------------------------------------------------------------

(deftest importance-resampling-converges-to-posterior
  (testing "Resampled particles have posterior mean ~ 1.0"
    (let [traces (is/importance-resampling
                  {:samples 200 :particles 500 :key (rng/fresh-key 42)}
                  conjugate-model [] conjugate-obs)
          x-vals (mapv extract-x traces)]
      (is (h/z-test-passes? posterior-mean x-vals)
          "importance-resampling posterior mean within 3.5 SE of 1.0"))))

;; ---------------------------------------------------------------------------
;; 2. mh-custom (custom proposal MH)
;; ---------------------------------------------------------------------------

(def ^:private rw-proposal
  (gen [current-choices]
       (let [cur-x (cm/get-choice current-choices [:x])]
         (trace :x (dist/gaussian cur-x 0.5)))))

(deftest mh-custom-converges-to-posterior
  (testing "Custom-proposal MH chain mean ~ 1.0"
    ;; MCMC chains have autocorrelation, so N_eff << N_samples.
    ;; Use h/close? with tolerance 0.3 (matches gfi_laws_test MH pattern).
    (let [traces (mcmc/mh-custom
                  {:samples 1000 :burn 500
                   :proposal-gf rw-proposal
                   :key (rng/fresh-key 99)}
                  conjugate-model [] conjugate-obs)
          x-vals (mapv extract-x traces)
          mean-x (/ (reduce + x-vals) (count x-vals))]
      (is (h/close? mean-x posterior-mean 0.3)
          (str "mh-custom posterior mean=" mean-x " expected=" posterior-mean)))))

;; ---------------------------------------------------------------------------
;; 3. smcp3
;; TODO: smcp3 is designed for sequential models with extend steps.
;;       The Normal-Normal conjugate model is single-observation, not sequential.
;;       A proper test would require an unfold/time-series model, which
;;       goes beyond the scope of this convergence suite.
;; ---------------------------------------------------------------------------

;; ---------------------------------------------------------------------------
;; 4. compiled-smc
;; TODO: compiled-smc operates on unfold-combinator models with
;;       pre-extracted kernel schemas and noise transforms. It requires
;;       a compiled unfold model with tensor traces, not a vanilla
;;       DynamicGF. Testing this on the conjugate model is architecturally
;;       infeasible without a dedicated unfold setup.
;; ---------------------------------------------------------------------------

;; ---------------------------------------------------------------------------
;; 5. compiled-vi-from-model
;; ---------------------------------------------------------------------------

(deftest compiled-vi-converges-to-posterior
  (testing "Compiled VI recovers posterior mean ~ 1.0, sigma ~ 0.707"
    (let [{:keys [mu sigma]} (vi/compiled-vi-from-model
                              {:iterations 500 :learning-rate 0.02
                               :elbo-samples 10 :key (rng/fresh-key 42)}
                              conjugate-model [] conjugate-obs [:x])
          mu-val (mx/item mu)
          sig-val (mx/item sigma)]
      (is (h/close? posterior-mean mu-val mean-tol)
          "compiled-vi posterior mean ~ 1.0")
      (is (h/close? posterior-sigma sig-val mean-tol)
          "compiled-vi posterior sigma ~ 0.707"))))

;; ---------------------------------------------------------------------------
;; 6. programmable-vi (ELBO objective)
;; ---------------------------------------------------------------------------

(defn- conjugate-log-joint
  "log p(x, y=2) = log N(x; 0,1) + log N(2; x,1)"
  [x-arr]
  (let [x (if (pos? (mx/ndim x-arr)) (mx/index x-arr 0) x-arr)]
    (mx/add (dist/log-prob (dist/gaussian 0 1) x)
            (dist/log-prob (dist/gaussian x 1) (mx/scalar 2.0)))))

(defn- guide-log-prob
  "log q(x; mu, exp(log-sigma)) = log N(x; mu, exp(log-sigma))"
  [x-arr params]
  (let [x (if (pos? (mx/ndim x-arr)) (mx/index x-arr 0) x-arr)
        mu (mx/index params 0)
        sigma (mx/exp (mx/index params 1))]
    (dist/log-prob (dist/gaussian mu sigma) x)))

(defn- guide-sample
  "Draw n reparameterized samples from q(x; params)."
  [params key n]
  (let [mu (mx/index params 0)
        sigma (mx/exp (mx/index params 1))
        eps (rng/normal (rng/ensure-key key) [n 1])]
    (mx/add mu (mx/multiply sigma eps))))

(deftest programmable-vi-converges-to-posterior
  (testing "Programmable VI with ELBO recovers posterior mean ~ 1.0"
    (let [{:keys [params]} (vi/programmable-vi
                            {:iterations 500 :learning-rate 0.02
                             :n-samples 10 :objective :elbo
                             :key (rng/fresh-key 42)}
                            conjugate-log-joint guide-log-prob
                            guide-sample (mx/array [0.0 0.0]))
          [mu-val log-sig] (mx/->clj params)
          sig-val (js/Math.exp log-sig)]
      (is (h/close? posterior-mean mu-val mean-tol)
          "programmable-vi posterior mean ~ 1.0")
      (is (h/close? posterior-sigma sig-val mean-tol)
          "programmable-vi posterior sigma ~ 0.707"))))

;; ---------------------------------------------------------------------------
;; 7. vectorized-importance-resampling
;; ---------------------------------------------------------------------------

(deftest vectorized-importance-resampling-converges-to-posterior
  (testing "Vectorized resampled particles have posterior mean ~ 1.0"
    (let [{:keys [vtrace]} (is/vectorized-importance-resampling
                            {:particles 500 :key (rng/fresh-key 42)}
                            conjugate-model [] conjugate-obs)
          x-vals (-> (:choices vtrace) (cm/get-choice [:x]) mx/->clj)]
      (is (h/z-test-passes? posterior-mean x-vals)
          "vectorized-importance-resampling posterior mean within 3.5 SE of 1.0"))))

;; ---------------------------------------------------------------------------
;; 8. tidy-importance-sampling (log-ML only)
;; ---------------------------------------------------------------------------

(deftest tidy-importance-sampling-estimates-log-ml
  (testing "Log-ML estimate matches analytical value"
    (let [{:keys [log-ml-estimate log-weights]}
          (is/tidy-importance-sampling
           {:samples 1000 :key (rng/fresh-key 42)}
           conjugate-model [] conjugate-obs)]
      (is (h/finite? log-ml-estimate)
          "tidy-importance-sampling log-ML is finite")
      (is (h/close? expected-log-ml log-ml-estimate log-ml-tol)
          "tidy-importance-sampling log-ML ~ -2.2655")
      (is (= 1000 (count log-weights))
          "tidy-importance-sampling returns all weights"))))

;; ---------------------------------------------------------------------------
;; 9. kernel-based proposal MH
;; ---------------------------------------------------------------------------

(deftest kernel-proposal-converges-to-posterior
  (testing "Kernel random-walk chain mean ~ 1.0"
    (let [model (dyn/auto-key conjugate-model)
          k (kern/random-walk :x 0.5)
          {:keys [trace]} (p/generate model [] conjugate-obs)
          traces (kern/run-kernel
                  {:samples 500 :burn 200 :key (rng/fresh-key 42)}
                  k trace)
          x-vals (mapv extract-x traces)]
      (is (h/z-test-passes? posterior-mean x-vals)
          "kernel random-walk posterior mean within 3.5 SE of 1.0")
      (is (pos? (:acceptance-rate (meta traces)))
          "kernel random-walk has nonzero acceptance rate"))))

(cljs.test/run-tests)
