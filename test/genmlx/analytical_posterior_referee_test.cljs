(ns genmlx.analytical-posterior-referee-test
  "Compare inference output to closed-form conjugate posteriors.
   The analytical posterior serves as referee — neither system's
   stochastic output is treated as ground truth.

   CORRECTNESS_PLAN section 1.3: analytical posterior as referee.

   Four conjugate families:
     1. Normal-Normal (known variance)
     2. Beta-Bernoulli
     3. Gamma-Poisson
     4. Normal-InverseGamma (unknown mean and variance)

   Each family tests IS convergence to the analytically derived
   posterior mean (and sometimes variance). MH is tested for
   Normal-Normal as a second inference algorithm."
  (:require [cljs.test :refer [deftest is testing run-tests]]
            [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.selection :as sel]
            [genmlx.test-helpers :as th])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- realize
  "Evaluate an MLX array and extract its JS number."
  [x]
  (mx/eval! x)
  (mx/item x))

(defn- observations-from-indexed
  "Build a choicemap :y0, :y1, ... from sequential data."
  [data]
  (apply cm/choicemap
         (mapcat (fn [i x] [(keyword (str "y" i)) (mx/scalar x)])
                 (range) data)))

(defn- normalize-log-weights
  "Convert log-weights to normalized probability weights (as JS numbers)."
  [log-weights]
  (let [ws (mapv realize log-weights)
        max-w (apply max ws)
        exp-ws (mapv #(js/Math.exp (- % max-w)) ws)
        total (reduce + exp-ws)]
    (mapv #(/ % total) exp-ws)))

(defn- weighted-mean
  "Compute weighted mean of a scalar address from IS results."
  [{:keys [traces log-weights]} addr]
  (let [norm-ws (normalize-log-weights log-weights)
        vals (mapv (fn [tr]
                     (-> (:choices tr) (cm/get-submap addr) cm/get-value realize))
                   traces)]
    (reduce + (map * norm-ws vals))))

(defn- weighted-variance
  "Compute weighted variance of a scalar address from IS results."
  [{:keys [traces log-weights]} addr expected-mean]
  (let [norm-ws (normalize-log-weights log-weights)
        vals (mapv (fn [tr]
                     (-> (:choices tr) (cm/get-submap addr) cm/get-value realize))
                   traces)]
    (reduce + (map (fn [w v] (* w (let [d (- v expected-mean)] (* d d))))
                   norm-ws vals))))

(defn- trace-samples
  "Extract a scalar address from a vector of traces as JS numbers."
  [traces addr]
  (mapv (fn [tr]
          (-> (:choices tr) (cm/get-submap addr) cm/get-value realize))
        traces))

;; ---------------------------------------------------------------------------
;; Ground truth: Normal-Normal (known variance)
;; ---------------------------------------------------------------------------
;;
;; Prior:      mu ~ N(0, 10)         => prior variance = 100
;; Likelihood: x_i ~ N(mu, 1)       => known variance = 1
;; Data:       [2.1, 1.8, 2.5, 1.9, 2.2]  (n=5, sum=10.5)
;;
;; Posterior:  N(mu_post, sigma_post^2) where
;;   sigma_post^2 = 1 / (1/100 + 5/1) = 1/5.01 = 0.199601...
;;   mu_post = sigma_post^2 * (0/100 + 10.5/1) = 0.199601 * 10.5 = 2.09581...

(def nn-ground-truth
  {:data [2.1 1.8 2.5 1.9 2.2]
   :prior-mean 0.0
   :prior-std 10.0
   :lik-std 1.0
   :posterior-mean (let [prior-prec (/ 1.0 100.0)
                         lik-prec (/ 5.0 1.0)
                         post-var (/ 1.0 (+ prior-prec lik-prec))]
                     (* post-var (+ (/ 0.0 100.0) (/ 10.5 1.0))))
   :posterior-variance (/ 1.0 (+ (/ 1.0 100.0) (/ 5.0 1.0)))})

;; ---------------------------------------------------------------------------
;; Ground truth: Beta-Bernoulli
;; ---------------------------------------------------------------------------
;;
;; Prior:      p ~ Beta(2, 2)
;; Likelihood: x_i ~ Bernoulli(p)
;; Data:       [1,1,1,0,1,0,1,1,0,1]  (7 successes, 3 failures)
;;
;; Posterior:  Beta(2+7, 2+3) = Beta(9, 5)
;; E[p] = 9/14 = 0.642857...
;; Var[p] = 9*5 / (14^2 * 15) = 45/2940 = 0.015306...

(def bb-ground-truth
  {:data [1 1 1 0 1 0 1 1 0 1]
   :alpha 2.0
   :beta-param 2.0
   :posterior-alpha 9.0
   :posterior-beta 5.0
   :posterior-mean (/ 9.0 14.0)
   :posterior-variance (/ (* 9.0 5.0) (* (* 14.0 14.0) 15.0))})

;; ---------------------------------------------------------------------------
;; Ground truth: Gamma-Poisson
;; ---------------------------------------------------------------------------
;;
;; Prior:      lambda ~ Gamma(3, 1)   (shape=3, rate=1)
;; Likelihood: x_i ~ Poisson(lambda)
;; Data:       [2, 3, 1, 4, 2, 3, 2, 1]  (n=8, sum=18)
;;
;; Posterior:  Gamma(3+18, 1+8) = Gamma(21, 9)
;; E[lambda] = 21/9 = 2.333...
;; Var[lambda] = 21/81 = 0.259259...

(def gp-ground-truth
  {:data [2 3 1 4 2 3 2 1]
   :shape 3.0
   :rate 1.0
   :posterior-shape 21.0
   :posterior-rate 9.0
   :posterior-mean (/ 21.0 9.0)
   :posterior-variance (/ 21.0 81.0)})

;; ---------------------------------------------------------------------------
;; Ground truth: Normal-InverseGamma (unknown mean and variance)
;; ---------------------------------------------------------------------------
;;
;; Prior:      mu ~ N(0, 10),  sigma^2 ~ InvGamma(3, 2)
;; Likelihood: x_i ~ N(mu, sigma)
;; Data:       [1.5, 2.0, 1.8, 2.2, 1.7]
;;
;; This is a joint posterior — we just test that IS converges to
;; reasonable values: E[mu] near sample mean, E[sigma^2] bounded.

(def nig-ground-truth
  {:data [1.5 2.0 1.8 2.2 1.7]
   :mu0 0.0
   :sigma0 10.0
   :alpha0 3.0
   :beta0 2.0
   :approx-mean-mu 1.84  ;; near sample mean
   :approx-mean-sig2-upper 2.0})  ;; E[sigma^2] should be small

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

(def nn-model
  "Normal-Normal: mu ~ N(prior-mean, prior-std), y_i ~ N(mu, lik-std)."
  (dyn/auto-key
   (gen [data prior-mean prior-std lik-std]
        (let [mu (trace :mu (dist/gaussian (mx/scalar prior-mean) (mx/scalar prior-std)))]
          (doseq [[i _] (map-indexed vector data)]
            (trace (keyword (str "y" i)) (dist/gaussian mu (mx/scalar lik-std))))
          mu))))

(def bb-model
  "Beta-Bernoulli: p ~ Beta(alpha, beta), y_i ~ Bernoulli(p)."
  (dyn/auto-key
   (gen [data alpha beta-param]
        (let [p (trace :p (dist/beta-dist (mx/scalar alpha) (mx/scalar beta-param)))]
          (doseq [[i _] (map-indexed vector data)]
            (trace (keyword (str "y" i)) (dist/bernoulli p)))
          p))))

(def gp-model
  "Gamma-Poisson: lambda ~ Gamma(shape, rate), y_i ~ Poisson(lambda)."
  (dyn/auto-key
   (gen [data shape rate]
        (let [lam (trace :lambda (dist/gamma-dist (mx/scalar shape) (mx/scalar rate)))]
          (doseq [[i _] (map-indexed vector data)]
            (trace (keyword (str "y" i)) (dist/poisson lam)))
          lam))))

(def nig-model
  "Normal-InverseGamma: sigma^2 ~ InvGamma, mu ~ N(0, sigma0), y_i ~ N(mu, sqrt(sigma^2))."
  (dyn/auto-key
   (gen [data mu0 sigma0 alpha0 beta0]
        (let [sigma-sq (trace :sigma-sq (dist/inv-gamma (mx/scalar alpha0) (mx/scalar beta0)))
              sigma (mx/sqrt sigma-sq)
              mu (trace :mu (dist/gaussian (mx/scalar mu0) (mx/scalar sigma0)))]
          (doseq [[i _] (map-indexed vector data)]
            (trace (keyword (str "y" i)) (dist/gaussian mu sigma)))
          {:mu mu :sigma-sq sigma-sq}))))

;; ===========================================================================
;; Family 1: Normal-Normal
;; ===========================================================================

(deftest nn-is-posterior-mean-converges
  (testing "IS weighted mean of mu converges to analytical posterior mean"
    (let [{:keys [data prior-mean prior-std lik-std posterior-mean]} nn-ground-truth
          obs (observations-from-indexed data)
          result (is/importance-sampling {:samples 1000 :key (rng/fresh-key 42)}
                                          nn-model [data prior-mean prior-std lik-std] obs)
          est-mean (weighted-mean result :mu)]
      (is (th/close? posterior-mean est-mean 0.09)
          (str "IS E[mu] = " est-mean " should be near " posterior-mean)))))

(deftest nn-is-posterior-variance-converges
  (testing "IS weighted variance of mu converges to analytical posterior variance"
    (let [{:keys [data prior-mean prior-std lik-std posterior-mean posterior-variance]}
          nn-ground-truth
          obs (observations-from-indexed data)
          result (is/importance-sampling {:samples 1000 :key (rng/fresh-key 43)}
                                          nn-model [data prior-mean prior-std lik-std] obs)
          est-mean (weighted-mean result :mu)
          est-var (weighted-variance result :mu est-mean)]
      (is (th/close? posterior-variance est-var 0.06)
          (str "IS Var[mu] = " est-var " should be near " posterior-variance)))))

(deftest nn-mh-posterior-mean-converges
  (testing "MH chain mean converges to analytical posterior mean"
    (let [{:keys [data prior-mean prior-std lik-std posterior-mean]} nn-ground-truth
          obs (observations-from-indexed data)
          traces (mcmc/mh {:samples 2000 :burn 500 :selection (sel/select :mu)
                           :key (rng/fresh-key 44)}
                          nn-model [data prior-mean prior-std lik-std] obs)
          samples (trace-samples traces :mu)
          chain-mean (th/sample-mean samples)]
      (is (th/close? posterior-mean chain-mean 0.09)
          (str "MH E[mu] = " chain-mean " should be near " posterior-mean)))))

(deftest nn-mh-posterior-variance-converges
  (testing "MH chain variance converges to analytical posterior variance"
    (let [{:keys [data prior-mean prior-std lik-std posterior-mean posterior-variance]}
          nn-ground-truth
          obs (observations-from-indexed data)
          traces (mcmc/mh {:samples 2000 :burn 500 :selection (sel/select :mu)
                           :key (rng/fresh-key 45)}
                          nn-model [data prior-mean prior-std lik-std] obs)
          samples (trace-samples traces :mu)
          chain-var (th/sample-variance samples)]
      (is (th/close? posterior-variance chain-var 0.06)
          (str "MH Var[mu] = " chain-var " should be near " posterior-variance)))))

(deftest nn-is-and-mh-agree
  (testing "IS and MH posterior means agree with each other"
    (let [{:keys [data prior-mean prior-std lik-std]} nn-ground-truth
          obs (observations-from-indexed data)
          is-result (is/importance-sampling {:samples 1000 :key (rng/fresh-key 46)}
                                            nn-model [data prior-mean prior-std lik-std] obs)
          is-mean (weighted-mean is-result :mu)
          mh-traces (mcmc/mh {:samples 2000 :burn 500 :selection (sel/select :mu)
                              :key (rng/fresh-key 47)}
                             nn-model [data prior-mean prior-std lik-std] obs)
          mh-mean (th/sample-mean (trace-samples mh-traces :mu))]
      (is (th/close? is-mean mh-mean 0.15)
          (str "IS mean " is-mean " and MH mean " mh-mean " should agree")))))

;; ===========================================================================
;; Family 2: Beta-Bernoulli
;; ===========================================================================

(deftest bb-is-posterior-mean-converges
  (testing "IS weighted mean of p converges to Beta(9,5) mean"
    (let [{:keys [data alpha beta-param posterior-mean]} bb-ground-truth
          obs (observations-from-indexed data)
          result (is/importance-sampling {:samples 1000 :key (rng/fresh-key 50)}
                                          bb-model [data alpha beta-param] obs)
          est-mean (weighted-mean result :p)]
      (is (th/close? posterior-mean est-mean 0.05)
          (str "IS E[p] = " est-mean " should be near " posterior-mean)))))

(deftest bb-is-posterior-variance-converges
  (testing "IS weighted variance of p converges to Beta(9,5) variance"
    (let [{:keys [data alpha beta-param posterior-mean posterior-variance]}
          bb-ground-truth
          obs (observations-from-indexed data)
          result (is/importance-sampling {:samples 1000 :key (rng/fresh-key 51)}
                                          bb-model [data alpha beta-param] obs)
          est-mean (weighted-mean result :p)
          est-var (weighted-variance result :p est-mean)]
      (is (th/close? posterior-variance est-var 0.01)
          (str "IS Var[p] = " est-var " should be near " posterior-variance)))))

(deftest bb-posterior-mean-between-prior-and-data
  (testing "Posterior mean lies between prior mean and data proportion"
    (let [{:keys [data alpha beta-param posterior-mean]} bb-ground-truth
          prior-mean (/ alpha (+ alpha beta-param))  ;; 0.5
          data-proportion (/ 7.0 10.0)]               ;; 0.7
      (is (< prior-mean posterior-mean data-proportion)
          "posterior mean between prior mean (0.5) and data proportion (0.7)"))))

;; ===========================================================================
;; Family 3: Gamma-Poisson
;; ===========================================================================

(deftest gp-is-posterior-mean-converges
  (testing "IS weighted mean of lambda converges to Gamma(21,9) mean"
    (let [{:keys [data shape rate posterior-mean]} gp-ground-truth
          obs (observations-from-indexed data)
          result (is/importance-sampling {:samples 1000 :key (rng/fresh-key 60)}
                                          gp-model [data shape rate] obs)
          est-mean (weighted-mean result :lambda)]
      (is (th/close? posterior-mean est-mean 0.15)
          (str "IS E[lambda] = " est-mean " should be near " posterior-mean)))))

(deftest gp-is-posterior-variance-converges
  (testing "IS weighted variance of lambda converges to Gamma(21,9) variance"
    (let [{:keys [data shape rate posterior-mean posterior-variance]} gp-ground-truth
          obs (observations-from-indexed data)
          result (is/importance-sampling {:samples 1000 :key (rng/fresh-key 61)}
                                          gp-model [data shape rate] obs)
          est-mean (weighted-mean result :lambda)
          est-var (weighted-variance result :lambda est-mean)]
      (is (th/close? posterior-variance est-var 0.10)
          (str "IS Var[lambda] = " est-var " should be near " posterior-variance)))))

(deftest gp-posterior-concentrates-near-data-mean
  (testing "Posterior mean is closer to data mean than prior mean"
    (let [{:keys [data shape rate posterior-mean]} gp-ground-truth
          prior-mean (/ shape rate)         ;; 3.0
          data-mean (/ 18.0 8.0)]           ;; 2.25
      ;; Posterior mean (2.333) should be closer to data mean (2.25) than prior (3.0)
      (is (< (js/Math.abs (- posterior-mean data-mean))
             (js/Math.abs (- prior-mean data-mean)))
          "posterior shrinks toward data mean"))))

;; ===========================================================================
;; Family 4: Normal-InverseGamma (unknown mean and variance)
;; ===========================================================================

(deftest nig-is-mean-mu-reasonable
  (testing "IS weighted mean of mu is near sample mean"
    (let [{:keys [data mu0 sigma0 alpha0 beta0 approx-mean-mu]} nig-ground-truth
          obs (observations-from-indexed data)
          result (is/importance-sampling {:samples 1000 :key (rng/fresh-key 70)}
                                          nig-model [data mu0 sigma0 alpha0 beta0] obs)
          est-mean (weighted-mean result :mu)]
      (is (th/close? approx-mean-mu est-mean 0.25)
          (str "IS E[mu] = " est-mean " should be near sample mean " approx-mean-mu)))))

(deftest nig-is-sigma-sq-bounded
  (testing "IS weighted mean of sigma^2 is positive and bounded"
    (let [{:keys [data mu0 sigma0 alpha0 beta0 approx-mean-sig2-upper]} nig-ground-truth
          obs (observations-from-indexed data)
          result (is/importance-sampling {:samples 1000 :key (rng/fresh-key 71)}
                                          nig-model [data mu0 sigma0 alpha0 beta0] obs)
          est-mean (weighted-mean result :sigma-sq)]
      (is (pos? est-mean)
          (str "E[sigma^2] = " est-mean " should be positive"))
      (is (< est-mean approx-mean-sig2-upper)
          (str "E[sigma^2] = " est-mean " should be < " approx-mean-sig2-upper)))))

;; ===========================================================================
;; Cross-family: IS log-ML monotonicity
;; ===========================================================================

(deftest is-log-ml-finite
  (testing "IS log-ML estimates are finite for all families"
    (let [nn-obs (observations-from-indexed (:data nn-ground-truth))
          bb-obs (observations-from-indexed (:data bb-ground-truth))
          gp-obs (observations-from-indexed (:data gp-ground-truth))
          nn-ml (realize (:log-ml-estimate
                          (is/importance-sampling {:samples 500 :key (rng/fresh-key 80)}
                                                  nn-model [(:data nn-ground-truth)
                                                            (:prior-mean nn-ground-truth)
                                                            (:prior-std nn-ground-truth)
                                                            (:lik-std nn-ground-truth)] nn-obs)))
          bb-ml (realize (:log-ml-estimate
                          (is/importance-sampling {:samples 500 :key (rng/fresh-key 81)}
                                                  bb-model [(:data bb-ground-truth)
                                                            (:alpha bb-ground-truth)
                                                            (:beta-param bb-ground-truth)] bb-obs)))
          gp-ml (realize (:log-ml-estimate
                          (is/importance-sampling {:samples 500 :key (rng/fresh-key 82)}
                                                  gp-model [(:data gp-ground-truth)
                                                            (:shape gp-ground-truth)
                                                            (:rate gp-ground-truth)] gp-obs)))]
      (is (js/isFinite nn-ml) (str "Normal-Normal log-ML = " nn-ml " should be finite"))
      (is (js/isFinite bb-ml) (str "Beta-Bernoulli log-ML = " bb-ml " should be finite"))
      (is (js/isFinite gp-ml) (str "Gamma-Poisson log-ML = " gp-ml " should be finite"))
      ;; All log-ML should be negative (log of a probability)
      (is (neg? nn-ml) "Normal-Normal log-ML should be negative")
      (is (neg? bb-ml) "Beta-Bernoulli log-ML should be negative")
      (is (neg? gp-ml) "Gamma-Poisson log-ML should be negative"))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(run-tests)
