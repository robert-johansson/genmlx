(ns genmlx.inference-is-test
  "Phase 4.2: Importance sampling convergence tests.
   All posteriors are analytically derived conjugate posteriors.

   Normal-Normal conjugacy:
     Prior: mu ~ N(0, 10), Likelihood: y_i ~ N(mu, 1), i=1..5
     y = [2.8, 3.1, 2.9, 3.2, 3.0], mean(y) = 3.0
     precision_prior = 1/100 = 0.01
     precision_obs   = 5/1 = 5.0
     precision_post  = 5.01
     mu_post = (0.01*0 + 5*3.0) / 5.01 = 2.9940
     sigma_post = 1/sqrt(5.01) = 0.4468

   Log marginal likelihood (analytically):
     y ~ MVN(0, I + 100*J) where J = ones(5,5)
     |C| = 1 + 100*5 = 501 (matrix determinant lemma)
     C^{-1} = I - (100/501)*J
     y^T C^{-1} y = sum(y_i^2) - (100/501)*(sum y_i)^2
                   = 45.1 - (100/501)*225 = 45.1 - 44.9102 = 0.1898
     log p(y) = -5/2*log(2pi) - 0.5*log(501) - 0.5*0.1898 = -7.7979

   Beta-Bernoulli conjugacy:
     Prior: p ~ Beta(1,1), Data: 10 heads / 10 flips
     Posterior: Beta(11, 1)
     E[p|x] = 11/12 = 0.9167
     Var[p|x] = 11/(12^2*13) = 0.005876

   Tolerance policy:
     IS with N=2000: conservative N_eff ~ 200 for Normal-Normal
     std_error = sigma_post / sqrt(N_eff) = 0.447/sqrt(200) = 0.0316
     3.5-sigma = 0.111. Use 0.15 for safety margin."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.importance :as is]
            [genmlx.inference.util :as u]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def normal-normal-model
  "mu ~ N(0, 10), y_i ~ N(mu, 1) for i=1..n."
  (dyn/auto-key
    (gen [ys]
      (let [mu (trace :mu (dist/gaussian 0 10))]
        (doseq [[i y] (map-indexed vector ys)]
          (trace (keyword (str "y" i))
                 (dist/gaussian mu 1)))
        mu))))

(def beta-bernoulli-model
  "p ~ Beta(1,1), x_i ~ Bernoulli(p) for i=1..n."
  (dyn/auto-key
    (gen [n-flips]
      (let [p (trace :p (dist/beta-dist 1 1))]
        (doseq [i (range n-flips)]
          (trace (keyword (str "x" i))
                 (dist/bernoulli p)))
        p))))

;; ---------------------------------------------------------------------------
;; Observation choicemaps
;; ---------------------------------------------------------------------------

(def ys [2.8 3.1 2.9 3.2 3.0])

(def normal-obs
  "Observed y values for Normal-Normal model."
  (reduce (fn [cm [i y]]
            (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector ys)))

(def bernoulli-obs
  "10 heads out of 10 flips."
  (reduce (fn [cm i]
            (cm/set-choice cm [(keyword (str "x" i))] (mx/scalar 1.0)))
          cm/EMPTY
          (range 10)))

;; ---------------------------------------------------------------------------
;; Analytical values
;; ---------------------------------------------------------------------------

;; Normal-Normal posterior: N(2.9940, 0.4468)
(def ^:private mu-post 2.9940119760479043)
(def ^:private sigma-post 0.4467670516087703)

;; Log marginal likelihood: -7.7979
(def ^:private analytical-log-ml -7.797905896206512)

;; Beta-Bernoulli posterior: Beta(11,1), E[p]=11/12
(def ^:private beta-mean-post (/ 11 12))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- is-weighted-mean
  "Compute weighted mean of an address from IS result."
  [result addr]
  (let [{:keys [probs]} (u/normalize-log-weights (:log-weights result))
        vals (mapv (fn [tr]
                     (let [v (cm/get-choice (:choices tr) [addr])]
                       (mx/eval! v) (mx/item v)))
                   (:traces result))]
    (reduce + (map * probs vals))))

(defn- is-ess
  "Compute effective sample size from IS log-weights."
  [log-weights]
  (let [{:keys [probs]} (u/normalize-log-weights log-weights)]
    (/ 1.0 (reduce + (map #(* % %) probs)))))

;; ==========================================================================
;; 1. IS convergence to Normal-Normal posterior
;; ==========================================================================
;; Tolerance: N=2000, conservative N_eff~200
;; std_error = 0.447/sqrt(200) = 0.0316, 3.5σ = 0.111

(deftest is-normal-normal-convergence
  (testing "IS weighted mean converges to analytical posterior mean"
    (let [result (is/importance-sampling
                   {:samples 2000 :key (rng/fresh-key 42)}
                   normal-normal-model [ys] normal-obs)
          weighted-mean (is-weighted-mean result :mu)]
      (is (h/close? mu-post weighted-mean 0.15)
          (str "weighted mean " weighted-mean " ≈ " mu-post)))))

;; ==========================================================================
;; 2. IS log marginal likelihood
;; ==========================================================================
;; log p(y) = -7.7979 (derived above)
;; IS log-ML has high variance; use generous tolerance

(deftest is-log-ml-estimate
  (testing "IS log-ML estimate is in the right ballpark"
    (let [result (is/importance-sampling
                   {:samples 2000 :key (rng/fresh-key 99)}
                   normal-normal-model [ys] normal-obs)
          log-ml (h/realize (:log-ml-estimate result))]
      ;; IS log-ML is unbiased on the linear scale but biased low on log scale
      ;; (Jensen's inequality). Tolerance of 1.0 is generous but tests
      ;; the estimate is in the right order of magnitude.
      (is (h/close? analytical-log-ml log-ml 1.0)
          (str "log-ML " log-ml " ≈ " analytical-log-ml))
      (is (js/isFinite log-ml) "log-ML is finite"))))

;; ==========================================================================
;; 3. IS effective sample size
;; ==========================================================================
;; ESS must be in (0, N]. For this model with N=1000,
;; ESS should be well above 0 (prior is broad, so many particles
;; will have reasonable weight).

(deftest is-ess-bounds
  (testing "ESS is in valid range (0, N]"
    (let [n 1000
          result (is/importance-sampling
                   {:samples n :key (rng/fresh-key 77)}
                   normal-normal-model [ys] normal-obs)
          ess (is-ess (:log-weights result))]
      (is (> ess 0) "ESS > 0")
      (is (<= ess n) (str "ESS " ess " <= N=" n))
      ;; With prior N(0,10) and observations around 3, ESS should be
      ;; reasonable (>10% of N for this well-behaved model)
      (is (> ess (* 0.05 n))
          (str "ESS " ess " > 5% of N (well-behaved model)")))))

;; ==========================================================================
;; 4. IS convergence to Beta-Bernoulli posterior
;; ==========================================================================
;; Beta(1,1) prior with 10/10 heads → Beta(11,1), E[p]=0.9167
;; IS on discrete observations is well-behaved

(deftest is-beta-bernoulli-convergence
  (testing "IS weighted mean converges to Beta-Bernoulli posterior"
    (let [result (is/importance-sampling
                   {:samples 2000 :key (rng/fresh-key 55)}
                   beta-bernoulli-model [10] bernoulli-obs)
          weighted-mean (is-weighted-mean result :p)]
      ;; Beta(11,1) has mean 0.9167 and std 0.0767
      ;; With N=2000 and good ESS, tolerance ~0.05
      (is (h/close? beta-mean-post weighted-mean 0.08)
          (str "weighted mean " weighted-mean " ≈ " beta-mean-post)))))

;; ==========================================================================
;; 5. Tidy IS agrees with standard IS
;; ==========================================================================

(deftest tidy-is-log-ml
  (testing "tidy IS log-ML is in correct range"
    (let [result (is/tidy-importance-sampling
                   {:samples 1000 :key (rng/fresh-key 42)}
                   normal-normal-model [ys] normal-obs)
          log-ml (:log-ml-estimate result)]
      (is (js/isFinite log-ml) "tidy IS log-ML is finite")
      (is (h/close? analytical-log-ml log-ml 1.5)
          (str "tidy IS log-ML " log-ml " ≈ " analytical-log-ml)))))

;; ==========================================================================
;; 6. IS returns correct number of traces
;; ==========================================================================

(deftest is-returns-n-traces
  (testing "IS returns exactly N traces and N weights"
    (let [n 50
          result (is/importance-sampling
                   {:samples n :key (rng/fresh-key 1)}
                   normal-normal-model [ys] normal-obs)]
      (is (= n (count (:traces result))) "N traces")
      (is (= n (count (:log-weights result))) "N weights"))))

;; ==========================================================================
;; Run tests
;; ==========================================================================

(cljs.test/run-tests)
