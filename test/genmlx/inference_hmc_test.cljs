(ns genmlx.inference-hmc-test
  "Phase 4.3: HMC convergence tests.

   Uses the Normal-Normal conjugate model:
     Prior: mu ~ N(0, 10), Likelihood: y_i ~ N(mu, 1), i=1..5
     y = [2.8, 3.1, 2.9, 3.2, 3.0]
     Posterior: mu | y ~ N(2.9940, 0.4468)

   HMC tolerance:
     HMC has lower autocorrelation than MH, so N_eff is closer to N.
     200 samples, burn 100 → ~200 effective samples.
     Conservative N_eff ~ 150.
     std_error = 0.447/sqrt(150) = 0.0365
     3.5-sigma = 0.128. Use 0.15."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test model
;; ---------------------------------------------------------------------------

(def normal-normal-model
  (dyn/auto-key
    (gen [ys]
      (let [mu (trace :mu (dist/gaussian 0 10))]
        (doseq [[i y] (map-indexed vector ys)]
          (trace (keyword (str "y" i))
                 (dist/gaussian mu 1)))
        mu))))

(def ys [2.8 3.1 2.9 3.2 3.0])

(def obs
  (reduce (fn [cm [i y]]
            (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector ys)))

;; Analytical posterior
(def ^:private mu-post 2.9940119760479043)
(def ^:private sigma-post 0.4467670516087703)

;; ==========================================================================
;; 1. HMC convergence
;; ==========================================================================

(deftest hmc-convergence
  (testing "HMC sample mean converges to analytical posterior mean"
    (let [samples (mcmc/hmc {:samples 200 :burn 100 :thin 1
                              :step-size 0.1 :leapfrog-steps 10
                              :addresses [:mu]
                              :key (rng/fresh-key 42)
                              :device :cpu :compile? false}
                             normal-normal-model [ys] obs)
          ;; HMC returns vectors of Clojure vectors (JS numbers)
          mu-vals (mapv first samples)
          mean-mu (h/sample-mean mu-vals)]
      (is (h/close? mu-post mean-mu 0.15)
          (str "HMC mean " mean-mu " ≈ " mu-post)))))

;; ==========================================================================
;; 2. HMC returns correct number of samples
;; ==========================================================================

(deftest hmc-sample-count
  (testing "HMC returns exactly N samples"
    (let [n 50
          samples (mcmc/hmc {:samples n :burn 20 :thin 1
                              :step-size 0.1 :leapfrog-steps 10
                              :addresses [:mu]
                              :key (rng/fresh-key 1)
                              :device :cpu :compile? false}
                             normal-normal-model [ys] obs)]
      (is (= n (count samples)) "N samples returned"))))

;; ==========================================================================
;; 3. HMC variance approximates posterior variance
;; ==========================================================================

(deftest hmc-variance
  (testing "HMC sample variance approximates posterior variance"
    (let [samples (mcmc/hmc {:samples 300 :burn 100 :thin 1
                              :step-size 0.1 :leapfrog-steps 10
                              :addresses [:mu]
                              :key (rng/fresh-key 77)
                              :device :cpu :compile? false}
                             normal-normal-model [ys] obs)
          mu-vals (mapv first samples)
          var-mu (h/sample-variance mu-vals)
          expected-var (* sigma-post sigma-post)]
      ;; sigma_post^2 = 0.1996
      (is (h/close? expected-var var-mu 0.15)
          (str "HMC variance " var-mu " ≈ " expected-var)))))

;; ==========================================================================
;; Run tests
;; ==========================================================================

(cljs.test/run-tests)
