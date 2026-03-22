(ns genmlx.inference-mh-test
  "Phase 4.3: Metropolis-Hastings convergence tests.

   Uses the Normal-Normal conjugate model:
     Prior: mu ~ N(0, 10), Likelihood: y_i ~ N(mu, 1), i=1..5
     y = [2.8, 3.1, 2.9, 3.2, 3.0]
     Posterior: mu | y ~ N(2.9940, 0.4468)

   MH tolerance:
     500 samples, burn 200, thin 2 → ~250 effective post-burn samples.
     Autocorrelation reduces N_eff further; conservatively N_eff ~ 100.
     std_error = 0.447/sqrt(100) = 0.0447
     3.5-sigma = 0.156. Use 0.20 for safety."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
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

;; Analytical posterior: N(2.994, 0.447)
(def ^:private mu-post 2.9940119760479043)
(def ^:private sigma-post 0.4467670516087703)

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- extract-mu-values
  "Extract :mu values from a vector of traces."
  [traces]
  (mapv (fn [tr]
          (let [v (cm/get-choice (:choices tr) [:mu])]
            (mx/eval! v) (mx/item v)))
        traces))

;; ==========================================================================
;; 1. MH convergence to Normal-Normal posterior
;; ==========================================================================

(deftest mh-convergence
  (testing "MH sample mean converges to analytical posterior mean"
    (let [traces (mcmc/mh {:samples 500 :burn 200 :thin 2
                           :selection (sel/select :mu)
                           :key (rng/fresh-key 42)}
                          normal-normal-model [ys] obs)
          mu-vals (extract-mu-values traces)
          mean-mu (h/sample-mean mu-vals)]
      (is (h/close? mu-post mean-mu 0.20)
          (str "sample mean " mean-mu " ≈ " mu-post))))

  (testing "MH sample variance approximates posterior variance"
    (let [traces (mcmc/mh {:samples 500 :burn 200 :thin 2
                           :selection (sel/select :mu)
                           :key (rng/fresh-key 99)}
                          normal-normal-model [ys] obs)
          mu-vals (extract-mu-values traces)
          var-mu (h/sample-variance mu-vals)
          ;; sigma_post^2 = 0.1996
          expected-var (* sigma-post sigma-post)]
      ;; Variance estimate has higher uncertainty; use generous tolerance
      (is (h/close? expected-var var-mu 0.15)
          (str "sample variance " var-mu " ≈ " expected-var)))))

;; ==========================================================================
;; 2. MH returns correct number of samples
;; ==========================================================================

(deftest mh-sample-count
  (testing "MH returns exactly N samples"
    (let [n 100
          traces (mcmc/mh {:samples n :burn 50 :thin 1
                           :selection (sel/select :mu)
                           :key (rng/fresh-key 1)}
                          normal-normal-model [ys] obs)]
      (is (= n (count traces)) "N traces returned"))))

;; ==========================================================================
;; 3. MH from different starting points converges to same posterior
;; ==========================================================================

(deftest mh-different-seeds-agree
  (testing "two MH chains from different seeds agree on posterior mean"
    (let [run-mh (fn [seed]
                   (let [traces (mcmc/mh {:samples 400 :burn 200 :thin 2
                                          :selection (sel/select :mu)
                                          :key (rng/fresh-key seed)}
                                         normal-normal-model [ys] obs)]
                     (h/sample-mean (extract-mu-values traces))))
          mean1 (run-mh 1)
          mean2 (run-mh 2)]
      ;; Both should be near mu_post; difference should be small
      ;; Each has std_error ~ 0.045, combined ~ 0.063
      ;; 3.5σ of combined ~ 0.22
      (is (h/close? mean1 mean2 0.30)
          (str "chain means agree: " mean1 " ≈ " mean2)))))

;; ==========================================================================
;; 4. MH with sel/all still works (regenerates all addresses)
;; ==========================================================================

(deftest mh-select-all
  (testing "MH with sel/all runs without error"
    ;; sel/all regenerates observations too — acceptance rate is very low.
    ;; We only check that the chain runs and produces traces, not convergence.
    (let [traces (mcmc/mh {:samples 50 :burn 10 :thin 1
                           :selection sel/all
                           :key (rng/fresh-key 42)}
                          normal-normal-model [ys] obs)]
      (is (= 50 (count traces)) "returns requested number of traces"))))

;; ==========================================================================
;; Run tests
;; ==========================================================================

(cljs.test/run-tests)
