(ns genmlx.pmcmc-test
  "Tests for Tier 3b: Particle MCMC (PMMH and Particle Gibbs)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.pmcmc :as pmcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Model: mu ~ N(0, 10), y_i ~ N(mu, 1) for i=1..K
;; Posterior: mu | y ~ N(y_bar * K/(K + 1/100), 1/(K + 1/100)) ~ N(3.0, 0.316)
;; ---------------------------------------------------------------------------

(def K-obs 10)
(def true-mu 3.0)
(def obs-data (mapv (fn [i] (+ true-mu (* 0.5 (- i 4.5)))) (range K-obs)))
(def y-bar (/ (reduce + obs-data) K-obs))

(def gaussian-model
  (gen [K]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (doseq [i (range K)]
        (trace (keyword (str "y" i)) (dist/gaussian mu 1)))
      mu)))

(def gaussian-obs
  (apply cm/choicemap
    (mapcat (fn [i] [(keyword (str "y" i)) (mx/scalar (nth obs-data i))])
            (range K-obs))))

;; ---------------------------------------------------------------------------
;; Beta-Bernoulli model
;; p ~ Beta(2, 2), y_i ~ Bernoulli(p), 15/20 successes
;; Posterior: p ~ Beta(17, 7), mean = 17/24 ~ 0.708
;; ---------------------------------------------------------------------------

(def bb-K 20)
(def bb-successes 15)

(def bb-model
  (gen [K]
    (let [p (trace :p (dist/beta-dist 2 2))]
      (doseq [i (range K)]
        (trace (keyword (str "y" i)) (dist/bernoulli p)))
      p)))

(def bb-obs
  (apply cm/choicemap
    (mapcat (fn [i]
              [(keyword (str "y" i))
               (mx/scalar (if (< i bb-successes) 1.0 0.0))])
            (range bb-K))))

;; ---------------------------------------------------------------------------
;; 2D model: mu ~ N(0, 10), log-sigma ~ N(0, 2), y_i ~ N(mu, exp(log-sigma))
;; ---------------------------------------------------------------------------

(def model-2d
  (gen [K]
    (let [mu (trace :mu (dist/gaussian 0 10))
          log-sigma (trace :log-sigma (dist/gaussian 0 2))
          sigma (mx/exp log-sigma)]
      (doseq [i (range K)]
        (trace (keyword (str "y" i)) (dist/gaussian mu sigma)))
      [mu sigma])))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest pmmh-gaussian-test
  (testing "PMMH Gaussian mean posterior"
    (let [result (pmcmc/pmmh
                   {:n-particles 200
                    :n-samples 200
                    :burn 100
                    :param-addrs [:mu]
                    :observations gaussian-obs
                    :proposal-std 0.5
                    :key (rng/fresh-key 42)}
                   gaussian-model [K-obs])
          samples (:samples result)
          mu-samples (mapv first samples)
          mean-mu (/ (reduce + mu-samples) (count mu-samples))
          std-mu (js/Math.sqrt
                   (/ (reduce + (map #(* (- % mean-mu) (- % mean-mu)) mu-samples))
                      (dec (count mu-samples))))
          expected-mean y-bar
          expected-std (/ 1.0 (js/Math.sqrt K-obs))]
      (is (h/close? expected-mean mean-mu 1.0) "Posterior mean ~ y-bar")
      (is (h/close? expected-std std-mu 0.3) "Posterior std ~ 1/sqrt(K)")
      (is (> (:acceptance-rate result) 0) "Acceptance rate > 0")
      (is (< (:acceptance-rate result) 1) "Acceptance rate < 1"))))

(deftest pmmh-beta-bernoulli-test
  (testing "PMMH Beta-Bernoulli posterior"
    (let [result (pmcmc/pmmh
                   {:n-particles 50
                    :n-samples 400
                    :burn 300
                    :param-addrs [:p]
                    :observations bb-obs
                    :proposal-std 0.1
                    :key (rng/fresh-key 77)}
                   bb-model [bb-K])
          samples (:samples result)
          p-samples (mapv first samples)
          mean-p (/ (reduce + p-samples) (count p-samples))
          expected-mean (/ 17.0 24.0)]
      (is (h/close? expected-mean mean-p 0.15) "Posterior mean ~ 17/24")
      (is (> (:acceptance-rate result) 0.05) "Acceptance rate reasonable"))))

(deftest pmmh-2d-test
  (testing "PMMH 2D Gaussian (mu, log-sigma)"
    (let [result (pmcmc/pmmh
                   {:n-particles 200
                    :n-samples 200
                    :burn 100
                    :param-addrs [:mu :log-sigma]
                    :observations gaussian-obs
                    :proposal-std [0.5 0.3]
                    :key (rng/fresh-key 99)}
                   model-2d [K-obs])
          samples (:samples result)
          mu-samples (mapv first samples)
          ls-samples (mapv second samples)
          mean-mu (/ (reduce + mu-samples) (count mu-samples))
          mean-ls (/ (reduce + ls-samples) (count ls-samples))]
      (is (h/close? y-bar mean-mu 1.5) "mu ~ y-bar")
      (is (h/close? 0.0 mean-ls 1.5) "log-sigma ~ 0"))))

(deftest particle-gibbs-test
  (testing "Particle Gibbs Gaussian mean"
    (let [result (pmcmc/particle-gibbs
                   {:n-particles 20
                    :n-samples 150
                    :burn 50
                    :param-addrs [:mu]
                    :observations gaussian-obs
                    :proposal-std 0.5
                    :key (rng/fresh-key 42)}
                   gaussian-model [K-obs])
          samples (:samples result)
          mu-samples (mapv first samples)
          mean-mu (/ (reduce + mu-samples) (count mu-samples))]
      (is (h/close? y-bar mean-mu 1.5) "PG posterior mean ~ y-bar")
      (is (pos? (count samples)) "PG collected samples"))))

(deftest pmmh-log-ml-trajectory-test
  (testing "PMMH log-ML trajectory"
    (let [result (pmcmc/pmmh
                   {:n-particles 200
                    :n-samples 50
                    :burn 50
                    :param-addrs [:mu]
                    :observations gaussian-obs
                    :proposal-std 0.5
                    :key (rng/fresh-key 42)}
                   gaussian-model [K-obs])
          log-mls (:log-mls result)
          last-10 (take-last 10 log-mls)
          mean-last (/ (reduce + last-10) (count last-10))]
      (is (every? js/isFinite log-mls) "All log-MLs are finite")
      (is (> mean-last -100) "log-MLs are reasonable"))))

(deftest pmmh-custom-extract-test
  (testing "PMMH custom extraction"
    (let [result (pmcmc/pmmh
                   {:n-particles 100
                    :n-samples 20
                    :burn 10
                    :param-addrs [:mu]
                    :observations gaussian-obs
                    :proposal-std 0.5
                    :key (rng/fresh-key 42)
                    :extract-fn (fn [vals]
                                  {:mu (mx/item (first vals))})}
                   gaussian-model [K-obs])
          samples (:samples result)]
      (is (map? (first samples)) "Extract returns maps")
      (is (contains? (first samples) :mu) "Maps have :mu key"))))

(cljs.test/run-tests)
