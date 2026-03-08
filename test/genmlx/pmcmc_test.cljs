(ns genmlx.pmcmc-test
  "Tests for Tier 3b: Particle MCMC (PMMH and Particle Gibbs)."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.pmcmc :as pmcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [label pred]
  (if pred
    (println (str "  PASS: " label))
    (println (str "  FAIL: " label))))

(defn assert-close [label expected actual tol]
  (let [ok (< (js/Math.abs (- expected actual)) tol)]
    (if ok
      (println (str "  PASS: " label " (expected " (.toFixed expected 3) ", got " (.toFixed actual 3) ")"))
      (println (str "  FAIL: " label " (expected " (.toFixed expected 3) ", got " (.toFixed actual 3) ")")))))

;; ---------------------------------------------------------------------------
;; Test 1: PMMH — Gaussian mean posterior
;; ---------------------------------------------------------------------------
;; Model: mu ~ N(0, 10), y_i ~ N(mu, 1) for i=1..K
;; Posterior: mu | y ~ N(y_bar * K/(K + 1/100), 1/(K + 1/100)) ≈ N(3.0, 0.316)

(println "\n=== Test 1: PMMH — Gaussian mean posterior ===")

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
      ;; Analytical posterior: mean ≈ y-bar ≈ 3.0, std ≈ 1/sqrt(K) ≈ 0.316
      expected-mean y-bar
      expected-std (/ 1.0 (js/Math.sqrt K-obs))]
  (println (str "  Posterior mean: " (.toFixed mean-mu 3) " (expected ≈ " (.toFixed expected-mean 3) ")"))
  (println (str "  Posterior std:  " (.toFixed std-mu 3) " (expected ≈ " (.toFixed expected-std 3) ")"))
  (println (str "  Acceptance rate: " (.toFixed (:acceptance-rate result) 3)))
  (println (str "  Samples collected: " (count samples)))
  (assert-close "Posterior mean ≈ y-bar" expected-mean mean-mu 1.0)
  (assert-close "Posterior std ≈ 1/√K" expected-std std-mu 0.3)
  (assert-true "Acceptance rate > 0" (> (:acceptance-rate result) 0))
  (assert-true "Acceptance rate < 1" (< (:acceptance-rate result) 1)))

;; ---------------------------------------------------------------------------
;; Test 2: PMMH — Beta-Bernoulli posterior
;; ---------------------------------------------------------------------------
;; Model: p ~ Beta(2, 2), y_i ~ Bernoulli(p) for i=1..20
;; Observe 15 successes out of 20
;; Posterior: p ~ Beta(2+15, 2+5) = Beta(17, 7), mean = 17/24 ≈ 0.708

(println "\n=== Test 2: PMMH — Beta-Bernoulli posterior ===")

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
      ;; Beta(17, 7) mean = 17/24
      expected-mean (/ 17.0 24.0)]
  (println (str "  Posterior mean: " (.toFixed mean-p 3) " (expected ≈ " (.toFixed expected-mean 3) ")"))
  (println (str "  Acceptance rate: " (.toFixed (:acceptance-rate result) 3)))
  (assert-close "Posterior mean ≈ 17/24" expected-mean mean-p 0.15)
  (assert-true "Acceptance rate reasonable" (> (:acceptance-rate result) 0.05)))

;; ---------------------------------------------------------------------------
;; Test 3: PMMH — 2D posterior (mu + log-sigma)
;; ---------------------------------------------------------------------------
;; Model: mu ~ N(0, 10), log-sigma ~ N(0, 2), y_i ~ N(mu, exp(log-sigma))
;; At MLE: mu ≈ y-bar, sigma ≈ sample std

(println "\n=== Test 3: PMMH — 2D Gaussian (mu, log-sigma) ===")

(def model-2d
  (gen [K]
    (let [mu (trace :mu (dist/gaussian 0 10))
          log-sigma (trace :log-sigma (dist/gaussian 0 2))
          sigma (mx/exp log-sigma)]
      (doseq [i (range K)]
        (trace (keyword (str "y" i)) (dist/gaussian mu sigma)))
      [mu sigma])))

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
  (println (str "  Posterior mean(mu): " (.toFixed mean-mu 3) " (expected ≈ " (.toFixed y-bar 3) ")"))
  (println (str "  Posterior mean(log-sigma): " (.toFixed mean-ls 3) " (expected ≈ 0)"))
  (println (str "  Acceptance rate: " (.toFixed (:acceptance-rate result) 3)))
  (assert-close "mu ≈ y-bar" y-bar mean-mu 1.5)
  ;; log-sigma should be near 0 (sigma ≈ 1, data spread ≈ 1)
  (assert-close "log-sigma ≈ 0" 0.0 mean-ls 1.5))

;; ---------------------------------------------------------------------------
;; Test 4: Particle Gibbs — Gaussian mean
;; ---------------------------------------------------------------------------

(println "\n=== Test 4: Particle Gibbs — Gaussian mean ===")

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
  (println (str "  Posterior mean: " (.toFixed mean-mu 3) " (expected ≈ " (.toFixed y-bar 3) ")"))
  (println (str "  Acceptance rate: " (.toFixed (:acceptance-rate result) 3)))
  (println (str "  Samples: " (count samples)))
  (assert-close "PG posterior mean ≈ y-bar" y-bar mean-mu 1.5)
  (assert-true "PG collected samples" (pos? (count samples))))

;; ---------------------------------------------------------------------------
;; Test 5: PMMH log-ML monotonicity
;; ---------------------------------------------------------------------------
;; Log-ML estimates should generally increase during burn-in as chain
;; moves toward high-posterior region

(println "\n=== Test 5: PMMH log-ML trajectory ===")

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
      first-10 (take 10 log-mls)
      last-10 (take-last 10 log-mls)
      mean-first (/ (reduce + first-10) (count first-10))
      mean-last (/ (reduce + last-10) (count last-10))]
  (println (str "  Early mean log-ML: " (.toFixed mean-first 2)))
  (println (str "  Late mean log-ML:  " (.toFixed mean-last 2)))
  (assert-true "All log-MLs are finite" (every? js/isFinite log-mls))
  ;; The chain should be in a reasonable region after burn-in
  (assert-true "log-MLs are reasonable" (> mean-last -100)))

;; ---------------------------------------------------------------------------
;; Test 6: PMMH with custom extract-fn
;; ---------------------------------------------------------------------------

(println "\n=== Test 6: PMMH custom extraction ===")

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
  (println (str "  Sample type: " (type (first samples))))
  (println (str "  Sample: " (first samples)))
  (assert-true "Extract returns maps" (map? (first samples)))
  (assert-true "Maps have :mu key" (contains? (first samples) :mu)))

(println "\nDone.")
