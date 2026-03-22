(ns genmlx.dist-normalization-property-test
  "Property-based tests for discrete distribution normalization laws.
   Verifies: probability axioms (sum to 1), interface consistency,
   categorical parameterization, support coverage, and Bayesian conjugacy."
  (:require [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [cljs.test :as t]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]
                   [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- s [v] (mx/scalar (double v)))

(defn- close? [a b tol]
  (< (js/Math.abs (- a b)) tol))

;; ---------------------------------------------------------------------------
;; Pre-built distribution pools (SCI shrink safety)
;; ---------------------------------------------------------------------------

(def bernoulli-pool
  [(dist/bernoulli (s 0.3))
   (dist/bernoulli (s 0.5))
   (dist/bernoulli (s 0.7))
   (dist/bernoulli (s 0.01))
   (dist/bernoulli (s 0.99))])

(def categorical-pool
  [(dist/categorical (mx/array [0.2 0.3 0.5]))
   (dist/categorical (mx/array [0.5 0.5]))
   (dist/categorical (mx/array [0.1 0.1 0.1 0.7]))
   (dist/categorical (mx/array [0.25 0.25 0.25 0.25]))])

(def binomial-pool
  [(dist/binomial (s 3) (s 0.5))
   (dist/binomial (s 4) (s 0.3))])

(def discrete-uniform-pool
  [(dist/discrete-uniform (s 0) (s 3))
   (dist/discrete-uniform (s 1) (s 5))])

;; All discrete distributions with finite, enumerable support (exclude geometric -- unbounded)
(def discrete-pool
  (vec (concat bernoulli-pool categorical-pool binomial-pool discrete-uniform-pool)))

;; Small-support pool (support size <= 5) for coverage tests
;; For support coverage: exclude extreme bernoulli (p=0.01/0.99) -- need ~500 samples
;; to reliably see the rare event. Use moderate-probability distributions only.
(def small-discrete-pool
  (vec (concat [(dist/bernoulli (s 0.3))
                (dist/bernoulli (s 0.5))
                (dist/bernoulli (s 0.7))]
               [(dist/categorical (mx/array [0.5 0.5]))
                (dist/categorical (mx/array [0.25 0.25 0.25 0.25]))]
               binomial-pool
               [(dist/discrete-uniform (s 0) (s 3))])))

;; Keys for sampling
(def key-pool
  (let [root (rng/fresh-key)]
    (vec (rng/split-n root 5))))

;; ---------------------------------------------------------------------------
;; E16.1: finite-support normalization -- sum(exp(log-prob(v))) = 1
;; Law: Probability axiom -- a valid probability measure sums to 1
;;       over its support.
;; ---------------------------------------------------------------------------

(defspec finite-support-normalization-sum-exp-log-prob-equals-1 50
  (prop/for-all [d (gen/elements discrete-pool)]
    (let [support (dc/dist-support d)
          log-probs (mapv #(dc/dist-log-prob d %) support)
          _ (apply mx/eval! log-probs)
          total (reduce + (map #(js/Math.exp (mx/item %)) log-probs))]
      (close? total 1.0 1e-4))))

;; ---------------------------------------------------------------------------
;; E16.2: log-prob-support consistency with dist-log-prob
;; Law: The bulk interface (dist-log-prob-support) and the pointwise
;;       interface (dist-log-prob) compute identical log-probabilities.
;; ---------------------------------------------------------------------------

(defspec log-prob-support-matches-per-value-dist-log-prob 50
  (prop/for-all [d (gen/elements discrete-pool)]
    (let [support (dc/dist-support d)
          bulk-lps (dc/dist-log-prob-support d)
          individual-lps (mapv #(dc/dist-log-prob d %) support)
          _ (apply mx/eval! bulk-lps individual-lps)]
      (every? true?
        (map-indexed
          (fn [i ind-lp]
            (let [bulk-val (mx/item (mx/index bulk-lps i))
                  ind-val (mx/item ind-lp)]
              (close? bulk-val ind-val 1e-5)))
          individual-lps)))))

;; ---------------------------------------------------------------------------
;; E16.3: categorical probabilities match normalized weights
;; Law: For categorical(logits), exp(log-prob(k)) = softmax(logits)[k].
;;       The parameterization defines the measure.
;; ---------------------------------------------------------------------------

;; Pre-build categorical specs with known weights for verification
(def categorical-specs
  [{:logits [0.2 0.3 0.5]  :dist (dist/categorical (mx/array [0.2 0.3 0.5]))}
   {:logits [0.5 0.5]      :dist (dist/categorical (mx/array [0.5 0.5]))}
   {:logits [0.1 0.1 0.1 0.7] :dist (dist/categorical (mx/array [0.1 0.1 0.1 0.7]))}
   {:logits [0.25 0.25 0.25 0.25] :dist (dist/categorical (mx/array [0.25 0.25 0.25 0.25]))}])

(defspec categorical-probabilities-match-normalized-weights 30
  (prop/for-all [spec (gen/elements categorical-specs)]
    (let [d (:dist spec)
          logits (:logits spec)
          ;; Compute expected softmax probabilities
          max-l (apply max logits)
          exps (mapv #(js/Math.exp (- % max-l)) logits)
          sum-exps (reduce + exps)
          expected-probs (mapv #(/ % sum-exps) exps)
          ;; Compute actual probabilities from dist-log-prob
          support (dc/dist-support d)
          actual-probs (mapv (fn [v]
                               (let [lp (dc/dist-log-prob d v)]
                                 (mx/eval! lp)
                                 (js/Math.exp (mx/item lp))))
                             support)]
      (every? true?
        (map (fn [expected actual]
               (close? expected actual 1e-4))
             expected-probs actual-probs)))))

;; ---------------------------------------------------------------------------
;; E16.4: sample support coverage (small discrete)
;; Law: Ergodicity -- every value in the support has positive probability,
;;       so sufficiently many samples must cover all support values.
;; ---------------------------------------------------------------------------

(defspec sample-support-coverage-all-support-values-appear 20
  (prop/for-all [d (gen/elements small-discrete-pool)
                 key (gen/elements key-pool)]
    (let [support (dc/dist-support d)
          n-support (count support)
          ;; Sample 200 times, collect unique values
          keys (rng/split-n key 200)
          samples (mapv (fn [k]
                          (let [v (dc/dist-sample d k)]
                            (mx/eval! v)
                            (mx/item v)))
                        keys)
          unique-vals (set samples)
          ;; Check all support values appear
          support-vals (set (map (fn [v] (mx/eval! v) (mx/item v)) support))]
      (every? #(contains? unique-vals %) support-vals))))

;; ---------------------------------------------------------------------------
;; E16.5: conjugate prior-posterior update (Beta-Bernoulli)
;; Law: Bayes' theorem for conjugate pairs -- Beta(a,b) prior with
;;       k successes in n Bernoulli trials yields Beta(a+k, b+n-k)
;;       posterior with mean (a+k)/(a+k+b+n-k).
;; ---------------------------------------------------------------------------

;; Beta(2,3) prior, 5 heads + 3 tails = 8 trials
;; Posterior: Beta(2+5, 3+3) = Beta(7, 6), mean = 7/13 ~ 0.538
(def beta-bern-model
  (gen []
    (let [p (trace :p (dist/beta-dist (s 2) (s 3)))]
      (doseq [i (range 8)]
        (trace (keyword (str "y" i))
               (dist/bernoulli p)))
      p)))

(def beta-bern-obs
  (cm/from-map {:y0 (s 1) :y1 (s 1) :y2 (s 1) :y3 (s 1) :y4 (s 1)
                :y5 (s 0) :y6 (s 0) :y7 (s 0)}))

(defspec beta-bernoulli-conjugate-posterior-mean 5
  (prop/for-all [key (gen/elements key-pool)]
    (let [;; Run MH for 200 samples with 50 burn-in
          traces (mcmc/mh {:samples 200 :burn 50 :selection (sel/select :p) :key key}
                          beta-bern-model [] beta-bern-obs)
          ;; Extract posterior samples of :p
          p-vals (mapv (fn [tr]
                         (let [v (cm/get-value (cm/get-submap (:choices tr) :p))]
                           (mx/eval! v)
                           (mx/item v)))
                       traces)
          mean-p (/ (reduce + p-vals) (count p-vals))]
      ;; Analytical posterior mean: 7/13 ~ 0.538
      (close? mean-p (/ 7.0 13.0) 0.15))))

(t/run-tests)
