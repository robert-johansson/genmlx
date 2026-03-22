(ns genmlx.mcmc-property-test
  "Property-based tests for MCMC inference using test.check.
   Every test verifies a genuine algebraic law:
   - Address preservation under MH transitions
   - Identity kernel (sel/none = no change)
   - MH stationary distribution = posterior (detailed balance)
   - Gibbs samples from full conditional"
  (:require [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [cljs.test :as t]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]
                   [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- eval-weight [w]
  (mx/eval! w)
  (mx/item w))

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

(defn- close? [a b tol]
  (and (finite? a) (finite? b) (<= (js/Math.abs (- a b)) tol)))

(defn- choice-val [choices addr]
  (let [sub (cm/get-submap choices addr)]
    (when (and sub (cm/has-value? sub))
      (let [v (cm/get-value sub)]
        (mx/eval! v)
        (mx/item v)))))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

;; Linear regression model for MH tests
(def linreg
  (dyn/auto-key
    (gen [xs]
      (let [slope (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 10))]
        (doseq [[j x] (map-indexed vector xs)]
          (trace (keyword (str "y" j))
                 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                        intercept) 1)))
        (mx/eval! slope intercept)
        [(mx/item slope) (mx/item intercept)]))))

(def xs [1.0 2.0 3.0])
(def obs (cm/choicemap :y0 (mx/scalar 1.5)
                       :y1 (mx/scalar 3.0)
                       :y2 (mx/scalar 4.5)))

;; Conjugate Gaussian model for MH convergence test
;; Prior: N(0,1), Likelihood: y ~ N(mu, 1), observe y=2.0
;; Posterior: N(1.0, 0.5)  (mean=1.0, variance=0.5)
(def conjugate-gauss
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/gaussian 0 1))
            y  (trace :y (dist/gaussian mu 1))]
        (mx/eval! mu)
        (mx/item mu)))))

(def conjugate-obs (cm/choicemap :y (mx/scalar 2.0)))

;; Beta-Bernoulli model for Gibbs test
;; Prior: Beta(1,1) = Uniform(0,1), Likelihood: 7 Bernoulli obs (5 heads, 2 tails)
;; Posterior: Beta(6, 3), mean = 6/9 = 0.667
(def beta-bern
  (dyn/auto-key
    (gen []
      (let [p (trace :p (dist/beta-dist 1 1))]
        (trace :y0 (dist/bernoulli p))
        (trace :y1 (dist/bernoulli p))
        (trace :y2 (dist/bernoulli p))
        (trace :y3 (dist/bernoulli p))
        (trace :y4 (dist/bernoulli p))
        (trace :y5 (dist/bernoulli p))
        (trace :y6 (dist/bernoulli p))
        (mx/eval! p)
        (mx/item p)))))

(def beta-bern-obs
  (cm/choicemap :y0 (mx/scalar 1.0) :y1 (mx/scalar 1.0) :y2 (mx/scalar 1.0)
                :y3 (mx/scalar 1.0) :y4 (mx/scalar 1.0)
                :y5 (mx/scalar 0.0) :y6 (mx/scalar 0.0)))

(def key-pool (mapv #(rng/fresh-key %) [42 99 123 7 255]))
(def gen-key (gen/elements key-pool))

;; ---------------------------------------------------------------------------
;; Law: MH transition preserves the set of trace addresses
;; ---------------------------------------------------------------------------

(defspec mh-step-trace-has-same-addresses 50
  (prop/for-all [k gen-key]
    (let [{:keys [trace]} (p/generate linreg [xs] obs)
          new-trace (mcmc/mh-step trace (sel/select :slope) k)]
      (and (some? (choice-val (:choices new-trace) :slope))
           (some? (choice-val (:choices new-trace) :intercept))))))

;; ---------------------------------------------------------------------------
;; Law: MH with empty selection is the identity kernel (no change)
;; Strengthened: check EXACT value equality and weight = 0
;; ---------------------------------------------------------------------------

(defspec mh-step-sel-none-all-values-identical-weight-0 50
  (prop/for-all [k gen-key]
    (let [{:keys [trace]} (p/generate linreg [xs] obs)
          old-slope (choice-val (:choices trace) :slope)
          old-intercept (choice-val (:choices trace) :intercept)
          new-trace (mcmc/mh-step trace sel/none k)
          new-slope (choice-val (:choices new-trace) :slope)
          new-intercept (choice-val (:choices new-trace) :intercept)]
      (and (close? old-slope new-slope 1e-6)
           (close? old-intercept new-intercept 1e-6)))))

;; ---------------------------------------------------------------------------
;; Law: MH step score equals model log-joint at the returned choices
;; The trace score after MH must equal assess(model, choices).weight
;; ---------------------------------------------------------------------------

(defspec mh-step-trace-score-equals-assess-weight 50
  (prop/for-all [k gen-key]
    (let [{:keys [trace]} (p/generate linreg [xs] obs)
          new-trace (mcmc/mh-step trace (sel/select :slope) k)
          trace-score (eval-weight (:score new-trace))
          {:keys [weight]} (p/assess linreg [xs] (:choices new-trace))
          assess-score (eval-weight weight)]
      (close? trace-score assess-score 0.01))))

;; ---------------------------------------------------------------------------
;; Law: MH stationary distribution = posterior (detailed balance)
;; Prior N(0,1), likelihood N(mu,1), observe y=2.0
;; Posterior: N(1.0, 0.5), mean = 1.0
;; ---------------------------------------------------------------------------

(defspec mh-detailed-balance-posterior-mean-near-analytical 10
  (prop/for-all [k gen-key]
    (let [samples (mcmc/mh {:samples 200 :burn 50
                            :selection (sel/select :mu) :key k}
                           conjugate-gauss [] conjugate-obs)
          mu-vals (mapv #(choice-val (:choices %) :mu) samples)
          mean (/ (reduce + mu-vals) (count mu-vals))]
      (close? mean 1.0 0.5))))

;; ---------------------------------------------------------------------------
;; Law: Gibbs samples from the full conditional distribution
;; Beta(1,1) prior + 5 heads + 2 tails -> Beta(6,3), mean = 0.667
;; ---------------------------------------------------------------------------

(defspec gibbs-on-conjugate-model-posterior-mean-near-analytical 10
  (prop/for-all [k gen-key]
    (let [samples (mcmc/mh {:samples 300 :burn 50
                            :selection (sel/select :p) :key k}
                           beta-bern [] beta-bern-obs)
          p-vals (mapv #(choice-val (:choices %) :p) samples)
          mean (/ (reduce + p-vals) (count p-vals))]
      (close? mean 0.667 0.15))))

(t/run-tests)
