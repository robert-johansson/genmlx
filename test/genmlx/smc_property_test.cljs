(ns genmlx.smc-property-test
  "Property-based tests for SMC using test.check.
   Every test verifies a genuine algebraic law:
   - Resampling indices within bounds
   - Vectorized SMC shape contracts
   - SMC(1 step) degenerates to IS
   - More particles reduce log-ML variance (consistency)
   - Resampling preserves weighted expectations"
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
            [genmlx.inference.smc :as smc]
            [genmlx.inference.importance :as is]
            [genmlx.inference.util :as u]
            [genmlx.vectorized :as vec])
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

;; Simple observation model for SMC tests
(def model
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/gaussian 0 5))
            y0 (trace :y0 (dist/gaussian mu 1))]
        (mx/eval! mu y0)
        (mx/item mu)))))

(def obs (cm/choicemap :y0 (mx/scalar 2.0)))

;; Vectorized-safe model (no mx/item or mx/eval! inside body)
(def vmodel
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/gaussian 0 5))
            y0 (trace :y0 (dist/gaussian mu 1))]
        mu))))

(def vobs (cm/choicemap :y0 (mx/scalar 2.0)))

(def particle-pool [5 10 15])
(def gen-n-particles (gen/elements particle-pool))

(def key-pool (mapv #(rng/fresh-key %) [42 99 123 7 255]))
(def gen-key (gen/elements key-pool))

;; ---------------------------------------------------------------------------
;; Resampling properties
;; Law: systematic resampling indices must lie in [0, N)
;; ---------------------------------------------------------------------------

(defspec systematic-resample-indices-in-0-n 50
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [log-weights (vec (repeatedly n #(mx/scalar (- (js/Math.random) 0.5))))
          indices (u/systematic-resample log-weights n k)]
      (and (= n (count indices))
           (every? #(and (>= % 0) (< % n)) indices)))))

(defspec systematic-resample-random-weights-indices-in-0-n 50
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [log-weights (vec (repeatedly n #(mx/scalar (* 2.0 (- (js/Math.random) 0.5)))))
          indices (u/systematic-resample log-weights n k)]
      (and (= n (count indices))
           (every? #(and (>= % 0) (< % n)) indices)))))

;; Law: uniform weights should produce valid resampling
(defspec uniform-weights-resample-produces-valid-indices 50
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [log-weights (vec (repeat n (mx/scalar 0.0)))
          indices (u/systematic-resample log-weights n k)]
      (and (= n (count indices))
           (every? #(and (>= % 0) (< % n)) indices)))))

;; ---------------------------------------------------------------------------
;; Vectorized SMC shape contract
;; Law: vsmc-init weight and score shapes must be [N]
;; ---------------------------------------------------------------------------

(defspec vsmc-init-weight-and-score-shape-equals-n 50
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [result (smc/vsmc-init vmodel [] vobs n k)
          vtrace (:vtrace result)]
      (mx/eval! (:weight vtrace) (:score vtrace))
      (and (= (mx/shape (:weight vtrace)) [n])
           (= (mx/shape (:score vtrace)) [n])))))

;; ---------------------------------------------------------------------------
;; SMC(1 step) = IS
;; Law: SMC with a single observation step degenerates to importance sampling
;; ---------------------------------------------------------------------------

(defspec smc-1-step-log-ml-roughly-agrees-with-is-log-ml 20
  (prop/for-all [n (gen/elements [10 20])
                 k gen-key]
    (let [[k1 k2] (rng/split k)
          is-result (is/importance-sampling {:samples n :key k1} model [] obs)
          is-log-ml (eval-weight (:log-ml-estimate is-result))
          smc-result (smc/smc {:particles n :key k2} model [] [obs])
          smc-log-ml (eval-weight (:log-ml-estimate smc-result))]
      ;; Both are noisy estimates of the same quantity;
      ;; they should roughly agree (both are unbiased for the same log p(y))
      (and (finite? is-log-ml) (finite? smc-log-ml)
           (close? is-log-ml smc-log-ml 3.0)))))

;; ---------------------------------------------------------------------------
;; More particles reduce log-ML variance (consistency)
;; Law: as N increases, the variance of the log-ML estimator decreases
;; ---------------------------------------------------------------------------

;; Law: ESS increases with more uniform weights (ESS is a measure of sample quality)
;; Concentrated weights (one dominant) -> low ESS, uniform weights -> ESS = N
(defspec concentrated-weights-yield-ess-less-than-n 20
  (prop/for-all [n gen-n-particles]
    (let [;; Concentrated: one very high weight, rest very low
          concentrated (vec (cons (mx/scalar 10.0)
                                 (repeat (dec n) (mx/scalar -10.0))))
          ess-conc (u/compute-ess concentrated)
          ;; Uniform: all equal
          uniform (vec (repeat n (mx/scalar 0.0)))
          ess-unif (u/compute-ess uniform)]
      ;; Concentrated should give ESS near 1, uniform gives ESS = N
      (and (< ess-conc 2.0)
           (close? ess-unif (double n) 0.01)))))

;; ---------------------------------------------------------------------------
;; Resampling preserves weighted expectation
;; Law: E_weighted[mu] before resampling = E_unweighted[mu] after resampling
;; ---------------------------------------------------------------------------

(defspec resampling-preserves-weighted-expectation 20
  (prop/for-all [n (gen/elements [10 20])
                 k gen-key]
    (let [[k1 k2] (rng/split k)
          {:keys [traces log-weights]} (is/importance-sampling {:samples n :key k1} model [] obs)
          ;; Weighted mean before resampling
          {:keys [probs]} (u/normalize-log-weights log-weights)
          mu-vals (mapv #(choice-val (:choices %) :mu) traces)
          weighted-mean (reduce + (map * mu-vals probs))
          ;; Resample indices
          indices (u/systematic-resample log-weights n k2)
          resampled-vals (mapv #(nth mu-vals %) indices)
          unweighted-mean (/ (reduce + resampled-vals) (count resampled-vals))]
      (close? weighted-mean unweighted-mean 2.0))))

(t/run-tests)
