(ns genmlx.test-helpers
  "Shared test utilities for the GenMLX test suite.
   Provides numerical comparison, MLX helpers, PRNG management,
   statistical testing, and cljs.test fixtures."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

;; ---------------------------------------------------------------------------
;; Numerical comparison
;; ---------------------------------------------------------------------------

(defn close?
  "True if |expected - actual| <= tolerance."
  ([expected actual] (close? expected actual 1e-6))
  ([expected actual tol]
   (<= (js/Math.abs (- expected actual)) tol)))

(defn all-close?
  "True if every element pair is within tolerance. Works on seqs."
  ([expected actual] (all-close? expected actual 1e-6))
  ([expected actual tol]
   (and (= (count expected) (count actual))
        (every? true? (map #(close? %1 %2 tol) expected actual)))))

;; ---------------------------------------------------------------------------
;; MLX helpers
;; ---------------------------------------------------------------------------

(defn realize
  "mx/eval! then mx/item. Returns JS number."
  [x]
  (mx/eval! x)
  (mx/item x))

(defn realize-shape
  "mx/eval! then mx/shape. Returns shape vector."
  [x]
  (mx/eval! x)
  (mx/shape x))

(defn realize-vec
  "mx/eval! then mx/->clj. Returns Clojure vector."
  [x]
  (mx/eval! x)
  (mx/->clj x))

;; ---------------------------------------------------------------------------
;; PRNG
;; ---------------------------------------------------------------------------

(defn deterministic-key
  "Fixed PRNG key for reproducible tests."
  ([] (rng/fresh-key 42))
  ([seed] (rng/fresh-key seed)))

;; ---------------------------------------------------------------------------
;; Statistical helpers
;; ---------------------------------------------------------------------------

(defn sample-mean
  "Mean of a seq of numbers."
  [xs]
  (/ (reduce + xs) (count xs)))

(defn sample-variance
  "Unbiased sample variance."
  [xs]
  (let [mu (sample-mean xs)
        n (count xs)]
    (/ (reduce + (map #(let [d (- % mu)] (* d d)) xs))
       (dec n))))

(defn sample-std-error
  "Standard error of the mean."
  [xs]
  (js/Math.sqrt (/ (sample-variance xs) (count xs))))

(defn z-test-passes?
  "True if sample mean is within z-sigma standard errors of expected.
   Default z=3.5 gives false-positive rate < 0.0005."
  ([expected xs] (z-test-passes? expected xs 3.5))
  ([expected xs z]
   (let [se (sample-std-error xs)]
     (if (zero? se)
       (close? expected (sample-mean xs) 1e-6)
       (<= (js/Math.abs (/ (- (sample-mean xs) expected) se)) z)))))

;; ---------------------------------------------------------------------------
;; cljs.test fixture
;; ---------------------------------------------------------------------------

;; ---------------------------------------------------------------------------
;; Analytical log-prob helpers
;; ---------------------------------------------------------------------------

(def LOG-2PI
  "log(2π) — the normalization constant for gaussian log-prob."
  (js/Math.log (* 2 js/Math.PI)))

(defn gaussian-lp
  "Analytically compute log N(x; mu, sigma).
   = -0.5*log(2π) - log(σ) - 0.5*((x-μ)/σ)²"
  [x mu sigma]
  (let [z (/ (- x mu) sigma)]
    (- (* -0.5 LOG-2PI) (js/Math.log sigma) (* 0.5 z z))))

(def mlx-cleanup-fixture
  "Reusable :each fixture for MLX cleanup.
   Usage: (t/use-fixtures :each test-helpers/mlx-cleanup-fixture)"
  {:before (fn [] nil)
   :after (fn [] nil)})
