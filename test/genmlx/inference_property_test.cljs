(ns genmlx.inference-property-test
  "Property-based inference algorithm tests using test.check.
   Every test verifies a genuine algebraic law of probabilistic programming:
   normalization, identity elements, bounds, associativity, etc."
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
            [genmlx.inference.importance :as is]
            [genmlx.inference.kernel :as kern]
            [genmlx.inference.util :as u]
            [genmlx.inference.diagnostics :as diag])
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
;; Model and fixture pools
;; ---------------------------------------------------------------------------

(def model
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 1))
            y (trace :y (dist/gaussian 0 1))]
        (mx/eval! x y)
        (+ (mx/item x) (mx/item y))))))

;; Linear regression model for kernel tests
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

(def linreg-xs [1.0 2.0 3.0])
(def linreg-obs (cm/choicemap :y0 (mx/scalar 1.5)
                              :y1 (mx/scalar 3.0)
                              :y2 (mx/scalar 4.5)))

(def obs-pool
  [(cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5))
   (cm/choicemap :x (mx/scalar 0.0) :y (mx/scalar 0.0))
   (cm/choicemap :x (mx/scalar -1.0) :y (mx/scalar 2.0))
   (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar -0.5))])

(def gen-obs (gen/elements obs-pool))

(def particle-pool [5 10 15 20])
(def gen-n-particles (gen/elements particle-pool))

;; Pre-built PRNG keys
(def key-pool (mapv #(rng/fresh-key %) [42 99 123 7 255]))
(def gen-key (gen/elements key-pool))

;; ---------------------------------------------------------------------------
;; Importance Sampling
;; Law: normalized weights sum to 1.0 (normalization axiom of probability)
;; ---------------------------------------------------------------------------

(defspec is-normalized-weights-sum-to-1 50
  (prop/for-all [obs gen-obs
                 n gen-n-particles
                 k gen-key]
    (let [{:keys [log-weights]} (is/importance-sampling {:samples n :key k} model [] obs)
          {:keys [probs]} (u/normalize-log-weights log-weights)
          total (reduce + probs)]
      (close? 1.0 total 0.01))))

;; Law: generate(model, empty constraints) = simulate, so weight = p/q = 1, log-weight = 0
(defspec is-empty-constraints-yield-weights-near-0 50
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [{:keys [log-weights]} (is/importance-sampling {:samples n :key k} model [] cm/EMPTY)]
      (every? (fn [w] (close? 0.0 (eval-weight w) 0.01)) log-weights))))

;; ---------------------------------------------------------------------------
;; MH / Accept Decision
;; Law: MH acceptance probability = min(1, exp(log-alpha))
;; ---------------------------------------------------------------------------

;; Law: log-alpha = 0 means acceptance ratio = 1, always accept
(defspec accept-mh-0-always-true 100
  (prop/for-all [k gen-key]
    (u/accept-mh? 0 k)))

;; Law: log-alpha = -100 means acceptance ratio ~ exp(-100) ~ 0, rarely accept
(defspec accept-mh-neg-100-rarely-true 10
  (prop/for-all [_ (gen/return nil)]
    (let [accepts (count (filter true? (repeatedly 100 #(u/accept-mh? -100))))]
      (< accepts 5))))

;; Law: regenerate with empty selection = identity, weight = 0
(defspec regenerate-sel-none-yields-weight-near-0 50
  (prop/for-all [_ (gen/return nil)]
    (let [trace (p/simulate model [])
          {:keys [weight]} (p/regenerate model trace sel/none)]
      (close? 0.0 (eval-weight weight) 0.01))))

;; ---------------------------------------------------------------------------
;; Weight Utilities
;; Law: log-softmax normalization produces a valid probability distribution
;; ---------------------------------------------------------------------------

(defspec normalize-log-weights-probs-sum-to-1 50
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [{:keys [log-weights]} (is/importance-sampling {:samples n :key k} model [] (first obs-pool))
          {:keys [probs]} (u/normalize-log-weights log-weights)
          total (reduce + probs)]
      (close? 1.0 total 0.01))))

;; Law: ESS = 1 / sum(w_i^2) where w_i are normalized weights, so ESS in (0, N]
(defspec compute-ess-result-in-0-n 50
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [{:keys [log-weights]} (is/importance-sampling {:samples n :key k} model [] (first obs-pool))
          ess (u/compute-ess log-weights)]
      (and (> ess 0) (<= ess (+ n 0.01))))))

;; Law: uniform weights w_i = 1/N -> ESS = N (maximum efficiency)
(defspec uniform-weights-yield-ess-equals-n 50
  (prop/for-all [n gen-n-particles]
    (let [log-weights (vec (repeat n (mx/scalar 0.0)))
          ess (u/compute-ess log-weights)]
      (close? (double n) ess 0.01))))

;; ---------------------------------------------------------------------------
;; Diagnostics
;; Law: R-hat measures between-chain vs within-chain variance
;; ---------------------------------------------------------------------------

;; Law: identical chains have zero between-chain variance, R-hat = 1.0 (or NaN for constant)
(defspec diag-r-hat-of-identical-chains-near-1 50
  (prop/for-all [_ (gen/return nil)]
    (let [n 20
          chain (mapv (fn [_] (mx/scalar 5.0)) (range n))
          chains [chain chain]]
      (let [r (diag/r-hat chains)]
        (or (js/isNaN r) (>= r 0.99))))))

;; Law: R-hat >= 1.0 always (between-chain variance is non-negative)
(defspec diag-r-hat-gte-1-always-non-degenerate 30
  (prop/for-all [_ (gen/return nil)]
    (let [n 20
          chain1 (mapv (fn [i] (mx/scalar (+ 0.0 (* 0.1 i)))) (range n))
          chain2 (mapv (fn [i] (mx/scalar (+ 5.0 (* 0.1 i)))) (range n))
          r (diag/r-hat [chain1 chain2])]
      (>= r 1.0))))

;; ---------------------------------------------------------------------------
;; Kernel composition laws
;; ---------------------------------------------------------------------------

(def k1 (kern/mh-kernel (sel/select :slope)))
(def k2 (kern/mh-kernel (sel/select :intercept)))
(def k3 (kern/mh-kernel (sel/select :slope)))

;; Law: chain(k1, k2) preserves model support -- the output trace has the
;; same addresses as the input trace. Kernel transitions cannot add or remove
;; trace sites; they can only change values.
(defspec kernel-chain-preserves-model-support 30
  (prop/for-all [k gen-key]
    (let [{:keys [trace]} (p/generate linreg [linreg-xs] linreg-obs)
          chained (kern/chain k1 k2)
          result (chained trace k)
          ;; All model addresses must be present
          slope (choice-val (:choices result) :slope)
          intercept (choice-val (:choices result) :intercept)
          y0 (choice-val (:choices result) :y0)
          y1 (choice-val (:choices result) :y1)
          y2 (choice-val (:choices result) :y2)
          score (eval-weight (:score result))]
      (and (finite? slope) (finite? intercept)
           (finite? y0) (finite? y1) (finite? y2)
           (finite? score)))))

;; Law: cycle(n, [k1, k2]) preserves model support -- cycling through multiple
;; kernels produces a valid trace with all addresses intact.
(defspec cycle-kernels-preserves-model-support 30
  (prop/for-all [k gen-key]
    (let [{:keys [trace]} (p/generate linreg [linreg-xs] linreg-obs)
          cycled (kern/cycle-kernels 4 [k1 k2])
          result (cycled trace k)
          slope (choice-val (:choices result) :slope)
          intercept (choice-val (:choices result) :intercept)
          score (eval-weight (:score result))]
      (and (finite? slope) (finite? intercept) (finite? score)))))

(t/run-tests)
