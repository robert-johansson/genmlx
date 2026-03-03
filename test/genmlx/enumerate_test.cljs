(ns genmlx.enumerate-test
  "Tests for enumerative / grid-based inference."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.enumerate :as enum])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn- assert-true [desc pred]
  (if pred
    (println (str "  PASS: " desc))
    (println (str "  FAIL: " desc))))

(defn- assert-close [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (println (str "  PASS: " desc " (expected=" expected " actual=" actual ")"))
      (println (str "  FAIL: " desc " (expected=" expected " actual=" actual " diff=" diff " tol=" tol ")")))))

;; ---------------------------------------------------------------------------
;; Two-coin model
;; ---------------------------------------------------------------------------

(def two-coins
  (gen []
    (let [c1 (trace :c1 (dist/bernoulli 0.5))
          c2 (trace :c2 (dist/bernoulli 0.5))]
      (mx/add c1 c2))))

(println "\n== Enumerative inference ==")

(println "\n-- enumerate-joint: two coins --")
(let [joint (enum/enumerate-joint (dyn/auto-key two-coins) [] nil
              {:c1 [(mx/scalar 0) (mx/scalar 1)]
               :c2 [(mx/scalar 0) (mx/scalar 1)]})
      total-prob (reduce + (map :prob joint))]
  (assert-close "joint probabilities sum to 1.0" 1.0 total-prob 1e-6)
  (assert-true "4 entries in joint" (= 4 (count joint)))
  ;; Each combination is equally likely for fair coins
  (doseq [{:keys [prob]} joint]
    (assert-close "each combination has prob 0.25" 0.25 prob 1e-6)))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; enumerate-marginals
;; ---------------------------------------------------------------------------

(println "\n-- enumerate-marginals: two coins --")
(let [marginals (enum/enumerate-marginals (dyn/auto-key two-coins) [] nil
                  {:c1 [(mx/scalar 0) (mx/scalar 1)]
                   :c2 [(mx/scalar 0) (mx/scalar 1)]})]
  (assert-close "P(c1=0) = 0.5" 0.5 (get-in marginals [:c1 0.0]) 1e-6)
  (assert-close "P(c1=1) = 0.5" 0.5 (get-in marginals [:c1 1.0]) 1e-6)
  (assert-close "P(c2=0) = 0.5" 0.5 (get-in marginals [:c2 0.0]) 1e-6)
  (assert-close "P(c2=1) = 0.5" 0.5 (get-in marginals [:c2 1.0]) 1e-6))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; Posterior with observations — biased coin model
;; ---------------------------------------------------------------------------

(def biased-coin-model
  (gen []
    (let [coin (trace :coin (dist/bernoulli 0.7))
          obs  (trace :obs (dist/bernoulli
                             (if (pos? (mx/item coin)) 0.9 0.1)))]
      coin)))

(println "\n-- enumerate-marginals: posterior with observation --")
(let [;; Condition on obs=1
      obs (cm/choicemap :obs (mx/scalar 1))
      marginals (enum/enumerate-marginals (dyn/auto-key biased-coin-model) [] obs
                  {:coin [(mx/scalar 0) (mx/scalar 1)]})
      ;; P(coin=1 | obs=1) = P(obs=1|coin=1)*P(coin=1) / P(obs=1)
      ;; = 0.9 * 0.7 / (0.9*0.7 + 0.1*0.3)
      ;; = 0.63 / (0.63 + 0.03) = 0.63 / 0.66 ≈ 0.9545
      expected-p1 (/ (* 0.9 0.7) (+ (* 0.9 0.7) (* 0.1 0.3)))]
  (assert-close "posterior P(coin=1|obs=1) matches Bayes' rule"
    expected-p1 (get-in marginals [:coin 1.0]) 0.01))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; enumerate-marginal-likelihood
;; ---------------------------------------------------------------------------

(println "\n-- enumerate-marginal-likelihood --")
(let [obs (cm/choicemap :obs (mx/scalar 1))
      log-ml (mx/item (enum/enumerate-marginal-likelihood
                         (dyn/auto-key biased-coin-model) [] obs
                         {:coin [(mx/scalar 0) (mx/scalar 1)]}))
      ;; P(obs=1) = 0.9*0.7 + 0.1*0.3 = 0.66
      expected-log-ml (js/Math.log 0.66)]
  (assert-close "marginal likelihood matches analytical"
    expected-log-ml log-ml 0.01))

(mx/clear-cache!)

;; ---------------------------------------------------------------------------
;; Cardinality guard
;; ---------------------------------------------------------------------------

(println "\n-- cardinality guard --")
(let [big-support (into {} (map (fn [i] [(keyword (str "x" i))
                                          (mapv mx/scalar (range 20))])
                                (range 5)))
      ;; 20^5 = 3.2M > 10000
      threw? (try
               (enum/enumerate-joint (dyn/auto-key two-coins) [] nil big-support)
               false
               (catch :default e
                 (let [msg (.-message e)]
                   (boolean (re-find #"10000" msg)))))]
  (assert-true "throws on too-large Cartesian product" threw?))

;; ---------------------------------------------------------------------------
;; Custom :max-combinations option
;; ---------------------------------------------------------------------------

(println "\n-- custom :max-combinations --")
;; two-coins has 2*2 = 4 combinations
(let [supports {:c1 [(mx/scalar 0) (mx/scalar 1)]
                :c2 [(mx/scalar 0) (mx/scalar 1)]}
      ;; max-combinations=3 should reject 4 combos
      threw? (try
               (enum/enumerate-joint (dyn/auto-key two-coins) [] nil supports
                                     {:max-combinations 3})
               false
               (catch :default _ true))]
  (assert-true ":max-combinations 3 rejects 4 combos" threw?))

(let [supports {:c1 [(mx/scalar 0) (mx/scalar 1)]
                :c2 [(mx/scalar 0) (mx/scalar 1)]}
      ;; max-combinations=100 should allow 4 combos
      joint (enum/enumerate-joint (dyn/auto-key two-coins) [] nil supports
                                  {:max-combinations 100})]
  (assert-true ":max-combinations 100 allows 4 combos" (= 4 (count joint))))

(mx/clear-cache!)

(println "\n-- enumerate-marginals accepts opts --")
(let [supports {:c1 [(mx/scalar 0) (mx/scalar 1)]
                :c2 [(mx/scalar 0) (mx/scalar 1)]}
      marginals (enum/enumerate-marginals (dyn/auto-key two-coins) [] nil supports
                                          {:max-combinations 100})]
  (assert-close "P(c1=0) = 0.5 with opts" 0.5 (get-in marginals [:c1 0.0]) 1e-6))

(mx/clear-cache!)

(println "\n-- enumerate-marginal-likelihood accepts opts --")
(let [supports {:c1 [(mx/scalar 0) (mx/scalar 1)]
                :c2 [(mx/scalar 0) (mx/scalar 1)]}
      log-ml (mx/item (enum/enumerate-marginal-likelihood
                         (dyn/auto-key two-coins) [] nil supports
                         {:max-combinations 100}))]
  (assert-true "marginal-likelihood is finite with opts" (js/isFinite log-ml)))

(println "\n== All enumerative inference tests complete ==")
