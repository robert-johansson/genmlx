(ns genmlx.enumerate-test
  "Tests for enumerative / grid-based inference."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.enumerate :as enum])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Model definitions
;; ---------------------------------------------------------------------------

(def two-coins
  (gen []
    (let [c1 (trace :c1 (dist/bernoulli 0.5))
          c2 (trace :c2 (dist/bernoulli 0.5))]
      (mx/add c1 c2))))

(def biased-coin-model
  (gen []
    (let [coin (trace :coin (dist/bernoulli 0.7))
          obs  (trace :obs (dist/bernoulli
                             (if (pos? (mx/item coin)) 0.9 0.1)))]
      coin)))

;; ---------------------------------------------------------------------------
;; enumerate-joint
;; ---------------------------------------------------------------------------

(deftest enumerate-joint-two-coins
  (testing "enumerate-joint: two coins"
    (let [joint (enum/enumerate-joint (dyn/auto-key two-coins) [] nil
                  {:c1 [(mx/scalar 0) (mx/scalar 1)]
                   :c2 [(mx/scalar 0) (mx/scalar 1)]})
          total-prob (reduce + (map :prob joint))]
      (is (h/close? 1.0 total-prob 1e-6) "joint probabilities sum to 1.0")
      (is (= 4 (count joint)) "4 entries in joint")
      (doseq [{:keys [prob]} joint]
        (is (h/close? 0.25 prob 1e-6) "each combination has prob 0.25"))))
  (mx/clear-cache!))

;; ---------------------------------------------------------------------------
;; enumerate-marginals
;; ---------------------------------------------------------------------------

(deftest enumerate-marginals-two-coins
  (testing "enumerate-marginals: two coins"
    (let [marginals (enum/enumerate-marginals (dyn/auto-key two-coins) [] nil
                      {:c1 [(mx/scalar 0) (mx/scalar 1)]
                       :c2 [(mx/scalar 0) (mx/scalar 1)]})]
      (is (h/close? 0.5 (get-in marginals [:c1 0.0]) 1e-6) "P(c1=0) = 0.5")
      (is (h/close? 0.5 (get-in marginals [:c1 1.0]) 1e-6) "P(c1=1) = 0.5")
      (is (h/close? 0.5 (get-in marginals [:c2 0.0]) 1e-6) "P(c2=0) = 0.5")
      (is (h/close? 0.5 (get-in marginals [:c2 1.0]) 1e-6) "P(c2=1) = 0.5")))
  (mx/clear-cache!))

;; ---------------------------------------------------------------------------
;; Posterior with observations — biased coin model
;; ---------------------------------------------------------------------------

(deftest enumerate-posterior-biased-coin
  (testing "enumerate-marginals: posterior with observation"
    (let [obs (cm/choicemap :obs (mx/scalar 1))
          marginals (enum/enumerate-marginals (dyn/auto-key biased-coin-model) [] obs
                      {:coin [(mx/scalar 0) (mx/scalar 1)]})
          expected-p1 (/ (* 0.9 0.7) (+ (* 0.9 0.7) (* 0.1 0.3)))]
      (is (h/close? expected-p1 (get-in marginals [:coin 1.0]) 0.01)
          "posterior P(coin=1|obs=1) matches Bayes' rule")))
  (mx/clear-cache!))

;; ---------------------------------------------------------------------------
;; enumerate-marginal-likelihood
;; ---------------------------------------------------------------------------

(deftest enumerate-marginal-likelihood-test
  (testing "enumerate-marginal-likelihood"
    (let [obs (cm/choicemap :obs (mx/scalar 1))
          log-ml (mx/item (enum/enumerate-marginal-likelihood
                             (dyn/auto-key biased-coin-model) [] obs
                             {:coin [(mx/scalar 0) (mx/scalar 1)]}))
          expected-log-ml (js/Math.log 0.66)]
      (is (h/close? expected-log-ml log-ml 0.01) "marginal likelihood matches analytical")))
  (mx/clear-cache!))

;; ---------------------------------------------------------------------------
;; Cardinality guard
;; ---------------------------------------------------------------------------

(deftest cardinality-guard
  (testing "throws on too-large Cartesian product"
    (let [big-support (into {} (map (fn [i] [(keyword (str "x" i))
                                              (mapv mx/scalar (range 20))])
                                    (range 5)))
          threw? (try
                   (enum/enumerate-joint (dyn/auto-key two-coins) [] nil big-support)
                   false
                   (catch :default e
                     (let [msg (.-message e)]
                       (boolean (re-find #"10000" msg)))))]
      (is threw? "throws on too-large Cartesian product"))))

;; ---------------------------------------------------------------------------
;; Custom :max-combinations option
;; ---------------------------------------------------------------------------

(deftest custom-max-combinations
  (testing ":max-combinations 3 rejects 4 combos"
    (let [supports {:c1 [(mx/scalar 0) (mx/scalar 1)]
                    :c2 [(mx/scalar 0) (mx/scalar 1)]}
          threw? (try
                   (enum/enumerate-joint (dyn/auto-key two-coins) [] nil supports
                                         {:max-combinations 3})
                   false
                   (catch :default _ true))]
      (is threw? ":max-combinations 3 rejects 4 combos")))

  (testing ":max-combinations 100 allows 4 combos"
    (let [supports {:c1 [(mx/scalar 0) (mx/scalar 1)]
                    :c2 [(mx/scalar 0) (mx/scalar 1)]}
          joint (enum/enumerate-joint (dyn/auto-key two-coins) [] nil supports
                                      {:max-combinations 100})]
      (is (= 4 (count joint)) ":max-combinations 100 allows 4 combos")))
  (mx/clear-cache!))

(deftest enumerate-opts-passthrough
  (testing "enumerate-marginals accepts opts"
    (let [supports {:c1 [(mx/scalar 0) (mx/scalar 1)]
                    :c2 [(mx/scalar 0) (mx/scalar 1)]}
          marginals (enum/enumerate-marginals (dyn/auto-key two-coins) [] nil supports
                                              {:max-combinations 100})]
      (is (h/close? 0.5 (get-in marginals [:c1 0.0]) 1e-6) "P(c1=0) = 0.5 with opts")))
  (mx/clear-cache!)

  (testing "enumerate-marginal-likelihood accepts opts"
    (let [supports {:c1 [(mx/scalar 0) (mx/scalar 1)]
                    :c2 [(mx/scalar 0) (mx/scalar 1)]}
          log-ml (mx/item (enum/enumerate-marginal-likelihood
                             (dyn/auto-key two-coins) [] nil supports
                             {:max-combinations 100}))]
      (is (js/isFinite log-ml) "marginal-likelihood is finite with opts")))
  (mx/clear-cache!))

(cljs.test/run-tests)
