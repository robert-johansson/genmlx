(ns genmlx.auto-wiring-test
  "WP-2 Gate 2: Auto-wiring tests — verify that DynamicGF automatically
   detects conjugate pairs and uses analytical handlers in p/generate.
   Side-by-side comparison: auto-analytical vs manual analytical vs standard."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.conjugacy :as conjugacy]
            [genmlx.inference.auto-analytical :as auto-analytical]))

;; ---------------------------------------------------------------------------
;; Model definitions
;; ---------------------------------------------------------------------------

(def nn-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 2))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      mu)))

(def bb-model
  (gen []
    (let [p (trace :p (dist/beta-dist 2 3))]
      (trace :x1 (dist/bernoulli p))
      (trace :x2 (dist/bernoulli p))
      p)))

(def gp-model
  (gen []
    (let [rate (trace :rate (dist/gamma-dist 3 2))]
      (trace :c1 (dist/poisson rate))
      rate)))

(def ge-model
  (gen []
    (let [rate (trace :rate (dist/gamma-dist 2 1))]
      (trace :t1 (dist/exponential rate))
      rate)))

(def non-conj-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 1))]
      (trace :x (dist/bernoulli (mx/sigmoid mu)))
      mu)))

(def nn-model-3obs
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 2))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian mu 1))
      mu)))

(def nn-5obs
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 2))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian mu 1))
      (trace :y4 (dist/gaussian mu 1))
      (trace :y5 (dist/gaussian mu 1))
      mu)))

(def mixed-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 2))
          p  (trace :p (dist/beta-dist 2 3))]
      (trace :y (dist/gaussian mu 1))
      (trace :x (dist/bernoulli p))
      [mu p])))

;; ---------------------------------------------------------------------------
;; Section 1: Schema augmentation in make-gen-fn
;; ---------------------------------------------------------------------------

(deftest schema-auto-augmentation
  (testing "Normal-Normal model schema"
    (let [schema (:schema nn-model)]
      (is (:has-conjugate? schema) "NN: has-conjugate?")
      (is (= 2 (count (:conjugate-pairs schema))) "NN: 2 conjugate pairs")
      (is (map? (:auto-handlers schema)) "NN: auto-handlers present")
      (is (fn? (get (:auto-handlers schema) :mu)) "NN: handler for :mu")
      (is (fn? (get (:auto-handlers schema) :y1)) "NN: handler for :y1")
      (is (fn? (get (:auto-handlers schema) :y2)) "NN: handler for :y2")))

  (testing "Beta-Bernoulli model schema"
    (let [schema (:schema bb-model)]
      (is (:has-conjugate? schema) "BB: has-conjugate?")
      (is (every? #(= :beta-bernoulli (:family %)) (:conjugate-pairs schema)) "BB: family is :beta-bernoulli")
      (is (= #{:p :x1 :x2} (set (keys (:auto-handlers schema)))) "BB: auto-handlers for :p :x1 :x2")))

  (testing "Gamma-Poisson model schema"
    (let [schema (:schema gp-model)]
      (is (:has-conjugate? schema) "GP: has-conjugate?")
      (is (= :gamma-poisson (-> schema :conjugate-pairs first :family)) "GP: family is :gamma-poisson")
      (is (= #{:rate :c1} (set (keys (:auto-handlers schema)))) "GP: auto-handlers for :rate :c1")))

  (testing "Gamma-Exponential model schema"
    (let [schema (:schema ge-model)]
      (is (:has-conjugate? schema) "GE: has-conjugate?")
      (is (= :gamma-exponential (-> schema :conjugate-pairs first :family)) "GE: family is :gamma-exponential")))

  (testing "Non-conjugate model — no auto-handlers"
    (let [schema (:schema non-conj-model)]
      (is (not (:has-conjugate? schema)) "non-conj: no conjugate pairs")
      (is (nil? (:auto-handlers schema)) "non-conj: no auto-handlers"))))

;; ---------------------------------------------------------------------------
;; Section 2: Gate 2 — Auto-Analytical Generate (NN)
;; ---------------------------------------------------------------------------

(deftest nn-auto-generate
  (testing "Normal-Normal auto generate"
    (let [model (dyn/auto-key nn-model)
          constraints (-> cm/EMPTY
                          (cm/set-value :y1 (mx/scalar 1.0))
                          (cm/set-value :y2 (mx/scalar 2.0)))
          {:keys [trace weight]} (p/generate model [] constraints)
          w (mx/item weight)
          s (mx/item (:score trace))
          log-2pi 1.8378770664093453
          ll-1 (* -0.5 (+ log-2pi (Math/log 5.0) (/ 1.0 5.0)))
          post-var-1 0.8
          post-mean-1 0.8
          marg-var-2 (+ post-var-1 1.0)
          diff-2 (- 2.0 post-mean-1)
          ll-2 (* -0.5 (+ log-2pi (Math/log marg-var-2) (/ (* diff-2 diff-2) marg-var-2)))
          expected (+ ll-1 ll-2)]
      (is (h/close? expected w 1e-5) "NN-auto: weight matches marginal LL")
      (is (h/close? w s 1e-8) "NN-auto: score = weight")
      (is (cm/has-value? (cm/get-submap (:choices trace) :mu)) "NN-auto: trace has :mu")
      (is (cm/has-value? (cm/get-submap (:choices trace) :y1)) "NN-auto: trace has :y1")
      (is (cm/has-value? (cm/get-submap (:choices trace) :y2)) "NN-auto: trace has :y2")
      (is (h/close? 1.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :y1))) 1e-8) "NN-auto: y1 = 1.0")
      (is (h/close? 2.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :y2))) 1e-8) "NN-auto: y2 = 2.0"))))

;; ---------------------------------------------------------------------------
;; Section 3: Gate 2 — Auto-Analytical Generate (BB)
;; ---------------------------------------------------------------------------

(deftest bb-auto-generate
  (testing "Beta-Bernoulli auto generate"
    (let [model (dyn/auto-key bb-model)
          constraints (-> cm/EMPTY
                          (cm/set-value :x1 (mx/scalar 1.0))
                          (cm/set-value :x2 (mx/scalar 0.0)))
          {:keys [trace weight]} (p/generate model [] constraints)
          w (mx/item weight)
          expected (+ (Math/log (/ 2.0 5.0)) (Math/log (/ 3.0 6.0)))]
      (is (h/close? expected w 1e-5) "BB-auto: weight matches marginal LL")
      (is (h/close? 1.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x1))) 1e-8) "BB-auto: x1 = 1.0")
      (is (h/close? 0.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x2))) 1e-8) "BB-auto: x2 = 0.0"))))

;; ---------------------------------------------------------------------------
;; Section 4: Gate 2 — Auto-Analytical Generate (GP)
;; ---------------------------------------------------------------------------

(deftest gp-auto-generate
  (testing "Gamma-Poisson auto generate"
    (let [model (dyn/auto-key gp-model)
          constraints (-> cm/EMPTY (cm/set-value :c1 (mx/scalar 5.0)))
          {:keys [trace weight]} (p/generate model [] constraints)
          w (mx/item weight)
          a 3.0 b 2.0 k 5.0
          expected (+ (- (mx/item (mx/lgamma (mx/scalar (+ a k))))
                         (mx/item (mx/lgamma (mx/scalar a)))
                         (mx/item (mx/lgamma (mx/scalar (+ k 1.0)))))
                      (* a (- (Math/log b) (Math/log (+ b 1.0))))
                      (* k (- 0 (Math/log (+ b 1.0)))))]
      (is (h/close? expected w 1e-5) "GP-auto: weight matches marginal LL"))))

;; ---------------------------------------------------------------------------
;; Section 5: Gate 2 — Auto-Analytical Generate (GE)
;; ---------------------------------------------------------------------------

(deftest ge-auto-generate
  (testing "Gamma-Exponential auto generate"
    (let [model (dyn/auto-key ge-model)
          constraints (-> cm/EMPTY (cm/set-value :t1 (mx/scalar 0.5)))
          {:keys [trace weight]} (p/generate model [] constraints)
          w (mx/item weight)
          a 2.0 b 1.0 x 0.5
          expected (+ (Math/log a) (* a (Math/log b))
                      (- (* (+ a 1.0) (Math/log (+ b x)))))]
      (is (h/close? expected w 1e-5) "GE-auto: weight matches marginal LL"))))

;; ---------------------------------------------------------------------------
;; Section 6: Fallthrough — unconstrained obs use standard handler
;; ---------------------------------------------------------------------------

(deftest fallthrough-behavior
  (testing "no obs constrained — standard handler"
    (let [model (dyn/auto-key nn-model)
          {:keys [trace weight]} (p/generate model [] cm/EMPTY)
          w (mx/item weight)
          s (mx/item (:score trace))]
      (is (h/close? 0.0 w 1e-8) "fallthrough: weight = 0 (no constraints)")
      (is (< s 0) "fallthrough: score < 0 (has log-probs)")))

  (testing "only prior constrained, no obs — standard handler"
    (let [model (dyn/auto-key nn-model)
          constraints (-> cm/EMPTY (cm/set-value :mu (mx/scalar 0.5)))
          {:keys [trace weight]} (p/generate model [] constraints)
          w (mx/item weight)
          s (mx/item (:score trace))]
      (is (not= 0.0 w) "prior-only: weight != 0 (prior constrained)")
      (is (> (Math/abs (- w s)) 0.01) "prior-only: weight != score")))

  (testing "partial obs constrained"
    (let [model (dyn/auto-key nn-model-3obs)
          constraints (-> cm/EMPTY
                          (cm/set-value :y1 (mx/scalar 1.0))
                          (cm/set-value :y3 (mx/scalar 2.0)))
          {:keys [trace weight]} (p/generate model [] constraints)]
      (is (and (cm/has-value? (cm/get-submap (:choices trace) :mu))
               (cm/has-value? (cm/get-submap (:choices trace) :y1))
               (cm/has-value? (cm/get-submap (:choices trace) :y2))
               (cm/has-value? (cm/get-submap (:choices trace) :y3))) "partial-obs: trace has all sites")
      (is (h/close? 1.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :y1))) 1e-8) "partial-obs: y1 = 1.0")
      (is (h/close? 2.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :y3))) 1e-8) "partial-obs: y3 = 2.0"))))

;; ---------------------------------------------------------------------------
;; Section 7: Multi-observation consistency
;; ---------------------------------------------------------------------------

(deftest multi-observation-consistency
  (testing "5-obs Normal-Normal"
    (let [model (dyn/auto-key nn-5obs)
          constraints (-> cm/EMPTY
                          (cm/set-value :y1 (mx/scalar 1.0))
                          (cm/set-value :y2 (mx/scalar 1.5))
                          (cm/set-value :y3 (mx/scalar 0.5))
                          (cm/set-value :y4 (mx/scalar 2.0))
                          (cm/set-value :y5 (mx/scalar 1.2)))
          {:keys [trace weight]} (p/generate model [] constraints)
          w (mx/item weight)]
      (is (js/isFinite w) "5-obs NN: weight is finite")
      (is (< w 0) "5-obs NN: weight < 0")
      (is (every? #(cm/has-value? (cm/get-submap (:choices trace) %))
                  [:y1 :y2 :y3 :y4 :y5]) "5-obs NN: all obs in trace"))))

;; ---------------------------------------------------------------------------
;; Section 8: Mixed conjugate families in same model
;; ---------------------------------------------------------------------------

(deftest mixed-families
  (testing "mixed model schema"
    (let [schema (:schema mixed-model)]
      (is (:has-conjugate? schema) "mixed: has-conjugate?")
      (is (= 2 (count (:conjugate-pairs schema))) "mixed: 2 pairs (NN + BB)")
      (is (= #{:mu :y :p :x} (set (keys (:auto-handlers schema)))) "mixed: handlers for all 4 addrs")))

  (testing "mixed model generate"
    (let [model (dyn/auto-key mixed-model)
          constraints (-> cm/EMPTY
                          (cm/set-value :y (mx/scalar 1.0))
                          (cm/set-value :x (mx/scalar 1.0)))
          {:keys [trace weight]} (p/generate model [] constraints)
          w (mx/item weight)
          log-2pi 1.8378770664093453
          nn-ll (* -0.5 (+ log-2pi (Math/log 5.0) (/ 1.0 5.0)))
          bb-ll (Math/log (/ 2.0 5.0))
          expected (+ nn-ll bb-ll)]
      (is (h/close? expected w 1e-5) "mixed: weight matches sum of marginal LLs"))))

(cljs.test/run-tests)
