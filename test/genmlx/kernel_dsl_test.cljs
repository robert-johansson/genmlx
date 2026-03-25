(ns genmlx.kernel-dsl-test
  "Tests for kernel DSL: random-walk, prior, proposal, gibbs."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.kernel :as kern])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Shared model: mu ~ N(0, 10), obs_i ~ N(mu, 1) for i=0..4, all obs=3.0
;; Posterior: mu ~ N(3, ~0.45)
(def model
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/gaussian 0 10))]
        (mx/eval! mu)
        (let [mu-val (mx/item mu)]
          (doseq [i (range 5)]
            (trace (keyword (str "obs" i))
                       (dist/gaussian mu-val 1)))
          mu-val)))))

(def observations
  (reduce (fn [cm i]
            (cm/set-choice cm [(keyword (str "obs" i))]
                           (mx/scalar 3.0)))
          cm/EMPTY (range 5)))

(defn extract-mu-mean [traces]
  (let [mu-vals (mapv (fn [t]
                        (mx/realize (cm/get-value (cm/get-submap (:choices t) :mu))))
                      traces)]
    (/ (reduce + mu-vals) (count mu-vals))))

(def model2
  (dyn/auto-key
    (gen [xs]
      (let [slope     (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 10))]
        (mx/eval! slope intercept)
        (let [s (mx/item slope) b (mx/item intercept)]
          (doseq [[j x] (map-indexed vector xs)]
            (trace (keyword (str "y" j))
                       (dist/gaussian (+ (* s x) b) 1)))
          s)))))

(deftest random-walk-single-test
  (testing "random-walk: single address"
    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/random-walk :mu 0.5)
          traces (kern/run-kernel {:samples 200 :burn 100} k trace)
          mu-mean (extract-mu-mean traces)
          ar (:acceptance-rate (meta traces))]
      (is (= 200 (count traces)) "random-walk: 200 samples")
      (is (h/close? 3.0 mu-mean 1.0) "random-walk: posterior mu near 3")
      (is (> ar 0) "random-walk: acceptance rate > 0"))))

(deftest random-walk-multi-test
  (testing "random-walk: multi-address (map form)"
    (let [xs [1.0 2.0 3.0 4.0 5.0]
          obs (reduce (fn [cm [j x]]
                        (cm/set-choice cm [(keyword (str "y" j))]
                                      (mx/scalar (+ (* 2.0 x) 1.0))))
                      cm/EMPTY (map-indexed vector xs))
          {:keys [trace]} (p/generate model2 [xs] obs)
          k (kern/random-walk {:slope 0.3 :intercept 0.3})
          traces (kern/run-kernel {:samples 200 :burn 300} k trace)
          slope-vals (mapv (fn [t] (mx/realize (cm/get-value (cm/get-submap (:choices t) :slope)))) traces)
          intercept-vals (mapv (fn [t] (mx/realize (cm/get-value (cm/get-submap (:choices t) :intercept)))) traces)
          slope-mean (/ (reduce + slope-vals) (count slope-vals))
          intercept-mean (/ (reduce + intercept-vals) (count intercept-vals))]
      (is (h/close? 2.0 slope-mean 1.5) "random-walk(map): slope near 2")
      (is (h/close? 1.0 intercept-mean 2.0) "random-walk(map): intercept near 1"))))

(deftest prior-test
  (testing "prior: resample from prior"
    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/prior :mu)
          traces (kern/run-kernel {:samples 200 :burn 100} k trace)
          mu-mean (extract-mu-mean traces)
          ar (:acceptance-rate (meta traces))]
      (is (= 200 (count traces)) "prior: 200 samples")
      (is (h/close? 3.0 mu-mean 1.0) "prior: posterior mu near 3")
      (is (> ar 0) "prior: acceptance rate > 0"))))

(def sym-proposal
  (gen [current-choices]
    (let [cur-mu (cm/get-value (cm/get-submap current-choices :mu))]
      (mx/eval! cur-mu)
      (trace :mu (dist/gaussian (mx/item cur-mu) 0.5)))))

(deftest proposal-symmetric-test
  (testing "proposal: symmetric custom proposal"
    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/proposal sym-proposal)
          traces (kern/run-kernel {:samples 200 :burn 100} k trace)
          mu-mean (extract-mu-mean traces)
          ar (:acceptance-rate (meta traces))]
      (is (= 200 (count traces)) "proposal(sym): 200 samples")
      (is (h/close? 3.0 mu-mean 1.0) "proposal(sym): posterior mu near 3")
      (is (> ar 0) "proposal(sym): acceptance rate > 0"))))

(def fwd-proposal
  (gen [current-choices]
    (let [cur-mu (cm/get-value (cm/get-submap current-choices :mu))]
      (mx/eval! cur-mu)
      (trace :mu (dist/gaussian (+ (mx/item cur-mu) 0.1) 0.5)))))

(def bwd-proposal
  (gen [current-choices]
    (let [cur-mu (cm/get-value (cm/get-submap current-choices :mu))]
      (mx/eval! cur-mu)
      (trace :mu (dist/gaussian (+ (mx/item cur-mu) 0.1) 0.5)))))

(deftest proposal-asymmetric-test
  (testing "proposal: asymmetric forward/backward"
    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/proposal fwd-proposal :backward bwd-proposal)
          traces (kern/run-kernel {:samples 200 :burn 100} k trace)
          mu-mean (extract-mu-mean traces)
          ar (:acceptance-rate (meta traces))]
      (is (= 200 (count traces)) "proposal(asym): 200 samples")
      (is (h/close? 3.0 mu-mean 1.0) "proposal(asym): posterior mu near 3")
      (is (> ar 0) "proposal(asym): acceptance rate > 0"))))

(deftest gibbs-keyword-test
  (testing "gibbs: keyword args (prior-based)"
    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/gibbs :mu)
          traces (kern/run-kernel {:samples 200 :burn 100} k trace)
          mu-mean (extract-mu-mean traces)
          ar (:acceptance-rate (meta traces))]
      (is (= 200 (count traces)) "gibbs(kw): 200 samples")
      (is (h/close? 3.0 mu-mean 1.0) "gibbs(kw): posterior mu near 3")
      (is (> ar 0) "gibbs(kw): acceptance rate > 0"))))

(deftest gibbs-std-map-test
  (testing "gibbs: std map (random-walk-based)"
    (let [xs [1.0 2.0 3.0 4.0 5.0]
          obs (reduce (fn [cm [j x]]
                        (cm/set-choice cm [(keyword (str "y" j))]
                                      (mx/scalar (+ (* 2.0 x) 1.0))))
                      cm/EMPTY (map-indexed vector xs))
          {:keys [trace]} (p/generate model2 [xs] obs)
          k (kern/gibbs {:slope 0.3 :intercept 0.3})
          traces (kern/run-kernel {:samples 200 :burn 100} k trace)
          slope-vals (mapv (fn [t] (mx/realize (cm/get-value (cm/get-submap (:choices t) :slope)))) traces)
          slope-mean (/ (reduce + slope-vals) (count slope-vals))]
      (is (h/close? 2.0 slope-mean 1.0) "gibbs(map): slope near 2"))))

(deftest compatibility-test
  (testing "compose with chain, repeat-kernel, mix-kernels"
    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/chain (kern/random-walk :mu 0.5) (kern/prior :mu))
          traces (kern/run-kernel {:samples 50 :burn 50} k trace)
          mu-mean (extract-mu-mean traces)]
      (is (h/close? 3.0 mu-mean 1.5) "chain(rw+prior): posterior mu near 3"))

    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/repeat-kernel 3 (kern/random-walk :mu 0.5))
          traces (kern/run-kernel {:samples 50 :burn 50} k trace)
          mu-mean (extract-mu-mean traces)]
      (is (h/close? 3.0 mu-mean 1.5) "repeat(rw): posterior mu near 3"))

    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/mix-kernels [[(kern/random-walk :mu 0.5) 0.7]
                               [(kern/prior :mu) 0.3]])
          traces (kern/run-kernel {:samples 50 :burn 50} k trace)
          mu-mean (extract-mu-mean traces)]
      (is (h/close? 3.0 mu-mean 1.5) "mix(rw+prior): posterior mu near 3"))

    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/random-walk :mu 0.5)
          callback-count (atom 0)
          traces (kern/run-kernel {:samples 10 :burn 0
                                   :callback (fn [_] (swap! callback-count inc))}
                                  k trace)]
      (is (= 10 @callback-count) "run-kernel callback fires"))))

(cljs.test/run-tests)
