(ns genmlx.kernel-combinator-test
  "Tests for kernel combinators: chain, cycle-kernels, mix-kernels, seed."
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

;; Shared model: x ~ N(0, 10), obs_i ~ N(x, 1) for i=0..4, all obs=3.0
;; Posterior: x ~ N(3, ~0.45)
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

(deftest chain-test
  (testing "chain: sequential kernel composition"
    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/chain (kern/mh-kernel (sel/select :mu))
                        (kern/mh-kernel (sel/select :mu)))
          traces (kern/run-kernel {:samples 100 :burn 50} k trace)
          mu-mean (extract-mu-mean traces)
          ar (:acceptance-rate (meta traces))]
      (is (= 100 (count traces)) "chain: 100 samples")
      (is (h/close? 3.0 mu-mean 1.0) "chain: posterior mu near 3")
      (is (> ar 0) "chain: acceptance rate > 0"))))

(deftest cycle-kernels-test
  (testing "cycle-kernels: round-robin cycling"
    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/cycle-kernels 6 [(kern/mh-kernel (sel/select :mu))])
          traces (kern/run-kernel {:samples 100 :burn 50} k trace)
          mu-mean (extract-mu-mean traces)
          ar (:acceptance-rate (meta traces))]
      (is (= 100 (count traces)) "cycle-kernels: 100 samples")
      (is (h/close? 3.0 mu-mean 1.0) "cycle-kernels: posterior mu near 3")
      (is (> ar 0) "cycle-kernels: acceptance rate > 0"))))

(deftest mix-kernels-single-test
  (testing "mix-kernels: single kernel weight 1.0"
    (let [{:keys [trace]} (p/generate model [] observations)
          k (kern/mix-kernels [[(kern/mh-kernel (sel/select :mu)) 1.0]])
          traces (kern/run-kernel {:samples 100 :burn 50} k trace)
          mu-mean (extract-mu-mean traces)]
      (is (= 100 (count traces)) "mix-kernels(1): 100 samples")
      (is (h/close? 3.0 mu-mean 1.0) "mix-kernels(1): posterior mu near 3"))))

(deftest mix-kernels-weighted-test
  (testing "mix-kernels: two kernels weighted"
    (let [{:keys [trace]} (p/generate model [] observations)
          sel (sel/select :mu)
          k (kern/mix-kernels [[(kern/mh-kernel sel) 0.7]
                               [(kern/mh-kernel sel) 0.3]])
          traces (kern/run-kernel {:samples 100 :burn 50} k trace)
          mu-mean (extract-mu-mean traces)]
      (is (= 100 (count traces)) "mix-kernels(2): 100 samples")
      (is (h/close? 3.0 mu-mean 1.0) "mix-kernels(2): posterior mu near 3"))))

(deftest seed-convergence-test
  (testing "seed: kernel with fixed key converges"
    (let [{:keys [trace]} (p/generate model [] observations)
          fixed-key (rng/fresh-key)
          k (kern/seed (kern/mh-kernel (sel/select :mu)) fixed-key)
          traces (kern/run-kernel {:samples 100 :burn 50} k trace)
          mu-mean (extract-mu-mean traces)]
      (is (= 100 (count traces)) "seed: 100 samples")
      (is (h/close? 3.0 mu-mean 1.0) "seed: posterior mu near 3"))))

(deftest seed-deterministic-test
  (testing "seed: deterministic with update-kernel"
    (let [{:keys [trace]} (p/generate model [] observations)
          constraints (cm/choicemap :mu (mx/scalar 2.5))
          k (kern/seed (kern/update-kernel constraints) (rng/fresh-key))
          traces1 (kern/run-kernel {:samples 1 :burn 0} k trace)
          traces2 (kern/run-kernel {:samples 1 :burn 0} k trace)
          mu1 (mx/realize (cm/get-value (cm/get-submap (:choices (first traces1)) :mu)))
          mu2 (mx/realize (cm/get-value (cm/get-submap (:choices (first traces2)) :mu)))]
      (is (h/close? mu1 mu2 1e-6) "seed(update): identical result"))))

(cljs.test/run-tests)
