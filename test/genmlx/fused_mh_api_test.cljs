(ns genmlx.fused-mh-api-test
  "Test fused-mh public API (M5)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.inference.mcmc :as mcmc]))

;; Static linreg (tensor-native score)
(def static-linreg
  (gen [x]
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 5))
          y-pred (mx/add (mx/multiply slope (mx/ensure-array x)) intercept)]
      (trace :y (dist/gaussian y-pred 1))
      slope)))

(def obs (cm/choicemap :y (mx/scalar 5.0)))

(deftest fused-mh-basic-usage-test
  (testing "basic usage"
    (let [result (mcmc/fused-mh
                   {:samples 500 :burn 200 :addresses [:slope :intercept]
                    :proposal-std 0.5}
                   static-linreg [2.0] obs)]
      (is (some? (:samples result)) "returns :samples")
      (is (some? (:final-params result)) "returns :final-params")
      (is (fn? (:chain-fn result)) "returns :chain-fn")
      (is (= [500 2] (mx/shape (:samples result))) "samples shape [500,2]")
      (is (= [2] (mx/shape (:final-params result))) "final-params shape [2]")
      (let [samples-js (mx/->clj (:samples result))
            slopes (mapv first samples-js)
            mean-slope (/ (reduce + slopes) (count slopes))]
        (is (js/isFinite mean-slope) "posterior slope finite")))))

(deftest fused-mh-chain-fn-reuse-test
  (testing "chain-fn reuse"
    (let [result1 (mcmc/fused-mh
                    {:samples 200 :burn 100 :addresses [:slope :intercept]
                     :proposal-std 0.5}
                    static-linreg [2.0] obs)
          cfn (:chain-fn result1)
          t0 (js/Date.now)
          result2 (mcmc/fused-mh
                    {:samples 200 :burn 100 :addresses [:slope :intercept]
                     :proposal-std 0.5 :chain-fn cfn}
                    static-linreg [2.0] obs)
          t1 (js/Date.now)]
      (is (= [200 2] (mx/shape (:samples result2))) "reuse returns samples")
      (is (< (- t1 t0) 500) "reuse is fast (<500ms)"))))

(deftest fused-mh-thinning-test
  (testing "thinning"
    (let [result (mcmc/fused-mh
                   {:samples 100 :burn 50 :thin 3 :addresses [:slope :intercept]
                    :proposal-std 0.5}
                   static-linreg [2.0] obs)]
      (is (= [100 2] (mx/shape (:samples result))) "thin=3 samples shape [100,2]"))))

(deftest fused-mh-benchmark-test
  (testing "benchmark: fused vs block-based"
    (let [warmup (mcmc/fused-mh {:samples 10 :burn 10 :addresses [:slope :intercept]
                                  :proposal-std 0.5}
                                static-linreg [2.0] obs)
          cfn (:chain-fn warmup)
          t0 (js/Date.now)
          _ (mcmc/fused-mh {:samples 1000 :burn 500 :addresses [:slope :intercept]
                             :proposal-std 0.5 :chain-fn cfn}
                           static-linreg [2.0] obs)
          t1 (js/Date.now)
          ms-fused (- t1 t0)
          _ (mcmc/compiled-mh {:samples 10 :burn 10 :addresses [:slope :intercept]
                               :proposal-std 0.5}
                              static-linreg [2.0] obs)
          t2 (js/Date.now)
          _ (mcmc/compiled-mh {:samples 1000 :burn 500 :addresses [:slope :intercept]
                               :proposal-std 0.5}
                              static-linreg [2.0] obs)
          t3 (js/Date.now)
          ms-block (- t3 t2)]
      (is (and (pos? ms-fused) (pos? ms-block)) "both complete"))))

(cljs.test/run-tests)
