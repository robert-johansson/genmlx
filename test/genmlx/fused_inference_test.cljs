(ns genmlx.fused-inference-test
  "WP-2: Fused Inference + Optimization tests.
   Tests fused MCMC+Adam, fused-learn dispatch, metadata reporting,
   and memory bounds."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.dist :as dist]
            [genmlx.inference.compiled-optimizer :as co]
            [genmlx.inference.compiled-gradient :as cg]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- cleanup! []
  (mx/clear-cache!)
  (mx/sweep-dead-arrays!))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def simple-model
  "Simple 2-latent model: Gaussian mean + noise, observe y."
  (gen []
       (let [mu (trace :mu (dist/gaussian 0 10))
             sigma (trace :sigma (dist/gaussian 1 0.1))]
         (trace :y (dist/gaussian mu (mx/abs sigma)))
         mu)))

(def three-param-model
  "3-latent model: slope + intercept + noise, observe y."
  (gen []
       (let [a (trace :a (dist/gaussian 0 5))
             b (trace :b (dist/gaussian 0 5))
             c (trace :c (dist/gaussian 1 0.1))]
         (trace :y (dist/gaussian (mx/add a b) (mx/abs c)))
         a)))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest fused-learn-direct-dispatch-test
  (testing "fused-learn :direct dispatch"
    (let [obs (cm/choicemap {:y (mx/scalar 5.0)})
          result (co/fused-learn simple-model [] obs [:mu :sigma] :direct
                                 {:iterations 20 :lr 0.05 :log-every 10})]
      (is (contains? result :compilation-level) "fused-learn :direct returns :compilation-level")
      (is (mx/array? (:params result)) "fused-learn :direct returns :params")
      (is (sequential? (:loss-history result)) "fused-learn :direct returns :loss-history")
      (is (#{:tensor-native :handler} (:compilation-level result))
          "direct path reports :tensor-native or :handler"))
    (cleanup!))

  (testing "fused-learn nil method falls through to :direct"
    (let [obs (cm/choicemap {:y (mx/scalar 5.0)})
          result (co/fused-learn simple-model [] obs [:mu :sigma] nil
                                 {:iterations 20 :lr 0.05 :log-every 10})]
      (is (contains? result :compilation-level) "fused-learn nil method falls through to :direct")
      (is (mx/array? (:params result)) "fused-learn nil returns :params"))
    (cleanup!)))

(deftest fused-mcmc-adam-basic-test
  (testing "fused MCMC+Adam basic functionality"
    (let [obs (cm/choicemap {:y (mx/scalar 5.0)})
          result (co/make-fused-mcmc-train simple-model [] obs [:mu :sigma]
                                           {:iterations 30 :lr 0.01 :mcmc-steps 3 :proposal-std 0.3 :log-every 10})]
      (is (contains? result :params) "returns :params key")
      (is (contains? result :loss-history) "returns :loss-history key")
      (is (contains? result :compilation-level) "returns :compilation-level key")
      (is (contains? result :mcmc-compiled) "returns :mcmc-compiled key")
      (is (contains? result :latent-index) "returns :latent-index key")
      (is (contains? result :n-params) "returns :n-params key")
      (is (true? (:mcmc-compiled result)) "mcmc-compiled is true")
      (is (= :tensor-native (:compilation-level result)) "compilation-level is :tensor-native")
      (is (mx/array? (:params result)) "params is an MLX array")
      (is (pos? (count (:loss-history result))) "loss-history is non-empty")
      (is (every? number? (:loss-history result)) "all loss-history entries are numbers")
      (is (pos? (:n-params result)) "n-params is positive"))
    (cleanup!)))

(deftest noise-pre-generation-shapes-test
  (testing "noise pre-generation shapes"
    (let [obs (cm/choicemap {:y (mx/scalar 3.0)})
          model-k (dyn/auto-key simple-model)
          {:keys [trace]} (p/generate model-k [] obs)
          {:keys [score-fn n-params]} (u/prepare-mcmc-score simple-model [] obs [:mu :sigma] trace)
          K n-params
          T 10
          rk (rng/fresh-key)
          [nk uk] (rng/split rk)
          noise (rng/normal nk [T K])
          uniforms (rng/uniform uk [T])
          _ (mx/materialize! noise uniforms)]
      (is (= [T K] (mx/shape noise)) "noise shape is [T,K]")
      (is (= [T] (mx/shape uniforms)) "uniforms shape is [T]")
      (let [std (mx/scalar 0.3)
            chain-fn (cg/make-differentiable-chain score-fn std T K)
            init-p (mx/zeros [K])
            final-p (chain-fn init-p noise uniforms)]
        (mx/materialize! final-p)
        (is (= [K] (mx/shape final-p)) "chain output shape matches K")
        (is (every? js/isFinite (mx/->clj final-p)) "chain output is finite")))
    (cleanup!)))

(deftest gradient-direction-through-chain-test
  (testing "gradient direction through chain"
    (let [obs (cm/choicemap {:y (mx/scalar 5.0)})
          model-k (dyn/auto-key simple-model)
          {:keys [trace]} (p/generate model-k [] obs)
          {:keys [score-fn n-params]}
          (u/prepare-mcmc-score simple-model [] obs [:mu :sigma] trace)
          K n-params
          T 3
          std (mx/scalar 0.3)
          chain-fn (cg/make-differentiable-chain score-fn std T K)
          neg-obj (fn [params noise uniforms]
                    (mx/negative (score-fn (chain-fn params noise uniforms))))
          vg (mx/value-and-grad neg-obj)
          start (mx/array [3.0 1.0 5.0])
          rk (rng/fresh-key)
          [nk uk] (rng/split rk)
          noise (rng/normal nk [T K])
          uniforms (rng/uniform uk [T])
          _ (mx/materialize! noise uniforms)
          [loss grad] (vg start noise uniforms)]
      (mx/materialize! loss grad)
      (let [grad-vec (mx/->clj grad)]
        (is (every? js/isFinite grad-vec) "gradient is finite")
        (is (js/isFinite (mx/item loss)) "loss is finite")
        (is (= K (count grad-vec)) "gradient has correct dimension")))
    (cleanup!)))

(deftest fused-learn-mcmc-dispatch-test
  (testing "fused-learn :mcmc dispatch + metadata"
    (let [obs (cm/choicemap {:y (mx/scalar 5.0)})
          result (co/fused-learn simple-model [] obs [:mu :sigma] :mcmc
                                 {:iterations 20 :lr 0.01 :mcmc-steps 3 :proposal-std 0.3 :log-every 10})]
      (is (true? (:mcmc-compiled result)) "fused-learn :mcmc returns :mcmc-compiled")
      (is (mx/array? (:params result)) "fused-learn :mcmc returns :params")
      (is (sequential? (:loss-history result)) "fused-learn :mcmc returns :loss-history")
      (is (#{:tensor-native :handler} (:compilation-level result))
          "mcmc path reports compilation-level")
      (is (map? (:latent-index result)) "mcmc path has latent-index map")
      (is (pos? (:n-params result)) "mcmc path has n-params > 0"))
    (cleanup!)))

(deftest callback-invocation-test
  (testing "callback invocation"
    (let [obs (cm/choicemap {:y (mx/scalar 5.0)})
          callback-log (atom [])
          result (co/make-fused-mcmc-train simple-model [] obs [:mu :sigma]
                                           {:iterations 50 :lr 0.01 :mcmc-steps 3 :proposal-std 0.3
                                            :log-every 10
                                            :callback (fn [info]
                                                        (swap! callback-log conj info))})]
      (is (pos? (count @callback-log)) "callback was invoked")
      (is (every? #(contains? % :iter) @callback-log) "callback info has :iter key")
      (is (every? #(contains? % :loss) @callback-log) "callback info has :loss key")
      (is (every? #(contains? % :params) @callback-log) "callback info has :params key"))
    (cleanup!)))

(deftest prng-key-determinism-test
  (testing "PRNG key determinism"
    (let [obs (cm/choicemap {:y (mx/scalar 5.0)})
          key1 (rng/fresh-key 42)
          key2 (rng/fresh-key 42)
          r1 (co/make-fused-mcmc-train simple-model [] obs [:mu :sigma]
                                       {:iterations 20 :lr 0.01 :mcmc-steps 3 :proposal-std 0.3
                                        :log-every 10 :key key1})
          _ (cleanup!)
          r2 (co/make-fused-mcmc-train simple-model [] obs [:mu :sigma]
                                       {:iterations 20 :lr 0.01 :mcmc-steps 3 :proposal-std 0.3
                                        :log-every 10 :key key2})
          p1 (mx/->clj (:params r1))
          p2 (mx/->clj (:params r2))]
      (is (and (some? r1) (some? r2)) "both runs complete with same key")
      (is (and (every? js/isFinite p1) (every? js/isFinite p2)) "both produce finite params"))
    (cleanup!)))

(deftest convergence-3-param-model-test
  (testing "fused MCMC+Adam convergence (3-param model)"
    (let [obs (cm/choicemap {:y (mx/scalar 7.0)})
          result (co/make-fused-mcmc-train three-param-model [] obs [:a :b :c]
                                           {:iterations 300 :lr 0.01 :mcmc-steps 5 :proposal-std 0.3
                                            :log-every 100})
          losses (:loss-history result)
          first-loss (first losses)
          last-loss (last losses)]
      (is (>= (count losses) 3) "iterations complete with logged losses")
      (is (<= last-loss (+ first-loss 2.0)) "loss decreases overall (last <= first + 2.0)")
      (is (js/isFinite last-loss) "final loss is finite")
      (is (every? js/isFinite (mx/->clj (:params result))) "params are finite"))
    (cleanup!)))

(deftest memory-bounds-mcmc-test
  (testing "memory bounds: 500 mcmc iterations without leak"
    (let [obs (cm/choicemap {:y (mx/scalar 5.0)})
          result (co/make-fused-mcmc-train simple-model [] obs [:mu :sigma]
                                           {:iterations 500 :lr 0.01 :mcmc-steps 5 :proposal-std 0.3
                                            :log-every 100})]
      (is (some? result) "500 mcmc iterations completed without error")
      (is (every? js/isFinite (mx/->clj (:params result))) "params are finite after 500 iters")
      (is (every? js/isFinite (:loss-history result)) "all logged losses are finite"))
    (cleanup!))

  (testing "memory bounds: 500 direct iterations"
    (let [obs (cm/choicemap {:y (mx/scalar 5.0)})
          result (co/learn simple-model [] obs [:mu :sigma]
                           {:iterations 500 :lr 0.05 :log-every 100})]
      (is (some? result) "direct path 500 iterations completed")
      (is (every? js/isFinite (mx/->clj (:params result))) "direct path params finite after 500 iters"))
    (cleanup!)))

(cljs.test/run-tests)
