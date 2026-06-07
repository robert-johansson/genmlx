(ns genmlx.categorical-batched-logprob-test
  "Regression tests for categorical log-prob under batched / per-particle logits
   (genmlx-ql6a). A plain (take log-probs v -1) on multi-dim [N,K] logits with a
   per-particle [N] index returns [N,N] (the full cross-product), silently
   corrupting the [N] score into [N,N] -> NaN ESS for vectorized HMM-style
   models. The diagonal gather (take-along-axis) must return [N]. The 1-D
   (LLM token-sampling, shared mixture weights) and scalar-observation paths
   must be unaffected."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn])
  (:require-macros [genmlx.gen :refer [gen]]))

(deftest categorical-1d-logits-batched
  (testing "1-D [K] logits with an [N] index -> [N] log-prob = log p"
    (let [logits (mx/log (mx/array [0.2 0.5 0.3]))          ;; [3]
          d (dist/categorical logits)
          v (mx/array [0 1 2 1] mx/int32)                   ;; [4]
          lp (dist/log-prob d v)]
      (mx/eval! lp)
      (is (= [4] (mx/shape lp)) "1-D logits + [N] v -> [N]")
      (let [logp (mapv #(js/Math.log %) [0.2 0.5 0.3])
            expected [(nth logp 0) (nth logp 1) (nth logp 2) (nth logp 1)]]
        (is (h/all-close? expected (mx/->clj lp) 1e-4) "1-D batched log-prob values")))))

(deftest categorical-perparticle-logits-batched
  (testing "[N,K] per-particle logits with an [N] index -> [N] (was [N,N])"
    (let [probs  [[0.9 0.1] [0.3 0.7] [0.5 0.5]]            ;; [3,2]
          logits (mx/log (mx/array probs))                  ;; [3,2]
          d (dist/categorical logits)
          v (mx/array [0 1 0] mx/int32)                     ;; [3] per-particle
          lp (dist/log-prob d v)]
      (mx/eval! lp)
      (is (= [3] (mx/shape lp)) "per-particle [N,K] + [N] v -> [N], not [N,N]")
      (let [expected [(js/Math.log 0.9) (js/Math.log 0.7) (js/Math.log 0.5)]]
        (is (h/all-close? expected (mx/->clj lp) 1e-4) "diagonal per-particle gather values")))))

(deftest categorical-perparticle-logits-scalar-obs
  (testing "[N,K] per-particle logits with a shared scalar index -> [N]"
    (let [probs  [[0.9 0.1] [0.3 0.7] [0.5 0.5]]
          logits (mx/log (mx/array probs))
          d (dist/categorical logits)
          v (mx/scalar 1 mx/int32)                          ;; shared category 1
          lp (dist/log-prob d v)]
      (mx/eval! lp)
      (is (= [3] (mx/shape lp)) "[N,K] + scalar v -> [N]")
      (let [expected [(js/Math.log 0.1) (js/Math.log 0.7) (js/Math.log 0.5)]]
        (is (h/all-close? expected (mx/->clj lp) 1e-4) "scalar-obs per-particle values")))))

(deftest categorical-llm-style-1d
  (testing "1-D large-K logits (LLM token sampling) -> [N] = -log K, unaffected"
    (let [K 5000
          logits (mx/zeros [K])                             ;; uniform over K
          d (dist/categorical logits)
          v (mx/array [0 17 4999 123] mx/int32)
          lp (dist/log-prob d v)]
      (mx/eval! lp)
      (is (= [4] (mx/shape lp)) "1-D large-K + [N] v -> [N]")
      (let [expected (vec (repeat 4 (- (js/Math.log K))))]
        (is (h/all-close? expected (mx/->clj lp) 1e-3) "uniform 1-D categorical = -log K")))))

(deftest categorical-perparticle-vectorized-integration
  (testing "vectorized model with per-particle categorical logits -> [N] finite score"
    ;; z0 ~ Cat(shared init); z1 ~ Cat(transition[z0]) — per-particle [N,K] logits.
    (let [trans (mx/log (mx/array [[0.8 0.2] [0.1 0.9]]))   ;; [2,2]
          init  (mx/log (mx/array [0.5 0.5]))
          model (dyn/auto-key
                  (gen []
                    (let [z0 (trace :z0 (dist/categorical init))
                          logits1 (mx/take-idx trans z0 0)   ;; [N,2] per-particle
                          z1 (trace :z1 (dist/categorical logits1))]
                      z1)))
          n 64
          vt (dyn/vsimulate model [] n (rng/fresh-key 7))
          score (:score vt)]
      (mx/eval! score)
      (is (= [n] (mx/shape score)) "per-particle categorical batched score is [N]")
      (is (every? h/finite? (mx/->clj score))
          "scores finite — no NaN from [N,N] corruption"))))

(cljs.test/run-tests)
