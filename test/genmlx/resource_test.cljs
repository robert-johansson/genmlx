(ns genmlx.resource-test
  "Stress tests for Metal resource management.
   Verifies that inference loops don't leak Metal buffers over extended runs."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.kernel :as kern])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Simple 5-site Gaussian model
(def model
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                          intercept) 1)))
      slope)))

(def xs [1.0 2.0 3.0])
(def observations
  (reduce (fn [cm [j y]]
            (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector [2.5 5.1 7.3])))

(deftest mh-stress-test
  (testing "MH stress — 500 iterations shouldn't grow memory linearly"
    (let [_ (mx/clear-cache!)
          _ (mx/reset-peak-memory!)
          {:keys [trace]} (p/generate (dyn/auto-key model) [xs] observations)
          _ (mx/eval! (:score trace))
          _ (mcmc/mh {:samples 50 :selection (sel/select :slope :intercept)}
                     model [xs] observations)
          mem-50 (mx/get-active-memory)
          _ (mcmc/mh {:samples 500 :selection (sel/select :slope :intercept)}
                     model [xs] observations)
          mem-500 (mx/get-active-memory)]
      (is (or (< mem-500 (* 5 (max mem-50 1024)))
              (< mem-500 (* 10 1024 1024)))
          "memory bounded (500 iters < 5x 50 iters)"))))

(deftest is-stress-test
  (testing "IS stress — 200 samples complete without crash"
    (let [_ (mx/clear-cache!)
          result (is/importance-sampling {:samples 200} model [xs] observations)]
      (is (= 200 (count (:traces result))) "IS completed 200 samples")
      (is (= 200 (count (:log-weights result))) "IS has weights"))))

(deftest collect-samples-resource-test
  (testing "collect-samples with array-heavy step-fn"
    (let [_ (mx/clear-cache!)
          step-fn (fn [state _key]
                    (let [a (mx/add state (mx/scalar 0.1))
                          b (mx/multiply a (mx/scalar 0.99))
                          c (mx/add b (rng/normal (rng/fresh-key) [10]))
                          d (mx/sum c)]
                      (mx/eval! d)
                      {:state d :accepted? true}))
          results (kern/collect-samples
                    {:samples 200 :burn 50}
                    step-fn
                    mx/item
                    (mx/scalar 0.0))]
      (is (= 200 (count results)) "collect-samples completed 200 samples")
      (let [active-mem (mx/get-active-memory)]
        (is (< active-mem (* 50 1024 1024)) "active memory bounded")))))

(deftest clear-cache-effect-test
  (testing "clear-cache effect"
    (let [_ (doseq [_ (range 100)]
              (let [a (rng/normal (rng/fresh-key) [100])]
                (mx/eval! a)))
          cache-before (mx/get-cache-memory)
          _ (mx/clear-cache!)
          cache-after (mx/get-cache-memory)]
      (is (<= cache-after cache-before) "clear-cache reduces cache"))))

(cljs.test/run-tests)
