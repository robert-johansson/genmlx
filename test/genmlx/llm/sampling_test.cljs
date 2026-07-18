;; @tier fast
(ns genmlx.llm.sampling-test
  "Native-parity sampling transforms (genmlx-djw6): the pi-provider decode
   loop must sample exactly like the native ChatSession sampler (mlx-core
   sampling.rs) at the same ChatConfig. Pins the greedy epsilon, the
   temperature->top-k->top-p->min-p filter order and each filter's edge
   semantics, and the asymmetric repetition / flat presence penalties with
   their 20-token default window. Tiny [4]-[5] logit vectors — GPU cost nil."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.llm.sampling :as s]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

(defn- ->vec [a] (vec (mx/->clj a)))

(deftest greedy-epsilon
  (testing "native GREEDY_TEMPERATURE_EPS boundary"
    (is (s/greedy? nil))
    (is (s/greedy? 0))
    (is (s/greedy? 1e-6))
    (is (not (s/greedy? 1e-5)))
    (is (not (s/greedy? 0.6)))))

(deftest top-k-value-threshold
  (let [logits (mx/array [1.0 5.0 3.0 2.0])
        out    (->vec (s/apply-top-k logits 2))]
    (testing "keeps the 2 largest, -inf elsewhere"
      (is (= [5.0 3.0] [(nth out 1) (nth out 2)]))
      (is (= js/Number.NEGATIVE_INFINITY (nth out 0)))
      (is (= js/Number.NEGATIVE_INFINITY (nth out 3))))
    (testing "k<=0 and k>=vocab disable"
      (is (= [1.0 5.0 3.0 2.0] (->vec (s/apply-top-k logits 0))))
      (is (= [1.0 5.0 3.0 2.0] (->vec (s/apply-top-k logits 4)))))))

(deftest top-p-nucleus
  (let [logits (mx/array [10.0 1.0 0.0 -5.0])] ; top prob ~0.99988
    (testing "p=0.5: only the dominant token survives (prev-cumsum rule)"
      (let [out (->vec (s/apply-top-p logits 0.5))]
        (is (= 10.0 (nth out 0)))
        (is (every? #(= js/Number.NEGATIVE_INFINITY %) (rest out)))))
    (testing "the p-crossing token is INCLUDED (prev < p, not cum < p)"
      ;; two equal tokens at prob ~0.5 each: p=0.6 keeps both (prev of 2nd = 0.5 < 0.6)
      (let [out (->vec (s/apply-top-p (mx/array [2.0 2.0 -20.0]) 0.6))]
        (is (= 2.0 (nth out 0)))
        (is (= 2.0 (nth out 1)))
        (is (= js/Number.NEGATIVE_INFINITY (nth out 2)))))
    (testing "p>=1 disables"
      (is (= [10.0 1.0 0.0 -5.0] (->vec (s/apply-top-p logits 1.0)))))))

(deftest min-p-relative-floor
  (let [logits (mx/array [3.0 2.0 -10.0])]
    (testing "keeps tokens with prob >= min-p * max-prob"
      (let [out (->vec (s/apply-min-p logits 0.1))]  ; p1/p0 = e^-1 ~ 0.37 >= 0.1; p2 ~ e^-13
        (is (= 3.0 (nth out 0)))
        (is (= 2.0 (nth out 1)))
        (is (= js/Number.NEGATIVE_INFINITY (nth out 2)))))
    (testing "min-p<=0 disables"
      (is (= [3.0 2.0 -10.0] (->vec (s/apply-min-p logits 0.0)))))))

(deftest repetition-penalty-asymmetric
  (let [logits (mx/array [2.0 -2.0 1.0])]
    (testing "positive logits divide, negative multiply; untouched ids unchanged"
      (is (= [1.0 -4.0 1.0] (->vec (s/apply-repetition-penalty logits [0 1] 2.0)))))
    (testing "penalty 1.0 and empty history are identity"
      (is (= [2.0 -2.0 1.0] (->vec (s/apply-repetition-penalty logits [0 1] 1.0))))
      (is (= [2.0 -2.0 1.0] (->vec (s/apply-repetition-penalty logits [] 2.0)))))
    (testing "context window: only the last context-size tokens penalize"
      (is (= [2.0 -4.0 1.0] (->vec (s/apply-repetition-penalty logits [0 1] 2.0 1)))))
    (testing "duplicate occurrences penalize once (write-identical)"
      (is (= [1.0 -2.0 1.0] (->vec (s/apply-repetition-penalty logits [0 0 0] 2.0)))))
    (testing "out-of-vocab ids are ignored"
      (is (= [1.0 -2.0 1.0] (->vec (s/apply-repetition-penalty logits [0 99] 2.0)))))))

(deftest presence-penalty-flat
  (let [logits (mx/array [2.0 -2.0 1.0])]
    (is (= [2.0 -2.0 0.5] (->vec (s/apply-presence-penalty logits [2] 0.5))))
    (is (= [2.0 -2.0 1.0] (->vec (s/apply-presence-penalty logits [2] 0))))
    (testing "duplicates subtract once, not per occurrence"
      (is (= [2.0 -2.0 0.5] (->vec (s/apply-presence-penalty logits [2 2] 0.5)))))))

(deftest filter-chain-order
  (testing "temperature scales BEFORE the filters (native apply_sampling order)"
    ;; temp 0.5 doubles logits; top-k then thresholds on the scaled values
    (let [out (->vec (s/filter-logits (mx/array [1.0 3.0 2.0]) {:temperature 0.5 :top-k 1}))]
      (is (= 6.0 (nth out 1)))
      (is (= js/Number.NEGATIVE_INFINITY (nth out 0)))
      (is (= js/Number.NEGATIVE_INFINITY (nth out 2))))))

(deftest sample-token-paths
  (let [logits (mx/array [1.0 5.0 3.0])
        key    (rng/fresh-key)]
    (testing "greedy path: argmax, key unchanged"
      (let [[tok k'] (s/sample-token key logits {:temperature 0} [])]
        (is (= 1 tok))
        (is (identical? key k'))))
    (testing "greedy respects penalties (repetition can flip the argmax)"
      (let [[tok _] (s/sample-token key (mx/array [1.0 5.0 4.9]) {:temperature 0 :repetition-penalty 5.0} [1])]
        (is (= 2 tok))))
    (testing "sampled path: deterministic under a fixed key, in-range, advances the key"
      (let [[t1 k1] (s/sample-token key logits {:temperature 0.8 :top-k 2} [])
            [t2 _]  (s/sample-token key logits {:temperature 0.8 :top-k 2} [])]
        (is (= t1 t2))
        (is (contains? #{1 2} t1))   ; top-k 2 masks id 0
        (is (not (identical? key k1)))))))

(cljs.test/run-tests)
