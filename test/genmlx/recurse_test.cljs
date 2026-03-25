(ns genmlx.recurse-test
  "Tests for the Recurse combinator."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

(deftest countdown-simulate
  (testing "Countdown (linear recursion)"
    (let [countdown
          (comb/recurse (fn [self]
            (dyn/auto-key (gen [depth]
              (let [v (trace :v (dist/gaussian 0 1))]
                (mx/eval! v)
                (if (> depth 0)
                  {:v (mx/item v)
                   :child (splice :child self (dec depth))}
                  {:v (mx/item v)}))))))
          trace (p/simulate countdown [2])]
      (is (instance? tr/Trace trace) "returns trace")
      (is (cm/has-value? (cm/get-submap (:choices trace) :v)) "has :v at root")
      (is (not= cm/EMPTY (cm/get-submap (:choices trace) :child)) "has :child sub")
      (is (not= cm/EMPTY
            (cm/get-submap (cm/get-submap (:choices trace) :child) :child))
          "has nested :child")
      (let [score (mx/item (:score trace))]
        (is (js/isFinite score) "score is finite")
        (is (< score 0) "score is negative (3 gaussians)")))))

(deftest binary-tree-simulate
  (testing "Binary tree"
    (let [tree
          (comb/recurse (fn [self]
            (dyn/auto-key (gen [depth]
              (let [v (trace :v (dist/gaussian 0 1))]
                (mx/eval! v)
                (if (> depth 0)
                  {:v (mx/item v)
                   :left (splice :left self (dec depth))
                   :right (splice :right self (dec depth))}
                  {:v (mx/item v)}))))))
          trace (p/simulate tree [1])]
      (is (instance? tr/Trace trace) "tree trace exists")
      (is (cm/has-value? (cm/get-submap (:choices trace) :v)) "root :v exists")
      (is (not= cm/EMPTY (cm/get-submap (:choices trace) :left)) ":left sub exists")
      (is (not= cm/EMPTY (cm/get-submap (:choices trace) :right)) ":right sub exists")
      (let [score (mx/item (:score trace))]
        (is (js/isFinite score) "tree score finite")))))

(deftest generate-with-constraints
  (testing "generate with constraints"
    (let [countdown
          (comb/recurse (fn [self]
            (dyn/auto-key (gen [depth]
              (let [v (trace :v (dist/gaussian 0 1))]
                (mx/eval! v)
                (if (> depth 0)
                  {:v (mx/item v)
                   :child (splice :child self (dec depth))}
                  {:v (mx/item v)}))))))
          obs (-> cm/EMPTY
                  (cm/set-choice [:v] (mx/scalar 0.5))
                  (cm/set-choice [:child] (cm/choicemap :v (mx/scalar -0.3))))
          {:keys [trace weight]} (p/generate countdown [1] obs)]
      (mx/eval! weight)
      (is (instance? tr/Trace trace) "generate returns trace")
      (is (js/isFinite (mx/item weight)) "weight is finite")
      (is (h/close? 0.5 (mx/item (cm/get-choice (:choices trace) [:v])) 1e-5)
          "root :v constrained")
      (is (h/close? -0.3 (mx/item (cm/get-choice (:choices trace) [:child :v])) 1e-5)
          "child :v constrained"))))

(deftest recurse-update
  (testing "update"
    (let [countdown
          (comb/recurse (fn [self]
            (dyn/auto-key (gen [depth]
              (let [v (trace :v (dist/gaussian 0 1))]
                (mx/eval! v)
                (if (> depth 0)
                  {:v (mx/item v)
                   :child (splice :child self (dec depth))}
                  {:v (mx/item v)}))))))
          obs (cm/choicemap :v (mx/scalar 1.0)
                             :child (cm/choicemap :v (mx/scalar 2.0)))
          {:keys [trace]} (p/generate countdown [1] obs)
          new-constraints (cm/choicemap :v (mx/scalar 0.0))
          result (p/update countdown trace new-constraints)
          new-trace (:trace result)]
      (mx/eval! (:weight result))
      (is (instance? tr/Trace new-trace) "update returns trace")
      (is (h/close? 0.0 (mx/item (cm/get-choice (:choices new-trace) [:v])) 1e-5)
          "updated root :v")
      (is (h/close? 2.0 (mx/item (cm/get-choice (:choices new-trace) [:child :v])) 1e-5)
          "child :v unchanged")
      (is (js/isFinite (mx/item (:weight result))) "update weight finite"))))

(deftest recurse-regenerate
  (testing "regenerate"
    (let [countdown
          (comb/recurse (fn [self]
            (dyn/auto-key (gen [depth]
              (let [v (trace :v (dist/gaussian 0 1))]
                (mx/eval! v)
                (if (> depth 0)
                  {:v (mx/item v)
                   :child (splice :child self (dec depth))}
                  {:v (mx/item v)}))))))
          obs (cm/choicemap :v (mx/scalar 1.0)
                             :child (cm/choicemap :v (mx/scalar 2.0)))
          {:keys [trace]} (p/generate countdown [1] obs)
          selection (sel/hierarchical :v sel/all)
          result (p/regenerate countdown trace selection)
          new-trace (:trace result)]
      (mx/eval! (:weight result))
      (is (instance? tr/Trace new-trace) "regenerate returns trace")
      (is (js/isFinite (mx/item (:weight result))) "regenerate weight finite")
      (is (h/close? 2.0 (mx/item (cm/get-choice (:choices new-trace) [:child :v])) 1e-5)
          "child :v unchanged after regen"))))

(deftest random-depth-recursion
  (testing "geometric stopping"
    (let [geo-list
          (comb/recurse (fn [self]
            (dyn/auto-key (gen [p]
              (let [v (trace :v (dist/gaussian 0 1))
                    cont (trace :cont (dist/bernoulli p))]
                (mx/eval! v)
                (mx/eval! cont)
                (if (> (mx/item cont) 0.5)
                  {:v (mx/item v)
                   :next (splice :next self p)}
                  {:v (mx/item v)}))))))
          trace (p/simulate geo-list [0.3])]
      (is (instance? tr/Trace trace) "geo-list trace exists")
      (is (js/isFinite (mx/item (:score trace))) "geo-list score finite"))))

(cljs.test/run-tests)
