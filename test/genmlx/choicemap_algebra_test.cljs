(ns genmlx.choicemap-algebra-test
  "Algebraic property tests for ChoiceMap operations."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]))

(defn cm= [a b]
  "Structural equality via to-map (handles Value wrapper comparison)."
  (= (cm/to-map a) (cm/to-map b)))

(deftest merge-identity
  (testing "merge identity laws"
    (let [v1 (mx/scalar 1.0)
          v2 (mx/scalar 2.0)
          cm1 (cm/choicemap :x v1 :y v2)]
      (is (cm= cm1 (cm/merge-cm cm/EMPTY cm1)) "left identity: merge(EMPTY, cm) = cm")
      (is (cm= cm1 (cm/merge-cm cm1 cm/EMPTY)) "right identity: merge(cm, EMPTY) = cm"))))

(deftest get-set-law
  (testing "get-set law"
    (let [v1 (mx/scalar 1.0)
          v2 (mx/scalar 2.0)
          v3 (mx/scalar 3.0)
          cm1 (cm/choicemap :x v1 :y v2)
          cm2 (cm/set-choice cm1 [:x] v3)]
      (is (= (mx/realize v3) (mx/realize (cm/get-choice cm2 [:x]))) "get-set: get(set(cm, a, v), a) = v"))))

(deftest set-get-non-interference
  (testing "set-get (non-interference)"
    (let [v1 (mx/scalar 1.0)
          v2 (mx/scalar 2.0)
          v3 (mx/scalar 3.0)
          cm1 (cm/choicemap :x v1 :y v2)
          cm2 (cm/set-choice cm1 [:x] v3)]
      (is (= (mx/realize v2) (mx/realize (cm/get-choice cm2 [:y]))) "set-get: set at :x doesn't change :y"))))

(deftest merge-override
  (testing "merge override"
    (let [v1 (mx/scalar 1.0)
          v2 (mx/scalar 2.0)
          v3 (mx/scalar 3.0)
          a (cm/choicemap :x v1 :y v2)
          b (cm/choicemap :x v3)
          merged (cm/merge-cm a b)]
      (is (= (mx/realize v3) (mx/realize (cm/get-choice merged [:x]))) "merge override: b's :x wins"))))

(deftest merge-preserves
  (testing "merge preserves"
    (let [v1 (mx/scalar 1.0)
          v2 (mx/scalar 2.0)
          v3 (mx/scalar 3.0)
          a (cm/choicemap :x v1 :y v2)
          b (cm/choicemap :z v3)
          merged (cm/merge-cm a b)]
      (is (= (mx/realize v1) (mx/realize (cm/get-choice merged [:x]))) "merge preserves: a's :x survives")
      (is (= (mx/realize v2) (mx/realize (cm/get-choice merged [:y]))) "merge preserves: a's :y survives")
      (is (= (mx/realize v3) (mx/realize (cm/get-choice merged [:z]))) "merge adds: b's :z added"))))

(deftest nested-paths
  (testing "nested paths"
    (let [v1 (mx/scalar 1.0)
          v2 (mx/scalar 2.0)
          cm1 (cm/set-choice cm/EMPTY [:a :b] v1)]
      (is (= (mx/realize v1) (mx/realize (cm/get-choice cm1 [:a :b]))) "nested set/get")
      (let [cm2 (cm/set-choice cm1 [:a :c] v2)]
        (is (= (mx/realize v1) (mx/realize (cm/get-choice cm2 [:a :b]))) "nested: original path survives")
        (is (= (mx/realize v2) (mx/realize (cm/get-choice cm2 [:a :c]))) "nested: new path accessible")))))

(deftest addresses-round-trip
  (testing "addresses round-trip"
    (let [v1 (mx/scalar 1.0)
          v2 (mx/scalar 2.0)
          v3 (mx/scalar 3.0)
          cm1 (cm/choicemap :x v1 :y v2)
          cm2 (cm/set-choice cm1 [:nested :deep] v3)
          addrs (cm/addresses cm2)]
      (is (every? (fn [path]
                    (some? (cm/get-choice cm2 path)))
                  addrs)
          "all addresses retrievable")
      (is (= 3 (count addrs)) "address count"))))

(deftest merge-associativity
  (testing "merge associativity"
    (let [v1 (mx/scalar 1.0)
          v2 (mx/scalar 2.0)
          v3 (mx/scalar 3.0)
          a (cm/choicemap :x v1)
          b (cm/choicemap :y v2)
          c (cm/choicemap :z v3)
          left  (cm/merge-cm (cm/merge-cm a b) c)
          right (cm/merge-cm a (cm/merge-cm b c))]
      (is (cm= left right) "associativity: merge(merge(a,b),c) = merge(a,merge(b,c))"))
    (let [a (cm/choicemap :x (mx/scalar 1.0))
          b (cm/choicemap :x (mx/scalar 2.0) :y (mx/scalar 3.0))
          c (cm/choicemap :y (mx/scalar 4.0) :z (mx/scalar 5.0))
          left  (cm/merge-cm (cm/merge-cm a b) c)
          right (cm/merge-cm a (cm/merge-cm b c))]
      (is (cm= left right) "associativity with overlapping keys"))))

(cljs.test/run-tests)
