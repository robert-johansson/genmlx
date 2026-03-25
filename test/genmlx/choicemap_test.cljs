(ns genmlx.choicemap-test
  "ChoiceMap data structure tests: EMPTY, Value, constructor, nested,
   set-choice, merge, to-map/from-map."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.choicemap :as cm]))

(deftest empty-choicemap-test
  (testing "EMPTY"
    (is (not (cm/has-value? cm/EMPTY)) "EMPTY has no value")
    (is (= [] (cm/addresses cm/EMPTY)) "EMPTY addresses")))

(deftest value-test
  (testing "Value"
    (let [v (cm/->Value 42)]
      (is (cm/has-value? v) "Value has value")
      (is (= 42 (cm/get-value v)) "Value get-value"))))

(deftest choicemap-constructor-test
  (testing "choicemap constructor"
    (let [m (cm/choicemap :x 1.0 :y 2.0)]
      (is (not (cm/has-value? m)) "Node has no value")
      (is (= 1.0 (cm/get-choice m [:x])) "get-choice :x")
      (is (= 2.0 (cm/get-choice m [:y])) "get-choice :y")
      (is (= #{[:x] [:y]} (set (cm/addresses m))) "addresses"))))

(deftest nested-choicemap-test
  (testing "nested choicemap"
    (let [m (cm/choicemap :params {:slope 2.0 :intercept 1.0} :noise 0.5)]
      (is (= 2.0 (cm/get-choice m [:params :slope])) "nested get-choice slope")
      (is (= 1.0 (cm/get-choice m [:params :intercept])) "nested get-choice intercept")
      (is (= 0.5 (cm/get-choice m [:noise])) "flat get-choice"))))

(deftest set-choice-test
  (testing "set-choice"
    (let [m (cm/set-choice cm/EMPTY [:x] 1.0)]
      (is (= 1.0 (cm/get-choice m [:x])) "set single"))
    (let [m (cm/set-choice cm/EMPTY [:a :b] 3.0)]
      (is (= 3.0 (cm/get-choice m [:a :b])) "set nested"))))

(deftest merge-cm-test
  (testing "merge-cm"
    (let [a (cm/choicemap :x 1.0 :y 2.0)
          b (cm/choicemap :y 3.0 :z 4.0)
          merged (cm/merge-cm a b)]
      (is (= 1.0 (cm/get-choice merged [:x])) "merge keeps a")
      (is (= 3.0 (cm/get-choice merged [:y])) "merge overrides")
      (is (= 4.0 (cm/get-choice merged [:z])) "merge adds b"))))

(deftest to-map-from-map-test
  (testing "to-map"
    (let [m (cm/choicemap :x 1.0 :y 2.0)
          plain (cm/to-map m)]
      (is (= {:x 1.0 :y 2.0} plain) "to-map")))
  (testing "from-map"
    (let [m (cm/from-map {:x 1.0 :y {:a 2.0}})]
      (is (= 1.0 (cm/get-choice m [:x])) "from-map flat")
      (is (= 2.0 (cm/get-choice m [:y :a])) "from-map nested"))))

(cljs.test/run-tests)
