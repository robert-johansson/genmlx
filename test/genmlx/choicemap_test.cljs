;; @tier fast core
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

(deftest merge-cm-b-wins-test
  (testing "b overrides a on leaf-vs-node conflicts (genmlx-ybw9)"
    (let [leaf-a (cm/choicemap :x 1.0)
          node-b (cm/choicemap :x {:sub 2.0})]
      ;; a leaf, b node: b wins (previously leaf-a silently won)
      (is (= 2.0 (cm/get-choice (cm/merge-cm leaf-a node-b) [:x :sub]))
          "node b replaces leaf a")
      ;; a node, b leaf: b wins
      (is (= 1.0 (cm/get-choice (cm/merge-cm node-b leaf-a) [:x]))
          "leaf b replaces node a")))
  (testing "non-choicemap args are coerced, never leak raw (genmlx-ybw9)"
    (let [merged (cm/merge-cm {:x 1.0} {:y 2.0})]
      (is (cm/choicemap? merged) "result is a choicemap")
      (is (= 1.0 (cm/get-choice merged [:x])) "raw map a coerced")
      (is (= 2.0 (cm/get-choice merged [:y])) "raw map b coerced"))))

(deftest get-choice-missing-path-test
  (testing "get-choice returns nil on missing or non-leaf paths (genmlx-ybw9)"
    (let [m (cm/choicemap :params {:slope 2.0})]
      (is (nil? (cm/get-choice m [:nope])) "missing top-level addr")
      (is (nil? (cm/get-choice m [:params :nope])) "missing nested addr")
      (is (nil? (cm/get-choice m [:params])) "non-leaf path")
      (is (nil? (cm/get-choice m [:nope :deeper])) "missing then deeper"))))

(deftest constructor-odd-args-test
  (testing "choicemap throws on odd argument count (genmlx-ybw9)"
    (is (thrown? js/Error (cm/choicemap :x 1.0 :dangling))
        "trailing odd arg throws instead of silently dropping")))

(deftest set-value-non-node-test
  (testing "set-value refuses to destroy a Value leaf (genmlx-ybw9)"
    (is (thrown? js/Error (cm/set-value (cm/->Value 1.0) :x 2.0))
        "set-value on a leaf throws instead of silently rebuilding")))

(deftest stack-choicemaps-precondition-test
  (testing "stack-choicemaps enforces uniform structure (genmlx-ybw9)"
    (let [a (cm/choicemap :x 1.0)
          b (cm/choicemap :x 2.0)
          stacked (cm/stack-choicemaps [a b] vec)]
      (is (= [1.0 2.0] (cm/get-choice stacked [:x])) "homogeneous stacks fine"))
    (let [a (cm/choicemap :x 1.0)
          b (cm/choicemap :x 2.0 :y 3.0)]
      (is (thrown? js/Error (cm/stack-choicemaps [a b] vec))
          "later-only address throws instead of being dropped")
      (is (thrown? js/Error (cm/stack-choicemaps [b a] vec))
          "first-only address throws instead of stacking nils"))
    (let [leaf (cm/->Value 1.0)
          node (cm/choicemap :x 1.0)]
      (is (thrown? js/Error (cm/stack-choicemaps [leaf node] vec))
          "leaf/node mismatch throws"))))

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
