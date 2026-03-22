(ns genmlx.choicemap-algebra-test2
  "Algebraic property tests for ChoiceMap — specification-driven, cljs.test."
  (:require [cljs.test :refer [deftest is are testing]]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]))

(defn- cm=
  "Structural equality via to-map (handles Value wrapper comparison)."
  [a b]
  (= (cm/to-map a) (cm/to-map b)))

;; ---------------------------------------------------------------------------
;; Round-trip: to-map / from-map
;; ---------------------------------------------------------------------------

(deftest to-map-from-map-round-trip
  (testing "flat choicemap round-trips through plain maps"
    (let [cm1 (cm/choicemap :x 1 :y 2)
          rt (cm/from-map (cm/to-map cm1))]
      (is (cm= cm1 rt))))

  (testing "nested choicemap round-trips"
    (let [cm1 (cm/set-choice cm/EMPTY [:a :b] 10)
          cm2 (cm/set-choice cm1 [:a :c] 20)
          rt (cm/from-map (cm/to-map cm2))]
      (is (cm= cm2 rt))))

  (testing "3-level deep nesting round-trips"
    (let [cm1 (-> cm/EMPTY
                  (cm/set-choice [:a :b :c] 1)
                  (cm/set-choice [:a :b :d] 2)
                  (cm/set-choice [:a :e] 3))
          rt (cm/from-map (cm/to-map cm1))]
      (is (cm= cm1 rt))))

  (testing "empty choicemap round-trips"
    (is (cm= cm/EMPTY (cm/from-map (cm/to-map cm/EMPTY))))))

;; ---------------------------------------------------------------------------
;; EMPTY as identity element for merge
;; ---------------------------------------------------------------------------

(deftest empty-is-merge-identity
  (are [desc cm]
       (and (cm= cm (cm/merge-cm cm/EMPTY cm))
            (cm= cm (cm/merge-cm cm cm/EMPTY)))

    "flat" (cm/choicemap :x 1 :y 2)
    "nested" (cm/set-choice cm/EMPTY [:a :b] 42)
    "empty" cm/EMPTY))

;; ---------------------------------------------------------------------------
;; Merge override semantics: b wins over a
;; ---------------------------------------------------------------------------

(deftest merge-right-overrides-left
  (let [a (cm/choicemap :x 1 :y 2)
        b (cm/choicemap :x 99)
        merged (cm/merge-cm a b)]
    (is (= 99 (cm/get-choice merged [:x]))
        "b's value at :x overrides a's")
    (is (= 2 (cm/get-choice merged [:y]))
        "a's uncontested key :y survives")))

(deftest merge-adds-new-keys
  (let [a (cm/choicemap :x 1)
        b (cm/choicemap :y 2)
        merged (cm/merge-cm a b)]
    (is (= 1 (cm/get-choice merged [:x])))
    (is (= 2 (cm/get-choice merged [:y])))))

(deftest merge-nested-override
  (testing "merge overrides at nested paths"
    (let [a (cm/set-choice cm/EMPTY [:a :b] 1)
          b (cm/set-choice cm/EMPTY [:a :b] 99)
          merged (cm/merge-cm a b)]
      (is (= 99 (cm/get-choice merged [:a :b]))))))

(deftest merge-nested-disjoint-union
  (testing "merge combines disjoint nested paths"
    (let [a (cm/set-choice cm/EMPTY [:a :b] 1)
          b (cm/set-choice cm/EMPTY [:a :c] 2)
          merged (cm/merge-cm a b)]
      (is (= 1 (cm/get-choice merged [:a :b])))
      (is (= 2 (cm/get-choice merged [:a :c]))))))

;; ---------------------------------------------------------------------------
;; Merge associativity
;; ---------------------------------------------------------------------------

(deftest merge-is-associative
  (testing "disjoint keys"
    (let [a (cm/choicemap :x 1)
          b (cm/choicemap :y 2)
          c (cm/choicemap :z 3)]
      (is (cm= (cm/merge-cm (cm/merge-cm a b) c)
               (cm/merge-cm a (cm/merge-cm b c))))))

  (testing "overlapping keys"
    (let [a (cm/choicemap :x 1)
          b (cm/choicemap :x 2 :y 3)
          c (cm/choicemap :y 4 :z 5)]
      (is (cm= (cm/merge-cm (cm/merge-cm a b) c)
               (cm/merge-cm a (cm/merge-cm b c)))))))

;; ---------------------------------------------------------------------------
;; Address enumeration
;; ---------------------------------------------------------------------------

(deftest addresses-returns-all-leaf-paths
  (testing "flat choicemap"
    (let [addrs (cm/addresses (cm/choicemap :x 1 :y 2 :z 3))]
      (is (= 3 (count addrs)))
      (is (= #{[:x] [:y] [:z]} (set addrs)))))

  (testing "nested choicemap"
    (let [cm1 (-> cm/EMPTY
                  (cm/set-choice [:a :b] 1)
                  (cm/set-choice [:a :c] 2)
                  (cm/set-choice [:d] 3))
          addrs (cm/addresses cm1)]
      (is (= 3 (count addrs)))
      (is (= #{[:a :b] [:a :c] [:d]} (set addrs)))))

  (testing "deep nesting"
    (let [cm1 (cm/set-choice cm/EMPTY [:a :b :c :d] 1)
          addrs (cm/addresses cm1)]
      (is (= [[:a :b :c :d]] addrs))))

  (testing "empty choicemap has no addresses"
    (is (= [] (cm/addresses cm/EMPTY)))))

;; ---------------------------------------------------------------------------
;; Get/set laws
;; ---------------------------------------------------------------------------

(deftest get-set-law
  (are [path val]
       (= val (cm/get-choice (cm/set-choice cm/EMPTY path val) path))
    [:x] 42
    [:a :b] "hello"
    [:a :b :c] :keyword))

(deftest set-preserves-other-paths
  (let [cm1 (-> cm/EMPTY
                (cm/set-choice [:x] 1)
                (cm/set-choice [:y] 2))
        cm2 (cm/set-choice cm1 [:x] 99)]
    (is (= 99 (cm/get-choice cm2 [:x])))
    (is (= 2 (cm/get-choice cm2 [:y]))
        "setting :x does not disturb :y")))

;; ---------------------------------------------------------------------------
;; Deep nesting (3+ levels)
;; ---------------------------------------------------------------------------

(deftest deep-nesting-operations
  (let [cm1 (-> cm/EMPTY
                (cm/set-choice [:level1 :level2 :level3] "deep")
                (cm/set-choice [:level1 :level2 :sibling] "sibling")
                (cm/set-choice [:level1 :other] "other"))]
    (is (= "deep" (cm/get-choice cm1 [:level1 :level2 :level3])))
    (is (= "sibling" (cm/get-choice cm1 [:level1 :level2 :sibling])))
    (is (= "other" (cm/get-choice cm1 [:level1 :other])))
    (is (= 3 (count (cm/addresses cm1))))))

;; ---------------------------------------------------------------------------
;; MLX scalar values
;; ---------------------------------------------------------------------------

(deftest mlx-values-survive-choicemap-operations
  (let [v (mx/scalar 3.14)
        cm1 (cm/choicemap :x v)
        retrieved (cm/get-choice cm1 [:x])]
    (is (identical? v retrieved)
        "MLX array identity preserved through set/get")))

(deftest mlx-values-survive-merge
  (let [v1 (mx/scalar 1.0)
        v2 (mx/scalar 2.0)
        a (cm/choicemap :x v1)
        b (cm/choicemap :y v2)
        {:keys [x y]} (cm/to-map (cm/merge-cm a b))]
    (is (identical? v1 x))
    (is (identical? v2 y))))

;; ---------------------------------------------------------------------------
;; Choicemap constructor from nested maps
;; ---------------------------------------------------------------------------

(deftest choicemap-constructor-handles-nested-maps
  (let [cm1 (cm/choicemap :params {:slope 2.0 :intercept 1.0})]
    (is (= 2.0 (cm/get-choice cm1 [:params :slope])))
    (is (= 1.0 (cm/get-choice cm1 [:params :intercept])))))

;; ---------------------------------------------------------------------------
;; has-value? / get-submap
;; ---------------------------------------------------------------------------

(deftest has-value-distinguishes-leaves-from-nodes
  (let [cm1 (cm/choicemap :x 1)]
    (is (not (cm/has-value? cm1))
        "Node is not a leaf")
    (is (cm/has-value? (cm/get-submap cm1 :x))
        "Value at :x is a leaf")
    (is (not (cm/has-value? cm/EMPTY))
        "EMPTY is not a leaf")))

(deftest get-submap-returns-empty-for-missing
  (is (cm= cm/EMPTY (cm/get-submap (cm/choicemap :x 1) :missing))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
