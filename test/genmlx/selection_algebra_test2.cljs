(ns genmlx.selection-algebra-test2
  "Algebraic property tests for Selection — specification-driven, cljs.test."
  (:require [cljs.test :refer [deftest is are testing]]
            [genmlx.selection :as sel]))

;; ---------------------------------------------------------------------------
;; all selects everything, none selects nothing
;; ---------------------------------------------------------------------------

(deftest all-selects-every-address
  (are [addr] (sel/selected? sel/all addr)
    :x :y :z :anything :deeply-nested-name))

(deftest none-rejects-every-address
  (are [addr] (not (sel/selected? sel/none addr))
    :x :y :z :anything :deeply-nested-name))

;; ---------------------------------------------------------------------------
;; Complement involution: complement(complement(s)) = s
;; ---------------------------------------------------------------------------

(defn complement-involution?
  "True if complement(complement(s)) agrees with s on all given addrs."
  [s addrs]
  (let [cc (-> s sel/complement-sel sel/complement-sel)]
    (every? #(= (sel/selected? s %) (sel/selected? cc %)) addrs)))

(deftest complement-involution-on-select
  (let [s (sel/select :x :y)]
    (is (complement-involution? s [:x :y :z :w])
        "complement(complement(select(:x :y))) = select(:x :y)")))

(deftest complement-involution-on-all
  (is (complement-involution? sel/all [:x :y :z])
      "complement(complement(all)) = all"))

(deftest complement-involution-on-none
  (is (complement-involution? sel/none [:x :y :z])
      "complement(complement(none)) = none"))

(deftest complement-involution-on-hierarchical
  (let [h (sel/hierarchical :a (sel/select :x :y) :b sel/all)]
    (is (complement-involution? h [:a :b :c :d]))))

;; ---------------------------------------------------------------------------
;; Complement semantics: complement(all) = none, complement(none) = all
;; ---------------------------------------------------------------------------

(deftest complement-all-behaves-as-none
  (let [comp-all (sel/complement-sel sel/all)]
    (are [addr] (not (sel/selected? comp-all addr))
      :x :y :z)))

(deftest complement-none-behaves-as-all
  (let [comp-none (sel/complement-sel sel/none)]
    (are [addr] (sel/selected? comp-none addr)
      :x :y :z)))

;; ---------------------------------------------------------------------------
;; Select: inclusion and exclusion
;; ---------------------------------------------------------------------------

(deftest select-includes-specified-excludes-others
  (let [s (sel/select :a :b :c)]
    (are [addr expected]
      (= expected (sel/selected? s addr))
      :a true
      :b true
      :c true
      :d false
      :e false)))

(deftest from-set-matches-select
  (let [s1 (sel/select :x :y)
        s2 (sel/from-set #{:x :y})]
    (are [addr]
      (= (sel/selected? s1 addr) (sel/selected? s2 addr))
      :x :y :z)))

;; ---------------------------------------------------------------------------
;; Complement of select: inverts membership
;; ---------------------------------------------------------------------------

(deftest complement-of-select-inverts
  (let [s (sel/select :x :y)
        c (sel/complement-sel s)]
    (are [addr s-expected c-expected]
      (and (= s-expected (sel/selected? s addr))
           (= c-expected (sel/selected? c addr)))
      :x true  false
      :y true  false
      :z false true
      :w false true)))

;; ---------------------------------------------------------------------------
;; Hierarchical selection
;; ---------------------------------------------------------------------------

(deftest hierarchical-selects-registered-addresses
  (let [h (sel/hierarchical :a (sel/select :x :y) :b sel/all)]
    (is (sel/selected? h :a))
    (is (sel/selected? h :b))
    (is (not (sel/selected? h :c)))))

(deftest hierarchical-subselection-propagates
  (let [h (sel/hierarchical :a (sel/select :x :y) :b sel/all)]
    (testing "subselection at :a is select(:x :y)"
      (let [sub-a (sel/get-subselection h :a)]
        (is (sel/selected? sub-a :x))
        (is (sel/selected? sub-a :y))
        (is (not (sel/selected? sub-a :z)))))
    (testing "subselection at :b is all"
      (let [sub-b (sel/get-subselection h :b)]
        (is (sel/selected? sub-b :anything))))
    (testing "subselection at missing address is none"
      (let [sub-c (sel/get-subselection h :c)]
        (is (not (sel/selected? sub-c :anything)))))))

(deftest nested-hierarchical-selection
  (let [inner (sel/hierarchical :p (sel/select :q))
        outer (sel/hierarchical :a inner)]
    (is (sel/selected? outer :a))
    (let [sub-a (sel/get-subselection outer :a)]
      (is (sel/selected? sub-a :p))
      (is (not (sel/selected? sub-a :z)))
      (let [sub-a-p (sel/get-subselection sub-a :p)]
        (is (sel/selected? sub-a-p :q))))))

;; ---------------------------------------------------------------------------
;; Complement distributes through hierarchy
;; ---------------------------------------------------------------------------

(deftest complement-of-hierarchical-subselection
  (let [h (sel/hierarchical :a (sel/select :x :y))
        c (sel/complement-sel h)]
    (testing "complement flips top-level"
      (is (not (sel/selected? c :a)))
      (is (sel/selected? c :z)))
    (testing "complement propagates into subselections"
      (let [c-sub-a (sel/get-subselection c :a)]
        (is (not (sel/selected? c-sub-a :x))
            "complement(hierarchical).subsel(:a) complements the inner")
        (is (sel/selected? c-sub-a :z))))))

;; ---------------------------------------------------------------------------
;; all/none subselection identity
;; ---------------------------------------------------------------------------

(deftest all-subselection-is-all
  (are [addr] (sel/selected? (sel/get-subselection sel/all addr) :anything)
    :x :y :deeply-nested))

(deftest none-subselection-is-none
  (are [addr] (not (sel/selected? (sel/get-subselection sel/none addr) :anything))
    :x :y :deeply-nested))

;; ---------------------------------------------------------------------------
;; select subselection is all (by design)
;; ---------------------------------------------------------------------------

(deftest select-subselection-is-all-for-selected-addrs
  (let [s (sel/select :a :b)]
    (is (sel/selected? (sel/get-subselection s :a) :any-sub-addr)
        "subselection of a selected addr in SelectAddrs is all")))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
