;; @tier fast core
(ns genmlx.selection-test
  "Selection algebra tests: all, none, select, from-set, hierarchical, complement."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.selection :as sel]))

(deftest all-selection-test
  (testing "all"
    (is (sel/selected? sel/all :x) "all selects :x")
    (is (sel/selected? sel/all :anything) "all selects :anything")))

(deftest none-selection-test
  (testing "none"
    (is (not (sel/selected? sel/none :x)) "none rejects :x")
    (is (not (sel/selected? sel/none :anything)) "none rejects :anything")))

(deftest select-test
  (testing "select"
    (let [s (sel/select :x :y)]
      (is (sel/selected? s :x) "select includes :x")
      (is (sel/selected? s :y) "select includes :y")
      (is (not (sel/selected? s :z)) "select excludes :z"))))

(deftest set-as-selection-test
  (testing "set as selection"
    (let [s (sel/from-set #{:a :b})]
      (is (sel/selected? s :a) "set includes :a")
      (is (not (sel/selected? s :c)) "set excludes :c"))))

(deftest hierarchical-test
  (testing "hierarchical"
    (let [s (sel/hierarchical :sub (sel/select :x :y))]
      ;; A PARTIAL subselection at :sub means "descend into :sub and select
      ;; :x/:y" — it does NOT select the leaf :sub itself. (Gen.jl semantics.)
      (is (not (sel/selected? s :sub)) "hierarchical with partial subsel does NOT select :sub as a leaf")
      (is (not (sel/selected? s :other)) "hierarchical excludes :other")
      (let [sub (sel/get-subselection s :sub)]
        (is (sel/selected? sub :x) "subselection includes :x")
        (is (not (sel/selected? sub :z)) "subselection excludes :z"))))
  (testing "an entry mapped to `all` selects its address as a leaf"
    (let [s (sel/hierarchical :a sel/all :sub (sel/select :z))]
      (is (sel/selected? s :a) "hierarchical with `all` subsel selects :a as a leaf")
      (is (not (sel/selected? s :sub)) "partial subsel does not select :sub as a leaf")
      (is (identical? sel/all (sel/get-subselection s :a)) "get-subselection :a => all")
      (is (not (sel/selected? (sel/get-subselection s :other) :anything))
          "unmapped address => none subselection"))))

(deftest get-subselection-select-test
  (testing "SelectAddrs.get-subselection: all iff selected, else none (genmlx-yey5)"
    (let [s (sel/select :x :y)]
      ;; Selected address => descend with `all` (select the whole subtree).
      (is (identical? sel/all (sel/get-subselection s :x)) ":x subselection is `all`")
      (is (identical? sel/all (sel/get-subselection s :y)) ":y subselection is `all`")
      ;; UNselected address => descend with `none` (resample nothing). This is
      ;; the fix: it previously returned `all` unconditionally, so
      ;; (regenerate (select :x)) resampled entire unselected splice subtrees.
      (is (identical? sel/none (sel/get-subselection s :z)) ":z subselection is `none`")
      (is (not (sel/selected? (sel/get-subselection s :z) :anything))
          "nothing under an unselected address is selected"))))

(deftest get-subselection-all-none-test
  (testing "all/none subselections are themselves"
    (is (identical? sel/all (sel/get-subselection sel/all :anything)))
    (is (identical? sel/none (sel/get-subselection sel/none :anything)))))

(deftest complement-test
  (testing "complement selected?"
    (let [s (sel/complement-sel (sel/select :x :y))]
      (is (not (sel/selected? s :x)) "complement excludes :x")
      (is (sel/selected? s :z) "complement includes :z")))
  (testing "complement get-subselection is consistent with selected? (mirror-bug regression)"
    ;; The Complement was a "mirror" of the SelectAddrs bug: selected? said an
    ;; address was selected while the descended subselection resampled nothing
    ;; (or vice versa). With the inner subselections fixed, the two now agree.
    (let [s (sel/complement-sel (sel/select :x))]
      ;; :x is excluded by the complement => nothing under :x is selected.
      (is (not (sel/selected? s :x)) "complement: :x excluded as leaf")
      (is (not (sel/selected? (sel/get-subselection s :x) :anything))
          "complement: nothing under excluded :x is selected")
      ;; :y is included by the complement => everything under :y is selected.
      (is (sel/selected? s :y) "complement: :y included as leaf")
      (is (sel/selected? (sel/get-subselection s :y) :anything)
          "complement: everything under included :y is selected"))))

(cljs.test/run-tests)
