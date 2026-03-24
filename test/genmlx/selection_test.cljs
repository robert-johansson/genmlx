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
      (is (sel/selected? s :sub) "hierarchical includes :sub")
      (is (not (sel/selected? s :other)) "hierarchical excludes :other")
      (let [sub (sel/get-subselection s :sub)]
        (is (sel/selected? sub :x) "subselection includes :x")
        (is (not (sel/selected? sub :z)) "subselection excludes :z")))))

(deftest complement-test
  (testing "complement"
    (let [s (sel/complement-sel (sel/select :x :y))]
      (is (not (sel/selected? s :x)) "complement excludes :x")
      (is (sel/selected? s :z) "complement includes :z"))))

(cljs.test/run-tests)
