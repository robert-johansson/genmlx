(ns genmlx.selection-test
  (:require [genmlx.selection :as sel]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-false [msg actual]
  (if (not actual)
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected falsy")))

(println "\n=== Selection Tests ===\n")

;; Test all
(println "-- all --")
(assert-true "all selects :x" (sel/selected? sel/all :x))
(assert-true "all selects :anything" (sel/selected? sel/all :anything))

;; Test none
(println "\n-- none --")
(assert-false "none rejects :x" (sel/selected? sel/none :x))
(assert-false "none rejects :anything" (sel/selected? sel/none :anything))

;; Test select
(println "\n-- select --")
(let [s (sel/select :x :y)]
  (assert-true "select includes :x" (sel/selected? s :x))
  (assert-true "select includes :y" (sel/selected? s :y))
  (assert-false "select excludes :z" (sel/selected? s :z)))

;; Test set wrapper as selection
(println "\n-- set as selection --")
(let [s (sel/from-set #{:a :b})]
  (assert-true "set includes :a" (sel/selected? s :a))
  (assert-false "set excludes :c" (sel/selected? s :c)))

;; Test hierarchical
(println "\n-- hierarchical --")
(let [s (sel/hierarchical :sub (sel/select :x :y))]
  (assert-true "hierarchical includes :sub" (sel/selected? s :sub))
  (assert-false "hierarchical excludes :other" (sel/selected? s :other))
  (let [sub (sel/get-subselection s :sub)]
    (assert-true "subselection includes :x" (sel/selected? sub :x))
    (assert-false "subselection excludes :z" (sel/selected? sub :z))))

;; Test complement
(println "\n-- complement --")
(let [s (sel/complement-sel (sel/select :x :y))]
  (assert-false "complement excludes :x" (sel/selected? s :x))
  (assert-true "complement includes :z" (sel/selected? s :z)))

(println "\nAll selection tests complete.")
