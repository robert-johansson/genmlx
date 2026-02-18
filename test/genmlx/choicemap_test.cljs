(ns genmlx.choicemap-test
  (:require [genmlx.choicemap :as cm]))

(defn assert= [msg expected actual]
  (if (= expected actual)
    (println "  PASS:" msg)
    (do (println "  FAIL:" msg)
        (println "    expected:" (pr-str expected))
        (println "    actual:  " (pr-str actual)))))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-false [msg actual]
  (if (not actual)
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected falsy")))

(println "\n=== ChoiceMap Tests ===\n")

;; Test EMPTY
(println "-- EMPTY --")
(assert-false "EMPTY has no value" (cm/has-value? cm/EMPTY))
(assert= "EMPTY addresses" [] (cm/addresses cm/EMPTY))

;; Test Value
(println "\n-- Value --")
(let [v (cm/->Value 42)]
  (assert-true "Value has value" (cm/has-value? v))
  (assert= "Value get-value" 42 (cm/get-value v)))

;; Test choicemap constructor
(println "\n-- choicemap constructor --")
(let [m (cm/choicemap :x 1.0 :y 2.0)]
  (assert-false "Node has no value" (cm/has-value? m))
  (assert= "get-choice :x" 1.0 (cm/get-choice m [:x]))
  (assert= "get-choice :y" 2.0 (cm/get-choice m [:y]))
  (assert= "addresses" #{[:x] [:y]} (set (cm/addresses m))))

;; Test nested choicemap
(println "\n-- nested choicemap --")
(let [m (cm/choicemap :params {:slope 2.0 :intercept 1.0} :noise 0.5)]
  (assert= "nested get-choice" 2.0 (cm/get-choice m [:params :slope]))
  (assert= "nested get-choice" 1.0 (cm/get-choice m [:params :intercept]))
  (assert= "flat get-choice" 0.5 (cm/get-choice m [:noise])))

;; Test set-choice
(println "\n-- set-choice --")
(let [m (cm/set-choice cm/EMPTY [:x] 1.0)]
  (assert= "set single" 1.0 (cm/get-choice m [:x])))
(let [m (cm/set-choice cm/EMPTY [:a :b] 3.0)]
  (assert= "set nested" 3.0 (cm/get-choice m [:a :b])))

;; Test merge-cm
(println "\n-- merge-cm --")
(let [a (cm/choicemap :x 1.0 :y 2.0)
      b (cm/choicemap :y 3.0 :z 4.0)
      merged (cm/merge-cm a b)]
  (assert= "merge keeps a" 1.0 (cm/get-choice merged [:x]))
  (assert= "merge overrides" 3.0 (cm/get-choice merged [:y]))
  (assert= "merge adds b" 4.0 (cm/get-choice merged [:z])))

;; Test to-map / from-map
(println "\n-- to-map / from-map --")
(let [m (cm/choicemap :x 1.0 :y 2.0)
      plain (cm/to-map m)]
  (assert= "to-map" {:x 1.0 :y 2.0} plain))
(let [m (cm/from-map {:x 1.0 :y {:a 2.0}})]
  (assert= "from-map flat" 1.0 (cm/get-choice m [:x]))
  (assert= "from-map nested" 2.0 (cm/get-choice m [:y :a])))

(println "\nAll choicemap tests complete.")
