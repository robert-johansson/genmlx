(ns genmlx.choicemap-algebra-test
  "Algebraic property tests for ChoiceMap operations."
  (:require [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert= [msg expected actual]
  (if (= expected actual)
    (println "  PASS:" msg)
    (do (println "  FAIL:" msg)
        (println "    expected:" (pr-str expected))
        (println "    actual:  " (pr-str actual)))))

(defn cm= [a b]
  "Structural equality via to-map (handles Value wrapper comparison)."
  (= (cm/to-map a) (cm/to-map b)))

(defn assert-cm= [msg expected actual]
  (if (cm= expected actual)
    (println "  PASS:" msg)
    (do (println "  FAIL:" msg)
        (println "    expected:" (pr-str (cm/to-map expected)))
        (println "    actual:  " (pr-str (cm/to-map actual))))))

(println "\n=== ChoiceMap Algebra Tests ===\n")

;; Use MLX scalars as values to match real usage
(let [v1 (mx/scalar 1.0)
      v2 (mx/scalar 2.0)
      v3 (mx/scalar 3.0)]

  ;; -- Identity laws --
  (println "-- merge identity --")
  (let [cm1 (cm/choicemap :x v1 :y v2)]
    (assert-cm= "left identity: merge(EMPTY, cm) = cm"
                cm1 (cm/merge-cm cm/EMPTY cm1))
    (assert-cm= "right identity: merge(cm, EMPTY) = cm"
                cm1 (cm/merge-cm cm1 cm/EMPTY)))

  ;; -- Get-set law --
  (println "\n-- get-set --")
  (let [cm1 (cm/choicemap :x v1 :y v2)
        cm2 (cm/set-choice cm1 [:x] v3)]
    (assert= "get-set: get(set(cm, a, v), a) = v"
             (mx/realize v3) (mx/realize (cm/get-choice cm2 [:x]))))

  ;; -- Set-get (non-interference) --
  (println "\n-- set-get (non-interference) --")
  (let [cm1 (cm/choicemap :x v1 :y v2)
        cm2 (cm/set-choice cm1 [:x] v3)]
    (assert= "set-get: set at :x doesn't change :y"
             (mx/realize v2) (mx/realize (cm/get-choice cm2 [:y]))))

  ;; -- Merge override --
  (println "\n-- merge override --")
  (let [a (cm/choicemap :x v1 :y v2)
        b (cm/choicemap :x v3)
        merged (cm/merge-cm a b)]
    (assert= "merge override: b's :x wins"
             (mx/realize v3) (mx/realize (cm/get-choice merged [:x]))))

  ;; -- Merge preserves --
  (println "\n-- merge preserves --")
  (let [a (cm/choicemap :x v1 :y v2)
        b (cm/choicemap :z v3)
        merged (cm/merge-cm a b)]
    (assert= "merge preserves: a's :x survives"
             (mx/realize v1) (mx/realize (cm/get-choice merged [:x])))
    (assert= "merge preserves: a's :y survives"
             (mx/realize v2) (mx/realize (cm/get-choice merged [:y])))
    (assert= "merge adds: b's :z added"
             (mx/realize v3) (mx/realize (cm/get-choice merged [:z]))))

  ;; -- Nested paths --
  (println "\n-- nested paths --")
  (let [cm1 (cm/set-choice cm/EMPTY [:a :b] v1)]
    (assert= "nested set/get"
             (mx/realize v1) (mx/realize (cm/get-choice cm1 [:a :b])))
    ;; Set deeper path, original still accessible
    (let [cm2 (cm/set-choice cm1 [:a :c] v2)]
      (assert= "nested: original path survives"
               (mx/realize v1) (mx/realize (cm/get-choice cm2 [:a :b])))
      (assert= "nested: new path accessible"
               (mx/realize v2) (mx/realize (cm/get-choice cm2 [:a :c])))))

  ;; -- Addresses round-trip --
  (println "\n-- addresses round-trip --")
  (let [cm1 (cm/choicemap :x v1 :y v2)
        cm2 (cm/set-choice cm1 [:nested :deep] v3)
        addrs (cm/addresses cm2)]
    (assert-true "all addresses retrievable"
                 (every? (fn [path]
                           (some? (cm/get-choice cm2 path)))
                         addrs))
    (assert= "address count"
             3 (count addrs)))

  ;; -- Associativity --
  (println "\n-- merge associativity --")
  (let [a (cm/choicemap :x v1)
        b (cm/choicemap :y v2)
        c (cm/choicemap :z v3)
        left  (cm/merge-cm (cm/merge-cm a b) c)
        right (cm/merge-cm a (cm/merge-cm b c))]
    (assert-cm= "associativity: merge(merge(a,b),c) = merge(a,merge(b,c))"
                left right))

  ;; Associativity with overlapping keys
  (let [a (cm/choicemap :x (mx/scalar 1.0))
        b (cm/choicemap :x (mx/scalar 2.0) :y (mx/scalar 3.0))
        c (cm/choicemap :y (mx/scalar 4.0) :z (mx/scalar 5.0))
        left  (cm/merge-cm (cm/merge-cm a b) c)
        right (cm/merge-cm a (cm/merge-cm b c))]
    (assert-cm= "associativity with overlapping keys"
                left right)))

(println "\nAll choicemap algebra tests complete.")
