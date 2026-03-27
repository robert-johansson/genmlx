(ns genmlx.tutorial.ch04-test
  "Test file for Tutorial Chapter 4: Choice Maps and Traces.
   Exercises all data structure operations."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.selection :as sel]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn])
  (:require-macros [genmlx.gen :refer [gen]]))

(def pass (atom 0))
(def fail (atom 0))

(defn assert-true [msg pred]
  (if pred
    (do (swap! pass inc) (println "  PASS:" msg))
    (do (swap! fail inc) (println "  FAIL:" msg))))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass inc) (println "  PASS:" msg))
      (do (swap! fail inc) (println "  FAIL:" msg (str "expected=" expected " actual=" actual))))))

;; ============================================================
;; Listing 4.1: Value and Node — the two types
;; ============================================================
(println "\n== Listing 4.1: Value and Node ==")

(let [v (cm/->Value (mx/scalar 3.0))]
  (assert-true "Value has-value? is true" (cm/has-value? v))
  (assert-close "Value get-value" 3.0 (mx/item (cm/get-value v)) 0.001)
  (assert-true "Value get-submap returns EMPTY" (= cm/EMPTY (cm/get-submap v :anything))))

(let [n (cm/->Node {:x (cm/->Value (mx/scalar 1.0))
                     :y (cm/->Value (mx/scalar 2.0))})]
  (assert-true "Node has-value? is false" (not (cm/has-value? n)))
  (assert-true "Node get-submap :x is a Value" (cm/has-value? (cm/get-submap n :x)))
  (assert-close "Node :x value" 1.0 (mx/item (cm/get-value (cm/get-submap n :x))) 0.001)
  (assert-true "Node get-submap :z returns EMPTY" (= cm/EMPTY (cm/get-submap n :z))))

(assert-true "EMPTY is a Node" (instance? cm/Node cm/EMPTY))
(assert-true "EMPTY has no submaps" (empty? (:m cm/EMPTY)))

;; ============================================================
;; Listing 4.2: cm/choicemap constructor
;; ============================================================
(println "\n== Listing 4.2: cm/choicemap ==")

(let [c (cm/choicemap :slope (mx/scalar 2.0)
                      :intercept (mx/scalar 1.0)
                      :y0 (mx/scalar 5.0))]
  (assert-true "choicemap is a Node" (instance? cm/Node c))
  (assert-true "has :slope" (cm/has-value? (cm/get-submap c :slope)))
  (assert-true "has :intercept" (cm/has-value? (cm/get-submap c :intercept)))
  (assert-true "has :y0" (cm/has-value? (cm/get-submap c :y0)))
  (assert-close ":slope value" 2.0 (mx/item (cm/get-choice c [:slope])) 0.001))

;; ============================================================
;; Listing 4.3: cm/from-map and cm/to-map
;; ============================================================
(println "\n== Listing 4.3: from-map / to-map ==")

(let [nested (cm/from-map {:params {:slope 2.0 :intercept 1.0}
                            :obs {:y0 5.0}})]
  (assert-true "from-map creates nested structure" (instance? cm/Node nested))
  (assert-close "nested access" 2.0 (cm/get-choice nested [:params :slope]) 0.001)
  (assert-close "nested obs" 5.0 (cm/get-choice nested [:obs :y0]) 0.001))

;; to-map round-trip
(let [c (cm/choicemap :x (mx/scalar 3.0) :y (mx/scalar 4.0))
      m (cm/to-map c)]
  (assert-true "to-map returns a plain map" (map? m))
  (assert-true "to-map has :x" (contains? m :x))
  (assert-true "to-map :x is MLX array" (mx/array? (:x m))))

;; ============================================================
;; Listing 4.4: get-submap, get-value, get-choice
;; ============================================================
(println "\n== Listing 4.4: access operations ==")

(let [c (cm/choicemap :params (cm/choicemap :slope (mx/scalar 2.0)
                                             :intercept (mx/scalar 1.0))
                      :y0 (mx/scalar 5.0))]
  ;; get-submap: returns sub-choicemap at address
  (let [params (cm/get-submap c :params)]
    (assert-true "get-submap returns a Node" (instance? cm/Node params))
    (assert-true "sub has :slope" (cm/has-value? (cm/get-submap params :slope))))
  ;; get-value: extracts the raw value from a Value
  (let [y0-sub (cm/get-submap c :y0)]
    (assert-true "y0 is a Value" (cm/has-value? y0-sub))
    (assert-close "get-value" 5.0 (mx/item (cm/get-value y0-sub)) 0.001))
  ;; get-choice: path-based access
  (assert-close "get-choice path" 2.0 (mx/item (cm/get-choice c [:params :slope])) 0.001)
  (assert-close "get-choice flat" 5.0 (mx/item (cm/get-choice c [:y0])) 0.001))

;; ============================================================
;; Listing 4.5: set-value, set-choice, merge-cm
;; ============================================================
(println "\n== Listing 4.5: modification operations ==")

;; set-value: set a single address on a Node
(let [c (cm/set-value cm/EMPTY :x (mx/scalar 7.0))]
  (assert-close "set-value" 7.0 (mx/item (cm/get-choice c [:x])) 0.001))

;; set-choice: set at a path
(let [c (cm/set-choice cm/EMPTY [:params :slope] (mx/scalar 3.0))]
  (assert-close "set-choice path" 3.0 (mx/item (cm/get-choice c [:params :slope])) 0.001))

;; merge-cm: values in b override values in a
(let [a (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
      b (cm/choicemap :y (mx/scalar 99.0) :z (mx/scalar 3.0))
      merged (cm/merge-cm a b)]
  (assert-close "merge keeps a's :x" 1.0 (mx/item (cm/get-choice merged [:x])) 0.001)
  (assert-close "merge b overrides :y" 99.0 (mx/item (cm/get-choice merged [:y])) 0.001)
  (assert-close "merge adds b's :z" 3.0 (mx/item (cm/get-choice merged [:z])) 0.001))

;; ============================================================
;; Listing 4.6: cm/addresses
;; ============================================================
(println "\n== Listing 4.6: addresses ==")

(let [c (cm/choicemap :slope (mx/scalar 2.0)
                      :intercept (mx/scalar 1.0)
                      :y0 (mx/scalar 5.0))
      addrs (cm/addresses c)]
  (assert-true "addresses returns a vector" (vector? addrs))
  (assert-true "has 3 addresses" (= 3 (count addrs)))
  (assert-true "each address is a path" (every? vector? addrs)))

;; Nested addresses
(let [c (cm/choicemap :params (cm/choicemap :slope (mx/scalar 2.0)
                                             :intercept (mx/scalar 1.0)))
      addrs (cm/addresses c)]
  (assert-true "nested: 2 addresses" (= 2 (count addrs)))
  (assert-true "paths are [:params :slope] and [:params :intercept]"
               (= (set addrs) #{[:params :slope] [:params :intercept]})))

;; ============================================================
;; Listing 4.7: stack/unstack for batching
;; ============================================================
(println "\n== Listing 4.7: stack / unstack ==")

(let [cm1 (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 10.0))
      cm2 (cm/choicemap :x (mx/scalar 2.0) :y (mx/scalar 20.0))
      cm3 (cm/choicemap :x (mx/scalar 3.0) :y (mx/scalar 30.0))
      stacked (cm/stack-choicemaps [cm1 cm2 cm3] mx/stack)]
  (assert-true "stacked is a Node" (instance? cm/Node stacked))
  ;; :x should be [3]-shaped
  (let [x-arr (cm/get-value (cm/get-submap stacked :x))]
    (assert-true "stacked :x is [3]-shaped" (= [3] (mx/shape x-arr)))
    (assert-close "stacked :x[0]" 1.0 (mx/item (mx/index x-arr 0)) 0.001)
    (assert-close "stacked :x[2]" 3.0 (mx/item (mx/index x-arr 2)) 0.001))
  ;; Unstack back
  (let [unstacked (cm/unstack-choicemap stacked 3 mx/index
                    (fn [v] (= [] (mx/shape v))))]
    (assert-true "unstack produces 3 choicemaps" (= 3 (count unstacked)))
    (assert-close "unstacked[0] :x" 1.0
                  (mx/item (cm/get-value (cm/get-submap (nth unstacked 0) :x))) 0.001)
    (assert-close "unstacked[2] :y" 30.0
                  (mx/item (cm/get-value (cm/get-submap (nth unstacked 2) :y))) 0.001)))

;; ============================================================
;; Listing 4.8: Trace record
;; ============================================================
(println "\n== Listing 4.8: Trace record ==")

(let [model (dyn/auto-key (gen [x]
              (let [v (trace :v (dist/gaussian x 1))]
                (mx/multiply v v))))
      t (p/simulate model [0])]
  (assert-true "trace has :gen-fn" (some? (:gen-fn t)))
  (assert-true "trace has :args" (= [0] (:args t)))
  (assert-true "trace has :choices" (instance? cm/Node (:choices t)))
  (assert-true "trace has :retval (v^2)" (>= (mx/item (:retval t)) 0))
  (assert-true "trace has :score" (js/Number.isFinite (mx/item (:score t)))))

;; ============================================================
;; Listing 4.9: Trace metadata (splice-scores)
;; ============================================================
(println "\n== Listing 4.9: trace metadata ==")

(let [sub-model (gen [] (trace :z (dist/gaussian 0 1)))
      parent-model (gen [] (splice :child sub-model []))
      model (dyn/auto-key parent-model)
      t (p/simulate model [])
      meta-data (meta t)]
  (assert-true "trace has metadata" (some? meta-data))
  ;; splice-scores may or may not be present depending on implementation
  ;; Just verify the trace works with splice
  (assert-true "parent trace has :child submap"
               (not= cm/EMPTY (cm/get-submap (:choices t) :child))))

;; ============================================================
;; Listing 4.10: Selections — select, all, none
;; ============================================================
(println "\n== Listing 4.10: selections ==")

;; select specific addresses
(let [s (sel/select :x :y)]
  (assert-true "select :x is selected" (sel/selected? s :x))
  (assert-true "select :y is selected" (sel/selected? s :y))
  (assert-true "select :z is not selected" (not (sel/selected? s :z))))

;; all selects everything
(assert-true "all selects :anything" (sel/selected? sel/all :anything))
(assert-true "all selects :whatever" (sel/selected? sel/all :whatever))

;; none selects nothing
(assert-true "none does not select :x" (not (sel/selected? sel/none :x)))

;; ============================================================
;; Listing 4.11: complement-sel
;; ============================================================
(println "\n== Listing 4.11: complement ==")

(let [s (sel/select :x :y)
      c (sel/complement-sel s)]
  (assert-true "complement: :x not selected" (not (sel/selected? c :x)))
  (assert-true "complement: :y not selected" (not (sel/selected? c :y)))
  (assert-true "complement: :z IS selected" (sel/selected? c :z))
  (assert-true "complement: :anything IS selected" (sel/selected? c :anything)))

;; ============================================================
;; Listing 4.12: hierarchical selections
;; ============================================================
(println "\n== Listing 4.12: hierarchical ==")

(let [s (sel/hierarchical :params (sel/select :slope))]
  (assert-true ":params is selected" (sel/selected? s :params))
  (assert-true ":other is not selected" (not (sel/selected? s :other)))
  ;; Sub-selection under :params
  (let [sub (sel/get-subselection s :params)]
    (assert-true "sub selects :slope" (sel/selected? sub :slope))
    (assert-true "sub does not select :intercept" (not (sel/selected? sub :intercept)))))

;; ============================================================
;; Listing 4.13: Selections as Boolean algebra
;; ============================================================
(println "\n== Listing 4.13: Boolean algebra ==")

;; all is the top element, none is the bottom
(assert-true "all contains everything" (sel/selected? sel/all :anything))
(assert-true "none contains nothing" (not (sel/selected? sel/none :anything)))

;; complement of all is none (functionally)
(let [comp-all (sel/complement-sel sel/all)]
  (assert-true "complement(all) selects nothing" (not (sel/selected? comp-all :x))))

;; complement of none is all (functionally)
(let [comp-none (sel/complement-sel sel/none)]
  (assert-true "complement(none) selects everything" (sel/selected? comp-none :x)))

;; ============================================================
;; Summary
;; ============================================================
(println (str "\n== Chapter 4 tests: " @pass " PASS, " @fail " FAIL =="))
(when (pos? @fail) (js/process.exit 1))
