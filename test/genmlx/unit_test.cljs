(ns genmlx.unit-test
  "Unit tests demoted from property test files.
   These test genuine laws but don't need random generation (test.check).
   Pure arithmetic, data-structure, and optimizer convergence checks."
  (:require [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.serialize :as ser]
            [genmlx.learning :as learn]
            [genmlx.inference.util :as u]))

;; ---------------------------------------------------------------------------
;; Test infrastructure
;; ---------------------------------------------------------------------------

(def pass-count (volatile! 0))
(def fail-count (volatile! 0))

(defn assert-true [name pred]
  (if pred
    (do (vswap! pass-count inc) (println "  PASS:" name))
    (do (vswap! fail-count inc) (println "  FAIL:" name))))

(defn- close? [a b tol]
  (and (number? a) (number? b)
       (js/isFinite a) (js/isFinite b)
       (<= (js/Math.abs (- a b)) tol)))

(println "\n=== Unit Tests (demoted from property tests) ===\n")

;; ---------------------------------------------------------------------------
;; Parameter Store (3 tests, from gradient_learning_property_test.cljs)
;; ---------------------------------------------------------------------------

(println "-- parameter store --")

;; 1. param-store: get/set round-trip
(let [store (-> (learn/make-param-store)
                (learn/set-param :a (mx/scalar 3.14)))
      v (learn/get-param store :a)]
  (mx/eval! v)
  (assert-true "param-store: get/set round-trip"
    (close? 3.14 (mx/item v) 1e-5)))

;; 2. param-store: param-names includes all stored keys
(let [store (-> (learn/make-param-store)
                (learn/set-param :a (mx/scalar 1.0))
                (learn/set-param :b (mx/scalar 2.0))
                (learn/set-param :c (mx/scalar 3.0)))
      names (set (learn/param-names store))]
  (assert-true "param-store: param-names includes all stored keys"
    (and (contains? names :a)
         (contains? names :b)
         (contains? names :c))))

;; 3. param-store: params->array / array->params round-trip
(let [store (-> (learn/make-param-store)
                (learn/set-param :x (mx/scalar 5.0))
                (learn/set-param :y (mx/scalar -2.0)))
      arr (learn/params->array store [:x :y])
      _ (mx/eval! arr)
      recovered (learn/array->params arr [:x :y])
      rx (do (mx/eval! (:x recovered)) (mx/item (:x recovered)))
      ry (do (mx/eval! (:y recovered)) (mx/item (:y recovered)))]
  (assert-true "param-store: params->array / array->params round-trip"
    (and (close? 5.0 rx 1e-5)
         (close? -2.0 ry 1e-5))))

;; ---------------------------------------------------------------------------
;; Optimizers (3 tests, from gradient_learning_property_test.cljs)
;; ---------------------------------------------------------------------------

(println "\n-- optimizers --")

;; 4. SGD: step moves params in negative gradient direction
(let [params (mx/array [5.0])
      grad-arr (mx/array [2.0])
      new-params (learn/sgd-step params grad-arr 0.1)]
  (mx/eval! new-params)
  ;; Should be 5.0 - 0.1*2.0 = 4.8
  (assert-true "SGD: step moves params in negative gradient direction"
    (close? 4.8 (mx/item (mx/index new-params 0)) 1e-4)))

;; 5. SGD: step magnitude proportional to lr
(let [params (mx/array [5.0])
      grad-arr (mx/array [2.0])
      p1 (learn/sgd-step params grad-arr 0.1)
      p2 (learn/sgd-step params grad-arr 0.2)
      _ (mx/eval! p1 p2)
      delta1 (- 5.0 (mx/item (mx/index p1 0)))  ;; 0.1*2.0 = 0.2
      delta2 (- 5.0 (mx/item (mx/index p2 0)))]  ;; 0.2*2.0 = 0.4
  ;; delta2 should be 2x delta1
  (assert-true "SGD: step magnitude proportional to lr"
    (close? (* 2.0 delta1) delta2 1e-5)))

;; 6. Adam: loss decreases on quadratic
(let [init-params (mx/array [5.0])
      result (learn/train
               {:iterations 20 :optimizer :adam :lr 0.1}
               (fn [params _key]
                 (let [loss (mx/sum (mx/square params))
                       grad (mx/multiply (mx/scalar 2.0) params)]
                   {:loss loss :grad grad}))
               init-params)
      history (:loss-history result)]
  (assert-true "Adam: loss decreases on quadratic (20 steps)"
    (< (last history) (first history))))

;; ---------------------------------------------------------------------------
;; ChoiceMap (1 test, from choicemap_property_test.cljs)
;; ---------------------------------------------------------------------------

(println "\n-- choicemap --")

;; 7. EMPTY has no addresses
(assert-true "EMPTY has no addresses"
  (empty? (cm/addresses cm/EMPTY)))

;; ---------------------------------------------------------------------------
;; Serialization (1 test, from serialize_property_test.cljs)
;; ---------------------------------------------------------------------------

(println "\n-- serialization --")

;; 8. EMPTY round-trips to empty
(let [fake-trace (tr/make-trace {:gen-fn nil :args []
                                  :choices cm/EMPTY
                                  :retval nil
                                  :score (mx/scalar 0.0)})
      json-str (ser/save-choices fake-trace)
      cm-restored (ser/load-choices json-str)]
  (assert-true "EMPTY round-trips to empty"
    (empty? (cm/addresses cm-restored))))

;; ---------------------------------------------------------------------------
;; MH Accept (2 tests, from inference_property_test.cljs)
;; ---------------------------------------------------------------------------

(println "\n-- MH accept --")

;; 9. accept-mh?(0) always true
;; log-alpha = 0 means acceptance ratio = 1, deterministically accept
(assert-true "accept-mh?(0) always true"
  (every? true? (repeatedly 20 #(u/accept-mh? 0))))

;; 10. accept-mh?(-100) rarely true
;; log-alpha = -100 means acceptance ratio ~ exp(-100) ~ 0
(let [accepts (count (filter true? (repeatedly 100 #(u/accept-mh? -100))))]
  (assert-true "accept-mh?(-100) rarely true"
    (< accepts 5)))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== Unit Tests Complete: " @pass-count " passed, " @fail-count " failed ==="))
(when (pos? @fail-count)
  (js/process.exit 1))
