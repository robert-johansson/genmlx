(ns genmlx.tutorial.ch09-test
  "Test file for Tutorial Chapter 9: Gradients, Learning, and Neural Models."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.gradients :as grad]
            [genmlx.learning :as learn])
  (:require-macros [genmlx.gen :refer [gen]]))

(def pass (atom 0))
(def fail (atom 0))

(defn assert-true [msg pred]
  (if pred
    (do (swap! pass inc) (println "  PASS:" msg))
    (do (swap! fail inc) (println "  FAIL:" msg))))

;; Simple model for gradient tests
(def grad-model
  (gen [x]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :obs (dist/gaussian mu 1))
      mu)))

;; ============================================================
;; Listing 9.1: Choice gradients
;; ============================================================
(println "\n== Listing 9.1: choice gradients ==")

(let [model (dyn/auto-key grad-model)
      trace (p/simulate model [0])
      grads (grad/choice-gradients model trace [:mu])]
  (assert-true "returns a map" (map? grads))
  (assert-true "has :mu gradient" (contains? grads :mu))
  (assert-true "gradient is MLX array" (mx/array? (:mu grads)))
  (assert-true "gradient is finite" (js/Number.isFinite (mx/item (:mu grads)))))

;; ============================================================
;; Listing 9.2: Parameter stores
;; ============================================================
(println "\n== Listing 9.2: parameter stores ==")

(let [ps (learn/make-param-store {:theta (mx/scalar 1.0) :sigma (mx/scalar 0.5)})]
  (assert-true "param store is a map" (map? ps))
  (assert-true "has :params" (contains? ps :params))
  (assert-true "has :theta" (some? (get-in ps [:params :theta])))
  (let [v (mx/item (get-in ps [:params :theta]))]
    (assert-true "theta = 1.0" (< (js/Math.abs (- v 1.0)) 0.001))))

;; ============================================================
;; Listing 9.3: param in models
;; ============================================================
(println "\n== Listing 9.3: param in models ==")

(def parameterized-model
  (gen []
    (let [mu (param :mu 0.0)]
      (trace :x (dist/gaussian mu 1))
      mu)))

(let [model (dyn/auto-key parameterized-model)
      ps (learn/make-param-store {:mu (mx/scalar 5.0)})
      trace (learn/simulate-with-params model [] ps)]
  (assert-true "simulate-with-params works" (some? trace))
  (assert-true "has :x" (cm/has-value? (cm/get-submap (:choices trace) :x))))

;; ============================================================
;; Listing 9.4: SGD and Adam
;; ============================================================
(println "\n== Listing 9.4: optimizers ==")

;; SGD and Adam operate on flat MLX arrays
(let [params (mx/scalar 5.0)
      grads (mx/scalar 1.0)
      updated (learn/sgd-step params grads 0.1)]
  (assert-true "SGD returns MLX array" (mx/array? updated))
  ;; theta should decrease: 5.0 - 0.1 * 1.0 = 4.9
  (assert-true "SGD: value decreased" (< (mx/item updated) 5.0)))

(let [params (mx/array [5.0 3.0])
      state (learn/adam-init params)
      grads (mx/array [1.0 0.5])
      [new-params new-state] (learn/adam-step params grads state {:lr 0.01})]
  (assert-true "Adam returns new params" (mx/array? new-params))
  (assert-true "Adam returns new state" (map? new-state))
  (assert-true "Adam state has :t" (contains? new-state :t))
  (assert-true "Adam state :t = 1" (= 1 (:t new-state))))

;; ============================================================
;; Listing 9.5: Training loop
;; ============================================================
(println "\n== Listing 9.5: training loop ==")

;; Simple loss: (theta - 3)^2, flat MLX array
(let [loss-grad-fn (fn [params _key]
                     (let [theta (mx/item params)
                           loss (mx/scalar (* (- theta 3.0) (- theta 3.0)))
                           grad-val (mx/scalar (* 2.0 (- theta 3.0)))]
                       {:loss loss :grad grad-val}))
      result (learn/train {:iterations 50 :optimizer :sgd :lr 0.1}
                           loss-grad-fn (mx/scalar 0.0))]
  (assert-true "train returns result" (map? result))
  (assert-true "result has :params" (contains? result :params))
  (assert-true "result has :loss-history" (contains? result :loss-history))
  (let [final-theta (mx/item (:params result))]
    (assert-true "theta converged near 3" (< (js/Math.abs (- final-theta 3.0)) 1.0))))

;; ============================================================
;; Summary
;; ============================================================
(println (str "\n== Chapter 9 tests: " @pass " PASS, " @fail " FAIL =="))
(when (pos? @fail) (js/process.exit 1))
