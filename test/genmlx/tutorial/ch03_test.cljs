(ns genmlx.tutorial.ch03-test
  "Test file for Tutorial Chapter 3: How It Works — The Handler Loop.
   Tests exercise the handler transitions directly, the runtime,
   and the gen macro internals."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.selection :as sel]
            [genmlx.protocols :as p]
            [genmlx.handler :as h]
            [genmlx.runtime :as rt]
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
;; Listing 3.1: simulate-transition is a pure function
;; ============================================================
(println "\n== Listing 3.1: simulate-transition ==")

(let [key (rng/fresh-key)
      init-state {:key key :choices cm/EMPTY :score (mx/scalar 0.0)}
      d (dist/gaussian 0 1)
      [value state'] (h/simulate-transition init-state :x d)]
  ;; Returns a value and a new state
  (assert-true "returns MLX value" (mx/array? value))
  (assert-true "state' has :choices with :x" (cm/has-value? (cm/get-submap (:choices state') :x)))
  (assert-true "state' has updated :score" (not= 0.0 (mx/item (:score state'))))
  (assert-true "state' has advanced :key" (not= key (:key state')))
  ;; The original state is unchanged (pure!)
  (assert-true "original state unchanged: choices still empty"
               (= cm/EMPTY (:choices init-state)))
  (assert-close "original state unchanged: score still 0"
                0.0 (mx/item (:score init-state)) 0.001))

;; ============================================================
;; Listing 3.2: generate-transition — constrained case
;; ============================================================
(println "\n== Listing 3.2: generate-transition (constrained) ==")

(let [key (rng/fresh-key)
      constraints (cm/choicemap :x (mx/scalar 3.0))
      init-state {:key key :choices cm/EMPTY :score (mx/scalar 0.0)
                  :weight (mx/scalar 0.0) :constraints constraints}
      d (dist/gaussian 0 1)
      [value state'] (h/generate-transition init-state :x d)]
  ;; Uses the constrained value
  (assert-close "value is the constraint (3.0)" 3.0 (mx/item value) 0.001)
  ;; Weight gets the log-prob
  (assert-true "weight is nonzero" (not= 0.0 (mx/item (:weight state'))))
  ;; Score also gets the log-prob
  (assert-close "score equals weight" (mx/item (:score state')) (mx/item (:weight state')) 0.001))

;; ============================================================
;; Listing 3.3: generate-transition — unconstrained case
;; ============================================================
(println "\n== Listing 3.3: generate-transition (unconstrained) ==")

(let [key (rng/fresh-key)
      init-state {:key key :choices cm/EMPTY :score (mx/scalar 0.0)
                  :weight (mx/scalar 0.0) :constraints cm/EMPTY}
      d (dist/gaussian 0 1)
      [value state'] (h/generate-transition init-state :x d)]
  ;; Falls through to simulate — samples a value
  (assert-true "value is sampled (MLX array)" (mx/array? value))
  ;; Weight stays at zero (no constraint was applied)
  (assert-close "weight stays zero" 0.0 (mx/item (:weight state')) 0.001)
  ;; Score gets the log-prob of the sampled value
  (assert-true "score is nonzero" (not= 0.0 (mx/item (:score state')))))

;; ============================================================
;; Listing 3.4: Handler state shape
;; ============================================================
(println "\n== Listing 3.4: handler state ==")

(let [key (rng/fresh-key)
      ;; Simulate state
      sim-state {:key key :choices cm/EMPTY :score (mx/scalar 0.0)}
      ;; Generate state (adds :weight and :constraints)
      gen-state {:key key :choices cm/EMPTY :score (mx/scalar 0.0)
                 :weight (mx/scalar 0.0)
                 :constraints (cm/choicemap :y0 (mx/scalar 5.0))}]
  (assert-true "simulate state has :key" (contains? sim-state :key))
  (assert-true "simulate state has :choices" (contains? sim-state :choices))
  (assert-true "simulate state has :score" (contains? sim-state :score))
  (assert-true "generate state has :weight" (contains? gen-state :weight))
  (assert-true "generate state has :constraints" (contains? gen-state :constraints)))

;; ============================================================
;; Listing 3.5: Chaining transitions (two trace sites)
;; ============================================================
(println "\n== Listing 3.5: chaining transitions ==")

(let [key (rng/fresh-key)
      init {:key key :choices cm/EMPTY :score (mx/scalar 0.0)}
      d1 (dist/gaussian 0 10)
      d2 (dist/gaussian 0 1)
      ;; First trace site
      [slope state1] (h/simulate-transition init :slope d1)
      ;; Second trace site — uses state1, not init
      [noise state2] (h/simulate-transition state1 :noise d2)]
  (assert-true "slope is MLX array" (mx/array? slope))
  (assert-true "noise is MLX array" (mx/array? noise))
  (assert-true "state2 has both :slope and :noise"
               (and (cm/has-value? (cm/get-submap (:choices state2) :slope))
                    (cm/has-value? (cm/get-submap (:choices state2) :noise))))
  ;; Score accumulated from both
  (let [lp1 (mx/item (dc/dist-log-prob d1 slope))
        lp2 (mx/item (dc/dist-log-prob d2 noise))
        total (mx/item (:score state2))]
    (assert-close "score = sum of log-probs" (+ lp1 lp2) total 0.001)))

;; ============================================================
;; Listing 3.6: run-handler threads state through volatile!
;; ============================================================
(println "\n== Listing 3.6: run-handler ==")

(let [key (rng/fresh-key)
      init-state {:key key :choices cm/EMPTY :score (mx/scalar 0.0)
                  :executor (fn [_ _ _] nil)}
      result (rt/run-handler
               h/simulate-transition
               init-state
               (fn [rt]
                 ;; Inside the body, trace is a closure
                 (let [trace-fn (.-trace rt)
                       x (trace-fn :x (dist/gaussian 0 1))
                       y (trace-fn :y (dist/gaussian 0 1))]
                   (mx/add x y))))]
  (assert-true "run-handler returns a map" (map? result))
  (assert-true "result has :choices" (some? (:choices result)))
  (assert-true "result has :retval" (some? (:retval result)))
  (assert-true "result has :score" (some? (:score result)))
  (assert-true "choices has :x" (cm/has-value? (cm/get-submap (:choices result) :x)))
  (assert-true "choices has :y" (cm/has-value? (cm/get-submap (:choices result) :y)))
  (assert-true "retval is x + y" (mx/array? (:retval result)))
  (assert-true "score is finite" (js/Number.isFinite (mx/item (:score result)))))

;; ============================================================
;; Listing 3.7: The gen macro produces a DynamicGF
;; ============================================================
(println "\n== Listing 3.7: gen macro ==")

(def my-model
  (gen [a b]
    (let [x (trace :x (dist/gaussian a b))]
      (mx/multiply x x))))

(assert-true "gen produces an object with :body-fn" (some? (:body-fn my-model)))
(assert-true "gen produces an object with :source" (some? (:source my-model)))

(let [model (dyn/auto-key my-model)
      trace (p/simulate model [0 1])]
  (assert-true "model simulates with args" (some? trace))
  (assert-true "retval is x^2" (>= (mx/item (:retval trace)) 0)))

;; ============================================================
;; Listing 3.8: trace/splice/param are local bindings
;; ============================================================
(println "\n== Listing 3.8: trace works with HOFs ==")

;; trace works inside map, for, closures — it's a local binding
(def hof-model
  (gen [n]
    (let [values (mapv (fn [i]
                         (trace (keyword (str "x" i)) (dist/gaussian 0 1)))
                       (range n))]
      values)))

(let [model (dyn/auto-key hof-model)
      trace (p/simulate model [4])]
  (assert-true "hof-model simulates" (some? trace))
  (assert-true "has :x0" (cm/has-value? (cm/get-submap (:choices trace) :x0)))
  (assert-true "has :x1" (cm/has-value? (cm/get-submap (:choices trace) :x1)))
  (assert-true "has :x2" (cm/has-value? (cm/get-submap (:choices trace) :x2)))
  (assert-true "has :x3" (cm/has-value? (cm/get-submap (:choices trace) :x3)))
  (assert-true "retval is vector of 4" (= 4 (count (:retval trace)))))

;; ============================================================
;; Listing 3.9: Purity enables composition
;; Same transition works with different initial states
;; ============================================================
(println "\n== Listing 3.9: same transition, different states ==")

(let [key (rng/fresh-key)
      d (dist/gaussian 0 1)
      ;; Simulate state
      sim-state {:key key :choices cm/EMPTY :score (mx/scalar 0.0)}
      [sim-val sim-state'] (h/simulate-transition sim-state :x d)
      ;; Generate state with constraint
      gen-state {:key key :choices cm/EMPTY :score (mx/scalar 0.0)
                 :weight (mx/scalar 0.0)
                 :constraints (cm/choicemap :x (mx/scalar 2.0))}
      [gen-val gen-state'] (h/generate-transition gen-state :x d)]
  ;; Simulate samples randomly
  (assert-true "simulate samples a value" (mx/array? sim-val))
  ;; Generate uses the constraint
  (assert-close "generate uses constraint" 2.0 (mx/item gen-val) 0.001)
  ;; Both produce valid states
  (assert-true "both have :x in choices"
               (and (cm/has-value? (cm/get-submap (:choices sim-state') :x))
                    (cm/has-value? (cm/get-submap (:choices gen-state') :x)))))

;; ============================================================
;; Summary
;; ============================================================
(println (str "\n== Chapter 3 tests: " @pass " PASS, " @fail " FAIL =="))
(when (pos? @fail) (js/process.exit 1))
