(ns genmlx.recurse-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tol]
  (if (<= (js/Math.abs (- expected actual)) tol)
    (println "  PASS:" msg)
    (do (println "  FAIL:" msg)
        (println "    expected:" expected "±" tol)
        (println "    actual:  " actual))))

(println "\n=== Recurse Combinator Tests ===\n")

;; --- Countdown model (linear recursion) ---
(println "-- Countdown (simulate) --")
(let [countdown
      (comb/recurse (fn [self]
        (gen [depth]
          (let [v (dyn/trace :v (dist/gaussian 0 1))]
            (mx/eval! v)
            (if (> depth 0)
              {:v (mx/item v)
               :child (dyn/splice :child self (dec depth))}
              {:v (mx/item v)})))))
      trace (p/simulate countdown [2])]
  (assert-true "returns trace" (instance? tr/Trace trace))
  (assert-true "has :v at root" (cm/has-value? (cm/get-submap (:choices trace) :v)))
  (assert-true "has :child sub" (not= cm/EMPTY (cm/get-submap (:choices trace) :child)))
  (assert-true "has nested :child" (not= cm/EMPTY
    (cm/get-submap (cm/get-submap (:choices trace) :child) :child)))
  (let [score (mx/item (:score trace))]
    (assert-true "score is finite" (js/isFinite score))
    (assert-true "score is negative (3 gaussians)" (< score 0))))

;; --- Binary tree ---
(println "\n-- Binary tree (simulate) --")
(let [tree
      (comb/recurse (fn [self]
        (gen [depth]
          (let [v (dyn/trace :v (dist/gaussian 0 1))]
            (mx/eval! v)
            (if (> depth 0)
              {:v (mx/item v)
               :left (dyn/splice :left self (dec depth))
               :right (dyn/splice :right self (dec depth))}
              {:v (mx/item v)})))))
      trace (p/simulate tree [1])]
  (assert-true "tree trace exists" (instance? tr/Trace trace))
  (assert-true "root :v exists" (cm/has-value? (cm/get-submap (:choices trace) :v)))
  (assert-true ":left sub exists" (not= cm/EMPTY (cm/get-submap (:choices trace) :left)))
  (assert-true ":right sub exists" (not= cm/EMPTY (cm/get-submap (:choices trace) :right)))
  (let [score (mx/item (:score trace))]
    (assert-true "tree score finite" (js/isFinite score))))

;; --- Generate with constraints ---
(println "\n-- Generate with constraints --")
(let [countdown
      (comb/recurse (fn [self]
        (gen [depth]
          (let [v (dyn/trace :v (dist/gaussian 0 1))]
            (mx/eval! v)
            (if (> depth 0)
              {:v (mx/item v)
               :child (dyn/splice :child self (dec depth))}
              {:v (mx/item v)})))))
      obs (-> cm/EMPTY
              (cm/set-choice [:v] (mx/scalar 0.5))
              (cm/set-choice [:child] (cm/choicemap :v (mx/scalar -0.3))))
      {:keys [trace weight]} (p/generate countdown [1] obs)]
  (mx/eval! weight)
  (assert-true "generate returns trace" (instance? tr/Trace trace))
  (assert-true "weight is finite" (js/isFinite (mx/item weight)))
  (assert-close "root :v constrained" 0.5
    (mx/item (cm/get-choice (:choices trace) [:v])) 1e-5)
  (assert-close "child :v constrained" -0.3
    (mx/item (cm/get-choice (:choices trace) [:child :v])) 1e-5))

;; --- Update ---
(println "\n-- Update --")
(let [countdown
      (comb/recurse (fn [self]
        (gen [depth]
          (let [v (dyn/trace :v (dist/gaussian 0 1))]
            (mx/eval! v)
            (if (> depth 0)
              {:v (mx/item v)
               :child (dyn/splice :child self (dec depth))}
              {:v (mx/item v)})))))
      obs (cm/choicemap :v (mx/scalar 1.0)
                         :child (cm/choicemap :v (mx/scalar 2.0)))
      {:keys [trace]} (p/generate countdown [1] obs)
      new-constraints (cm/choicemap :v (mx/scalar 0.0))
      result (p/update countdown trace new-constraints)
      new-trace (:trace result)]
  (mx/eval! (:weight result))
  (assert-true "update returns trace" (instance? tr/Trace new-trace))
  (assert-close "updated root :v" 0.0
    (mx/item (cm/get-choice (:choices new-trace) [:v])) 1e-5)
  (assert-close "child :v unchanged" 2.0
    (mx/item (cm/get-choice (:choices new-trace) [:child :v])) 1e-5)
  (assert-true "update weight finite" (js/isFinite (mx/item (:weight result)))))

;; --- Regenerate ---
(println "\n-- Regenerate --")
(let [countdown
      (comb/recurse (fn [self]
        (gen [depth]
          (let [v (dyn/trace :v (dist/gaussian 0 1))]
            (mx/eval! v)
            (if (> depth 0)
              {:v (mx/item v)
               :child (dyn/splice :child self (dec depth))}
              {:v (mx/item v)})))))
      obs (cm/choicemap :v (mx/scalar 1.0)
                         :child (cm/choicemap :v (mx/scalar 2.0)))
      {:keys [trace]} (p/generate countdown [1] obs)
      ;; Use hierarchical selection to only resample root :v, not child's
      selection (sel/hierarchical :v sel/all)
      result (p/regenerate countdown trace selection)
      new-trace (:trace result)]
  (mx/eval! (:weight result))
  (assert-true "regenerate returns trace" (instance? tr/Trace new-trace))
  (assert-true "regenerate weight finite" (js/isFinite (mx/item (:weight result))))
  (assert-close "child :v unchanged after regen" 2.0
    (mx/item (cm/get-choice (:choices new-trace) [:child :v])) 1e-5))

;; --- Random-depth recursion (geometric stopping) ---
(println "\n-- Random-depth recursion --")
(let [geo-list
      (comb/recurse (fn [self]
        (gen [p]
          (let [v (dyn/trace :v (dist/gaussian 0 1))
                cont (dyn/trace :cont (dist/bernoulli p))]
            (mx/eval! v)
            (mx/eval! cont)
            (if (> (mx/item cont) 0.5)
              {:v (mx/item v)
               :next (dyn/splice :next self p)}
              {:v (mx/item v)})))))
      trace (p/simulate geo-list [0.3])]
  (assert-true "geo-list trace exists" (instance? tr/Trace trace))
  (assert-true "geo-list score finite" (js/isFinite (mx/item (:score trace)))))

(println "\nAll recurse combinator tests complete.")
