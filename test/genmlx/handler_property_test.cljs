(ns genmlx.handler-property-test
  "Property-based tests for handler.cljs state transitions using test.check.
   Verifies scalar and batched handler transitions are pure and correct."
  (:require [cljs.test :as t]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.handler :as h])
  (:require-macros [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- eval-weight [w]
  (mx/eval! w)
  (mx/item w))

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

(defn- close? [a b tol]
  (and (finite? a) (finite? b) (<= (js/Math.abs (- a b)) tol)))

;; ---------------------------------------------------------------------------
;; Generators
;; ---------------------------------------------------------------------------

(def addr-pool [:a :b :c :x :y])
(def gen-addr (gen/elements addr-pool))

(def dist-pool
  [(dist/gaussian 0 1)
   (dist/gaussian 3 0.5)
   (dist/uniform 0 1)])
(def gen-dist (gen/elements dist-pool))

(def key-pool (mapv #(rng/fresh-key %) [42 99 123 7 255]))
(def gen-key (gen/elements key-pool))

(def gen-batch-size (gen/elements [5 10 20]))

(defn- make-simulate-state [key]
  {:key key
   :choices cm/EMPTY
   :score (mx/scalar 0.0)
   :executor nil})

(defn- make-generate-state [key constraints]
  {:key key
   :choices cm/EMPTY
   :score (mx/scalar 0.0)
   :weight (mx/scalar 0.0)
   :constraints constraints
   :executor nil})

(defn- make-update-state [key old-choices constraints]
  {:key key
   :choices cm/EMPTY
   :score (mx/scalar 0.0)
   :weight (mx/scalar 0.0)
   :constraints constraints
   :old-choices old-choices
   :discard cm/EMPTY
   :executor nil})

;; ---------------------------------------------------------------------------
;; Properties
;; ---------------------------------------------------------------------------

(defspec simulate-transition-score-equals-log-prob 50
  (prop/for-all [k gen-key
                 addr gen-addr
                 d gen-dist]
    (let [state (make-simulate-state k)
          [value state'] (h/simulate-transition state addr d)
          _ (mx/eval! value (:score state'))
          score (mx/item (:score state'))
          lp (mx/item (dc/dist-log-prob d value))]
      (close? score lp 0.01))))

(defspec simulate-transition-key-changes 50
  (prop/for-all [k gen-key
                 addr gen-addr
                 d gen-dist]
    (let [state (make-simulate-state k)
          [_ state'] (h/simulate-transition state addr d)]
      (not (identical? (:key state) (:key state'))))))

(defspec generate-transition-constrained-value-matches 50
  (prop/for-all [k gen-key]
    (let [constraint-val (mx/scalar 2.5)
          constraints (cm/choicemap :x constraint-val)
          state (make-generate-state k constraints)
          d (dist/gaussian 0 1)
          [value state'] (h/generate-transition state :x d)]
      (mx/eval! value)
      (close? (mx/item value) 2.5 1e-6))))

(defspec generate-transition-unconstrained-accumulates-score 50
  (prop/for-all [k gen-key]
    (let [state (make-generate-state k cm/EMPTY)
          d (dist/gaussian 0 1)
          [_ state'] (h/generate-transition state :x d)
          score (eval-weight (:score state'))
          weight (eval-weight (:weight state'))]
      (and (finite? score)
           (close? 0.0 weight 0.01)))))

(defspec batched-simulate-score-shape-equals-N 50
  (prop/for-all [k gen-key
                 n gen-batch-size]
    (let [state {:key k :choices cm/EMPTY :score (mx/zeros [n])
                 :executor nil :batch-size n :batched? true}
          d (dist/gaussian 0 1)
          [value state'] (h/batched-simulate-transition state :x d)]
      (mx/eval! (:score state'))
      (= (mx/shape (:score state')) [n]))))

(defspec batched-generate-constrained-value-is-scalar 50
  (prop/for-all [k gen-key
                 n gen-batch-size]
    (let [constraint-val (mx/scalar 2.5)
          constraints (cm/choicemap :x constraint-val)
          state {:key k :choices cm/EMPTY :score (mx/zeros [n])
                 :weight (mx/zeros [n]) :constraints constraints
                 :executor nil :batch-size n :batched? true}
          d (dist/gaussian 0 1)
          [value _] (h/batched-generate-transition state :x d)]
      (mx/eval! value)
      (close? (mx/item value) 2.5 1e-6))))

(defspec update-transition-discard-contains-old-value 50
  (prop/for-all [k gen-key]
    (let [old-val (mx/scalar 1.0)
          new-val (mx/scalar 3.0)
          old-choices (cm/choicemap :x old-val)
          constraints (cm/choicemap :x new-val)
          state (make-update-state k old-choices constraints)
          d (dist/gaussian 0 1)
          [value state'] (h/update-transition state :x d)
          discard-sub (cm/get-submap (:discard state') :x)]
      (and
        (do (mx/eval! value) (close? (mx/item value) 3.0 1e-6))
        (when (cm/has-value? discard-sub)
          (let [dv (cm/get-value discard-sub)]
            (mx/eval! dv)
            (close? (mx/item dv) 1.0 1e-6)))))))

(defspec merge-sub-result-preserves-parent-key-and-fields 50
  (prop/for-all [k gen-key]
    (let [parent-state {:key k
                        :choices cm/EMPTY
                        :score (mx/scalar 1.0)
                        :weight (mx/scalar 0.5)
                        :splice-scores nil}
          sub-result {:choices (cm/choicemap :inner (mx/scalar 2.0))
                      :score (mx/scalar 0.3)
                      :weight (mx/scalar 0.1)}
          state' (h/merge-sub-result parent-state :child sub-result)]
      (and
        (identical? (:key state') k)
        (let [s (eval-weight (:score state'))]
          (close? s 1.3 0.01))
        (let [w (eval-weight (:weight state'))]
          (close? w 0.6 0.01))))))

(t/run-tests)
