(ns genmlx.trace-immutability-property-test
  "Property-based tests verifying trace immutability through update and regenerate.
   After p/update or p/regenerate, the ORIGINAL trace must be completely unmodified.
   This catches bugs where update/regenerate mutate the input trace."
  (:require [cljs.test :as t]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]
                   [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- eval-item [x]
  (mx/eval! x)
  (mx/item x))

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

(defn- close? [a b tol]
  (and (finite? a) (finite? b) (<= (js/Math.abs (- a b)) tol)))

(defn- choice-val
  "Extract a JS number from a choicemap at addr."
  [choices addr]
  (let [sub (cm/get-submap choices addr)]
    (when (and sub (cm/has-value? sub))
      (let [v (cm/get-value sub)]
        (mx/eval! v)
        (mx/item v)))))

(defn- snapshot-choices
  "Deep-snapshot all choice values at given addrs, returning {addr -> JS number}."
  [choices addrs]
  (into {} (map (fn [a] [a (choice-val choices a)]) addrs)))

(defn- snapshot-trace
  "Capture score, retval, and choice values as JS numbers."
  [trace addrs]
  {:score (eval-item (:score trace))
   :choices (snapshot-choices (:choices trace) addrs)
   :args (:args trace)
   :gen-fn (:gen-fn trace)})

(defn- snapshots-equal?
  "True if two trace snapshots are identical (choices within tolerance)."
  [snap1 snap2 tol]
  (and (close? (:score snap1) (:score snap2) tol)
       (= (:args snap1) (:args snap2))
       (identical? (:gen-fn snap1) (:gen-fn snap2))
       (every? (fn [[addr v1]]
                 (let [v2 (get (:choices snap2) addr)]
                   (close? v1 v2 tol)))
               (:choices snap1))))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def simple-2site
  {:model (dyn/auto-key (gen []
            (let [x (trace :x (dist/gaussian 0 1))
                  y (trace :y (dist/gaussian 0 1))]
              (mx/eval! x y)
              (+ (mx/item x) (mx/item y)))))
   :args []
   :addrs [:x :y]
   :label "simple-2site"})

(def dependent-sites
  {:model (dyn/auto-key (gen []
            (let [x (trace :x (dist/gaussian 0 2))]
              (mx/eval! x)
              (let [y (trace :y (dist/gaussian (mx/item x) 1))]
                (mx/eval! y)
                (mx/item y)))))
   :args []
   :addrs [:x :y]
   :label "dependent-sites"})

(def three-site
  {:model (dyn/auto-key (gen []
            (let [a (trace :a (dist/gaussian 0 1))
                  b (trace :b (dist/gaussian 0 1))
                  c (trace :c (dist/gaussian 0 1))]
              (mx/eval! a b c)
              (+ (mx/item a) (mx/item b) (mx/item c)))))
   :args []
   :addrs [:a :b :c]
   :label "three-site"})

(def mixed-dists
  {:model (dyn/auto-key (gen []
            (let [x (trace :x (dist/gaussian 0 1))
                  y (trace :y (dist/exponential 1))]
              (mx/eval! x y)
              (+ (mx/item x) (mx/item y)))))
   :args []
   :addrs [:x :y]
   :label "mixed-dists"})

(def model-pool [simple-2site dependent-sites three-site mixed-dists])
(def gen-model (gen/elements model-pool))

;; ---------------------------------------------------------------------------
;; Property 1: update does not mutate original trace
;; ---------------------------------------------------------------------------

(defspec update-preserves-original-trace 50
  (prop/for-all [m gen-model]
    (let [model (:model m)
          addrs (:addrs m)
          orig-trace (p/simulate model (:args m))
          snap-before (snapshot-trace orig-trace addrs)
          ;; Update with new constraints on first address
          first-addr (first addrs)
          constraint (cm/choicemap first-addr (mx/scalar 42.0))
          _result (p/update model orig-trace constraint)
          snap-after (snapshot-trace orig-trace addrs)]
      (snapshots-equal? snap-before snap-after 1e-10))))

;; ---------------------------------------------------------------------------
;; Property 2: regenerate does not mutate original trace
;; ---------------------------------------------------------------------------

(defspec regenerate-preserves-original-trace 50
  (prop/for-all [m gen-model]
    (let [model (:model m)
          addrs (:addrs m)
          orig-trace (p/simulate model (:args m))
          snap-before (snapshot-trace orig-trace addrs)
          ;; Regenerate first address
          first-addr (first addrs)
          _result (p/regenerate model orig-trace (sel/select first-addr))
          snap-after (snapshot-trace orig-trace addrs)]
      (snapshots-equal? snap-before snap-after 1e-10))))

;; ---------------------------------------------------------------------------
;; Property 3: multiple updates do not mutate original trace
;; ---------------------------------------------------------------------------

(defspec multiple-updates-preserve-original 30
  (prop/for-all [m gen-model]
    (let [model (:model m)
          addrs (:addrs m)
          orig-trace (p/simulate model (:args m))
          snap-before (snapshot-trace orig-trace addrs)
          ;; Do 3 sequential updates with different constraints
          first-addr (first addrs)
          _ (p/update model orig-trace (cm/choicemap first-addr (mx/scalar 1.0)))
          _ (p/update model orig-trace (cm/choicemap first-addr (mx/scalar -5.0)))
          _ (p/update model orig-trace (cm/choicemap first-addr (mx/scalar 100.0)))
          snap-after (snapshot-trace orig-trace addrs)]
      (snapshots-equal? snap-before snap-after 1e-10))))

;; ---------------------------------------------------------------------------
;; Property 4: regenerate with all-selection does not mutate original
;; ---------------------------------------------------------------------------

(defspec regenerate-all-preserves-original 50
  (prop/for-all [m gen-model]
    (let [model (:model m)
          addrs (:addrs m)
          orig-trace (p/simulate model (:args m))
          snap-before (snapshot-trace orig-trace addrs)
          _result (p/regenerate model orig-trace sel/all)
          snap-after (snapshot-trace orig-trace addrs)]
      (snapshots-equal? snap-before snap-after 1e-10))))

;; ---------------------------------------------------------------------------
;; Property 5: regenerate with none-selection does not mutate original
;; ---------------------------------------------------------------------------

(defspec regenerate-none-preserves-original 50
  (prop/for-all [m gen-model]
    (let [model (:model m)
          addrs (:addrs m)
          orig-trace (p/simulate model (:args m))
          snap-before (snapshot-trace orig-trace addrs)
          _result (p/regenerate model orig-trace sel/none)
          snap-after (snapshot-trace orig-trace addrs)]
      (snapshots-equal? snap-before snap-after 1e-10))))

;; ---------------------------------------------------------------------------
;; Property 6: update on map combinator does not mutate original
;; ---------------------------------------------------------------------------

(def map-kernel
  (dyn/auto-key (gen [x]
    (let [y (trace :y (dist/gaussian x 1))]
      (mx/eval! y)
      (mx/item y)))))

(def map-model (comb/map-combinator map-kernel))

(defn- map-choice-val [choices idx addr]
  (let [sub (cm/get-submap choices idx)]
    (when sub
      (let [inner (cm/get-submap sub addr)]
        (when (and inner (cm/has-value? inner))
          (let [v (cm/get-value inner)]
            (mx/eval! v)
            (mx/item v)))))))

(defn- snapshot-map-choices [choices n]
  (into {} (for [i (range n)] [i (map-choice-val choices i :y)])))

(defspec map-combinator-update-preserves-original 30
  (prop/for-all [_dummy (gen/return nil)]
    (let [trace (p/simulate map-model [[1.0 2.0 3.0]])
          snap-score (eval-item (:score trace))
          snap-choices (snapshot-map-choices (:choices trace) 3)
          ;; Update element 0
          constraint (cm/set-choice cm/EMPTY [0] (cm/choicemap :y (mx/scalar 99.0)))
          _result (p/update map-model trace constraint)
          after-score (eval-item (:score trace))
          after-choices (snapshot-map-choices (:choices trace) 3)]
      (and (close? snap-score after-score 1e-10)
           (every? (fn [[i v]] (close? v (get after-choices i) 1e-10))
                   snap-choices)))))

;; ---------------------------------------------------------------------------
;; Property 7: update on unfold combinator does not mutate original
;; ---------------------------------------------------------------------------

(def unfold-step
  (dyn/auto-key (gen [t state]
    (let [x (trace :x (dist/gaussian state 0.1))]
      (mx/eval! x)
      (mx/item x)))))

(def unfold-model (comb/unfold-combinator unfold-step))

(defn- unfold-choice-val [choices t addr]
  (let [sub (cm/get-submap choices t)]
    (when sub
      (let [inner (cm/get-submap sub addr)]
        (when (and inner (cm/has-value? inner))
          (let [v (cm/get-value inner)]
            (mx/eval! v)
            (mx/item v)))))))

(defn- snapshot-unfold-choices [choices n]
  (into {} (for [t (range n)] [t (unfold-choice-val choices t :x)])))

(defspec unfold-combinator-update-preserves-original 30
  (prop/for-all [_dummy (gen/return nil)]
    (let [trace (p/simulate unfold-model [3 0.0])
          snap-score (eval-item (:score trace))
          snap-choices (snapshot-unfold-choices (:choices trace) 3)
          ;; Update step 1
          constraint (cm/set-choice cm/EMPTY [1] (cm/choicemap :x (mx/scalar 99.0)))
          _result (p/update unfold-model trace constraint)
          after-score (eval-item (:score trace))
          after-choices (snapshot-unfold-choices (:choices trace) 3)]
      (and (close? snap-score after-score 1e-10)
           (every? (fn [[t v]] (close? v (get after-choices t) 1e-10))
                   snap-choices)))))

;; ---------------------------------------------------------------------------
;; Property 8: update result is a valid trace (new trace, not mutated original)
;; ---------------------------------------------------------------------------

(defspec update-result-is-distinct-trace 50
  (prop/for-all [m gen-model]
    (let [model (:model m)
          addrs (:addrs m)
          orig-trace (p/simulate model (:args m))
          first-addr (first addrs)
          constraint (cm/choicemap first-addr (mx/scalar 42.0))
          {:keys [trace]} (p/update model orig-trace constraint)
          ;; The new trace should have the constrained value
          new-val (choice-val (:choices trace) first-addr)]
      (close? new-val 42.0 1e-6))))

;; ---------------------------------------------------------------------------
;; Property 9: regenerate result changes selected, preserves unselected
;; ---------------------------------------------------------------------------

(defspec regenerate-result-preserves-unselected 50
  (prop/for-all [m (gen/such-that #(> (count (:addrs %)) 1) gen-model)]
    (let [model (:model m)
          addrs (:addrs m)
          orig-trace (p/simulate model (:args m))
          selected-addr (first addrs)
          unselected (rest addrs)
          orig-unsel (into {} (map (fn [a] [a (choice-val (:choices orig-trace) a)])
                                   unselected))
          {:keys [trace]} (p/regenerate model orig-trace (sel/select selected-addr))
          new-unsel (into {} (map (fn [a] [a (choice-val (:choices trace) a)])
                                  unselected))]
      (every? (fn [a]
                (close? (get orig-unsel a) (get new-unsel a) 1e-6))
              unselected))))

(t/run-tests)
