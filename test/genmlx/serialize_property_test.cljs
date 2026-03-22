(ns genmlx.serialize-property-test
  "Property-based tests for serialization round-trip laws.
   Verifies: choicemap round-trip, trace round-trip, dtype preservation,
   nested structure preservation, and empty identity."
  (:require [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [cljs.test :as t]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.serialize :as ser])
  (:require-macros [genmlx.gen :refer [gen]]
                   [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- s [v] (mx/scalar (double v)))

(defn- close? [a b tol]
  (< (js/Math.abs (- a b)) tol))

(defn- choicemap-values-match?
  "Check that two choicemaps have the same addresses and values within tolerance."
  [cm1 cm2 tol]
  (let [addrs1 (set (cm/addresses cm1))
        addrs2 (set (cm/addresses cm2))]
    (and (= addrs1 addrs2)
         (every? true?
           (for [path addrs1]
             (let [v1 (cm/get-choice cm1 path)
                   v2 (cm/get-choice cm2 path)]
               (mx/eval! v1 v2)
               (close? (mx/item v1) (mx/item v2) tol)))))))

;; ---------------------------------------------------------------------------
;; Pre-built pools (SCI shrink safety)
;; ---------------------------------------------------------------------------

;; Choicemap pool with varying structure
(def flat-cm-pool
  [(cm/from-map {:a (mx/scalar 1.0) :b (mx/scalar 2.0)})
   (cm/from-map {:x (mx/scalar -3.5) :y (mx/scalar 7.2) :z (mx/scalar 0.0)})
   (cm/from-map {:alpha (mx/scalar 0.001) :beta (mx/scalar 999.9)})])

(def single-cm-pool
  [(cm/from-map {:x (mx/scalar 5.0)})
   (cm/from-map {:val (mx/scalar -1.23)})])

(def nested-cm-pool
  [(cm/set-choice (cm/set-choice cm/EMPTY [:a :b] (mx/scalar 1.0))
                  [:a :c] (mx/scalar 2.0))
   (cm/set-choice (cm/set-choice cm/EMPTY [:params :slope] (mx/scalar 3.0))
                  [:params :intercept] (mx/scalar -1.0))])

(def deep-cm-pool
  [(-> cm/EMPTY
       (cm/set-choice [:a :b :c] (mx/scalar 1.0))
       (cm/set-choice [:a :b :d] (mx/scalar 2.0))
       (cm/set-choice [:a :e] (mx/scalar 3.0)))
   (-> cm/EMPTY
       (cm/set-choice [:x :y :z] (mx/scalar -5.0))
       (cm/set-choice [:x :y :w] (mx/scalar 10.0))
       (cm/set-choice [:x :q] (mx/scalar 0.5))
       (cm/set-choice [:r] (mx/scalar 99.0)))])

;; All choicemaps combined for general round-trip tests
(def all-cm-pool
  (vec (concat flat-cm-pool single-cm-pool nested-cm-pool deep-cm-pool)))

;; Multi-dtype choicemap pool
(def dtype-cm-pool
  [(cm/from-map {:f (mx/scalar 3.14)
                 :i (mx/scalar 42 mx/int32)
                 :b (mx/array true mx/bool-dt)})
   (cm/from-map {:float-val (mx/scalar -2.5)
                 :int-val (mx/scalar 7 mx/int32)})])

;; Model pool for trace serialization
(def simple-model
  (gen [] (trace :x (dist/gaussian (s 0) (s 1)))))

(def two-site-model
  (gen []
    (let [x (trace :x (dist/gaussian (s 0) (s 1)))]
      (trace :y (dist/gaussian x (s 1))))))

(def model-pool [simple-model two-site-model])

(def key-pool
  (let [root (rng/fresh-key)]
    (vec (rng/split-n root 5))))

;; ---------------------------------------------------------------------------
;; E13.1: choicemap -> save-choices -> load-choices round-trip
;; Law: Serialization is an isomorphism -- the serialized-then-deserialized
;;       choicemap has the same addresses and values as the original.
;; ---------------------------------------------------------------------------

(defspec choicemap-round-trip-via-save-load-choices 50
  (prop/for-all [cm-orig (gen/elements all-cm-pool)]
    ;; Wrap choicemap in a minimal trace for save-choices API
    (let [fake-trace (tr/make-trace {:gen-fn nil :args []
                                     :choices cm-orig
                                     :retval nil
                                     :score (mx/scalar 0.0)})
          json-str (ser/save-choices fake-trace)
          cm-restored (ser/load-choices json-str)]
      (choicemap-values-match? cm-orig cm-restored 1e-6))))

;; ---------------------------------------------------------------------------
;; E13.2: save-choices / load-choices round-trip for model traces
;; Law: Trace serialization preserves all choice values -- a trace
;;       simulated from a model can be saved and reconstructed.
;; ---------------------------------------------------------------------------

(defspec save-choices-load-choices-preserves-trace-choices 30
  (prop/for-all [model (gen/elements model-pool)
                 key (gen/elements key-pool)]
    (let [trace (p/simulate (dyn/with-key model key) [])
          json-str (ser/save-choices trace)
          cm-restored (ser/load-choices json-str)]
      (choicemap-values-match? (:choices trace) cm-restored 1e-6))))

;; ---------------------------------------------------------------------------
;; E13.3: dtype preservation
;; Law: Serialization preserves MLX dtype metadata -- float32 stays
;;       float32, int32 stays int32, bool stays bool.
;; ---------------------------------------------------------------------------

(defspec dtype-preserved-through-serialization-round-trip 20
  (prop/for-all [cm-orig (gen/elements dtype-cm-pool)]
    (let [fake-trace (tr/make-trace {:gen-fn nil :args []
                                     :choices cm-orig
                                     :retval nil
                                     :score (mx/scalar 0.0)})
          json-str (ser/save-choices fake-trace)
          cm-restored (ser/load-choices json-str)
          addrs (cm/addresses cm-orig)]
      (every? true?
        (for [path addrs]
          (let [v-orig (cm/get-choice cm-orig path)
                v-rest (cm/get-choice cm-restored path)]
            (= (str (mx/dtype v-orig)) (str (mx/dtype v-rest)))))))))

;; ---------------------------------------------------------------------------
;; E13.4: nested structure preservation
;; Law: Serialization preserves tree structure -- the set of address
;;       paths is identical before and after, and all values match.
;; ---------------------------------------------------------------------------

(defspec nested-structure-preserved-through-serialization 30
  (prop/for-all [cm-orig (gen/elements (vec (concat nested-cm-pool deep-cm-pool)))]
    (let [fake-trace (tr/make-trace {:gen-fn nil :args []
                                     :choices cm-orig
                                     :retval nil
                                     :score (mx/scalar 0.0)})
          json-str (ser/save-choices fake-trace)
          cm-restored (ser/load-choices json-str)
          orig-addrs (set (cm/addresses cm-orig))
          rest-addrs (set (cm/addresses cm-restored))]
      (and (= orig-addrs rest-addrs)
           (choicemap-values-match? cm-orig cm-restored 1e-6)))))

;; ---------------------------------------------------------------------------
;; E13.5: EMPTY round-trip
;; Law: Serialization preserves the identity element -- the empty
;;       choicemap serializes and deserializes to an empty choicemap.
;; ---------------------------------------------------------------------------

(defspec empty-choicemap-round-trips-to-empty 10
  (prop/for-all [_ (gen/return nil)]
    (let [fake-trace (tr/make-trace {:gen-fn nil :args []
                                     :choices cm/EMPTY
                                     :retval nil
                                     :score (mx/scalar 0.0)})
          json-str (ser/save-choices fake-trace)
          cm-restored (ser/load-choices json-str)]
      (empty? (cm/addresses cm-restored)))))

(t/run-tests)
