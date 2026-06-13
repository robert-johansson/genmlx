;; @tier fast
(ns genmlx.tensor-trace-property-test
  "Property-based TensorTrace round-trip tests (genmlx-ota8) using test.check.

   Complements the deterministic examples in tensor_trace_test.cljs with
   generated value maps / address indices:
     1. pack-values then unpack-values is the identity on values.
     2. trace -> tensor-trace -> trace preserves choices and all scalar fields.

   Run: bun run --bun nbb test/genmlx/tensor_trace_property_test.cljs"
  (:require [cljs.test :as t]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.tensor-trace :as tt])
  (:require-macros [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Generators
;; ---------------------------------------------------------------------------

(def addr-pool [:a :b :c :d :e :f :g :h])

(def gen-num
  (gen/double* {:min -50.0 :max 50.0 :NaN? false :infinite? false}))

(def gen-addr-num-map
  "Non-empty {addr -> number} map with DISTINCT addresses (into {} dedups by
   key, last-wins). 1-8 entries drawn from the pool."
  (gen/such-that seq
    (gen/fmap (fn [pairs] (into {} pairs))
              (gen/vector (gen/tuple (gen/elements addr-pool) gen-num) 1 8))
    100))

(defn- close? [a b tol]
  (and (number? a) (number? b) (js/isFinite a) (js/isFinite b)
       (<= (js/Math.abs (- a b)) tol)))

(defn- ->scalars [m] (into {} (map (fn [[a v]] [a (mx/scalar v)]) m)))

(defn- contiguous-index
  "addr-index {addr -> 0..K-1} in key order (a bijection onto the [K] tensor
   positions, exactly as make-addr-index assigns)."
  [addrs]
  (zipmap addrs (range (count addrs))))

;; ---------------------------------------------------------------------------
;; Property 1: pack-values then unpack-values is the identity
;; ---------------------------------------------------------------------------

(defspec pack-unpack-roundtrips-values 100
  (prop/for-all [m gen-addr-num-map]
    (let [addrs (vec (keys m))
          ai (contiguous-index addrs)
          packed (tt/pack-values (->scalars m) ai)
          unpacked (tt/unpack-values packed ai)]
      (and (= [(count addrs)] (mx/shape packed))
           (every? (fn [a] (close? (get m a) (mx/item (get unpacked a)) 1e-5))
                   addrs)))))

;; ---------------------------------------------------------------------------
;; Property 2: trace -> tensor-trace -> trace preserves choices + fields
;; ---------------------------------------------------------------------------

(defspec tensor-trace->trace-roundtrips 100
  (prop/for-all [m       gen-addr-num-map
                 score-n gen-num
                 retval-n gen-num]
    (let [addrs   (vec (keys m))
          ai      (contiguous-index addrs)
          choices (cm/from-flat-map (->scalars m))
          trace   (tr/make-trace {:gen-fn :test-gfn
                                  :args   [1.0 2.0]
                                  :choices choices
                                  :retval (mx/scalar retval-n)
                                  :score  (mx/scalar score-n)})
          tt-trace (tt/trace->tensor-trace trace ai)
          back     (tt/tensor-trace->trace tt-trace)]
      (and (instance? tt/TensorTrace tt-trace)
           (instance? tr/Trace back)
           (= :test-gfn (:gen-fn back))
           (= [1.0 2.0] (:args back))
           (close? score-n  (mx/item (:score back)) 1e-5)
           (close? retval-n (mx/item (:retval back)) 1e-5)
           (every? (fn [a]
                     (close? (get m a)
                             (mx/item (cm/get-value (cm/get-submap (:choices back) a)))
                             1e-5))
                   addrs)))))

(t/run-tests)
