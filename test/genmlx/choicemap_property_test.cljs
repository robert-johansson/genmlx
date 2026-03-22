(ns genmlx.choicemap-property-test
  "Property-based choicemap algebra tests using test.check.
   Verifies set/get round-trips, to-map/from-map isomorphism,
   addresses enumeration, merge identity and override semantics."
  (:require [cljs.test :as t]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.choicemap :as cm])
  (:require-macros [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Generators
;; ---------------------------------------------------------------------------

(def addr-pool
  [:a :b :c :d :e :f :g :h :i :j
   :k :l :m :n :o :p :q :r :s :t])

(def gen-addr
  "Generator for a random keyword address."
  (gen/elements addr-pool))

(def gen-path
  "Generator for a random path of 1-3 keywords."
  (gen/not-empty (gen/vector gen-addr 1 3)))

(def gen-value
  "Generator for a random leaf value (plain number)."
  (gen/double* {:min -100.0 :max 100.0 :NaN? false :infinite? false}))

(def gen-path-value
  "Generator for a [path value] pair."
  (gen/tuple gen-path gen-value))

(defn gen-choicemap-from-paths
  "Generator for a choicemap built from 1-5 random path-value pairs."
  []
  (gen/fmap
    (fn [pvs]
      (reduce (fn [cm [path val]]
                (cm/set-choice cm path val))
              cm/EMPTY pvs))
    (gen/not-empty (gen/vector gen-path-value 1 5))))

(def gen-flat-map
  "Generator for a flat {:keyword value} map (1-5 entries)."
  (gen/fmap
    (fn [kvs] (into {} kvs))
    (gen/not-empty (gen/vector (gen/tuple gen-addr gen-value) 1 5))))

;; ---------------------------------------------------------------------------
;; Properties
;; ---------------------------------------------------------------------------

(defspec set-choice-get-choice-round-trips 100
  (prop/for-all [path gen-path
                 val gen-value]
    (let [cm (cm/set-choice cm/EMPTY path val)]
      (= val (cm/get-choice cm path)))))

(defspec to-map-from-map-to-map-round-trips 100
  (prop/for-all [cm (gen-choicemap-from-paths)]
    (let [m1 (cm/to-map cm)
          cm2 (cm/from-map m1)
          m2 (cm/to-map cm2)]
      (= m1 m2))))

(defspec from-map-to-map-round-trips 100
  (prop/for-all [m gen-flat-map]
    (= m (cm/to-map (cm/from-map m)))))

(defspec addresses-returns-exactly-the-set-paths 100
  (prop/for-all [path gen-path
                 val gen-value]
    (let [cm (cm/set-choice cm/EMPTY path val)
          addrs (cm/addresses cm)]
      (= [path] addrs))))

(defspec has-value-true-on-leaves-false-on-nodes 50
  (prop/for-all [path gen-path
                 val gen-value]
    (let [cm (cm/set-choice cm/EMPTY path val)
          leaf (reduce cm/get-submap cm path)
          root-is-node (not (cm/has-value? cm))]
      (and root-is-node (cm/has-value? leaf)))))

(defspec get-submap-of-missing-addr-returns-EMPTY 50
  (prop/for-all [cm (gen-choicemap-from-paths)
                 addr gen-addr]
    (let [missing-addr (keyword (str "__missing__" (name addr)))
          sub (cm/get-submap cm missing-addr)]
      (= sub cm/EMPTY))))

(defspec EMPTY-has-no-addresses 1
  (prop/for-all [_ (gen/return nil)]
    (empty? (cm/addresses cm/EMPTY))))

(defspec merge-cm-a-EMPTY-equals-a 100
  (prop/for-all [cm (gen-choicemap-from-paths)]
    (= (cm/to-map (cm/merge-cm cm cm/EMPTY))
       (cm/to-map cm))))

(defspec merge-cm-EMPTY-b-equals-b 100
  (prop/for-all [cm (gen-choicemap-from-paths)]
    (= (cm/to-map (cm/merge-cm cm/EMPTY cm))
       (cm/to-map cm))))

(defspec merge-b-overrides-a-at-shared-paths 100
  (prop/for-all [path gen-path
                 val-a gen-value
                 val-b gen-value]
    (let [a (cm/set-choice cm/EMPTY path val-a)
          b (cm/set-choice cm/EMPTY path val-b)
          merged (cm/merge-cm a b)]
      (= val-b (cm/get-choice merged path)))))

(defspec merge-preserves-values-at-disjoint-paths 100
  (prop/for-all [val-a gen-value
                 val-b gen-value]
    (let [path-a [:__left :val]
          path-b [:__right :val]
          a (cm/set-choice cm/EMPTY path-a val-a)
          b (cm/set-choice cm/EMPTY path-b val-b)
          merged (cm/merge-cm a b)]
      (and (= val-a (cm/get-choice merged path-a))
           (= val-b (cm/get-choice merged path-b))))))

(t/run-tests)
