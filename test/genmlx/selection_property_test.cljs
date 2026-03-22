(ns genmlx.selection-property-test
  "Property-based selection algebra tests using test.check.
   Verifies all/none semantics, from-set membership, complement involution,
   hierarchical subselection, and interaction with choicemaps."
  (:require [cljs.test :as t]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.selection :as sel]
            [genmlx.choicemap :as cm])
  (:require-macros [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Generators
;; ---------------------------------------------------------------------------

(def addr-pool [:a :b :c :d :e :f :g :h :i :j])

(def gen-addr (gen/elements addr-pool))

(def gen-addr-set
  "Generator for a non-empty set of addresses."
  (gen/fmap set (gen/not-empty (gen/vector gen-addr 1 5))))

(def gen-flat-selection
  "Generator for a flat selection from random addresses."
  (gen/fmap #(apply sel/select %) (gen/not-empty (gen/vector gen-addr 1 5))))

;; ---------------------------------------------------------------------------
;; Properties
;; ---------------------------------------------------------------------------

(defspec all-selects-any-address 50
  (prop/for-all [addr gen-addr]
    (sel/selected? sel/all addr)))

(defspec none-selects-no-address 50
  (prop/for-all [addr gen-addr]
    (not (sel/selected? sel/none addr))))

(defspec from-set-selects-k-iff-k-in-S 100
  (prop/for-all [s gen-addr-set
                 k gen-addr]
    (let [sel (sel/from-set s)]
      (= (contains? s k) (sel/selected? sel k)))))

(defspec complement-complement-is-identity 100
  (prop/for-all [s gen-addr-set
                 k gen-addr]
    (let [sel (sel/from-set s)
          cc (sel/complement-sel (sel/complement-sel sel))]
      (= (sel/selected? sel k) (sel/selected? cc k)))))

(defspec complement-all-behaves-as-none 50
  (prop/for-all [addr gen-addr]
    (not (sel/selected? (sel/complement-sel sel/all) addr))))

(defspec complement-none-behaves-as-all 50
  (prop/for-all [addr gen-addr]
    (sel/selected? (sel/complement-sel sel/none) addr)))

(defspec hierarchical-selects-its-keys 50
  (prop/for-all [addrs gen-addr-set]
    (let [h (apply sel/hierarchical (mapcat (fn [a] [a sel/all]) addrs))]
      (every? #(sel/selected? h %) addrs))))

(defspec subselection-returns-nested-selection 100
  (prop/for-all [outer-addr gen-addr
                 inner-addr gen-addr]
    (let [inner-sel (sel/select inner-addr)
          h (sel/hierarchical outer-addr inner-sel)
          sub (sel/get-subselection h outer-addr)]
      (sel/selected? sub inner-addr))))

(defspec subselection-of-missing-addr-is-none 50
  (prop/for-all [present-addr gen-addr]
    (let [h (sel/hierarchical present-addr sel/all)
          missing (keyword (str "__missing__" (name present-addr)))
          sub (sel/get-subselection h missing)]
      (not (sel/selected? sub :anything)))))

(defspec all-addresses-of-choicemap-selected-by-all 50
  (prop/for-all [addrs (gen/not-empty (gen/vector gen-addr 1 4))]
    (let [cm (reduce (fn [c a] (cm/set-choice c [a] 1.0)) cm/EMPTY addrs)
          leaf-addrs (cm/addresses cm)]
      (every? (fn [[a]] (sel/selected? sel/all a)) leaf-addrs))))

(defspec no-addresses-of-choicemap-selected-by-none 50
  (prop/for-all [addrs (gen/not-empty (gen/vector gen-addr 1 4))]
    (let [cm (reduce (fn [c a] (cm/set-choice c [a] 1.0)) cm/EMPTY addrs)
          leaf-addrs (cm/addresses cm)]
      (not-any? (fn [[a]] (sel/selected? sel/none a)) leaf-addrs))))

(t/run-tests)
