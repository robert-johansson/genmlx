;; @tier fast
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

;; ---------------------------------------------------------------------------
;; get-subselection descent semantics (genmlx-yey5)
;;
;; These pin the property that descent (used by splice/combinator-element
;; recursion) resamples a subtree iff its address is selected. The prior bug
;; was SelectAddrs.get-subselection returning `all` unconditionally, so EVERY
;; unselected subtree was resampled.
;; ---------------------------------------------------------------------------

(defspec select-subselection-all-iff-selected 100
  ;; A selected address descends with `all`; an unselected one with `none`.
  (prop/for-all [s gen-addr-set
                 k gen-addr]
    (let [sel (sel/from-set s)
          sub (sel/get-subselection sel k)]
      (if (contains? s k)
        (sel/selected? sub :anything)         ;; selected => everything under k selected
        (not (sel/selected? sub :anything)))))) ;; unselected => nothing under k selected

(defspec select-descent-consistent-with-leaf 100
  ;; The leaf check and the descent agree: descending into k selects a leaf
  ;; iff k itself is selected as a leaf.
  (prop/for-all [s gen-addr-set
                 k gen-addr]
    (let [sel (sel/from-set s)]
      (= (sel/selected? sel k)
         (sel/selected? (sel/get-subselection sel k) :anything)))))

(defspec complement-descent-consistent-with-leaf 100
  ;; Complement mirror-bug regression: selected?(comp, k) agrees with whether
  ;; everything under k is selected via the descended subselection.
  (prop/for-all [s gen-addr-set
                 k gen-addr]
    (let [comp (sel/complement-sel (sel/from-set s))]
      (= (sel/selected? comp k)
         (sel/selected? (sel/get-subselection comp k) :anything)))))

(defspec hierarchical-partial-subsel-not-leaf-selected 100
  ;; Hierarchical conflation regression: an address mapped to a PARTIAL
  ;; subselection is NOT selected as a leaf; mapped to `all` it IS.
  (prop/for-all [k     gen-addr
                 inner gen-addr]
    (and (not (sel/selected? (sel/hierarchical k (sel/select inner)) k))
         (sel/selected? (sel/hierarchical k sel/all) k))))

(t/run-tests)
