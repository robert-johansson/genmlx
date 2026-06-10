(ns genmlx.selection
  "Composable address selection algebra for GenMLX.
   Selections identify subsets of addresses in a choice map
   for operations like regenerate.")

(defprotocol ISelection
  (selected?        [s addr] "Is this address selected?")
  (get-subselection [s addr] "Get the selection for addresses under this one."))

(def all
  "Selection that selects every address; its subselection is itself."
  (reify ISelection
    (selected? [_ _] true)
    (get-subselection [_ _] all)))

(def none
  "Selection that selects no address; its subselection is itself."
  (reify ISelection
    (selected? [_ _] false)
    (get-subselection [_ _] none)))

(defrecord SelectAddrs [addrs]
  ISelection
  (selected? [_ addr] (contains? addrs addr))
  ;; Gen.jl semantics: selecting a flat address selects EVERYTHING under it,
  ;; but ONLY that address. So the subselection is `all` iff the address is
  ;; selected, otherwise `none`. Returning `all` unconditionally (the prior
  ;; bug) made every splice/combinator-element descent resample its whole
  ;; subtree regardless of which address was actually selected.
  (get-subselection [_ addr] (if (contains? addrs addr) all none)))

(defn select
  "Create a flat selection of specific addresses.
   (select :x :y :z)"
  [& addrs]
  (->SelectAddrs (set addrs)))

(defn from-set
  "Create a selection from a set of addresses."
  [s]
  (->SelectAddrs s))

;; Hierarchical selection: address -> sub-selection
(defrecord Hierarchical [m]
  ISelection
  ;; A leaf at `addr` is selected iff its subselection selects everything,
  ;; i.e. the entry maps to the canonical `all` selection. Mapping `addr` to a
  ;; PARTIAL subselection (e.g. (hierarchical :sub (select :z))) means "descend
  ;; into :sub and select :z" — it does NOT select the leaf :sub itself. The
  ;; prior `(contains? m addr)` conflated partial-descent with full-leaf
  ;; selection. Descent into sub-structures uses get-subselection, not
  ;; selected?, so this only affects genuine leaf checks.
  (selected? [_ addr] (identical? all (get m addr none)))
  (get-subselection [_ addr] (get m addr none)))

(defn hierarchical
  "Create a hierarchical selection.
   (hierarchical :sub1 (select :x :y) :sub2 all)"
  [& kvs]
  (->Hierarchical (apply hash-map kvs)))

;; Complement selection
(defrecord Complement [inner]
  ISelection
  (selected? [_ addr] (not (selected? inner addr)))
  (get-subselection [_ addr] (->Complement (get-subselection inner addr))))

(defn complement-sel
  "Create the complement of a selection."
  [s]
  (->Complement s))
