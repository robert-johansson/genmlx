(ns genmlx.selection
  "Composable address selection algebra for GenMLX.
   Selections identify subsets of addresses in a choice map
   for operations like regenerate.")

(defprotocol ISelection
  (selected?        [s addr] "Is this address selected?")
  (get-subselection [s addr] "Get the selection for addresses under this one."))

;; Select all addresses
(def all
  (reify ISelection
    (selected? [_ _] true)
    (get-subselection [_ _] all)))

;; Select no addresses
(def none
  (reify ISelection
    (selected? [_ _] false)
    (get-subselection [_ _] none)))

;; Select specific addresses (flat)
(defrecord SelectAddrs [addrs]
  ISelection
  (selected? [_ addr] (contains? addrs addr))
  (get-subselection [_ _] all))

(defn select
  "Create a selection of specific addresses.
   (select :x :y :z)"
  [& addrs]
  (->SelectAddrs (set addrs)))

;; Support plain sets as selections via SelectSet wrapper
(defrecord SelectSet [s]
  ISelection
  (selected? [_ addr] (contains? s addr))
  (get-subselection [_ _] all))

(defn from-set
  "Create a selection from a set of addresses."
  [s]
  (->SelectSet s))

;; Hierarchical selection: address -> sub-selection
(defrecord Hierarchical [m]
  ISelection
  (selected? [_ addr] (contains? m addr))
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
