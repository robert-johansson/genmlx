(ns genmlx.choicemap
  "Choice maps as persistent ClojureScript data with a thin protocol.
   Choice maps ARE nested Clojure maps with Value leaf wrappers.
   This gives us get-in, assoc-in, structural sharing, equality,
   and serialization for free.")

(defprotocol IChoiceMap
  (-has-value? [cm])
  (-get-value  [cm])
  (-get-submap [cm addr])
  (-submaps    [cm]))

;; Leaf: wraps a single random choice value
(defrecord Value [v]
  IChoiceMap
  (-has-value? [_] true)
  (-get-value  [_] v)
  (-get-submap [_ _] nil)
  (-submaps    [_] nil))

;; Node: persistent map of addr -> IChoiceMap
(defrecord Node [m]
  IChoiceMap
  (-has-value? [_] false)
  (-get-value  [_] (throw (ex-info (str "ChoiceMap node is not a leaf value. "
                                        "Available sub-addresses: " (vec (keys m)) ". "
                                        "Use get-submap or get-choice to access a specific sub-address.")
                                       {:sub-addresses (vec (keys m))})))
  (-get-submap [_ addr] (get m addr))
  (-submaps    [_] (seq m)))

(def EMPTY (->Node {}))

;; ---------------------------------------------------------------------------
;; Smart constructor from flat key-value pairs
;; ---------------------------------------------------------------------------

(defn choicemap
  "Create a choice map from keyword-value pairs.
   Values are wrapped as leaves; maps become nested nodes.
   (choicemap :x 1.0 :y 2.0)
   (choicemap :params {:slope 2.0 :intercept 1.0})"
  [& kvs]
  (->Node
    (into {}
      (map (fn [[k v]]
             [k (cond
                  (satisfies? IChoiceMap v) v
                  (map? v) (apply choicemap (mapcat identity v))
                  :else (->Value v))])
           (partition 2 kvs)))))

;; ---------------------------------------------------------------------------
;; Access
;; ---------------------------------------------------------------------------

(defn has-value?
  "Does this choice map node hold a leaf value?"
  [cm]
  (and (some? cm) (satisfies? IChoiceMap cm) (-has-value? cm)))

(defn get-value
  "Get the leaf value from a choice map node."
  [cm]
  (when (and cm (satisfies? IChoiceMap cm) (-has-value? cm))
    (-get-value cm)))

(defn get-submap
  "Get the sub-choice-map at the given address."
  [cm addr]
  (or (when (and cm (satisfies? IChoiceMap cm))
        (-get-submap cm addr))
      EMPTY))

;; ---------------------------------------------------------------------------
;; Hierarchical access by path
;; ---------------------------------------------------------------------------

(defn get-choice
  "Get the value at a path of addresses.
   (get-choice cm [:params :slope])"
  [cm path]
  (-get-value (reduce get-submap cm path)))

;; ---------------------------------------------------------------------------
;; Functional update — returns new choicemap (persistent)
;; ---------------------------------------------------------------------------

(defn set-value
  "Fast-path: set a Value at a single keyword address in a Node.
   Avoids path-length check, instance? check, and protocol check.
   Used by handler transitions where cm is always a Node and value is always raw."
  [cm addr value]
  (->Node (assoc (:m cm) addr (->Value value))))

(defn set-submap
  "Fast-path: set a sub-choicemap at a single address in a Node.
   Value must already be an IChoiceMap."
  [cm addr sub-cm]
  (->Node (assoc (:m cm) addr sub-cm)))

(defn set-choice
  "Set a value at the given path, returning a new choice map."
  [cm path value]
  (if (= 1 (count path))
    (->Node (assoc (if (instance? Node cm) (:m cm) {})
                   (first path)
                   (if (satisfies? IChoiceMap value) value (->Value value))))
    (let [addr (first path)
          child (get-submap cm addr)
          updated (set-choice child (rest path) value)]
      (->Node (assoc (if (instance? Node cm) (:m cm) {})
                     addr updated)))))

;; ---------------------------------------------------------------------------
;; Merge: values in b override values in a
;; ---------------------------------------------------------------------------

(defn merge-cm
  "Merge two choice maps. Values in b override values in a."
  [a b]
  (cond
    (nil? b) a
    (nil? a) b
    (not (satisfies? IChoiceMap b)) b
    (not (satisfies? IChoiceMap a)) a
    (has-value? b) b
    (has-value? a) a
    :else
    (->Node
      (merge-with merge-cm
        (when (instance? Node a) (:m a))
        (when (instance? Node b) (:m b))))))

;; ---------------------------------------------------------------------------
;; Enumeration
;; ---------------------------------------------------------------------------

(defn addresses
  "All leaf address paths as a vector of vectors."
  [cm]
  (cond
    (nil? cm) []
    (not (satisfies? IChoiceMap cm)) []
    (has-value? cm) [[]]
    (instance? Node cm)
    (into []
      (mapcat (fn [[addr sub]]
                (mapv #(into [addr] %) (addresses sub)))
              (-submaps cm)))
    :else []))

;; ---------------------------------------------------------------------------
;; Conversion to/from plain maps
;; ---------------------------------------------------------------------------

(defn to-map
  "Convert a choice map to a plain nested Clojure map."
  [cm]
  (cond
    (nil? cm) {}
    (not (satisfies? IChoiceMap cm)) {}
    (has-value? cm) (-get-value cm)
    :else (into {} (map (fn [[k v]] [k (to-map v)])) (-submaps cm))))

(defn from-map
  "Convert a plain nested map to a choice map."
  [m]
  (if (map? m)
    (->Node (into {} (map (fn [[k v]] [k (from-map v)])) m))
    (->Value m)))

;; ---------------------------------------------------------------------------
;; Stack/Unstack for batched execution
;; ---------------------------------------------------------------------------

(defn stack-choicemaps
  "Stack N choicemaps into one with [N]-shaped leaves.
   mlx-stack-fn: function that stacks a vector of arrays → [N,...]-shaped array.
   All N choicemaps must have the same address structure."
  [cms mlx-stack-fn]
  (cond
    (every? #(= % EMPTY) cms) EMPTY

    (has-value? (first cms))
    (->Value (mlx-stack-fn (mapv get-value cms)))

    (instance? Node (first cms))
    (let [addrs (keys (:m (first cms)))]
      (->Node
        (into {}
          (map (fn [addr]
                 [addr (stack-choicemaps
                         (mapv #(get-submap % addr) cms)
                         mlx-stack-fn)])
               addrs))))

    :else EMPTY))

(defn unstack-choicemap
  "Split a choicemap with [N]-shaped leaves into N scalar choicemaps.
   mlx-index-fn: function (array, i) → element at index i.
   scalar-leaf-fn: (value) → true if value is scalar (not [N]-shaped)."
  [cm n mlx-index-fn scalar-leaf-fn]
  (cond
    (= cm EMPTY) (vec (repeat n EMPTY))

    (has-value? cm)
    (let [v (get-value cm)]
      (if (scalar-leaf-fn v)
        (vec (repeat n cm))
        (mapv #(->Value (mlx-index-fn v %)) (range n))))

    (instance? Node cm)
    (let [addrs (keys (:m cm))
          addr-vecs (map (fn [addr]
                           [addr (unstack-choicemap
                                   (get-submap cm addr) n
                                   mlx-index-fn scalar-leaf-fn)])
                         addrs)]
      (mapv (fn [i]
              (->Node
                (into {}
                  (map (fn [[addr v]] [addr (nth v i)])
                       addr-vecs))))
            (range n)))

    :else (vec (repeat n EMPTY))))
