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

(declare from-map)

;; ---------------------------------------------------------------------------
;; Smart constructor from flat key-value pairs
;; ---------------------------------------------------------------------------

(defn choicemap
  "Create a choice map from keyword-value pairs.
   Values are wrapped as leaves; maps become nested nodes.
   (choicemap :x 1.0 :y 2.0)
   (choicemap :params {:slope 2.0 :intercept 1.0})
   For a single plain map use from-map. NOTE: (choicemap {m}) used to return
   EMPTY silently — 11 test sites passed vacuous EMPTY constraints that way
   (genmlx-ybw9); the odd-args throw keeps that mistake loud."
  [& kvs]
  (when (odd? (count kvs))
    (throw (ex-info "choicemap expects an even number of key-value arguments (for a single map, use from-map)"
                    {:arg-count (count kvs) :trailing (last kvs)})))
  (->Node
    (into {}
      (map (fn [[k v]]
             [k (cond
                  (satisfies? IChoiceMap v) v
                  (map? v) (from-map v)
                  :else (->Value v))])
           (partition 2 kvs)))))

;; ---------------------------------------------------------------------------
;; Access
;; ---------------------------------------------------------------------------

(defn choicemap?
  "Is x a non-nil value satisfying the IChoiceMap protocol?"
  [x]
  (and (some? x) (satisfies? IChoiceMap x)))

(defn has-value?
  "Does this choice map node hold a leaf value?"
  [cm]
  (and (choicemap? cm) (-has-value? cm)))

(defn get-value
  "Get the leaf value from a choice map node."
  [cm]
  (when (and (choicemap? cm) (-has-value? cm))
    (-get-value cm)))

(defn get-submap
  "Get the sub-choice-map at the given address."
  [cm addr]
  (or (when (choicemap? cm)
        (-get-submap cm addr))
      EMPTY))

;; ---------------------------------------------------------------------------
;; Hierarchical access by path
;; ---------------------------------------------------------------------------

(defn get-choice
  "Get the value at a path of addresses, or nil when the path is missing
   or resolves to a non-leaf node (same convention as get-value).
   (get-choice cm [:params :slope])"
  [cm path]
  (get-value (reduce get-submap cm path)))

;; ---------------------------------------------------------------------------
;; Functional update — returns new choicemap (persistent)
;; ---------------------------------------------------------------------------

(defn set-value
  "Fast-path: set a Value at a single keyword address in a Node.
   Used by handler transitions where cm is always a Node and value is always raw.
   nil cm is treated as empty (callers build choices via (update m :choices ...)
   where the key may be absent). Throws on a Value leaf or any other non-Node —
   silently rebuilding from a leaf would discard it."
  [cm addr value]
  (cond
    (instance? Node cm) (->Node (assoc (:m cm) addr (->Value value)))
    (nil? cm)           (->Node {addr (->Value value)})
    :else (throw (ex-info "set-value expects a Node choicemap (or nil)"
                          {:cm-type (type cm) :addr addr}))))

(defn set-submap
  "Fast-path: set a sub-choicemap at a single address in a Node.
   Value must already be an IChoiceMap."
  [cm addr sub-cm]
  (->Node (assoc (:m cm) addr sub-cm)))

(defn set-choice
  "Set a value at the given path, returning a new choice map."
  [cm path value]
  (let [node-m (if (instance? Node cm) (:m cm) {})
        [addr & more] path]
    (if (seq more)
      (let [child (get-submap cm addr)
            updated (set-choice child more value)]
        (->Node (assoc node-m addr updated)))
      (->Node (assoc node-m
                     addr
                     (if (satisfies? IChoiceMap value) value (->Value value)))))))

;; ---------------------------------------------------------------------------
;; Merge: values in b override values in a
;; ---------------------------------------------------------------------------

(defn merge-cm
  "Merge two choice maps. Entries in b override entries in a: on any
   conflict (leaf-vs-leaf, leaf-vs-node, node-vs-leaf) b's entry wins and
   replaces a's entire subtree at that address. Non-choicemap arguments are
   coerced (maps -> nested nodes, other values -> leaves)."
  [a b]
  (let [a (if (or (nil? a) (choicemap? a)) a (from-map a))
        b (if (or (nil? b) (choicemap? b)) b (from-map b))]
    (cond
      (nil? b) a
      (nil? a) b
      (has-value? b) b
      (has-value? a) b
      :else
      (->Node
        (merge-with merge-cm
          (into {} (-submaps a))
          (into {} (-submaps b)))))))

;; ---------------------------------------------------------------------------
;; Enumeration
;; ---------------------------------------------------------------------------

(defn addresses
  "All leaf address paths as a vector of vectors."
  [cm]
  (cond
    (nil? cm) []
    (not (choicemap? cm)) []
    (has-value? cm) [[]]
    :else
    (into []
      (mapcat (fn [[addr sub]]
                (mapv #(into [addr] %) (addresses sub)))
              (-submaps cm)))))

;; ---------------------------------------------------------------------------
;; Conversion to/from plain maps
;; ---------------------------------------------------------------------------

(defn to-map
  "Convert a choice map to a plain nested Clojure map."
  [cm]
  (cond
    (nil? cm) {}
    (not (choicemap? cm)) {}
    (has-value? cm) (-get-value cm)
    :else (into {} (map (fn [[k v]] [k (to-map v)])) (-submaps cm))))

(defn from-map
  "Convert a plain nested map to a choice map."
  [m]
  (if (map? m)
    (->Node (into {} (map (fn [[k v]] [k (from-map v)])) m))
    (->Value m)))

(defn from-flat-map
  "Build a ChoiceMap from a flat {keyword -> value} map.
   Each key becomes a top-level Value entry."
  [m]
  (reduce-kv
    (fn [cm addr val] (set-value cm addr val))
    EMPTY
    m))

;; ---------------------------------------------------------------------------
;; Stack/Unstack for batched execution
;; ---------------------------------------------------------------------------

(defn stack-choicemaps
  "Stack N choicemaps into one with [N]-shaped leaves.
   mlx-stack-fn: function that stacks a vector of arrays → [N,...]-shaped array.
   All N choicemaps must have the same address structure; throws when they
   diverge (an address present in one cm but not another, or leaf-vs-node
   mismatch) instead of silently dropping addresses or stacking nils."
  [cms mlx-stack-fn]
  (cond
    (every? #(= % EMPTY) cms) EMPTY

    (has-value? (first cms))
    (if (every? has-value? cms)
      (->Value (mlx-stack-fn (mapv get-value cms)))
      (throw (ex-info "stack-choicemaps: leaf/node mismatch across choicemaps"
                      {:leaf-indices (keep-indexed #(when (has-value? %2) %1) cms)})))

    (instance? Node (first cms))
    (let [addr-sets (mapv #(set (keys (:m %))) cms)
          addrs (keys (:m (first cms)))]
      (when-not (apply = addr-sets)
        (throw (ex-info "stack-choicemaps: address structure differs across choicemaps"
                        {:first-addrs (first addr-sets)
                         :divergent (into {}
                                          (keep-indexed
                                            (fn [i s] (when (not= s (first addr-sets)) [i s]))
                                            addr-sets))})))
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
