(ns genmlx.diff
  "Change-tagged values for incremental computation.
   Argdiffs and retdiffs enable `update` to skip unchanged computation,
   which is critical for MCMC performance where each MH step only changes
   one or a few addresses."
  (:require [clojure.set]))

;; ---------------------------------------------------------------------------
;; Diff types
;; ---------------------------------------------------------------------------

(def no-change
  "Indicates a value has not changed."
  {:diff-type :no-change})

(def unknown-change
  "Indicates a value may have changed (conservative)."
  {:diff-type :unknown-change})

(defn value-change
  "Indicates a value changed from old to new."
  [old-val new-val]
  {:diff-type :value-change :old old-val :new new-val})

(defn vector-diff
  "Indicates changes to specific indices of a vector.
   changed-indices: set of indices that changed."
  [changed-indices]
  {:diff-type :vector-diff :changed changed-indices})

(defn map-diff
  "Indicates changes to specific keys of a map.
   changed-keys: set of keys that changed.
   added-keys: set of keys that were added.
   removed-keys: set of keys that were removed."
  [changed-keys added-keys removed-keys]
  {:diff-type :map-diff
   :changed changed-keys
   :added added-keys
   :removed removed-keys})

;; ---------------------------------------------------------------------------
;; Diff predicates
;; ---------------------------------------------------------------------------

(defn no-change?
  "Returns true if the diff indicates no change."
  [d]
  (= (:diff-type d) :no-change))

(defn unknown-change?
  "Returns true if the diff type is unknown (must assume everything changed)."
  [d]
  (or (nil? d) (= (:diff-type d) :unknown-change)))

(defn changed?
  "Returns true if the diff indicates any change."
  [d]
  (not (no-change? d)))

;; ---------------------------------------------------------------------------
;; Diff computation
;; ---------------------------------------------------------------------------

(defn compute-diff
  "Compute the diff between two values."
  [old-val new-val]
  (if (identical? old-val new-val)
    no-change
    (if (= old-val new-val)
      no-change
      (value-change old-val new-val))))

(defn compute-vector-diff
  "Compute the diff between two vectors."
  [old-vec new-vec]
  (if (identical? old-vec new-vec)
    no-change
    (let [n (max (count old-vec) (count new-vec))
          changed (into #{}
                    (filter (fn [i]
                              (not= (get old-vec i) (get new-vec i))))
                    (range n))]
      (if (empty? changed)
        no-change
        (vector-diff changed)))))

(defn compute-map-diff
  "Compute the diff between two maps."
  [old-map new-map]
  (if (identical? old-map new-map)
    no-change
    (let [old-keys (set (keys old-map))
          new-keys (set (keys new-map))
          added (clojure.set/difference new-keys old-keys)
          removed (clojure.set/difference old-keys new-keys)
          common (clojure.set/intersection old-keys new-keys)
          changed (into #{}
                    (filter (fn [k] (not= (get old-map k) (get new-map k))))
                    common)]
      (if (and (empty? added) (empty? removed) (empty? changed))
        no-change
        (map-diff changed added removed)))))

;; ---------------------------------------------------------------------------
;; Diff-aware update support
;; ---------------------------------------------------------------------------

(defn should-recompute?
  "Given an argdiff and an address, determine if the value at that address
   needs to be recomputed."
  [argdiff addr]
  (cond
    (no-change? argdiff) false
    (unknown-change? argdiff) true
    (= (:diff-type argdiff) :vector-diff)
    (contains? (:changed argdiff) addr)
    (= (:diff-type argdiff) :map-diff)
    (or (contains? (:changed argdiff) addr)
        (contains? (:added argdiff) addr))
    :else true))
