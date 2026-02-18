(ns genmlx.mlx.random
  "Functional PRNG key management for GenMLX.
   Every sample consumes a key — no hidden PRNG state.
   Keys are MLX arrays that split deterministically."
  (:require [genmlx.mlx :as mx]))

(defn fresh-key
  "Create a fresh random key from an optional integer seed."
  ([]    (.key mx/random (js/Math.floor (* (js/Math.random) 2147483647))))
  ([seed] (.key mx/random seed)))

(defn- extract-row
  "Extract row i from a 2D MLX array as a 1D array."
  [arr i]
  (mx/squeeze (.index arr (new (.-Slice mx/core) i (inc i)))))

(defn split
  "Split a key into two independent sub-keys. Returns [k1 k2]."
  [key]
  (let [ks (.split mx/random key)]
    [(extract-row ks 0) (extract-row ks 1)]))

(defn split-n
  "Split a key into n independent sub-keys. Returns vector of n keys."
  [key n]
  (let [ks (.split mx/random key n)]
    (mapv #(extract-row ks %) (range n))))

;; ---------------------------------------------------------------------------
;; Key-based sampling (functional — no global PRNG state)
;; ---------------------------------------------------------------------------

(defn normal
  "Sample from standard normal using the given key."
  [key shape]
  (.normal mx/random (clj->js shape) nil key))

(defn uniform
  "Sample from uniform [0,1) using the given key."
  [key shape]
  (.uniform mx/random 0 1 (clj->js shape) nil key))

(defn bernoulli
  "Sample Bernoulli(p) using the given key."
  [key p shape]
  (.bernoulli mx/random p (clj->js shape) key))

(defn categorical
  "Sample from categorical distribution (log-probabilities) using the given key."
  [key logits]
  (.categorical mx/random logits key))

(defn randint
  "Sample random integers in [lo, hi) using the given key."
  [key lo hi shape]
  (.randint mx/random lo hi (clj->js shape) nil key))

(defn gumbel
  "Sample from standard Gumbel distribution using the given key."
  [key shape]
  (.gumbel mx/random (clj->js shape) nil key))

(defn laplace
  "Sample from standard Laplace distribution using the given key."
  [key shape]
  (.laplace mx/random (clj->js shape) nil key))
