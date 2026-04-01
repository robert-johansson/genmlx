(ns genmlx.mlx.random
  "Functional PRNG key management for GenMLX.
   Every sample consumes a key — no hidden PRNG state.
   Keys are MLX arrays that split deterministically.

   Calls mlx-node MxArray methods directly — no bridge.js dependency."
  (:require [genmlx.mlx :as mx]))

;; ---------------------------------------------------------------------------
;; MxArray class reference (for static methods like randomKey)
;; ---------------------------------------------------------------------------

(defonce ^:private M (.-MxArray (js/require "@mlx-node/core")))

;; ---------------------------------------------------------------------------
;; Key management
;; ---------------------------------------------------------------------------

(defn fresh-key
  "Create a fresh random key from an optional integer seed."
  ([] (.randomKey M (js/Math.floor (* (js/Math.random) 2147483647))))
  ([seed] (.randomKey M seed)))

(defn key->seed
  "Derive a non-negative integer seed from a PRNG key array.
   Combines both uint32 elements via XOR, then masks to 31 bits."
  [key]
  (mx/eval! key)
  (let [arr (mx/->clj key)]
    (bit-and (bit-xor (int (nth arr 0)) (int (nth arr 1)))
             0x7FFFFFFF)))

(defn seed!
  "No-op — mlx-node uses key-based sampling, no global PRNG state."
  [_key]
  nil)

(defn split
  "Split a key into two independent sub-keys. Returns [k1 k2]."
  [key]
  (let [ks (.randomSplit key)]
    [(aget ks 0) (aget ks 1)]))

(defn split-n
  "Split a key into n independent sub-keys. Returns vector of n keys."
  [key n]
  (let [ks (.randomSplitN key n)]
    (mapv #(mx/index ks %) (range n))))

(defn ensure-key
  "Return key if non-nil, otherwise a fresh random key."
  [key]
  (or key (fresh-key)))

(defn split-or-nils
  "Split key into [k1 k2] if non-nil, otherwise [nil nil]."
  [key]
  (if key (split key) [nil nil]))

(defn split-n-or-nils
  "Split key into n sub-keys if non-nil, otherwise a vector of n nils."
  [key n]
  (if key (split-n key n) (vec (repeat n nil))))

;; ---------------------------------------------------------------------------
;; Key-based sampling — direct mlx-node MxArray instance methods
;; ---------------------------------------------------------------------------

(defn normal
  "Sample from standard normal using the given key."
  [key shape]
  (.keyNormal key (mx/to-big-shape shape) nil))

(defn uniform
  "Sample from uniform [0,1) using the given key."
  [key shape]
  (.keyUniform key (mx/to-big-shape shape) nil nil nil))

(defn bernoulli
  "Sample Bernoulli(p) using the given key."
  [key p shape]
  (.keyBernoulli key p (mx/to-big-shape shape)))

(defn categorical
  "Sample from categorical distribution (log-probabilities) using the given key."
  [key logits]
  (.keyCategorical key logits nil))

(defn randint
  "Sample random integers in [lo, hi) using the given key."
  [key lo hi shape]
  (.keyRandint key lo hi (mx/to-big-shape shape) nil))

(defn gumbel
  "Sample from standard Gumbel distribution using the given key."
  [key shape]
  (.keyGumbel key (mx/to-big-shape shape) nil))

(defn laplace
  "Sample from standard Laplace distribution using the given key."
  [key shape]
  (.keyLaplace key (mx/to-big-shape shape) nil))

(defn truncated-normal
  "Sample from truncated normal distribution using the given key.
   Values are clipped to [lower, upper]."
  [key lower upper shape]
  (.keyTruncatedNormal key
    (mx/ensure-array lower) (mx/ensure-array upper)
    (mx/to-big-shape shape) nil))

(defn multivariate-normal
  "Sample from multivariate normal N(mean, cov) using the given key.
   mean: [k] array, cov: [k k] positive definite array."
  ([key mean cov]
   (.keyMultivariateNormal key
     (if (mx/array? mean) mean (mx/array mean))
     (if (mx/array? cov) cov (mx/array cov))
     (mx/to-big-shape []) nil))
  ([key mean cov shape]
   (.keyMultivariateNormal key
     (if (mx/array? mean) mean (mx/array mean))
     (if (mx/array? cov) cov (mx/array cov))
     (mx/to-big-shape shape) nil)))

(defn permutation
  "Return a random permutation of integers [0, n).
   Implemented via argsort of uniform random values."
  ([key n]
   (let [rand-vals (uniform key [n])]
     (.argsort rand-vals nil)))
  ([key arr axis]
   (let [n (nth (mx/shape arr) axis)
         perm (permutation key n)]
     (mx/take-idx arr perm axis))))
