(ns genmlx.mlx.random
  "Functional PRNG key management for GenMLX.
   Every sample consumes a key -- no hidden PRNG state.
   Keys are MLX arrays that split deterministically.

   Uses mlx-node module-level exports (genmlx.rs) -- no BigInt64Array
   shape conversion, no instance method calls."
  (:require [genmlx.mlx :as mx]))

;; =========================================================================
;; Module reference -- genmlx.rs module-level random functions.
;; Shapes are JS number[] (no BigInt64Array conversion needed).
;; =========================================================================

(defonce ^:private c (js/require "@mlx-node/core"))

;; =========================================================================
;; Key management
;; =========================================================================

(defn fresh-key
  "Create a fresh random key from an optional integer seed."
  ([] (.randomKey c (js/Math.floor (* (js/Math.random) 2147483647))))
  ([seed] (.randomKey c seed)))

(defn key->seed
  "Derive a non-negative integer seed from a PRNG key array."
  [key]
  (mx/eval! key)
  (let [arr (mx/->clj key)]
    (bit-and (bit-xor (int (nth arr 0)) (int (nth arr 1)))
             0x7FFFFFFF)))

(defn seed! [_key] nil)

(defn split
  "Split a key into two independent sub-keys. Returns [k1 k2]."
  [key]
  (let [ks (.randomSplit c key)]
    [(aget ks 0) (aget ks 1)]))

(defn split-n
  "Split a key into n independent sub-keys. Returns vector of n keys."
  [key n]
  (let [ks (.randomSplitN c key n)]
    (mapv #(mx/index ks %) (range n))))

(defn ensure-key
  "Return key if non-nil, otherwise a fresh random key."
  [key]
  (or key (fresh-key)))

(defn split-or-nils [key]
  (if key (split key) [nil nil]))

(defn split-n-or-nils [key n]
  (if key (split-n key n) (vec (repeat n nil))))

;; =========================================================================
;; Key-based sampling -- module-level genmlx.rs functions.
;; Shapes are plain clj->js number[] -- no to-big-shape conversion.
;; =========================================================================

(defn normal [key shape]
  (.keyNormal c key (clj->js shape)))

(defn uniform [key shape]
  (.keyUniform c key (clj->js shape)))

(defn bernoulli [key p shape]
  (.keyBernoulli c key p (clj->js shape)))

(defn categorical [key logits]
  (.keyCategorical c key logits))

(defn randint [key lo hi shape]
  (.keyRandint c key lo hi (clj->js shape)))

(defn gumbel [key shape]
  (.keyGumbel c key (clj->js shape)))

(defn laplace [key shape]
  (.keyLaplace c key (clj->js shape)))

(defn truncated-normal [key lower upper shape]
  (.keyTruncatedNormal c key
    (mx/ensure-array lower) (mx/ensure-array upper)
    (clj->js shape)))

(defn multivariate-normal
  ([key mean cov]
   (.keyMultivariateNormal c key
     (if (mx/array? mean) mean (mx/array mean))
     (if (mx/array? cov) cov (mx/array cov))
     #js []))
  ([key mean cov shape]
   (.keyMultivariateNormal c key
     (if (mx/array? mean) mean (mx/array mean))
     (if (mx/array? cov) cov (mx/array cov))
     (clj->js shape))))

(defn permutation
  ([key n]
   (let [rand-vals (uniform key [n])]
     (mx/argsort rand-vals)))
  ([key arr axis]
   (let [n (nth (mx/shape arr) axis)
         perm (permutation key n)]
     (mx/take-idx arr perm axis))))
