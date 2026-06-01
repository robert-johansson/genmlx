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

(defn- key->seed
  "Derive a non-negative integer seed from a PRNG key array."
  [key]
  (mx/eval! key)
  (let [arr (mx/->clj key)]
    (bit-and (bit-xor (int (nth arr 0)) (int (nth arr 1)))
             0x7FFFFFFF)))

(defn valid-key?
  "True if k is a well-formed MLX PRNG key: an MLX array of shape [2] whose
   dtype is not float32. A fresh key is uint32[2]; the float 0-scalar that an
   autograd boundary can produce (shape [], float32) is rejected. Shape/dtype
   are lazy-graph metadata, so this is a cheap check (no GPU eval)."
  [k]
  (and (mx/array? k)
       (= [2] (mx/shape k))
       (not= mx/float32 (mx/dtype k))))

(defn- check-key
  "Raise a clear error if k is non-nil but not a valid PRNG key. Catches a
   mis-typed key (e.g. a float scalar from a coerced-nil autograd arg) at the
   rng boundary with an actionable message, instead of letting it reach NAPI
   and crash Metal with a C++ exception (SIGTRAP)."
  [k where]
  (when (and (some? k) (not (valid-key? k)))
    (throw (ex-info (str "rng/" where ": malformed PRNG key — expected a uint32 "
                         "array of shape [2], got "
                         (if (mx/array? k)
                           (str "shape " (mx/shape k) " dtype " (mx/dtype k))
                           (pr-str k))
                         ". A float scalar here usually means a nil key was "
                         "coerced at an autograd boundary; thread a real key.")
                    {:key-shape (when (mx/array? k) (mx/shape k))
                     :key-dtype (when (mx/array? k) (mx/dtype k))}))))

(defn split
  "Split a key into two independent sub-keys. Returns [k1 k2]."
  [key]
  (check-key key "split")
  (let [ks (.randomSplit c key)]
    [(aget ks 0) (aget ks 1)]))

(defn split-n
  "Split a key into n independent sub-keys. Returns vector of n keys."
  [key n]
  (check-key key "split-n")
  (let [ks (.randomSplitN c key n)]
    (mapv #(mx/index ks %) (range n))))

(defn ensure-key
  "Return key if it is a valid PRNG key, or a fresh key if nil. A non-nil but
   malformed key (e.g. a float scalar from a coerced-nil autograd arg) is a
   caller bug and raises here rather than silently producing garbage samples
   (the (or key (fresh-key)) shorthand let a truthy float scalar slip through)."
  [key]
  (if (nil? key)
    (fresh-key)
    (do (check-key key "ensure-key") key)))

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

(defn- permutation
  ([key n]
   (let [rand-vals (uniform key [n])]
     (mx/argsort rand-vals)))
  ([key arr axis]
   (let [n (nth (mx/shape arr) axis)
         perm (permutation key n)]
     (mx/take-idx arr perm axis))))
