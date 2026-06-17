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

(defn valid-key?
  "True if k is a well-formed MLX PRNG key: an MLX array of shape [2] whose
   dtype is not float32. A fresh key is uint32[2]; the float 0-scalar that an
   autograd boundary can produce (shape [], float32) is rejected. Shape/dtype
   are lazy-graph metadata, so this is a cheap check (no GPU eval)."
  [key]
  (and (mx/array? key)
       (= [2] (mx/shape key))
       (not= mx/float32 (mx/dtype key))))

(defn- check-key
  "Raise a clear error if k is non-nil but not a valid PRNG key. Catches a
   mis-typed key (e.g. a float scalar from a coerced-nil autograd arg) at the
   rng boundary with an actionable message, instead of letting it reach NAPI
   and crash Metal with a C++ exception (SIGTRAP)."
  [key where]
  (when (and (some? key) (not (valid-key? key)))
    (throw (ex-info (str "rng/" where ": malformed PRNG key — expected a uint32 "
                         "array of shape [2], got "
                         (if (mx/array? key)
                           (str "shape " (mx/shape key) " dtype " (mx/dtype key))
                           (pr-str key))
                         ". A float scalar here usually means a nil key was "
                         "coerced at an autograd boundary; thread a real key.")
                    {:key-shape (when (mx/array? key) (mx/shape key))
                     :key-dtype (when (mx/array? key) (mx/dtype key))}))))

(defn- check-key-present
  "Like check-key, but nil is also an error: split/split-n require a real key.
   Raising here gives an actionable message instead of the raw NAPI error a
   nil reaching .randomSplit produces. Nil-tolerant call sites use
   split-or-nils / split-n-or-nils."
  [key where]
  (when (nil? key)
    (throw (ex-info (str "rng/" where ": key is nil — thread a real PRNG key "
                         "(rng/fresh-key), or use rng/" where "-or-nils for "
                         "nil-as-no-entropy call sites.")
                    {:key nil})))
  (check-key key where))

(defn split
  "Split a key into two independent sub-keys. Returns [k1 k2]."
  [key]
  (check-key-present key "split")
  (let [ks (.randomSplit c key)]
    [(aget ks 0) (aget ks 1)]))

(defn split-n
  "Split a key into n independent sub-keys. Returns vector of n keys."
  [key n]
  (check-key-present key "split-n")
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

(defn split-or-nils
  "Split key like split, but return [nil nil] when key is nil (no-key mode).
   Lets callers thread sub-keys unconditionally while preserving nil-as-no-
   entropy."
  [key]
  (if key (split key) [nil nil]))

(defn split-n-or-nils
  "Split key into n sub-keys like split-n, but return n nils when key is nil
   (no-key mode), preserving nil-as-no-entropy for unconditional threading."
  [key n]
  (if key (split-n key n) (vec (repeat n nil))))

;; =========================================================================
;; Key-based sampling -- module-level genmlx.rs functions.
;; Shapes are plain clj->js number[] -- no to-big-shape conversion.
;; =========================================================================

(defn normal
  "Standard-normal N(0,1) samples of the given shape (a clj shape vector)."
  [key shape]
  (.keyNormal c key (clj->js shape)))

(defn uniform
  "Uniform [0,1) samples of the given shape (a clj shape vector)."
  [key shape]
  (.keyUniform c key (clj->js shape)))

(defn bernoulli
  "Bernoulli(p) 0/1 samples of the given shape (a clj shape vector)."
  [key p shape]
  (.keyBernoulli c key p (clj->js shape)))

(defn categorical
  "Sample category indices from unnormalized logits along its last axis;
   returns one index per leading-batch row."
  [key logits]
  (.keyCategorical c key logits))

(defn randint
  "Uniform integers in [lo, hi) of the given shape (a clj shape vector)."
  [key lo hi shape]
  (.keyRandint c key lo hi (clj->js shape)))

(defn gumbel
  "Standard Gumbel samples of the given shape (a clj shape vector)."
  [key shape]
  (.keyGumbel c key (clj->js shape)))

(defn laplace
  "Standard Laplace samples of the given shape (a clj shape vector)."
  [key shape]
  (.keyLaplace c key (clj->js shape)))

(defn truncated-normal
  "Standard-normal samples truncated to [lower, upper], of the given shape
   (a clj shape vector). Bounds are coerced to MLX arrays."
  [key lower upper shape]
  (.keyTruncatedNormal c key
    (mx/ensure-array lower) (mx/ensure-array upper)
    (clj->js shape)))

(defn multivariate-normal
  "Multivariate-normal samples with the given mean vector and covariance
   matrix; shape (a clj shape vector, default []) gives extra leading batch
   dims. mean/cov are coerced to MLX arrays."
  ;; mx/array passes MLX arrays through unchanged, so no array?-guard is needed.
  ([key mean cov] (multivariate-normal key mean cov []))
  ([key mean cov shape]
   (.keyMultivariateNormal c key
     (mx/array mean) (mx/array cov) (clj->js shape))))
