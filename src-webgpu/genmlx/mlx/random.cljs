(ns genmlx.mlx.random
  "Functional PRNG key management — jax-js/WebGPU backend.
   Drop-in replacement for the MLX-native genmlx.mlx.random namespace.
   Every sample consumes a key — no hidden PRNG state.
   Keys are jax-js arrays that split deterministically (Threefry2x32)."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.bootstrap :as boot]))

(defn- rng []
  (.-random (boot/jax)))

(defn- np []
  (.-numpy (boot/jax)))

(defn fresh-key
  "Create a fresh random key from an optional integer seed."
  ([]    (.key (rng) (js/Math.floor (* (js/Math.random) 2147483647))))
  ([seed] (.key (rng) seed)))

(defn key->seed
  "Derive a non-negative integer seed from a PRNG key array.
   Combines both uint32 elements via XOR, then masks to 31 bits."
  [key]
  (let [d (.dataSync key)
        a (aget d 0)
        b (aget d 1)]
    (bit-and (bit-xor (int a) (int b)) 0x7FFFFFFF)))

(defn seed!
  "Seed — no-op for jax-js (functional PRNG, no global state)."
  [_key]
  nil)

(defn- extract-row
  "Extract row i from a 2D jax-js array as a 1D array."
  [arr i]
  (.take (np) (mx/ensure-ref arr) (.array (np) i) 0))

(defn split
  "Split a key into two independent sub-keys. Returns [k1 k2]."
  [key]
  (let [ks (.split (rng) key)
        k1 (extract-row (.-ref ks) 0)
        k2 (extract-row ks 1)]
    [k1 k2]))

(defn split-n
  "Split a key into n independent sub-keys. Returns vector of n keys."
  [key n]
  (let [ks (.split (rng) key n)]
    (mapv (fn [i]
            (extract-row (if (< i (dec n)) (.-ref ks) ks) i))
          (range n))))

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
;; Key-based sampling (functional — no global PRNG state)
;; ---------------------------------------------------------------------------

(defn normal
  "Sample from standard normal using the given key."
  [key shape]
  (.normal (rng) key (clj->js shape)))

(defn uniform
  "Sample from uniform [0,1) using the given key."
  [key shape]
  (.uniform (rng) key (clj->js shape)))

(defn bernoulli
  "Sample Bernoulli(p) using the given key."
  [key p shape]
  (.bernoulli (rng) key p (clj->js shape)))

(defn categorical
  "Sample from categorical distribution (log-probabilities) using the given key."
  [key logits]
  (.categorical (rng) key logits))

(defn randint
  "Sample random integers in [lo, hi) using the given key.
   Implemented via uniform sampling since jax-js may not have randint."
  [key lo hi shape]
  (let [u (uniform key shape)
        range-size (- hi lo)]
    (mx/add (mx/scalar lo mx/int32)
            (mx/astype (mx/multiply u (mx/scalar range-size)) mx/int32))))

(defn gumbel
  "Sample from standard Gumbel distribution using the given key."
  [key shape]
  (.gumbel (rng) key (clj->js shape)))

(defn laplace
  "Sample from standard Laplace distribution using the given key."
  [key shape]
  (.laplace (rng) key (clj->js shape)))

(defn truncated-normal
  "Sample from truncated normal distribution using the given key.
   Values are clipped to [lower, upper].
   Implemented via rejection: sample normal, clip to bounds."
  [key lower upper shape]
  (let [samples (normal key shape)]
    (mx/clip samples (mx/ensure-array lower) (mx/ensure-array upper))))

(defn multivariate-normal
  "Sample from multivariate normal N(mean, cov) using the given key."
  ([key mean cov]
   (.multivariateNormal (rng) key
     (if (mx/array? mean) mean (mx/array mean))
     (if (mx/array? cov)  cov  (mx/array cov))))
  ([key mean cov shape]
   (.multivariateNormal (rng) key
     (if (mx/array? mean) mean (mx/array mean))
     (if (mx/array? cov)  cov  (mx/array cov))
     (clj->js shape))))

(defn permutation
  "Return a random permutation of integers [0, n) or shuffle array along axis.
   Implemented via argsort of uniform random values."
  ([key n]
   (let [u (uniform key [n])]
     (mx/argsort u)))
  ([key arr axis]
   (let [n (nth (mx/shape arr) axis)
         u (uniform key [n])
         indices (mx/argsort u)]
     (mx/take-idx arr indices axis))))
