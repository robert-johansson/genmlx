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

(defn truncated-normal
  "Sample from truncated normal distribution using the given key.
   Values are clipped to [lower, upper]."
  [key lower upper shape]
  (.truncatedNormal mx/random
    (mx/ensure-array lower) (mx/ensure-array upper)
    (clj->js shape) nil key))

(defonce ^:private cpu-stream (.newStream mx/core (.-cpu mx/core)))

(defn multivariate-normal
  "Sample from multivariate normal N(mean, cov) using the given key.
   mean: [k] array, cov: [k k] positive definite array.
   Uses CPU stream because MLX's internal SVD/Cholesky requires it.
   Note: for high dimensions (k>10), manual Cholesky+matmul is faster.
   Returns [k] array."
  ([key mean cov]
   (.multivariateNormal mx/random
     (if (mx/array? mean) mean (mx/array mean))
     (if (mx/array? cov) cov (mx/array cov))
     #js [] nil key cpu-stream))
  ([key mean cov shape]
   (.multivariateNormal mx/random
     (if (mx/array? mean) mean (mx/array mean))
     (if (mx/array? cov) cov (mx/array cov))
     (clj->js shape) nil key cpu-stream)))

(defn permutation
  "Return a random permutation of integers [0, n) or shuffle array along axis."
  ([key n]
   (.permutation mx/random n key))
  ([key arr axis]
   (.permutation mx/random arr axis key)))
