(ns genmlx.mlx.random
  "Functional PRNG key management for GenMLX.
   Every sample consumes a key — no hidden PRNG state.
   Keys are MLX arrays that split deterministically."
  (:require [genmlx.mlx :as mx]))

;; ---------------------------------------------------------------------------
;; Bridge and MxArray references for PRNG ops
;; ---------------------------------------------------------------------------

(defonce ^:private bridge (js/require (str (.cwd js/process) "/src/genmlx/llm/bridge.js")))
(defonce ^:private M (.-MxArray (js/require "@mlx-node/core")))

;; ---------------------------------------------------------------------------
;; Key management
;; ---------------------------------------------------------------------------

(defn fresh-key
  "Create a fresh random key from an optional integer seed."
  ([]    (.randomKey bridge (js/Math.floor (* (js/Math.random) 2147483647))))
  ([seed] (.randomKey bridge seed)))

(defn key->seed
  "Derive a non-negative integer seed from a PRNG key array.
   Combines both uint32 elements via XOR, then masks to 31 bits."
  [key]
  (mx/eval! key)
  (let [arr (mx/->clj key)]
    (bit-and (bit-xor (int (nth arr 0)) (int (nth arr 1)))
             0x7FFFFFFF)))

(defn seed!
  "Seed the global MLX PRNG state from a key array.
   Note: mlx-node's key-based sampling doesn't use global state,
   but this is kept for compatibility."
  [key]
  ;; mlx-node doesn't have a global seed function — no-op
  nil)

(defn split
  "Split a key into two independent sub-keys. Returns [k1 k2]."
  [key]
  (let [ks (.randomSplit bridge key)]
    [(aget ks 0) (aget ks 1)]))

(defn split-n
  "Split a key into n independent sub-keys. Returns vector of n keys."
  [key n]
  (let [ks (.randomSplitN bridge key n)]
    ;; splitN returns [n, 2] array — extract each row
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
;; Key-based sampling (functional — no global PRNG state)
;; ---------------------------------------------------------------------------

(defn normal
  "Sample from standard normal using the given key."
  [key shape]
  (.keyNormal bridge key (clj->js shape) nil))

(defn uniform
  "Sample from uniform [0,1) using the given key."
  [key shape]
  (.keyUniform bridge key (clj->js shape) nil nil nil))

(defn bernoulli
  "Sample Bernoulli(p) using the given key."
  [key p shape]
  (.keyBernoulli bridge key p (clj->js shape)))

(defn categorical
  "Sample from categorical distribution (log-probabilities) using the given key."
  [key logits]
  (.keyCategorical bridge key logits nil))

(defn randint
  "Sample random integers in [lo, hi) using the given key."
  [key lo hi shape]
  (.keyRandint bridge key lo hi (clj->js shape) nil))

(defn gumbel
  "Sample from standard Gumbel distribution using the given key."
  [key shape]
  (.keyGumbel bridge key (clj->js shape) nil))

(defn laplace
  "Sample from standard Laplace distribution using the given key."
  [key shape]
  (.keyLaplace bridge key (clj->js shape) nil))

(defn truncated-normal
  "Sample from truncated normal distribution using the given key.
   Values are clipped to [lower, upper]."
  [key lower upper shape]
  (.keyTruncatedNormal bridge key
    (mx/ensure-array lower) (mx/ensure-array upper)
    (clj->js shape) nil))

(defn multivariate-normal
  "Sample from multivariate normal N(mean, cov) using the given key.
   mean: [k] array, cov: [k k] positive definite array."
  ([key mean cov]
   (.keyMultivariateNormal bridge key
     (if (mx/array? mean) mean (mx/array mean))
     (if (mx/array? cov) cov (mx/array cov))
     #js [] nil))
  ([key mean cov shape]
   (.keyMultivariateNormal bridge key
     (if (mx/array? mean) mean (mx/array mean))
     (if (mx/array? cov) cov (mx/array cov))
     (clj->js shape) nil)))

(defn permutation
  "Return a random permutation of integers [0, n).
   Note: mlx-node permutation not yet implemented — falls back to
   Fisher-Yates shuffle using key-based randint."
  ([key n]
   ;; Simple implementation: generate random indices and argsort
   (let [rand-vals (uniform key [n])]
     (.argsort rand-vals nil)))
  ([key arr axis]
   ;; Shuffle array along axis by permuting indices
   (let [n (nth (mx/shape arr) axis)
         perm (permutation key n)]
     (mx/take-idx arr perm axis))))
