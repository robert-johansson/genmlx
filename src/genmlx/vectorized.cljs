(ns genmlx.vectorized
  "Vectorized trace and utilities for batched inference.
   A VectorizedTrace holds [N]-shaped arrays at each choice site,
   enabling N particles to be processed in a single model execution."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]))

;; ---------------------------------------------------------------------------
;; VectorizedTrace record
;; ---------------------------------------------------------------------------

(defrecord VectorizedTrace [gen-fn args choices score weight n-particles retval])

;; ---------------------------------------------------------------------------
;; Systematic resampling (returns indices as MLX int32 array)
;; ---------------------------------------------------------------------------

(defn systematic-resample-indices
  "Systematic resampling from [N]-shaped log-weights.
   Returns [N] int32 MLX array of ancestor indices."
  [log-weights n key]
  (let [log-probs (mx/subtract log-weights (mx/logsumexp log-weights))
        probs     (mx/exp log-probs)
        _         (mx/eval! probs)
        probs-clj (mx/->clj probs)
        ;; Single uniform offset
        u0        (if key
                    (/ (mx/realize (mx/random-uniform [1])) n)
                    (/ (js/Math.random) n))
        indices   (loop [i 0 cumsum 0.0 j 0 acc (transient [])]
                    (if (>= j n)
                      (persistent! acc)
                      (let [threshold (+ u0 (/ j n))
                            cumsum'   (+ cumsum (nth probs-clj i))]
                        (if (>= cumsum' threshold)
                          (recur i cumsum (inc j) (conj! acc i))
                          (recur (inc i) cumsum' j acc)))))]
    (mx/array indices mx/int32)))

(defn systematic-resample-indices-gpu
  "GPU-native systematic resampling from [N]-shaped log-weights.
   Pure MLX operations — no CLJS loop. O(N^2) memory, suitable for N ≤ 10,000.
   Returns [N] int32 MLX array of ancestor indices."
  [log-weights n key]
  (let [probs    (mx/exp (mx/subtract log-weights (mx/logsumexp log-weights)))
        cumprobs (mx/cumsum probs)
        ;; Single uniform offset in [0, 1/N)
        u0       (mx/divide (rng/uniform (rng/ensure-key key) [1])
                            (mx/scalar n))
        ;; Thresholds: u0 + i/N for i in [0, N)
        thresholds (mx/add u0 (mx/divide (mx/arange n) (mx/scalar n)))
        ;; Broadcasting: cumprobs [1,N] >= thresholds [N,1] → [N,N] bool matrix
        ;; Each row i has true where cumprobs[j] >= threshold[i]
        ;; argmax on axis 1 gives first true index per threshold → ancestor index
        comparison (mx/greater-equal
                     (mx/reshape cumprobs [1 n])
                     (mx/reshape thresholds [n 1]))
        indices (mx/argmax comparison 1)]
    (mx/eval! indices)
    indices))

;; ---------------------------------------------------------------------------
;; Reindex choicemap leaves
;; ---------------------------------------------------------------------------

(defn- reindex-value
  "Reindex a single [N]-shaped array by ancestor indices."
  [arr indices]
  (mx/take-idx arr indices))

(defn- reindex-choicemap
  "Recursively reindex all leaves of a choicemap using ancestor indices."
  [choice-map indices]
  (cond
    (nil? choice-map) choice-map
    (cm/has-value? choice-map)
    (let [v (cm/get-value choice-map)]
      (if (mx/array? v)
        (cm/->Value (reindex-value v indices))
        choice-map))
    (instance? cm/Node choice-map)
    (cm/->Node (into {} (map (fn [[k sub]]
                               [k (reindex-choicemap sub indices)])
                             (cm/-submaps choice-map))))
    :else choice-map))

(defn resample-vtrace
  "Resample a VectorizedTrace using systematic resampling.
   Returns a new VectorizedTrace with reindexed choices, uniform weights."
  [vtrace key]
  (let [{:keys [weight n-particles choices score]} vtrace
        indices (systematic-resample-indices weight n-particles key)]
    (let [new-choices (reindex-choicemap choices indices)
          new-score (mx/take-idx score indices)
          new-weight (mx/zeros [n-particles])]
      (mx/eval! new-score new-weight)
      (assoc vtrace
             :choices new-choices
             :score   new-score
             :weight  new-weight))))

;; ---------------------------------------------------------------------------
;; Diagnostics
;; ---------------------------------------------------------------------------

(defn vtrace-log-ml-estimate
  "Log marginal likelihood estimate from a VectorizedTrace's [N] weights."
  [vtrace]
  (mx/subtract (mx/logsumexp (:weight vtrace))
               (mx/scalar (js/Math.log (:n-particles vtrace)))))

;; ---------------------------------------------------------------------------
;; Merge two VectorizedTraces by boolean mask (for per-particle MH accept/reject)
;; ---------------------------------------------------------------------------

(defn- merge-choicemap-by-mask
  "Recursively merge two identically-structured choicemaps using an [N] boolean mask.
   Where mask=true takes from proposed, where false keeps current."
  [current proposed mask]
  (cond
    (cm/has-value? current)
    (cm/->Value (mx/where mask (cm/get-value proposed) (cm/get-value current)))

    (instance? cm/Node current)
    (cm/->Node (into {} (map (fn [[k sub]]
                               [k (merge-choicemap-by-mask
                                    sub (cm/-get-submap proposed k) mask)])
                             (cm/-submaps current))))
    :else current))

(defn merge-vtraces-by-mask
  "Merge two VectorizedTraces per-particle using an [N] boolean mask.
   Where mask=true takes from proposed, where false keeps current.
   Preserves the current vtrace's :weight (rejuvenation is weight-preserving)."
  [current proposed mask]
  (assoc current
    :choices (merge-choicemap-by-mask (:choices current) (:choices proposed) mask)
    :score   (mx/where mask (:score proposed) (:score current))))

;; ---------------------------------------------------------------------------
;; Diagnostics
;; ---------------------------------------------------------------------------

(defn vtrace-ess
  "Effective sample size from a VectorizedTrace's [N] weights."
  [vtrace]
  (let [w (:weight vtrace)
        log-probs (mx/subtract w (mx/logsumexp w))
        probs     (mx/exp log-probs)
        _         (mx/eval! probs)
        probs-clj (mx/->clj probs)]
    (/ 1.0 (reduce + (map #(* % %) probs-clj)))))
