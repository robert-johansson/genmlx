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
   Returns [N] int32 MLX array of ancestor indices.
   GPU-accelerated: uses cumsum + broadcasting (no JS loops).
   Note: creates [N,N] temporary matrix, so O(N^2) memory.
   Fine for N up to ~20K. For larger N, would need searchsorted."
  [log-weights n key]
  (let [log-probs (mx/subtract log-weights (mx/logsumexp log-weights))
        probs     (mx/exp log-probs)
        ;; Cumulative sum on GPU: [N] array
        cdf       (mx/cumsum probs)
        ;; Systematic thresholds: u0 + j/N for j=0..N-1
        u0        (mx/divide (rng/uniform (rng/ensure-key key) [1])
                             (mx/scalar n))
        thresholds (mx/add u0 (mx/divide (mx/arange 0 n 1)
                                          (mx/scalar (float n))))
        ;; For each threshold j, find smallest i where cdf[i] >= threshold[j].
        ;; Equivalent: count how many cdf values are strictly less than threshold.
        ;; cdf [N,1] < thresholds [1,N] => [N,N] bool matrix
        ;; sum along axis 0 gives the index for each threshold.
        lt      (mx/less (mx/reshape cdf [n 1])
                         (mx/reshape thresholds [1 n]))
        indices (mx/sum lt 0)
        ;; Clamp to [0, N-1] for safety
        indices (mx/minimum indices (mx/scalar (dec n)))]
    (.astype indices mx/int32)))

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
      (if (and (mx/array? v) (pos? (count (mx/shape v))))
        (cm/->Value (reindex-value v indices))
        choice-map))  ;; scalar constraints stay unchanged
    (instance? cm/Node choice-map)
    (cm/->Node (into {} (map (fn [[k sub]]
                               [k (reindex-choicemap sub indices)])
                             (cm/-submaps choice-map))))
    :else choice-map))

(defn reindex-state
  "Reindex a structured state (plain ClojureScript map or MLX array) by ancestor indices.
   Walks the structure recursively: maps have their values reindexed,
   MLX arrays are reindexed via take-idx, other values pass through.
   Used by SMC unfold where state is {:logit-succ [N], :neg-bias [N], ...}."
  [state indices]
  (cond
    (mx/array? state)
    (mx/take-idx state indices)

    (map? state)
    (into {} (map (fn [[k v]] [k (reindex-state v indices)]) state))

    (vector? state)
    (mapv #(reindex-state % indices) state)

    :else state))

(defn resample-vtrace
  "Resample a VectorizedTrace using systematic resampling.
   Returns a new VectorizedTrace with reindexed choices, uniform weights."
  [vtrace key]
  (let [{:keys [weight n-particles choices score]} vtrace
        indices (systematic-resample-indices weight n-particles key)]
    (let [new-choices (reindex-choicemap choices indices)
          new-score (mx/take-idx score indices)
          new-weight (mx/zeros [n-particles])]
      (mx/materialize! new-score new-weight)
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
    (let [cv (cm/get-value current)]
      (if (and (mx/array? cv) (pos? (count (mx/shape cv))))
        (cm/->Value (mx/where mask (cm/get-value proposed) cv))
        current))  ;; scalar constraints stay unchanged

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
        _         (mx/materialize! probs)
        probs-clj (mx/->clj probs)]
    (/ 1.0 (reduce + (map #(* % %) probs-clj)))))
