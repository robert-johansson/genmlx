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

(defn- resample-from-u0
  "Systematic resampling core: [N] log-weights + pre-scaled u0 ([1] MLX array
   already divided by n) → [N] int32 ancestor indices.
   GPU-accelerated: uses cumsum + searchsorted (O(N) memory and compute)."
  [log-weights n u0-scaled]
  (let [log-probs (mx/subtract log-weights (mx/logsumexp log-weights))
        probs     (mx/exp log-probs)
        cdf       (mx/cumsum probs)
        thresholds (mx/add u0-scaled (mx/divide (mx/arange 0 n 1)
                                                (mx/scalar (float n))))
        indices   (mx/searchsorted cdf thresholds)
        indices   (mx/minimum indices (mx/scalar (dec n)))]
    (.astype indices mx/int32)))

(defn systematic-resample-indices
  "Systematic resampling from [N]-shaped log-weights.
   Returns [N] int32 MLX array of ancestor indices."
  [log-weights n key]
  (resample-from-u0 log-weights n
                    (mx/divide (rng/uniform (rng/ensure-key key) [1])
                               (mx/scalar n))))

(defn systematic-resample-indices-deterministic
  "Systematic resampling with pre-generated uniform u0.
   Like systematic-resample-indices but takes u0 [1]-shaped MLX array
   instead of a PRNG key. Suitable for use inside mx/compile-fn where
   random generation must happen outside the compiled function."
  [log-weights n u0]
  (resample-from-u0 log-weights n (mx/divide u0 (mx/scalar n))))

;; ---------------------------------------------------------------------------
;; Reindex choicemap leaves
;; ---------------------------------------------------------------------------

(defn- walk-choicemap
  "Recursively walk a choicemap, dispatching on has-value?/Node/else.
   leaf-fn is called on a [N]-shaped array leaf value and returns its
   replacement; scalar leaves (failing the array+shape guard) pass through.
   child-fn is called with [k sub] for each Node entry to produce the
   recursed child."
  [choice-map leaf-fn child-fn]
  (cond
    (cm/has-value? choice-map)
    (let [v (cm/get-value choice-map)]
      (if (and (mx/array? v) (pos? (count (mx/shape v))))
        (cm/->Value (leaf-fn v))
        choice-map))  ;; scalar constraints stay unchanged

    (instance? cm/Node choice-map)
    (cm/->Node (reduce-kv (fn [acc k sub] (assoc acc k (child-fn k sub)))
                          {} (:m choice-map)))

    :else choice-map))

(defn- reindex-choicemap
  "Recursively reindex all leaves of a choicemap using ancestor indices."
  [choice-map indices]
  (when (some? choice-map)
    (walk-choicemap choice-map
                    (fn [v] (mx/take-idx v indices))
                    (fn [_ sub] (reindex-choicemap sub indices)))))

(defn reindex-state
  "Reindex a structured state (plain ClojureScript map or MLX array) by ancestor indices.
   Walks the structure recursively: maps have their values reindexed,
   MLX arrays are reindexed via take-idx, other values pass through.
   Used by SMC unfold where state is {:logit-succ [N], :neg-bias [N], ...}."
  [state indices]
  (cond
    (mx/array? state)
    ;; 0-d arrays (e.g. a scalar constraint passed through as the model's
    ;; return value) are particle-invariant — permutation is identity, and
    ;; a gather on a 0-d array is an MLX shape error that aborts the
    ;; process through NAPI.
    (if (pos? (count (mx/shape state)))
      (mx/take-idx state indices)
      state)

    (map? state)
    (update-vals state #(reindex-state % indices))

    (vector? state)
    (mapv #(reindex-state % indices) state)

    :else state))

(defn resample-vtrace
  "Resample a VectorizedTrace using systematic resampling.
   Returns a new VectorizedTrace with reindexed choices, retval, and
   uniform weights. The retval MUST follow the same ancestor permutation
   as the choices — leaving it unpermuted silently pairs particle i's
   choices with ancestor j's return value (genmlx-v740)."
  [vtrace key]
  (let [{:keys [weight n-particles choices score retval]} vtrace
        indices (systematic-resample-indices weight n-particles key)
        new-choices (reindex-choicemap choices indices)
        new-score (mx/take-idx score indices)
        new-weight (mx/zeros [n-particles])]
    (mx/materialize! new-score new-weight)
    (assoc vtrace
           :choices new-choices
           :score   new-score
           :weight  new-weight
           :retval  (reindex-state retval indices))))

;; ---------------------------------------------------------------------------
;; Diagnostics
;; ---------------------------------------------------------------------------

(defn vtrace-log-ml-estimate
  "Log marginal likelihood estimate from a VectorizedTrace's [N] weights."
  [vtrace]
  (mx/subtract (mx/logsumexp (:weight vtrace))
               (mx/scalar (js/Math.log (:n-particles vtrace)))))

(defn vtrace-ess
  "Effective sample size from a VectorizedTrace's [N] weights."
  [vtrace]
  (let [w (:weight vtrace)
        log-probs (mx/subtract w (mx/logsumexp w))
        probs     (mx/exp log-probs)
        _         (mx/materialize! probs)
        probs-clj (mx/->clj probs)]
    (/ 1.0 (transduce (map #(* % %)) + probs-clj))))

;; ---------------------------------------------------------------------------
;; Merge two VectorizedTraces by boolean mask (for per-particle MH accept/reject)
;; ---------------------------------------------------------------------------

(defn- merge-choicemap-by-mask
  "Recursively merge two identically-structured choicemaps using an [N] boolean mask.
   Where mask=true takes from proposed, where false keeps current."
  [current proposed mask]
  (walk-choicemap current
                  (fn [cv]
                    (let [pv (cm/get-value proposed)]
                      (mx/where mask pv cv)))
                  (fn [k sub] (merge-choicemap-by-mask
                                sub (cm/-get-submap proposed k) mask))))

(defn- merge-state-by-mask
  "Per-particle merge of two structured retvals by an [N] boolean mask:
   arrays via mx/where, maps/vectors recursively, anything else keeps
   current. Mirrors reindex-state's shape handling."
  [cur prop mask]
  (cond
    (and (mx/array? cur) (mx/array? prop))
    (mx/where mask prop cur)

    (and (map? cur) (map? prop))
    (into {} (map (fn [[k v]] [k (merge-state-by-mask v (get prop k) mask)])) cur)

    (and (vector? cur) (vector? prop))
    (mapv #(merge-state-by-mask %1 %2 mask) cur prop)

    :else cur))

(defn merge-vtraces-by-mask
  "Merge two VectorizedTraces per-particle using an [N] boolean mask.
   Where mask=true takes from proposed, where false keeps current.
   Preserves the current vtrace's :weight (rejuvenation is weight-preserving).
   The retval merges with the same mask — leaving it unmerged silently
   pairs accepted choices with rejected return values (genmlx-v740)."
  [current proposed mask]
  (assoc current
    :choices (merge-choicemap-by-mask (:choices current) (:choices proposed) mask)
    :score   (mx/where mask (:score proposed) (:score current))
    :retval  (merge-state-by-mask (:retval current) (:retval proposed) mask)))
