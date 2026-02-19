(ns genmlx.vectorized
  "Vectorized trace and utilities for batched inference.
   A VectorizedTrace holds [N]-shaped arrays at each choice site,
   enabling N particles to be processed in a single model execution."
  (:require [genmlx.mlx :as mx]
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
    (assoc vtrace
           :choices (reindex-choicemap choices indices)
           :score   (mx/take-idx score indices)
           :weight  (mx/zeros [n-particles]))))

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
        _         (mx/eval! probs)
        probs-clj (mx/->clj probs)]
    (/ 1.0 (reduce + (map #(* % %) probs-clj)))))
