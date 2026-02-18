(ns genmlx.inference.util
  "Shared utilities for inference algorithms.
   Eliminates duplicated weight-normalization, score-function construction,
   parameter extraction, and accept/reject logic."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]))

(defn materialize-weights
  "Evaluate a vector of MLX log-weight scalars and return them as a single
   MLX 1-D array of JS doubles.  Replaces the repeated
   `(mx/array (mapv (fn [w] (mx/eval! w) (mx/item w)) ws))` pattern."
  [log-weights]
  (mx/array (mapv mx/realize log-weights)))

(defn normalize-log-weights
  "Given a vector of MLX log-weight scalars, return
   {:log-probs <MLX array>, :probs <clj vector of doubles>}
   after log-softmax normalization."
  [log-weights]
  (let [w-arr    (materialize-weights log-weights)
        log-probs (mx/subtract w-arr (mx/logsumexp w-arr))
        _         (mx/eval! log-probs)
        probs     (mx/->clj (mx/exp log-probs))]
    {:log-probs log-probs :probs probs}))

(defn make-score-fn
  "Build a compiled score function from a model + observations + addresses.
   Returns a fn: (params-array) -> MLX scalar log-weight."
  [model args observations addresses]
  (let [indexed-addrs (mapv vector (range) addresses)]
    (fn [params]
      (let [cm (reduce
                 (fn [cm [i addr]]
                   (cm/set-choice cm [addr] (mx/index params i)))
                 observations
                 indexed-addrs)]
        (:weight (p/generate model args cm))))))

(defn extract-params
  "Extract parameter values from a trace at the given addresses.
   Returns an MLX 1-D array of realized scalar values."
  [trace addresses]
  (mx/array (mapv #(let [v (cm/get-choice (tr/get-choices trace) [%])]
                     (mx/realize v))
                  addresses)))

(defn accept-mh?
  "Metropolis-Hastings accept/reject decision.
   `log-accept` is a JS number (the log acceptance ratio).
   Optional `key` uses the functional PRNG; nil falls back to js/Math.random."
  ([log-accept]
   (accept-mh? log-accept nil))
  ([log-accept key]
   (or (>= log-accept 0)
       (let [u (if key
                 (mx/realize (rng/uniform key []))
                 (js/Math.random))]
         (< (js/Math.log u) log-accept)))))
