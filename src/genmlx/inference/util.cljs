(ns genmlx.inference.util
  "Shared utilities for inference algorithms.
   Eliminates duplicated weight-normalization, score-function construction,
   parameter extraction, and accept/reject logic."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]))

(defn materialize-weights
  "Evaluate a vector of MLX log-weight scalars and return them as a single
   MLX 1-D array.  Uses mx/stack to combine in one MLX call instead of
   realizing each weight individually."
  [log-weights]
  (let [stacked (mx/stack (vec log-weights))]
    (mx/eval! stacked)
    stacked))

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
  "Build a score function from a model + observations + addresses.
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

(defn make-vectorized-score-fn
  "Build a vectorized score function for N parallel chains.
   Returns a fn: (params [N,D]) -> [N]-shaped MLX log-weight array."
  [model args observations addresses]
  (let [indexed-addrs (mapv vector (range) addresses)
        idx-scalars (mapv #(mx/scalar % mx/int32) (range (count addresses)))]
    (mx/compile-fn
      (fn [params]
        (let [params-t (mx/transpose params)
              cm (reduce
                   (fn [cm [i addr]]
                     ;; take-idx with axis=0 extracts row i of [D,N] → [N]-shaped
                     (cm/set-choice cm [addr]
                       (mx/take-idx params-t (nth idx-scalars i) 0)))
                   observations
                   indexed-addrs)]
          (:weight (p/generate model args cm)))))))

(defn make-compiled-score-fn
  "Build a compiled score function from a model + observations + addresses.
   Returns a compiled fn: (params-array) -> MLX scalar log-weight."
  [model args observations addresses]
  (mx/compile-fn (make-score-fn model args observations addresses)))

(defn make-compiled-grad-score
  "Build a compiled gradient of the score function.
   Returns a compiled fn: (params-array) -> MLX gradient array."
  [model args observations addresses]
  (mx/compile-fn (mx/grad (make-score-fn model args observations addresses))))

(defn make-compiled-val-grad
  "Build a compiled value-and-grad of the score function.
   Returns a compiled fn: (params-array) -> [score grad]."
  [model args observations addresses]
  (mx/compile-fn (mx/value-and-grad (make-score-fn model args observations addresses))))

(defn extract-params
  "Extract parameter values from a trace at the given addresses.
   Returns an MLX 1-D array of realized scalar values."
  [trace addresses]
  (mx/array (mapv #(let [v (cm/get-choice (:choices trace) [%])]
                     (mx/realize v))
                  addresses)))

(defn- make-differentiable-vectorized-score-fn
  "Like make-vectorized-score-fn but uses differentiable column extraction.
   The standard vectorized-score-fn uses transpose+index which doesn't
   differentiate properly in MLX. This version uses matmul with one-hot
   selectors, which is fully differentiable.
   Returns fn: [N,D] -> [N]-shaped scores."
  [model args observations addresses]
  (let [d (count addresses)
        indexed-addrs (mapv vector (range) addresses)
        ;; Pre-build one-hot column selectors [D,1] for each address
        one-hots (mapv (fn [i]
                         (let [v (vec (repeat d 0.0))]
                           (mx/array (assoc v i 1.0))))
                       (range d))]
    (fn [params]
      (let [;; Extract column i via dot product: params [N,D] . one-hot [D] -> [N]
            cm (reduce
                 (fn [cm [i addr]]
                   ;; matmul [N,D] x [D,1] -> [N,1], squeeze -> [N]
                   (cm/set-choice cm [addr]
                     (mx/squeeze (mx/matmul params (mx/reshape (nth one-hots i) [d 1])))))
                 observations
                 indexed-addrs)]
        (:weight (p/generate model args cm))))))

(defn make-vectorized-grad-score
  "Per-chain gradients for N parallel chains via the sum trick.
   Since score_n depends only on params[n,:], grad(sum(scores))[n,:] = grad(score_n).
   Returns fn: [N,D] -> [N,D]."
  [model args observations addresses]
  (let [diff-score-fn (make-differentiable-vectorized-score-fn model args observations addresses)
        summed-fn (fn [params] (mx/sum (diff-score-fn params)))]
    (mx/grad summed-fn)))

(defn make-compiled-vectorized-score-and-grad
  "Vectorized score fn + vectorized gradient fn, both compiled.
   Returns {:score-fn ([N,D]->[N]), :grad-fn ([N,D]->[N,D])}."
  [model args observations addresses]
  {:score-fn (make-vectorized-score-fn model args observations addresses)
   :grad-fn  (mx/compile-fn (make-vectorized-grad-score model args observations addresses))})

(defn make-compiled-vectorized-val-grad
  "Compiled vectorized value-and-grad via sum trick.
   Returns fn: [N,D] -> [scalar, [N,D]] where scalar = sum(scores).
   Per-chain scores can be obtained separately via score-fn."
  [model args observations addresses]
  (let [diff-score-fn (make-differentiable-vectorized-score-fn model args observations addresses)
        summed-fn (fn [params] (mx/sum (diff-score-fn params)))]
    (mx/compile-fn (mx/value-and-grad summed-fn))))

(defn init-vectorized-params
  "Initialize [N,D] parameter matrix from N independent generates."
  [model args observations addresses n-chains]
  (mx/stack
    (mapv (fn [_]
            (let [{:keys [trace]} (p/generate model args observations)]
              (extract-params trace addresses)))
          (range n-chains))))

(defn systematic-resample
  "Systematic resampling of particles. Returns vector of indices.
   Optional `key` uses functional PRNG; nil falls back to js/Math.random."
  [log-weights n key]
  (let [{:keys [probs]} (normalize-log-weights log-weights)
        u (if key
            (/ (mx/realize (rng/uniform key [])) n)
            (/ (js/Math.random) n))]
    (loop [i 0, cumsum 0.0, j 0, indices (transient [])]
      (if (>= j n)
        (persistent! indices)
        (let [threshold (+ u (/ j n))
              cumsum' (+ cumsum (nth probs i))]
          (if (>= cumsum' threshold)
            (recur i cumsum (inc j) (conj! indices i))
            (recur (inc i) cumsum' j indices)))))))

(defn compute-ess
  "Compute effective sample size from log-weights."
  [log-weights]
  (let [{:keys [probs]} (normalize-log-weights log-weights)]
    (/ 1.0 (reduce + (map #(* % %) probs)))))

(defn eval-state!
  "Evaluate key arrays in an inference state to materialize computation graphs.
   Handles both MLX arrays (param vectors) and Trace records."
  [state]
  (if (mx/array? state)
    (mx/eval! state)
    (let [score  (:score state)
          retval (:retval state)]
      (cond
        (and score (mx/array? retval)) (mx/eval! score retval)
        score                          (mx/eval! score)
        (mx/array? retval)             (mx/eval! retval)
        :else nil))))

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
