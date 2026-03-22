(ns genmlx.inference.util
  "Shared utilities for inference algorithms.
   Eliminates duplicated weight-normalization, score-function construction,
   parameter extraction, and accept/reject logic."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.compiled-ops :as cops]
            [genmlx.tensor-trace :as tt]))

(defn materialize-weights
  "Evaluate a vector of MLX log-weight scalars and return them as a single
   MLX 1-D array.  Uses mx/stack to combine in one MLX call instead of
   realizing each weight individually."
  [log-weights]
  (let [stacked (mx/stack (vec log-weights))]
    (mx/materialize! stacked)
    stacked))

(defn normalize-log-weights
  "Given a vector of MLX log-weight scalars, return
   {:log-probs <MLX array>, :probs <clj vector of doubles>}
   after log-softmax normalization."
  [log-weights]
  (let [w-arr (materialize-weights log-weights)
        log-probs (mx/subtract w-arr (mx/logsumexp w-arr))
        _ (mx/materialize! log-probs)
        probs (mx/->clj (mx/exp log-probs))]
    {:log-probs log-probs :probs probs}))

(defn compute-param-layout
  "Compute the flatten/unflatten layout for addresses that may hold array-valued choices.
   Returns {:layout [{:addr :shape :offset :size} ...] :total-size int :array-valued? bool}."
  [trace addresses]
  (loop [addrs addresses, offset 0, layout [], any-array? false]
    (if (empty? addrs)
      {:layout layout :total-size offset :array-valued? any-array?}
      (let [addr (first addrs)
            v (cm/get-choice (:choices trace) [addr])
            sh (mx/shape v)
            arr? (pos? (count sh))
            size (if arr? (reduce * sh) 1)]
        (recur (rest addrs) (+ offset size)
               (conj layout {:addr addr :shape sh :offset offset :size size})
               (or any-array? arr?))))))

(defn make-score-fn
  "Build a score function from a model + observations + addresses.
   Returns a fn: (params-array) -> MLX scalar log-weight.
   Supports both scalar and array-valued choices at each address.
   If layout is provided (from compute-param-layout), uses it to unflatten
   the 1-D params array into original shapes. Otherwise assumes all scalar."
  ([model args observations addresses]
   (make-score-fn model args observations addresses nil))
  ([model args observations addresses layout]
   (let [model (dyn/auto-key model)]
     (if (and layout (:array-valued? layout))
       ;; Array-valued path: unflatten params into original shapes
       (let [entries (:layout layout)]
         (fn [params]
           (let [cm (reduce
                     (fn [cm {:keys [addr shape offset size]}]
                       (let [v (if (= size 1)
                                 (mx/index params offset)
                                 (mx/reshape (mx/slice params offset (+ offset size)) shape))]
                         (cm/set-choice cm [addr] v)))
                     observations
                     entries)]
             (:weight (p/generate model args cm)))))
       ;; Scalar-only path (original, unchanged)
       (let [indexed-addrs (mapv vector (range) addresses)]
         (fn [params]
           (let [cm (reduce
                     (fn [cm [i addr]]
                       (cm/set-choice cm [addr] (mx/index params i)))
                     observations
                     indexed-addrs)]
             (:weight (p/generate model args cm)))))))))

(defn make-batched-score-fn
  "Build a batched score function via shape-based batching.
   Takes [N,D] params, passes [N]-shaped constraints to a single p/generate call,
   and returns [N]-shaped weights. No mx/vmap — uses MLX broadcasting instead.
   Not wrapped in compile-fn, so it can be nested inside mx/grad and mx/compile-fn."
  [model args observations addresses]
  (let [model (dyn/auto-key model)
        indexed-addrs (mapv vector (range) addresses)
        idx-scalars (mapv #(mx/scalar % mx/int32) (range (count addresses)))]
    (fn [params]
      (let [params-t (mx/transpose params)
            cm (reduce
                (fn [cm [i addr]]
                   ;; take-idx with axis=0 extracts row i of [D,N] → [N]-shaped
                  (cm/set-choice cm [addr]
                                 (mx/take-idx params-t (nth idx-scalars i) 0)))
                observations
                indexed-addrs)]
        (:weight (p/generate model args cm))))))

(defn make-vectorized-score-fn
  "Build a compiled vectorized score function for N parallel chains.
   Returns a compiled fn: (params [N,D]) -> [N]-shaped MLX log-weight array."
  [model args observations addresses]
  (mx/compile-fn (make-batched-score-fn model args observations addresses)))

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
   Returns an MLX 1-D array. Handles both scalar and array-valued choices:
   scalar choices contribute 1 element, array choices are flattened."
  ([trace addresses]
   (extract-params trace addresses nil))
  ([trace addresses layout]
   (if (and layout (:array-valued? layout))
     ;; Array-valued path: flatten all choices into single 1-D array
     (let [parts (mapv (fn [{:keys [addr size]}]
                         (let [v (cm/get-choice (:choices trace) [addr])]
                           (mx/eval! v)
                           (if (= size 1)
                             (mx/reshape v [1])
                             (mx/reshape v [-1]))))
                       (:layout layout))]
       (mx/concatenate parts))
     ;; Scalar-only path (original)
     (mx/array (mapv #(let [v (cm/get-choice (:choices trace) [%])]
                        (mx/realize v))
                     addresses)))))

(defn- make-differentiable-vectorized-score-fn
  "Like make-vectorized-score-fn but uses differentiable column extraction.
   The standard vectorized-score-fn uses transpose+index which doesn't
   differentiate properly in MLX. This version uses matmul with one-hot
   selectors, which is fully differentiable.
   Returns fn: [N,D] -> [N]-shaped scores."
  [model args observations addresses]
  (let [model (dyn/auto-key model)
        d (count addresses)
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
   :grad-fn (mx/compile-fn (make-vectorized-grad-score model args observations addresses))})

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
  (let [model (dyn/auto-key model)]
    (mx/stack
     (mapv (fn [_]
             (let [{:keys [trace]} (p/generate model args observations)]
               (extract-params trace addresses)))
           (range n-chains)))))

(defn systematic-resample
  "Systematic resampling of particles. Returns vector of indices.
   Uses functional PRNG key (falls back to fresh-key if nil)."
  [log-weights n key]
  (let [{:keys [probs]} (normalize-log-weights log-weights)
        rk (rng/ensure-key key)
        u (/ (mx/realize (rng/uniform rk [])) n)]
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

(defn- walk-value-arrays
  "Recursively find all MLX arrays in a value that may be a scalar, vector, or map."
  [v arrays]
  (cond
    (mx/array? v) (vswap! arrays conj! v)
    (map? v) (doseq [[_ val] v] (walk-value-arrays val arrays))
    (sequential? v) (doseq [item v] (walk-value-arrays item arrays))
    :else nil))

(defn collect-trace-arrays
  "Collect all MLX arrays from a trace for bulk evaluation.
   Recursively walks retval to find arrays inside maps/vectors (e.g., Unfold state)."
  [trace]
  (let [arrays (volatile! (transient []))]
    (letfn [(walk [cm]
              (cond
                (nil? cm) nil
                (cm/has-value? cm)
                (let [v (cm/get-value cm)]
                  (when (mx/array? v) (vswap! arrays conj! v)))
                (instance? cm/Node cm)
                (doseq [[_ sub] (cm/-submaps cm)]
                  (walk sub))
                :else nil))]
      (when-let [choices (:choices trace)]
        (walk choices)))
    (when-let [s (:score trace)]
      (when (mx/array? s) (vswap! arrays conj! s)))
    (when-let [r (:retval trace)]
      (walk-value-arrays r arrays))
    (persistent! @arrays)))

(defn materialize-state
  "Evaluate ALL arrays in an inference state to materialize computation graphs
   and detach graph nodes, making intermediate arrays GC-eligible.
   Handles both MLX arrays (param vectors) and Trace records (walks choices)."
  [state]
  (if (mx/array? state)
    (mx/materialize! state)
    (let [arrays (collect-trace-arrays state)]
      (when (seq arrays)
        (apply mx/materialize! arrays)))))

(defn accept-mh?
  "Metropolis-Hastings accept/reject decision.
   `log-accept` is a JS number (the log acceptance ratio).
   Uses functional PRNG key (falls back to fresh-key if nil)."
  ([log-accept]
   (accept-mh? log-accept nil))
  ([log-accept key]
   (or (>= log-accept 0)
       (let [key (rng/ensure-key key)
             u (mx/realize (rng/uniform key []))]
         (< (js/Math.log u) log-accept)))))

(defn collect-choicemap-arrays
  "Collect all MLX arrays from a choicemap (e.g., observations).
   Returns a JS Set of arrays (by identity) for fast lookup."
  [choicemap]
  (let [seen (js/Set.)]
    (letfn [(walk [cm]
              (cond
                (nil? cm) nil
                (cm/has-value? cm)
                (let [v (cm/get-value cm)]
                  (when (mx/array? v) (.add seen v)))
                (instance? cm/Node cm)
                (doseq [[_ sub] (cm/-submaps cm)]
                  (walk sub))
                :else nil))]
      (walk choicemap))
    seen))

(defn dispose-trace
  "Dispose all MLX arrays in a trace (or collection of traces), freeing Metal
   buffers immediately. Handles shared arrays safely by deduplicating first.

   Optional second arg `preserve` can be:
   - a choicemap (e.g., the observation choicemap) — arrays in it are skipped
   - a JS Set of arrays to skip
   - nil (dispose everything, original behavior)"
  ([trace-or-traces]
   (dispose-trace trace-or-traces nil))
  ([trace-or-traces preserve]
   (let [traces (if (sequential? trace-or-traces) trace-or-traces [trace-or-traces])
         skip (cond
                (nil? preserve) nil
                (instance? js/Set preserve) preserve
                ;; Assume it's a choicemap
                :else (collect-choicemap-arrays preserve))
         seen (js/Set.)]
     (doseq [t traces
             a (collect-trace-arrays t)]
       (when-not (.has seen a)
         (.add seen a)))
     (.forEach seen (fn [a]
                      (when-not (and skip (.has skip a))
                        (mx/dispose! a)))))))

(defn tidy-step
  "Run step-fn inside mx/tidy, preserving all arrays in the returned state.
   step-fn must return {:state new-state :accepted? bool}.
   State can be a Trace (with :choices/:score/:retval) or a plain MLX array.
   mx/tidy disposes all intermediate arrays not in the return value, but
   cannot walk CLJS data structures. We work around this by:
   1. Running the step inside tidy
   2. Evaluating all state arrays (detaches computation graphs from intermediates)
   3. Returning those arrays as a JS array for tidy to preserve"
  [step-fn state key]
  (mx/tidy-run
   #(step-fn state key)
   (fn [result]
     (let [s (:state result)]
       (if (mx/array? s) [s] (collect-trace-arrays s))))))

;; ===========================================================================
;; Level 3.5: Conjugate-aware score functions
;; ===========================================================================

(defn get-eliminated-addresses
  "Return the set of addresses analytically eliminated by L3 auto-handlers.
   Returns nil if the model has no analytical plan."
  [model]
  (get-in (:schema model) [:analytical-plan :rewrite-result :eliminated]))

(defn filter-addresses
  "Remove analytically eliminated addresses from an address list.
   If no addresses are eliminated, returns the original list unchanged."
  [addresses eliminated]
  (if (seq eliminated)
    (vec (remove eliminated addresses))
    addresses))

(defn make-conjugate-aware-score-fn
  "Build a score function that excludes analytically marginalized parameters.
   The auto-handlers in p/generate handle eliminated addresses automatically,
   so they need not be in the parameter vector.

   Returns {:score-fn      (fn [params] -> scalar)
            :addresses     filtered address list
            :eliminated    set of eliminated addresses (may be nil)
            :reduced?      true if dimension was reduced}."
  [model args observations addresses]
  (let [eliminated (get-eliminated-addresses model)
        filtered (filter-addresses addresses eliminated)
        reduced? (< (count filtered) (count addresses))]
    {:score-fn (make-score-fn model args observations filtered)
     :addresses filtered
     :eliminated eliminated
     :reduced? reduced?}))

;; ===========================================================================
;; Level 2: Tensor-native score function (tries compiled, falls back to GFI)
;; ===========================================================================

(defn make-compiled-generate-score-fn
  "Build a score function from compiled generate.
   All sites (latent + observed) are constrained — no sampling, no key needed.
   Returns {:score-fn (fn [K-tensor] -> scalar) :latent-index {addr -> int}}
   or nil if model has no compiled-generate."
  [model args observations addresses]
  (when-let [compiled-gen (:compiled-generate (:schema model))]
    (let [latent-index (into {} (map-indexed (fn [i a] [a i]) addresses))
          mlx-args (vec args)
          indexed-addrs (mapv vector (range) addresses)]
      {:score-fn
       (fn [params]
         (let [constraints (reduce
                            (fn [cm [i addr]]
                              (cm/set-choice cm [addr] (mx/index params i)))
                            observations
                            indexed-addrs)]
           (:weight (compiled-gen (rng/fresh-key) mlx-args constraints))))
       :latent-index latent-index})))

(defn make-tensor-score-fn
  "Try to build a tensor-native score function from model schema.
   Falls back through: tensor-native → compiled-generate → GFI handler.

   Returns {:score-fn (fn [K-tensor] -> scalar)
            :latent-index {addr -> int}
            :tensor-native? bool
            :compiled-generate? bool}

   When tensor-native? is true, score-fn takes a [K]-shaped latent tensor.
   When false, score-fn takes a [D]-shaped params array (same as make-score-fn)
   and latent-index is built from the addresses."
  [model args observations addresses]
  (let [schema (:schema model)
        source (:source model)]
    (if-let [result (cops/make-tensor-score-with-index
                     schema source (vec args) observations)]
      (assoc result :tensor-native? true :compiled-generate? false)
      ;; Try compiled-generate before falling back to GFI handler
      (if-let [cg-result (make-compiled-generate-score-fn
                          model args observations addresses)]
        (assoc cg-result :tensor-native? false :compiled-generate? true)
        ;; Fall back to GFI-based score
        (let [gfi-fn (make-score-fn model args observations addresses)
              latent-index (into {} (map-indexed (fn [i a] [a i]) addresses))]
          {:score-fn gfi-fn
           :latent-index latent-index
           :tensor-native? false
           :compiled-generate? false})))))

(defn extract-params-by-index
  "Extract parameter values from a trace using latent-index ordering.
   Returns [K] MLX array matching the latent-index mapping."
  [trace latent-index]
  (let [choices (:choices trace)
        pairs (sort-by val latent-index)]
    (mx/stack (mapv (fn [[addr _]]
                      (cm/get-value (cm/get-submap choices addr)))
                    pairs))))

(defn prepare-mcmc-score
  "Prepare score function + init params for compiled MCMC.
   Tries tensor-native score first (bypasses GFI), falls back to GFI-based.
   Automatically filters out analytically eliminated addresses (L3.5).

   Returns {:score-fn    (fn [D-tensor] -> scalar)
            :init-params [D] MLX array
            :n-params    int
            :tensor-native? bool}

   The returned score-fn and init-params always use the same indexing,
   whether tensor-native or GFI-based."
  [model args observations addresses trace]
  (let [;; L3.5: filter out conjugate prior addresses
        eliminated (get-eliminated-addresses model)
        addresses (filter-addresses addresses eliminated)
        {:keys [score-fn latent-index tensor-native?]}
        (make-tensor-score-fn model args observations addresses)]
    (if tensor-native?
      ;; Tensor-native: params packed in dep-order (latent-index)
      {:score-fn score-fn
       :init-params (extract-params-by-index trace latent-index)
       :n-params (count latent-index)
       :tensor-native? true
       :latent-index latent-index}
      ;; GFI fallback: use existing layout machinery
      (let [layout (compute-param-layout trace addresses)]
        {:score-fn score-fn
         :init-params (extract-params trace addresses layout)
         :n-params (:total-size layout)
         :tensor-native? false
         :latent-index latent-index}))))
