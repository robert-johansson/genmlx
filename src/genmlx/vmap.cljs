(ns genmlx.vmap
  "Vmap combinator: applies a kernel GF independently to N elements,
   storing choices with [N]-shaped leaf arrays instead of integer-indexed
   sub-choicemaps. Composable as a GFI combinator inside gen bodies."
  (:require [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.selection :as sel]
            [genmlx.handler :as h]
            [genmlx.runtime :as rt]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(def ^:private mlx-arr?
  "Check if x is an MLX array."
  mx/array?)

(defn- axis-size-of
  "Get the leading dimension size of an arg."
  [a]
  (cond
    (nil? a) nil
    (mlx-arr? a) (first (mx/shape a))
    (sequential? a) (count a)
    :else nil))

(defn- resolve-axis-size
  "Determine N from args and in-axes.
   in-axes=nil: all args batched, N from first arg's leading dim.
   in-axes=[0 nil ...]: N from first arg with axis=0.
   All nil axes or empty args: use explicit axis-size."
  [args in-axes axis-size]
  (cond
    ;; Explicit axis-size always wins if provided
    axis-size axis-size

    ;; No in-axes: all args batched, N from first arg
    (nil? in-axes)
    (let [a (first args)]
      (or (axis-size-of a)
          (throw (ex-info "vmap: cannot determine axis-size from arg" {:arg a}))))

    ;; in-axes provided: find first non-nil axis
    :else
    (let [idx (first (keep-indexed (fn [i ax] (when ax i)) in-axes))]
      (if idx
        (let [a (nth args idx)]
          (or (axis-size-of a)
              (throw (ex-info "vmap: cannot determine axis-size" {:arg a}))))
        (throw (ex-info "vmap: all in-axes are nil and no axis-size provided" {}))))))

(defn- index-arg
  "Index into an arg at position i."
  [a i]
  (cond
    (mlx-arr? a) (mx/index a i)
    (sequential? a) (nth a i)
    :else a))

(defn- extract-element-args
  "Slice args for element i given in-axes.
   axis=0: mx/index for arrays, nth for sequences.
   axis=nil: pass through unchanged.
   Empty args are returned as-is."
  [args in-axes i]
  (cond
    (empty? args)
    args

    (nil? in-axes)
    (mapv #(index-arg % i) args)

    :else
    (mapv (fn [a ax] (if (nil? ax) a (index-arg a i)))
          args in-axes)))

(defn- batched-args?
  "Check if args are suitable for batched fast path.
   All axis=0 args must be MLX arrays (not sequences)."
  [args in-axes]
  (cond
    (empty? args)
    true

    (nil? in-axes)
    (every? mlx-arr? args)

    :else
    (every? (fn [[a ax]] (or (nil? ax) (mlx-arr? a)))
            (map vector args in-axes))))

(defn- scalar-leaf?
  "Check if all leaves in a choicemap are scalar (not [N]-shaped)."
  [cm]
  (cond
    (= cm cm/EMPTY) true
    (cm/has-value? cm)
    (let [v (cm/get-value cm)]
      (or (not (mlx-arr? v)) (= [] (mx/shape v))))
    (instance? cm/Node cm)
    (every? (fn [[_ sub]] (scalar-leaf? sub)) (:m cm))
    :else true))

(defn- scalar-constraints?
  "Check if constraints are scalar (can be broadcast to all elements)."
  [constraints]
  (or (= constraints cm/EMPTY) (scalar-leaf? constraints)))

(defn- extract-element-selection
  "Extract the selection for element i.
   Integer-indexed hierarchical selections (keys are integers) are per-element.
   Keyword-indexed or non-hierarchical selections are shared across all elements."
  [selection i]
  (if (instance? sel/Hierarchical selection)
    (let [m (:m selection)]
      (if (some integer? (keys m))
        ;; Integer-indexed → per-element
        (get m i sel/none)
        ;; Keyword-indexed → shared
        selection))
    ;; Non-hierarchical (SelectAddrs, all, none, etc.) → shared
    selection))

(defn- scalar-value-leaf?
  "Check if a single leaf value is scalar (not [N]-shaped)."
  [v]
  (or (not (mlx-arr? v)) (= [] (mx/shape v))))

(defn- stack-choices
  "Stack N choicemaps into one with [N]-shaped leaves.
   All N choicemaps must have the same address structure."
  [cms]
  (cm/stack-choicemaps cms mx/stack))

(defn- unstack-choices
  "Split a choicemap with [N]-shaped leaves into N scalar choicemaps.
   Scalar leaves are replicated to all N elements."
  [cm n]
  (cm/unstack-choicemap cm n mx/index scalar-value-leaf?))

(defn- stack-retvals
  "Stack N return values into a single value."
  [retvals]
  (cond
    (every? mx/array? retvals) (mx/stack retvals)
    (every? number? retvals) (mx/array retvals)
    :else (vec retvals)))

(defn- mx-sum-by
  "Sum (f r) over results, starting from a scalar-0.0 accumulator."
  [f results]
  (reduce (fn [acc r] (mx/add acc (f r))) (mx/scalar 0.0) results))

(defn- capture-nested-meta
  "Per-element vector of each inner trace's OWN combinator metadata (nil entries
   for plain-kernel elements; nil overall when no element carries any). Stored on
   a vmap trace as ::nested-element-meta and re-attached by element-trace so that a
   reconstructed inner trace — when the kernel is ITSELF a combinator — carries
   its own ::element-scores. The general retained-only regenerate/update weight is
   project-based and NOT linear in the element :score, so without the inner
   metadata a nested (vmap-of-vmap / combinator-of-combinator) general-path
   regenerate mis-weights by the whole inner old score (genmlx-nt0c). Selecting
   ::nested-element-meta too makes the propagation recursive (3+ levels)."
  [traces]
  (let [v (mapv #(not-empty (select-keys (meta %)
                              [::element-scores ::n ::nested-element-meta ::batched?]))
                traces)]
    (when (some some? v) v)))

(defn- element-trace
  "Reconstruct the kernel trace for element i from stored per-element state.
   Score falls back to scalar 0.0 when no element-scores were recorded. nested-meta
   (when present) re-attaches element i's own combinator metadata so a nested
   kernel's regenerate/update/project sees its ::element-scores (genmlx-nt0c)."
  ([kernel args choices element-scores i]
   (element-trace kernel args choices element-scores i nil))
  ([kernel args choices element-scores i nested-meta]
   (let [t (tr/make-trace
            {:gen-fn kernel :args args
             :choices choices :retval nil
             :score (if element-scores (nth element-scores i) (mx/scalar 0.0))})
         m (when nested-meta (nth nested-meta i))]
     (if m (vary-meta t merge m) t))))

(defn- kernel-update
  "Update a kernel trace. Throws when the kernel does not implement IUpdate:
   the old generate fallback RESAMPLED every unconstrained site — not update
   semantics — and reported the score delta as the weight (genmlx-v740)."
  [kernel trace constraints]
  (if (satisfies? p/IUpdate kernel)
    (p/update kernel trace constraints)
    (throw (ex-info "vmap kernel does not implement IUpdate — no semantically honest fallback exists"
                    {:kernel-type (type kernel)}))))

(defn- kernel-regenerate
  "Regenerate a kernel trace. Throws when the kernel does not implement
   IRegenerate: the old simulate fallback resampled EVERY site regardless of
   the selection (genmlx-v740)."
  [kernel trace selection]
  (if (satisfies? p/IRegenerate kernel)
    (p/regenerate kernel trace selection)
    (throw (ex-info "vmap kernel does not implement IRegenerate — no semantically honest fallback exists"
                    {:kernel-type (type kernel)}))))

;; ---------------------------------------------------------------------------
;; VmapCombinator
;; ---------------------------------------------------------------------------

(defrecord VmapCombinator [kernel in-axes axis-size]
  p/IGenerativeFunction
  (simulate [this args]
    (let [n (resolve-axis-size args in-axes axis-size)]
      (if (and (:body-fn kernel) (batched-args? args in-axes))
        ;; Fast path: run kernel body once with batched handler
        (let [key (rng/fresh-key)
              result (rt/run-handler h/batched-simulate-transition
                                     {:choices cm/EMPTY :score (mx/scalar 0.0)
                                      :key key :batch-size n :batched? true
                                      ;; Trained params live on the kernel's
                                      ;; metadata; omitting them made the fast
                                      ;; path silently use param DEFAULTS
                                      ;; (genmlx-v740).
                                      :param-store (:genmlx.dynamic/param-store
                                                    (meta kernel))}
                                     (fn [rt] (apply (:body-fn kernel) rt args)))
              ;; score is [N]-shaped, sum for total
              total-score (mx/sum (:score result))
              element-scores (mapv #(mx/index (:score result) %) (range n))]
          (with-meta
            (tr/make-trace {:gen-fn this :args args
                            :choices (:choices result) :retval (:retval result)
                            :score total-score})
            {::element-scores element-scores ::n n ::batched? true}))
        ;; Slow path: loop N times
        (let [results (mapv (fn [i]
                              (p/simulate kernel (extract-element-args args in-axes i)))
                            (range n))
              choices (stack-choices (mapv :choices results))
              retvals (stack-retvals (mapv :retval results))
              score (mx-sum-by :score results)
              element-scores (mapv :score results)]
          (with-meta
            (tr/make-trace {:gen-fn this :args args
                            :choices choices :retval retvals :score score})
            {::element-scores element-scores
             ::nested-element-meta (capture-nested-meta results)
             ::n n})))))

  p/IGenerate
  (generate [this args constraints]
    (let [n (resolve-axis-size args in-axes axis-size)
          scalar? (scalar-constraints? constraints)]
      (if (and (:body-fn kernel) (= constraints cm/EMPTY) (batched-args? args in-axes))
        ;; Fast path: batched generate with no constraints (all sites simulated)
        (let [key (rng/fresh-key)
              result (rt/run-handler h/batched-generate-transition
                                     {:choices cm/EMPTY :score (mx/scalar 0.0)
                                      :weight (mx/scalar 0.0)
                                      :key key :constraints cm/EMPTY
                                      :batch-size n :batched? true
                                      ;; See simulate fast path (genmlx-v740).
                                      :param-store (:genmlx.dynamic/param-store
                                                    (meta kernel))}
                                     (fn [rt] (apply (:body-fn kernel) rt args)))
              total-score (mx/sum (:score result))
              total-weight (mx/sum (:weight result))
              element-scores (mapv #(mx/index (:score result) %) (range n))]
          {:trace (with-meta
                    (tr/make-trace {:gen-fn this :args args
                                    :choices (:choices result) :retval (:retval result)
                                    :score total-score})
                    {::element-scores element-scores ::n n ::batched? true})
           :weight total-weight})
        ;; Slow path: loop N times
        (let [per-constraints (when-not scalar? (unstack-choices constraints n))
              results (mapv (fn [i]
                              (p/generate kernel
                                          (extract-element-args args in-axes i)
                                          (if scalar? constraints (nth per-constraints i))))
                            (range n))
              choices (stack-choices (mapv (comp :choices :trace) results))
              retvals (stack-retvals (mapv (comp :retval :trace) results))
              score (mx-sum-by (comp :score :trace) results)
              weight (mx-sum-by :weight results)
              element-scores (mapv (comp :score :trace) results)]
          {:trace (with-meta
                    (tr/make-trace {:gen-fn this :args args
                                    :choices choices :retval retvals :score score})
                    {::element-scores element-scores
                     ::nested-element-meta (capture-nested-meta (mapv :trace results))
                     ::n n})
           :weight weight}))))

  p/IUpdate
  (update [this trace constraints]
    (let [n (or (::n (meta trace))
                (resolve-axis-size (:args trace) in-axes axis-size))
          old-per-choices (unstack-choices (:choices trace) n)
          scalar-c? (scalar-constraints? constraints)
          new-per-constraints (when-not scalar-c? (unstack-choices constraints n))
          old-element-scores (::element-scores (meta trace))
          old-nested-meta (::nested-element-meta (meta trace))
          results (mapv (fn [i]
                          (let [elem-args (extract-element-args (:args trace) in-axes i)
                                old-trace (element-trace kernel elem-args
                                                         (nth old-per-choices i)
                                                         old-element-scores i old-nested-meta)]
                            (kernel-update kernel old-trace
                                           (if scalar-c? constraints (nth new-per-constraints i)))))
                        (range n))
          choices (stack-choices (mapv (comp :choices :trace) results))
          retvals (stack-retvals (mapv (comp :retval :trace) results))
          score (mx-sum-by (comp :score :trace) results)
          ;; Thesis update weights are additive across elements, but each
          ;; element weight was computed against the constructed old score
          ;; (recorded element score, or 0 without metadata). Re-base against
          ;; the true total old score so both cases stay exact:
          ;; W = Σ w_i + Σ constructed_old_i - old_total.
          constructed-old (if old-element-scores
                            (reduce mx/add (mx/scalar 0.0) old-element-scores)
                            (mx/scalar 0.0))
          weight (mx/subtract (mx/add (mx-sum-by :weight results) constructed-old)
                              (:score trace))
          discard (let [discards (mapv :discard results)]
                    (if (every? #(= % cm/EMPTY) discards)
                      cm/EMPTY
                      (stack-choices discards)))
          element-scores (mapv (comp :score :trace) results)]
      {:trace (with-meta
                (tr/make-trace {:gen-fn this :args (:args trace)
                                :choices choices :retval retvals :score score})
                {::element-scores element-scores
                 ::nested-element-meta (capture-nested-meta (mapv :trace results))
                 ::n n})
       :weight weight
       :discard discard}))

  p/IRegenerate
  (regenerate [this trace selection]
    (let [n (or (::n (meta trace))
                (resolve-axis-size (:args trace) in-axes axis-size))
          old-per-choices (unstack-choices (:choices trace) n)
          old-element-scores (::element-scores (meta trace))
          old-nested-meta (::nested-element-meta (meta trace))
          results (mapv (fn [i]
                          (let [elem-args (extract-element-args (:args trace) in-axes i)
                                old-trace (element-trace kernel elem-args
                                                         (nth old-per-choices i)
                                                         old-element-scores i old-nested-meta)]
                            (kernel-regenerate kernel old-trace
                                               (extract-element-selection selection i))))
                        (range n))
          choices (stack-choices (mapv (comp :choices :trace) results))
          retvals (stack-retvals (mapv (comp :retval :trace) results))
          score (mx-sum-by (comp :score :trace) results)
          ;; Regenerate weights are additive across independent elements, but each
          ;; element weight was computed against the constructed old score
          ;; (recorded element score, or 0 without metadata). Re-base against the
          ;; true total old score: W = Σ wᵢ + Σ constructed_oldᵢ − old_total.
          ;; NOTE (genmlx-nt0c): ::element-scores is now propagated through nested
          ;; reconstructions (::nested-element-meta + element-trace), so every
          ;; reconstructed kernel old-trace — including the inner trace of a
          ;; vmap-of-vmap — carries its own metadata and the re-base term is 0 for
          ;; BOTH fast and general kernel paths. The expression is kept verbatim as
          ;; a 0-valued safety net: a fast-eligible kernel still degrades exactly
          ;; (−old_total compensates) if a trace's metadata is ever externally
          ;; stripped; only a GENERAL retained-only kernel on an externally-stripped
          ;; trace (no longer produced anywhere) would mis-weight.
          constructed-old (if old-element-scores
                            (reduce mx/add (mx/scalar 0.0) old-element-scores)
                            (mx/scalar 0.0))
          weight (mx/subtract (mx/add (mx-sum-by :weight results) constructed-old)
                              (:score trace))
          element-scores (mapv (comp :score :trace) results)]
      {:trace (with-meta
                (tr/make-trace {:gen-fn this :args (:args trace)
                                :choices choices :retval retvals :score score})
                {::element-scores element-scores
                 ::nested-element-meta (capture-nested-meta (mapv :trace results))
                 ::n n})
       :weight weight}))

  p/IAssess
  (assess [this args choices]
    (let [n (resolve-axis-size args in-axes axis-size)
          scalar? (scalar-constraints? choices)
          per-choices (when-not scalar? (unstack-choices choices n))
          results (mapv (fn [i]
                          (p/assess kernel
                                    (extract-element-args args in-axes i)
                                    (if scalar? choices (nth per-choices i))))
                        (range n))
          weight (mx-sum-by :weight results)]
      {:retval (stack-retvals (mapv :retval results))
       :weight weight}))

  p/IPropose
  (propose [this args]
    (let [n (resolve-axis-size args in-axes axis-size)
          results (mapv (fn [i]
                          (p/propose kernel (extract-element-args args in-axes i)))
                        (range n))
          choices (stack-choices (mapv :choices results))
          weight (mx-sum-by :weight results)]
      {:choices choices
       :weight weight
       :retval (stack-retvals (mapv :retval results))})))

;; ---------------------------------------------------------------------------
;; IProject via extend-type
;; ---------------------------------------------------------------------------

(extend-type VmapCombinator
  p/IProject
  (project [this trace selection]
    (let [n (or (::n (meta trace))
                (resolve-axis-size (:args trace) (:in-axes this) (:axis-size this)))
          old-per-choices (unstack-choices (:choices trace) n)
          old-element-scores (::element-scores (meta trace))
          old-nested-meta (::nested-element-meta (meta trace))]
      (reduce
       (fn [acc i]
         (let [elem-args (extract-element-args (:args trace) (:in-axes this) i)
               elem-trace (element-trace (:kernel this) elem-args
                                         (nth old-per-choices i)
                                         old-element-scores i old-nested-meta)]
           (mx/add acc (p/project (:kernel this) elem-trace
                                  (extract-element-selection selection i)))))
       (mx/scalar 0.0)
       (range n)))))

;; ---------------------------------------------------------------------------
;; Constructors
;; ---------------------------------------------------------------------------

(defn vmap-gf
  "Create a Vmap combinator from a kernel generative function.
   Options:
     :in-axes   - vector of axis specs (0 or nil) per arg. nil = broadcast.
                  Default: nil (all args batched along axis 0).
     :axis-size - explicit N (required when all in-axes are nil or args empty)."
  [kernel & {:keys [in-axes axis-size]}]
  (->VmapCombinator kernel in-axes axis-size))

(defn repeat-gf
  "Create a Vmap combinator that runs kernel n times with no batched args.
   Shorthand for (vmap-gf kernel :axis-size n)."
  [kernel n]
  (->VmapCombinator kernel nil n))
