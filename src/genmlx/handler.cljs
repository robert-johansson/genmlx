(ns genmlx.handler
  "Pure state transitions for GenMLX handler execution.

   ## Architecture

   Each GFI operation (simulate, generate, update, regenerate, project) is
   implemented as a **pure state transition**:

       (fn [state addr dist] -> [value state'])

   These transitions are used by runtime.cljs, which provides the
   orchestration (volatile! + closures) and passes a runtime object
   to gen body functions.

   ## Handler state schemas

   Each handler mode uses a plain map with a specific set of keys.  These
   correspond to the handler state H(σ, τ) in the λ_MLX formalization:

   | Mode       | Keys                                                          |
   |------------|---------------------------------------------------------------|
   | simulate   | :key :choices :score :executor                                |
   | generate   | :key :choices :score :weight :constraints :executor           |
   | update     | :key :choices :score :weight :constraints :old-choices        |
   |            |   :discard :executor                                          |
   | regenerate | :key :choices :score :weight :old-choices :selection :executor |
   | project    | :key :choices :score :weight :old-choices :selection          |
   |            |   :constraints :executor                                      |

   Batched variants add :batch-size (int) and :batched? (true).
   All other keys and semantics are identical — MLX broadcasting handles
   the shape difference between scalar and [N]-shaped values.

   ## Purity

   This module is purely functional. No mutable state, no dynamic vars,
   no side effects. All mutation is isolated in runtime.cljs."
  (:require [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.mlx.constants :refer [ZERO]]
            [genmlx.selection :as sel]
            [genmlx.dist.core :as dc]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]))

;; ---------------------------------------------------------------------------
;; Pure state transitions
;; ---------------------------------------------------------------------------

(defn simulate-transition
  "Pure: given state, addr, dist -> [value state']."
  [state addr dist]
  (let [[k1 k2] (rng/split (:key state))
        value (dc/dist-sample dist k2)
        lp (dc/dist-log-prob dist value)]
    [value (-> state
               (assoc :key k1)
               (update :choices cm/set-value addr value)
               (update :score mx/add lp))]))

(defn generate-transition
  "Pure: if constrained at addr, use constraint; otherwise simulate."
  [state addr dist]
  (let [constraint (cm/get-submap (:constraints state) addr)]
    (if (cm/has-value? constraint)
      (let [value (cm/get-value constraint)
            lp (dc/dist-log-prob dist value)]
        [value (-> state
                   (update :choices cm/set-value addr value)
                   (update :score mx/add lp)
                   (update :weight mx/add lp))])
      (simulate-transition state addr dist))))

(defn assess-transition
  "Pure: all addresses must be constrained. Throws if not."
  [state addr dist]
  (let [constraint (cm/get-submap (:constraints state) addr)]
    (if (cm/has-value? constraint)
      (let [value (cm/get-value constraint)
            lp (dc/dist-log-prob dist value)]
        [value (-> state
                   (update :choices cm/set-value addr value)
                   (update :score mx/add lp)
                   (update :weight mx/add lp))])
      (throw (ex-info (str "assess: address " addr " not found in provided choices. "
                           "All addresses must be constrained for assess.")
                      {:addr addr})))))

(defn update-transition
  "Pure: use new constraint, keep old, or sample fresh.

   :weight accumulates the NON-FRESH score: the log-prob of every
   constrained or retained site under the new parameters. Freshly
   sampled sites (new addresses) contribute only to :score. The caller
   computes the thesis update weight as non-fresh-score minus the
   recorded old trace score: fresh choices are drawn from the internal
   proposal and cancel; removed and overwritten old choices are charged
   via the old score."
  [state addr dist]
  (let [constraint (cm/get-submap (:constraints state) addr)
        old-choice (cm/get-submap (:old-choices state) addr)]
    (cond
      ;; New constraint provided
      (cm/has-value? constraint)
      (let [new-val (cm/get-value constraint)
            new-lp (dc/dist-log-prob dist new-val)
            old-val (when (cm/has-value? old-choice) (cm/get-value old-choice))]
        [new-val (-> state
                     (update :choices cm/set-value addr new-val)
                     (update :score mx/add new-lp)
                     (update :weight mx/add new-lp)
                     (cond->
                      old-val (update :discard cm/set-value addr old-val)))])

      ;; Keep old value
      (cm/has-value? old-choice)
      (let [val (cm/get-value old-choice)
            lp (dc/dist-log-prob dist val)]
        [val (-> state
                 (update :choices cm/set-value addr val)
                 (update :score mx/add lp)
                 (update :weight mx/add lp))])

      ;; New address: sample fresh
      :else (simulate-transition state addr dist))))

(defn regenerate-transition
  "Pure: resample selected addresses, keep unselected."
  [state addr dist]
  (let [sel (:selection state)
        old-choice (cm/get-submap (:old-choices state) addr)]
    (if (and sel (sel/selected? sel addr))
      ;; Selected: resample, compute weight adjustment
      (let [[k1 k2] (rng/split (:key state))
            new-val (dc/dist-sample dist k2)
            new-lp (dc/dist-log-prob dist new-val)
            old-val (when (cm/has-value? old-choice) (cm/get-value old-choice))
            old-lp (if old-val (dc/dist-log-prob dist old-val) ZERO)]
        [new-val (-> state
                     (assoc :key k1)
                     (update :choices cm/set-value addr new-val)
                     (update :score mx/add new-lp)
                     (update :weight mx/add (mx/subtract new-lp old-lp)))])
      ;; Not selected: keep old
      (let [val (when (cm/has-value? old-choice)
                  (cm/get-value old-choice))]
        (when (nil? val)
          (throw (ex-info (str "regenerate: address " addr " not found in previous trace choices. "
                               "Cannot keep old value for an address that was never sampled.")
                          {:addr addr})))
        (let [lp (dc/dist-log-prob dist val)]
          [val (-> state
                   (update :choices cm/set-value addr val)
                   (update :score mx/add lp))])))))

(defn project-transition
  "Pure: replay old value, accumulate log-prob for selected addresses."
  [state addr dist]
  (let [old-choice (cm/get-submap (:old-choices state) addr)
        val (when (cm/has-value? old-choice)
              (cm/get-value old-choice))]
    (when (nil? val)
      (throw (ex-info (str "project: address not found in previous trace choices. "
                           "Cannot replay a value for an address that was never sampled.")
                      {})))
    (let [lp (dc/dist-log-prob dist val)
          sel (:selection state)]
      [val (-> state
               (update :choices cm/set-value addr val)
               (update :score mx/add lp)
               (cond-> (and sel (sel/selected? sel addr))
                 (update :weight mx/add lp)))])))

;; ---------------------------------------------------------------------------
;; Batched state transitions (vectorized: [N]-shaped values)
;; ---------------------------------------------------------------------------

(defn- check-batched-lp!
  "Batched-execution invariant: every trace site's log-prob is a per-particle
   scalar — shape [] or [batch-size]. A higher-rank (or wrong-sized) log-prob
   means the model broadcast this site's distribution parameters across the
   particle axis (e.g. a mis-shaped per-particle gather producing an [N,N]
   array instead of [N]). Unguarded, that silently corrupts the [N] score and
   weight into [N,N], surfacing only downstream as a NaN ESS / wrong log-ML.
   Fail loudly at the offending site instead. Shape is lazy-graph metadata, so
   this is a no-eval, O(1) check — safe to run per site in the batched hot path."
  [addr n lp]
  (let [sh   (mx/shape lp)
        rank (count sh)]
    (when (or (> rank 1)
              (and (= rank 1)
                   (not (or (== (first sh) 1) (== (first sh) n)))))
      (throw (ex-info (str "Batched log-prob at " (pr-str addr) " has shape "
                           (vec sh) " — expected [] or [" n "]. The model "
                           "broadcast this trace site's distribution parameters "
                           "across the particle axis (likely a mis-shaped "
                           "per-particle index/gather producing an [N,N] array).")
                      {:address addr :lp-shape (vec sh) :batch-size n})))))

(defn batched-simulate-transition
  "Pure: sample [N] values, accumulate [N]-shaped score."
  [state addr dist]
  (let [n (:batch-size state)
        [k1 k2] (rng/split (:key state))
        value (dc/dist-sample-n dist k2 n)
        lp (dc/dist-log-prob dist value)]
    (check-batched-lp! addr n lp)
    [value (-> state
               (assoc :key k1)
               (update :choices cm/set-value addr value)
               (update :score mx/add lp))]))

(defn batched-generate-transition
  "Pure: constrained sites use scalar observation (broadcasts into [N] score),
   unconstrained sites delegate to batched-simulate-transition."
  [state addr dist]
  (let [constraint (cm/get-submap (:constraints state) addr)]
    (if (cm/has-value? constraint)
      ;; Constrained: scalar observation, scalar log-prob → broadcasts into [N] score/weight.
      ;; ensure-array so the stored choice is an MxArray (a raw JS number is rejected by
      ;; the v0.31.2 binary on downstream array ops like mx/shape / value_and_grad).
      (let [value (mx/ensure-array (cm/get-value constraint))
            lp (dc/dist-log-prob dist value)]
        (check-batched-lp! addr (:batch-size state) lp)
        [value (-> state
                   (update :choices cm/set-value addr value)
                   (update :score mx/add lp)
                   (update :weight mx/add lp))])
      (batched-simulate-transition state addr dist))))

(defn batched-update-transition
  "Pure: batched update with [N]-shaped old values and scalar/[N]-shaped new
   constraints. Same :weight convention as update-transition: accumulate the
   non-fresh score (constrained + retained site log-probs); the caller
   subtracts the old [N]-shaped score to obtain the thesis update weight."
  [state addr dist]
  (let [n (:batch-size state)
        constraint (cm/get-submap (:constraints state) addr)
        old-choice (cm/get-submap (:old-choices state) addr)]
    (cond
      ;; New constraint provided
      (cm/has-value? constraint)
      (let [new-val (cm/get-value constraint)
            new-lp (dc/dist-log-prob dist new-val)
            old-val (when (cm/has-value? old-choice) (cm/get-value old-choice))]
        (check-batched-lp! addr n new-lp)
        [new-val (-> state
                     (update :choices cm/set-value addr new-val)
                     (update :score mx/add new-lp)
                     (update :weight mx/add new-lp)
                     (cond->
                      old-val (update :discard cm/set-value addr old-val)))])

      ;; Keep old [N]-shaped values
      (cm/has-value? old-choice)
      (let [val (cm/get-value old-choice)
            lp (dc/dist-log-prob dist val)]
        (check-batched-lp! addr n lp)
        [val (-> state
                 (update :choices cm/set-value addr val)
                 (update :score mx/add lp)
                 (update :weight mx/add lp))])

      ;; New address: sample [N] fresh values
      :else (batched-simulate-transition state addr dist))))

(defn batched-regenerate-transition
  "Pure: batched regenerate with [N]-shaped values."
  [state addr dist]
  (let [n (:batch-size state)
        sel (:selection state)
        old-choice (cm/get-submap (:old-choices state) addr)]
    (if (and sel (sel/selected? sel addr))
      ;; Selected: resample [N] values, compute [N]-shaped weight adjustment
      (let [[k1 k2] (rng/split (:key state))
            new-val (dc/dist-sample-n dist k2 n)
            new-lp (dc/dist-log-prob dist new-val)
            old-val (when (cm/has-value? old-choice) (cm/get-value old-choice))
            old-lp (if old-val (dc/dist-log-prob dist old-val) ZERO)]
        (check-batched-lp! addr n new-lp)
        [new-val (-> state
                     (assoc :key k1)
                     (update :choices cm/set-value addr new-val)
                     (update :score mx/add new-lp)
                     (update :weight mx/add (mx/subtract new-lp old-lp)))])
      ;; Not selected: keep old [N]-shaped values
      (let [val (cm/get-value old-choice)
            lp (dc/dist-log-prob dist val)]
        (check-batched-lp! addr n lp)
        [val (-> state
                 (update :choices cm/set-value addr val)
                 (update :score mx/add lp))]))))

;; ---------------------------------------------------------------------------
;; Pure helpers used by runtime.cljs
;; ---------------------------------------------------------------------------

(defn merge-sub-result
  "Pure: merge a sub-generative-function result into parent state."
  [state addr sub-result]
  (-> state
      (update :choices cm/set-submap addr (:choices sub-result))
      (update :score mx/add (:score sub-result))
      (update :splice-scores (fn [ss] (assoc (or ss {}) addr (:score sub-result))))
      (cond->
       (and (contains? state :weight) (:weight sub-result))
        (update :weight mx/add (:weight sub-result))

        (:discard sub-result)
        (update :discard cm/set-submap addr (:discard sub-result))

        (or (:splice-scores sub-result) (:nested-splice-scores sub-result))
        (update :nested-splice-scores
                (fn [nss]
                  (let [sub-meta (cond-> {}
                                  (:splice-scores sub-result)
                                  (assoc :splice-scores (:splice-scores sub-result))
                                  (:nested-splice-scores sub-result)
                                  (assoc :nested-splice-scores (:nested-splice-scores sub-result)))]
                    (assoc (or nss {}) addr sub-meta)))))))

(defn- mlx-arr-batched?
  "Check if x is an MLX array with at least 1 dimension."
  [x]
  (and (mx/array? x) (pos? (count (mx/shape x)))))

(defn- scalar-leaf-val?
  "Check if a value is scalar (0-d or not an MLX array)."
  [v]
  (or (not (mx/array? v))
      (= [] (mx/shape v))))

(defn- run-batched-particle
  "Run one particle's GFI op against `gf`, dispatching on which scoped state
   is present: selection => regenerate, old-choices => update, constraints =>
   generate, else simulate. Returns {:choices :score :weight :discard :retval}."
  [gf elem-args sub-selection old-i cons-i]
  (cond
    ;; Regenerate mode
    sub-selection
    (let [old-choices (or old-i cm/EMPTY)
          elem-trace (tr/make-trace
                      {:gen-fn gf :args elem-args
                       :choices old-choices :retval nil
                       :score (mx/scalar 0.0)})
          {:keys [trace weight]} (p/regenerate gf elem-trace sub-selection)]
      ;; The child returns its final MH weight w = ΔS - pr, computed against
      ;; the constructed old score of 0. The parent's batched regenerate
      ;; accumulator holds proposal ratios (its final weight is
      ;; ΔS_total - accumulator), so convert back: pr = S'_child - w.
      {:choices (:choices trace) :score (:score trace)
       :weight (mx/subtract (:score trace) weight) :retval (:retval trace)})

    ;; Update mode
    (and old-i (not= old-i cm/EMPTY))
    (let [c (or cons-i cm/EMPTY)
          elem-trace (tr/make-trace
                      {:gen-fn gf :args elem-args
                       :choices old-i :retval nil
                       :score (mx/scalar 0.0)})
          {:keys [trace weight discard]} (p/update gf elem-trace c)]
      ;; The child update weight is non-fresh-score minus its constructed
      ;; old score of 0, i.e. exactly the non-fresh-score contribution the
      ;; parent's batched update accumulator expects — merge as-is.
      {:choices (:choices trace) :score (:score trace)
       :weight weight :discard discard :retval (:retval trace)})

    ;; Generate with constraints
    (and cons-i (not= cons-i cm/EMPTY))
    (let [{:keys [trace weight]} (p/generate gf elem-args cons-i)]
      {:choices (:choices trace) :score (:score trace)
       :weight weight :retval (:retval trace)})

    ;; Simulate
    :else
    (let [trace (p/simulate gf elem-args)]
      {:choices (:choices trace) :score (:score trace)
       :retval (:retval trace)})))

(defn- stack-particle-results
  "Stack a vector of per-particle result maps back into a single [N]-batched
   sub-result {:choices :score :weight :discard :retval}."
  [results]
  {:choices (cm/stack-choicemaps (mapv :choices results) mx/stack)
   :score (mx/stack (mapv :score results))
   :weight (when (some :weight results)
             (mx/stack (mapv #(or (:weight %) (mx/scalar 0.0)) results)))
   :discard (when (some :discard results)
              (let [discards (mapv #(or (:discard %) cm/EMPTY) results)]
                (if (every? #(= % cm/EMPTY) discards)
                  cm/EMPTY
                  (cm/stack-choicemaps discards mx/stack))))
   :retval (let [rvs (mapv :retval results)]
             (if (every? mx/array? rvs) (mx/stack rvs) (vec rvs)))})

(defn combinator-batched-fallback
  "Fallback for splicing a non-DynamicGF (e.g. VmapCombinator) in batched mode.
   Unstacks [N]-particle state, runs combinator GFI N times, stacks results."
  [state addr gf args]
  (let [n (:batch-size state)
        [k1 k2] (rng/split (:key state))
        ;; Extract scoped state
        sub-constraints (cm/get-submap (:constraints state) addr)
        sub-old-choices (cm/get-submap (:old-choices state) addr)
        sub-selection (when-let [s (:selection state)]
                        (sel/get-subselection s addr))
        ;; Extract per-particle args: only unstack if leading dim == N (batched)
        extract-scalar-arg (fn [a i]
                             (if (and (mlx-arr-batched? a)
                                      (= (first (mx/shape a)) n))
                               (mx/index a i)
                               a))
        ;; Unstack old-choices if present
        per-old-choices (when (and sub-old-choices (not= sub-old-choices cm/EMPTY))
                          (cm/unstack-choicemap sub-old-choices n mx/index scalar-leaf-val?))
        ;; Unstack constraints if present and not scalar
        per-constraints (when (and sub-constraints (not= sub-constraints cm/EMPTY))
                          (if (every? (fn [[_ sub]]
                                        (or (not (cm/has-value? sub))
                                            (scalar-leaf-val? (cm/get-value sub))))
                                      (when (instance? cm/Node sub-constraints)
                                        (:m sub-constraints)))
                            ;; Scalar constraints: replicate
                            (vec (repeat n sub-constraints))
                            (cm/unstack-choicemap sub-constraints n mx/index scalar-leaf-val?)))
        ;; Run per-particle
        results (mapv
                 (fn [i]
                   (run-batched-particle
                    gf
                    (mapv #(extract-scalar-arg % i) args)
                    sub-selection
                    (when per-old-choices (nth per-old-choices i))
                    (when per-constraints (nth per-constraints i))))
                 (range n))
        ;; Stack results and merge into parent state
        sub-result (stack-particle-results results)
        state' (-> state
                   (assoc :key k1)
                   (merge-sub-result addr sub-result))]
    [state' (:retval sub-result)]))
