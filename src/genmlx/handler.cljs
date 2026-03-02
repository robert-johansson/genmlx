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
            [genmlx.selection :as sel]
            [genmlx.dist.core :as dc]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]))

;; Cached zero constant for init states and fallback log-probs
(def ^:private ZERO (mx/scalar 0.0))

;; ---------------------------------------------------------------------------
;; Pure state transitions
;; ---------------------------------------------------------------------------

(defn simulate-transition
  "Pure: given state, addr, dist -> [value state']."
  [state addr dist]
  (let [[k1 k2] (rng/split (:key state))
        value (dc/dist-sample dist k2)
        lp    (dc/dist-log-prob dist value)]
    [value (-> state
             (assoc :key k1)
             (update :choices cm/set-value addr value)
             (update :score #(mx/add % lp)))]))

(defn generate-transition
  "Pure: if constrained at addr, use constraint; otherwise simulate."
  [state addr dist]
  (let [constraint (cm/get-submap (:constraints state) addr)]
    (if (cm/has-value? constraint)
      (let [value (cm/get-value constraint)
            lp    (dc/dist-log-prob dist value)]
        [value (-> state
                 (update :choices cm/set-value addr value)
                 (update :score  #(mx/add % lp))
                 (update :weight #(mx/add % lp)))])
      (simulate-transition state addr dist))))

(defn assess-transition
  "Pure: all addresses must be constrained. Throws if not."
  [state addr dist]
  (let [constraint (cm/get-submap (:constraints state) addr)]
    (if (cm/has-value? constraint)
      (let [value (cm/get-value constraint)
            lp    (dc/dist-log-prob dist value)]
        [value (-> state
                 (update :choices cm/set-value addr value)
                 (update :score  #(mx/add % lp))
                 (update :weight #(mx/add % lp)))])
      (throw (ex-info (str "assess: address " addr " not found in provided choices. "
                           "All addresses must be constrained for assess.")
                      {:addr addr})))))

(defn update-transition
  "Pure: use new constraint, keep old, or sample fresh."
  [state addr dist]
  (let [constraint (cm/get-submap (:constraints state) addr)
        old-choice (cm/get-submap (:old-choices state) addr)]
    (cond
      ;; New constraint provided
      (cm/has-value? constraint)
      (let [new-val (cm/get-value constraint)
            new-lp  (dc/dist-log-prob dist new-val)
            old-val (when (cm/has-value? old-choice) (cm/get-value old-choice))
            old-lp  (if old-val (dc/dist-log-prob dist old-val) ZERO)]
        [new-val (-> state
                   (update :choices cm/set-value addr new-val)
                   (update :score  #(mx/add % new-lp))
                   (update :weight #(mx/add % (mx/subtract new-lp old-lp)))
                   (cond->
                     old-val (update :discard cm/set-value addr old-val)))])

      ;; Keep old value
      (cm/has-value? old-choice)
      (let [val (cm/get-value old-choice)
            lp  (dc/dist-log-prob dist val)]
        [val (-> state
               (update :choices cm/set-value addr val)
               (update :score #(mx/add % lp)))])

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
            new-lp  (dc/dist-log-prob dist new-val)
            old-val (when (cm/has-value? old-choice) (cm/get-value old-choice))
            old-lp  (if old-val (dc/dist-log-prob dist old-val) ZERO)]
        [new-val (-> state
                   (assoc :key k1)
                   (update :choices cm/set-value addr new-val)
                   (update :score  #(mx/add % new-lp))
                   (update :weight #(mx/add % (mx/subtract new-lp old-lp))))])
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
                 (update :score #(mx/add % lp)))])))))

(defn project-transition
  "Pure: replay old value, accumulate log-prob for selected addresses."
  [state addr dist]
  (let [old-choice (cm/get-submap (:old-choices state) addr)
        val (when (cm/has-value? old-choice)
              (cm/get-value old-choice))
        _ (when (nil? val)
            (throw (ex-info (str "project: address not found in previous trace choices. "
                                 "Cannot replay a value for an address that was never sampled.")
                            {})))
        lp (dc/dist-log-prob dist val)
        sel (:selection state)]
    [val (-> state
           (update :choices cm/set-value addr val)
           (update :score #(mx/add % lp))
           (cond-> (and sel (sel/selected? sel addr))
             (update :weight #(mx/add % lp))))]))

;; ---------------------------------------------------------------------------
;; Batched state transitions (vectorized: [N]-shaped values)
;; ---------------------------------------------------------------------------

(defn batched-simulate-transition
  "Pure: sample [N] values, accumulate [N]-shaped score."
  [state addr dist]
  (let [n (:batch-size state)
        [k1 k2] (rng/split (:key state))
        value (dc/dist-sample-n dist k2 n)
        lp    (dc/dist-log-prob dist value)]
    [value (-> state
             (assoc :key k1)
             (update :choices cm/set-value addr value)
             (update :score #(mx/add % lp)))]))

(defn batched-generate-transition
  "Pure: constrained sites use scalar observation (broadcasts into [N] score),
   unconstrained sites delegate to batched-simulate-transition."
  [state addr dist]
  (let [constraint (cm/get-submap (:constraints state) addr)]
    (if (cm/has-value? constraint)
      ;; Constrained: scalar observation, scalar log-prob → broadcasts into [N] score/weight
      (let [value (cm/get-value constraint)
            lp    (dc/dist-log-prob dist value)]
        [value (-> state
                 (update :choices cm/set-value addr value)
                 (update :score  #(mx/add % lp))
                 (update :weight #(mx/add % lp)))])
      (batched-simulate-transition state addr dist))))

(defn batched-update-transition
  "Pure: batched update with [N]-shaped old values and scalar/[N]-shaped new constraints."
  [state addr dist]
  (let [n (:batch-size state)
        constraint (cm/get-submap (:constraints state) addr)
        old-choice (cm/get-submap (:old-choices state) addr)]
    (cond
      ;; New constraint provided — score difference against old values
      (cm/has-value? constraint)
      (let [new-val (cm/get-value constraint)
            new-lp  (dc/dist-log-prob dist new-val)
            old-val (when (cm/has-value? old-choice) (cm/get-value old-choice))
            old-lp  (if old-val (dc/dist-log-prob dist old-val) ZERO)]
        [new-val (-> state
                   (update :choices cm/set-value addr new-val)
                   (update :score  #(mx/add % new-lp))
                   (update :weight #(mx/add % (mx/subtract new-lp old-lp)))
                   (cond->
                     old-val (update :discard cm/set-value addr old-val)))])

      ;; Keep old [N]-shaped values
      (cm/has-value? old-choice)
      (let [val (cm/get-value old-choice)
            lp  (dc/dist-log-prob dist val)]
        [val (-> state
               (update :choices cm/set-value addr val)
               (update :score #(mx/add % lp)))])

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
            new-lp  (dc/dist-log-prob dist new-val)
            old-val (when (cm/has-value? old-choice) (cm/get-value old-choice))
            old-lp  (if old-val (dc/dist-log-prob dist old-val) ZERO)]
        [new-val (-> state
                   (assoc :key k1)
                   (update :choices cm/set-value addr new-val)
                   (update :score  #(mx/add % new-lp))
                   (update :weight #(mx/add % (mx/subtract new-lp old-lp))))])
      ;; Not selected: keep old [N]-shaped values
      (let [val (cm/get-value old-choice)
            lp  (dc/dist-log-prob dist val)]
        [val (-> state
               (update :choices cm/set-value addr val)
               (update :score #(mx/add % lp)))]))))

;; ---------------------------------------------------------------------------
;; Pure helpers used by runtime.cljs
;; ---------------------------------------------------------------------------

(defn merge-sub-result
  "Pure: merge a sub-generative-function result into parent state."
  [state addr sub-result]
  (-> state
    (update :choices cm/set-submap addr (:choices sub-result))
    (update :score (fn [sc] (mx/add sc (:score sub-result))))
    (update :splice-scores (fn [ss] (assoc (or ss {}) addr (:score sub-result))))
    (cond->
      (and (contains? state :weight) (:weight sub-result))
      (update :weight (fn [w] (mx/add w (:weight sub-result))))

      (:discard sub-result)
      (update :discard cm/set-submap addr (:discard sub-result)))))

(defn mlx-arr-batched?
  "Check if x is an MLX array with at least 1 dimension."
  [x]
  (and (some? x) (some? (.-shape x)) (.-item x)
       (pos? (count (mx/shape x)))))

(defn scalar-leaf-val?
  "Check if a value is scalar (0-d or not an MLX array)."
  [v]
  (or (not (and (some? v) (some? (.-shape v)) (.-item v)))
      (= [] (mx/shape v))))

(defn combinator-batched-fallback
  "Fallback for splicing a non-DynamicGF (e.g. VmapCombinator) in batched mode.
   Unstacks [N]-particle state, runs combinator GFI N times, stacks results."
  [state addr gf args]
  (let [n (:batch-size state)
        [k1 k2] (rng/split (:key state))
        ;; Extract scoped state
        sub-constraints (cm/get-submap (:constraints state) addr)
        sub-old-choices (cm/get-submap (:old-choices state) addr)
        sub-selection   (when-let [s (:selection state)]
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
        results
        (mapv
          (fn [i]
            (let [elem-args (mapv #(extract-scalar-arg % i) args)]
              (cond
                ;; Regenerate mode
                sub-selection
                (let [old-choices (or (and per-old-choices (nth per-old-choices i)) cm/EMPTY)
                      elem-trace (tr/make-trace
                                   {:gen-fn gf :args elem-args
                                    :choices old-choices :retval nil
                                    :score (mx/scalar 0.0)})
                      {:keys [trace weight]} (p/regenerate gf elem-trace sub-selection)]
                  {:choices (:choices trace) :score (:score trace)
                   :weight weight :retval (:retval trace)})

                ;; Update mode
                (and per-old-choices (nth per-old-choices i)
                     (not= (nth per-old-choices i) cm/EMPTY))
                (let [c (or (and per-constraints (nth per-constraints i)) cm/EMPTY)
                      elem-trace (tr/make-trace
                                   {:gen-fn gf :args elem-args
                                    :choices (nth per-old-choices i) :retval nil
                                    :score (mx/scalar 0.0)})
                      {:keys [trace weight discard]} (p/update gf elem-trace c)]
                  {:choices (:choices trace) :score (:score trace)
                   :weight weight :discard discard :retval (:retval trace)})

                ;; Generate with constraints
                (and per-constraints (nth per-constraints i)
                     (not= (nth per-constraints i) cm/EMPTY))
                (let [{:keys [trace weight]} (p/generate gf elem-args (nth per-constraints i))]
                  {:choices (:choices trace) :score (:score trace)
                   :weight weight :retval (:retval trace)})

                ;; Simulate
                :else
                (let [trace (p/simulate gf elem-args)]
                  {:choices (:choices trace) :score (:score trace)
                   :retval (:retval trace)}))))
          (range n))
        ;; Stack results
        stacked-choices (cm/stack-choicemaps (mapv :choices results) mx/stack)
        stacked-scores (mx/stack (mapv :score results))
        stacked-weights (when (some :weight results)
                          (mx/stack (mapv #(or (:weight %) (mx/scalar 0.0)) results)))
        stacked-retval (let [rvs (mapv :retval results)]
                         (if (every? mx/array? rvs) (mx/stack rvs) (vec rvs)))
        stacked-discard (when (some :discard results)
                          (let [discards (mapv #(or (:discard %) cm/EMPTY) results)]
                            (if (every? #(= % cm/EMPTY) discards)
                              cm/EMPTY
                              (cm/stack-choicemaps discards mx/stack))))
        ;; Merge into parent state
        sub-result {:choices stacked-choices :score stacked-scores
                    :weight stacked-weights :discard stacked-discard
                    :retval stacked-retval}
        state' (-> state
                 (assoc :key k1)
                 (merge-sub-result addr sub-result))]
    [state' stacked-retval]))
