(ns genmlx.handler
  "Handler-based execution — the heart of GenMLX.

   ## Architecture

   Each GFI operation (simulate, generate, update, regenerate, project) is
   implemented as a **pure state transition**:

       (fn [state addr dist] -> [value state'])

   wrapped in a thin dispatcher that reads/writes a volatile!.

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

   ## Mutable boundary

   The **only** mutable state in GenMLX is the volatile! created inside
   `run-handler`.  It is allocated, written, and dereferenced within a
   single `binding` block and never escapes.  The handler functions
   (`simulate-handler`, `generate-handler`, etc.) are the only code that
   touches it via `vreset!` / `vswap!`.

   One external module (inference/adev.cljs) also writes `*state*` via
   `vreset!` following the same handler pattern.

   ## Dynamic scope

   Four dynamic vars constitute the complete dynamic scope of the system:

     *handler*     — active handler fn, bound by `run-handler`
     *state*       — volatile! holding the handler state map, bound by `run-handler`
     *param-store* — optional parameter store for trainable params,
                     bound by `learning.cljs` and `inference/adev.cljs`
     *prng-key*    — optional volatile! holding a PRNG key for reproducibility,
                     bound by `with-key` in dynamic.cljs

   No other dynamic vars or global mutable state exists in GenMLX."
  (:require [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.selection :as sel]
            [genmlx.dist.core :as dc]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]))

;; The four dynamic vars below are the complete dynamic scope of GenMLX.
;; *handler* and *state* are bound exclusively by run-handler.
;; *param-store* is bound by learning.cljs and inference/adev.cljs.
;; rng/*prng-key* is bound by with-key in dynamic.cljs for reproducible inference.
(def ^:dynamic *handler* nil)
(def ^:dynamic *state*   nil)
(def ^:dynamic *param-store* nil)

;; Cached zero constant for init states and fallback log-probs
(def ^:private ZERO (mx/scalar 0.0))

(declare run-handler)

;; ---------------------------------------------------------------------------
;; Pure state transitions
;; ---------------------------------------------------------------------------

(defn- simulate-transition
  "Pure: given state, addr, dist -> [value state']."
  [state addr dist]
  (let [[k1 k2] (rng/split (:key state))
        value (dc/dist-sample dist k2)
        lp    (dc/dist-log-prob dist value)]
    [value (-> state
             (assoc :key k1)
             (update :choices cm/set-value addr value)
             (update :score #(mx/add % lp)))]))

(defn- generate-transition
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

(defn- assess-transition
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

(defn- update-transition
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

(defn- regenerate-transition
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

(defn- project-transition
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

(defn- batched-simulate-transition
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

(defn- batched-generate-transition
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

(defn- batched-update-transition
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

(defn- batched-regenerate-transition
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
;; Handler implementations (thin wrappers over pure transitions)
;; ---------------------------------------------------------------------------

(defn simulate-handler
  "Sample from dist at addr, accumulate score."
  [addr dist]
  (let [[value state'] (simulate-transition @*state* addr dist)]
    (vreset! *state* state')
    value))

(defn generate-handler
  "If constrained at addr, use constraint; otherwise simulate."
  [addr dist]
  (let [[value state'] (generate-transition @*state* addr dist)]
    (vreset! *state* state')
    value))

(defn assess-handler
  "All addresses must be constrained. Errors on unconstrained."
  [addr dist]
  (let [[value state'] (assess-transition @*state* addr dist)]
    (vreset! *state* state')
    value))

(defn update-handler
  "Update: use new constraint, keep old, or sample fresh."
  [addr dist]
  (let [[value state'] (update-transition @*state* addr dist)]
    (vreset! *state* state')
    value))

(defn regenerate-handler
  "Resample selected addresses, keep unselected."
  [addr dist]
  (let [[value state'] (regenerate-transition @*state* addr dist)]
    (vreset! *state* state')
    value))

(defn project-handler
  "Replay old values, accumulate log-prob for selected addresses."
  [addr dist]
  (let [[value state'] (project-transition @*state* addr dist)]
    (vreset! *state* state')
    value))

(defn batched-simulate-handler
  "Batched sample from dist at addr, accumulate [N]-shaped score."
  [addr dist]
  (let [[value state'] (batched-simulate-transition @*state* addr dist)]
    (vreset! *state* state')
    value))

(defn batched-generate-handler
  "Batched generate: constrained or batched-simulate."
  [addr dist]
  (let [[value state'] (batched-generate-transition @*state* addr dist)]
    (vreset! *state* state')
    value))

(defn batched-update-handler
  "Batched update: new constraint, keep old [N]-shaped, or sample fresh."
  [addr dist]
  (let [[value state'] (batched-update-transition @*state* addr dist)]
    (vreset! *state* state')
    value))

(defn batched-regenerate-handler
  "Batched regenerate: resample selected [N]-shaped addresses."
  [addr dist]
  (let [[value state'] (batched-regenerate-transition @*state* addr dist)]
    (vreset! *state* state')
    value))

;; ---------------------------------------------------------------------------
;; Entry points called by trace/splice inside gen bodies
;; ---------------------------------------------------------------------------

(defn trace-choice!
  "Dispatch to active handler. Outside any handler, sample directly."
  [addr dist]
  (if *handler*
    (*handler* addr dist)
    ;; Direct execution mode — sample and materialize
    (let [v (dc/dist-sample dist (rng/next-key))]
      (mx/eval! v)
      (mx/item v))))

(defn trace-param!
  "Read a trainable parameter value. If a param store is bound, reads from it.
   Otherwise returns the default value as an MLX array."
  [name default-value]
  (let [default (if (mx/array? default-value) default-value (mx/scalar default-value))]
    (if *param-store*
      (or (get-in *param-store* [:params name]) default)
      default)))

(defn- merge-sub-result
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

(defn- batched-splice-transition
  "Pure: execute a DynamicGF sub-GF in batched mode at the given address.
   Splits the parent key, runs the sub-GF body under the appropriate batched
   handler, and merges the sub-result back into parent state."
  [state addr gf args]
  (let [[k1 k2] (rng/split (:key state))
        n (:batch-size state)
        ;; Extract scoped state for this splice address
        sub-constraints (cm/get-submap (:constraints state) addr)
        sub-old-choices (cm/get-submap (:old-choices state) addr)
        sub-selection   (when-let [s (:selection state)]
                          (sel/get-subselection s addr))
        body-fn (:body-fn gf)
        ;; Choose handler + build init-state based on mode
        [handler init-state]
        (cond
          ;; Regenerate mode
          sub-selection
          [batched-regenerate-handler
           {:choices cm/EMPTY :score ZERO
            :weight ZERO :key k2
            :selection sub-selection
            :old-choices (or sub-old-choices cm/EMPTY)
            :batch-size n :batched? true}]

          ;; Update mode
          (and sub-old-choices (not= sub-old-choices cm/EMPTY))
          [batched-update-handler
           {:choices cm/EMPTY :score ZERO
            :weight ZERO :key k2
            :constraints (or sub-constraints cm/EMPTY)
            :old-choices sub-old-choices
            :discard cm/EMPTY
            :batch-size n :batched? true}]

          ;; Generate mode
          (and sub-constraints (not= sub-constraints cm/EMPTY))
          [batched-generate-handler
           {:choices cm/EMPTY :score ZERO
            :weight ZERO :key k2
            :constraints sub-constraints
            :batch-size n :batched? true}]

          ;; Simulate mode
          :else
          [batched-simulate-handler
           {:choices cm/EMPTY :score ZERO
            :key k2 :batch-size n :batched? true}])
        ;; Run sub-GF body under the chosen batched handler
        sub-result (run-handler handler init-state
                     #(apply body-fn args))
        ;; Merge sub-result into parent state, advance parent key
        state' (-> state
                 (assoc :key k1)
                 (merge-sub-result addr sub-result))]
    [state' (:retval sub-result)]))

(defn- mlx-arr-batched?
  "Check if x is an MLX array with at least 1 dimension."
  [x]
  (and (some? x) (some? (.-shape x)) (.-item x)
       (pos? (count (mx/shape x)))))

(defn- scalar-leaf-val?
  "Check if a value is scalar (0-d or not an MLX array)."
  [v]
  (or (not (and (some? v) (some? (.-shape v)) (.-item v)))
      (= [] (mx/shape v))))

(defn- combinator-batched-fallback
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

(defn trace-gf!
  "Call a sub-generative-function at the given address namespace."
  [addr gf args]
  (if *handler*
    (if (:batched? @*state*)
      ;; Batched mode
      (cond
        ;; DynamicGF: batched splice transition (runs body once)
        (:body-fn gf)
        (let [[state' retval] (batched-splice-transition @*state* addr gf args)]
          (vreset! *state* state')
          retval)

        ;; Combinator / other GFI: scalar fallback per particle
        (satisfies? p/IGenerativeFunction gf)
        (let [[state' retval] (combinator-batched-fallback @*state* addr gf args)]
          (vreset! *state* state')
          retval)

        :else
        (throw (ex-info "splice of unsupported gen-fn type in batched mode"
                        {:addr addr})))
      ;; Scalar mode: delegate to sub-execution with scoped constraints/state
      (let [state @*state*
            sub-constraints (cm/get-submap (:constraints state) addr)
            sub-old-choices (cm/get-submap (:old-choices state) addr)
            sub-selection   (when-let [s (:selection state)]
                              (sel/get-subselection s addr))
            old-splice-score (get (:old-splice-scores state) addr)
            sub-result ((:executor state) gf args
                         {:constraints sub-constraints
                          :old-choices sub-old-choices
                          :selection   sub-selection
                          :key         (:key state)
                          :old-splice-score old-splice-score})]
        (vswap! *state* merge-sub-result addr sub-result)
        (:retval sub-result)))
    ;; Direct mode — simulate and return value
    (let [trace (p/simulate gf args)]
      (:retval trace))))

;; ---------------------------------------------------------------------------
;; Orchestrator
;; ---------------------------------------------------------------------------

(defn run-handler
  "Execute body-fn under a handler, returning final state map (immutable)."
  [handler-fn init-state body-fn]
  (binding [*handler* handler-fn
            *state*   (volatile! init-state)
            mx/*batched-exec?* (boolean (:batched? init-state))]
    (let [retval (body-fn)]
      (assoc @*state* :retval retval))))
