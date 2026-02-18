(ns genmlx.handler
  "Handler-based execution — the heart of GenMLX.
   Dynamic vars replace GenJAX's mutable handler stack. State flows through
   a volatile! that never escapes the execution boundary.

   Each handler is split into a pure transition function
   (fn [state addr dist] -> [value state']) and a thin dispatch wrapper
   that reads/writes the volatile."
  (:require [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.selection :as sel]
            [genmlx.dist.core :as dc]))

(def ^:dynamic *handler* nil)
(def ^:dynamic *state*   nil)

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
             (update :choices #(cm/set-choice % [addr] value))
             (update :score #(mx/add % lp)))]))

(defn- generate-transition
  "Pure: if constrained at addr, use constraint; otherwise simulate."
  [state addr dist]
  (let [constraint (cm/get-submap (:constraints state) addr)]
    (if (cm/has-value? constraint)
      (let [value (cm/get-value constraint)
            lp    (dc/dist-log-prob dist value)]
        [value (-> state
                 (update :choices #(cm/set-choice % [addr] value))
                 (update :score  #(mx/add % lp))
                 (update :weight #(mx/add % lp)))])
      (simulate-transition state addr dist))))

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
            old-lp  (if old-val (dc/dist-log-prob dist old-val) (mx/scalar 0.0))]
        [new-val (-> state
                   (update :choices #(cm/set-choice % [addr] new-val))
                   (update :score  #(mx/add % new-lp))
                   (update :weight #(mx/add % (mx/subtract new-lp old-lp)))
                   (cond->
                     old-val (update :discard #(cm/set-choice % [addr] old-val))))])

      ;; Keep old value
      (cm/has-value? old-choice)
      (let [val (cm/get-value old-choice)
            lp  (dc/dist-log-prob dist val)]
        [val (-> state
               (update :choices #(cm/set-choice % [addr] val))
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
            old-lp  (if old-val (dc/dist-log-prob dist old-val) (mx/scalar 0.0))]
        [new-val (-> state
                   (assoc :key k1)
                   (update :choices #(cm/set-choice % [addr] new-val))
                   (update :score  #(mx/add % new-lp))
                   (update :weight #(mx/add % (mx/subtract new-lp old-lp))))])
      ;; Not selected: keep old
      (let [val (cm/get-value old-choice)
            lp  (dc/dist-log-prob dist val)]
        [val (-> state
               (update :choices #(cm/set-choice % [addr] val))
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

;; ---------------------------------------------------------------------------
;; Entry points called by trace/splice inside gen bodies
;; ---------------------------------------------------------------------------

(defn trace-choice!
  "Dispatch to active handler. Outside any handler, sample directly."
  [addr dist]
  (if *handler*
    (*handler* addr dist)
    ;; Direct execution mode — sample and materialize
    (let [v (dc/dist-sample dist nil)]
      (mx/eval! v)
      (mx/item v))))

(defn- merge-sub-result
  "Pure: merge a sub-generative-function result into parent state."
  [state addr sub-result]
  (-> state
    (update :choices #(cm/set-choice % [addr] (:choices sub-result)))
    (update :score (fn [sc] (mx/add sc (:score sub-result))))
    (cond->
      (and (contains? state :weight) (:weight sub-result))
      (update :weight (fn [w] (mx/add w (:weight sub-result))))

      (:discard sub-result)
      (update :discard #(cm/set-choice % [addr] (:discard sub-result))))))

(defn trace-gf!
  "Call a sub-generative-function at the given address namespace."
  [addr gf args]
  (if *handler*
    ;; Delegate to sub-execution with scoped constraints/state
    (let [state @*state*
          sub-constraints (cm/get-submap (:constraints state) addr)
          sub-old-choices (cm/get-submap (:old-choices state) addr)
          sub-selection   (when-let [s (:selection state)]
                            (sel/get-subselection s addr))
          sub-result ((:executor state) gf args
                       {:constraints sub-constraints
                        :old-choices sub-old-choices
                        :selection   sub-selection
                        :key         (:key state)})]
      (vswap! *state* merge-sub-result addr sub-result)
      (:retval sub-result))
    ;; Direct mode — simulate and return value
    (let [p (requiring-resolve 'genmlx.protocols/simulate)
          trace (p gf args)]
      (:retval trace))))

;; ---------------------------------------------------------------------------
;; Orchestrator
;; ---------------------------------------------------------------------------

(defn run-handler
  "Execute body-fn under a handler, returning final state map (immutable)."
  [handler-fn init-state body-fn]
  (binding [*handler* handler-fn
            *state*   (volatile! init-state)]
    (let [retval (body-fn)]
      (assoc @*state* :retval retval))))
