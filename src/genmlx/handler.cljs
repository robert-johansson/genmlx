(ns genmlx.handler
  "Handler-based execution — the heart of GenMLX.
   Dynamic vars replace GenJAX's mutable handler stack. State flows through
   a volatile! that never escapes the execution boundary.

   Five handlers implement the five GFI operations. Each handler is a
   function (fn [addr dist] -> value) that also side-effects into *state*."
  (:require [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.selection :as sel]))

(def ^:dynamic *handler* nil)
(def ^:dynamic *state*   nil)

;; Forward declarations for dist protocol functions
(declare dist-sample dist-log-prob)

;; ---------------------------------------------------------------------------
;; Key management
;; ---------------------------------------------------------------------------

(defn- consume-key!
  "Split the current key, update state with one half, return the other."
  []
  (let [[k1 k2] (rng/split (:key @*state*))]
    (vswap! *state* assoc :key k1)
    k2))

;; ---------------------------------------------------------------------------
;; Handler implementations
;; ---------------------------------------------------------------------------

(defn simulate-handler
  "Sample from dist at addr, accumulate score."
  [addr dist]
  (let [key   (consume-key!)
        value (dist-sample dist key)
        lp    (dist-log-prob dist value)]
    (vswap! *state*
      (fn [s] (-> s
                (update :choices #(cm/set-choice % [addr] value))
                (update :score #(mx/add % lp)))))
    value))

(defn generate-handler
  "If constrained at addr, use constraint; otherwise simulate."
  [addr dist]
  (let [constraint (cm/get-submap (:constraints @*state*) addr)]
    (if (cm/has-value? constraint)
      (let [value (cm/get-value constraint)
            lp    (dist-log-prob dist value)]
        (vswap! *state*
          (fn [s] (-> s
                    (update :choices #(cm/set-choice % [addr] value))
                    (update :score  #(mx/add % lp))
                    (update :weight #(mx/add % lp)))))
        value)
      (simulate-handler addr dist))))

(defn update-handler
  "Update: use new constraint, keep old, or sample fresh."
  [addr dist]
  (let [state @*state*
        constraint (cm/get-submap (:constraints state) addr)
        old-choice (cm/get-submap (:old-choices state) addr)]
    (cond
      ;; New constraint provided
      (cm/has-value? constraint)
      (let [new-val (cm/get-value constraint)
            new-lp  (dist-log-prob dist new-val)
            old-val (when (cm/has-value? old-choice) (cm/get-value old-choice))
            old-lp  (if old-val (dist-log-prob dist old-val) (mx/scalar 0.0))]
        (vswap! *state*
          (fn [s] (-> s
                    (update :choices #(cm/set-choice % [addr] new-val))
                    (update :score  #(mx/add % new-lp))
                    (update :weight #(mx/add % (mx/subtract new-lp old-lp)))
                    (cond->
                      old-val (update :discard #(cm/set-choice % [addr] old-val))))))
        new-val)

      ;; Keep old value
      (cm/has-value? old-choice)
      (let [val (cm/get-value old-choice)
            lp  (dist-log-prob dist val)]
        (vswap! *state*
          (fn [s] (-> s
                    (update :choices #(cm/set-choice % [addr] val))
                    (update :score #(mx/add % lp)))))
        val)

      ;; New address: sample fresh
      :else (simulate-handler addr dist))))

(defn regenerate-handler
  "Resample selected addresses, keep unselected."
  [addr dist]
  (let [state @*state*
        sel (:selection state)
        old-choice (cm/get-submap (:old-choices state) addr)]
    (if (and sel (sel/selected? sel addr))
      ;; Selected: resample, compute weight adjustment
      (let [key     (consume-key!)
            new-val (dist-sample dist key)
            new-lp  (dist-log-prob dist new-val)
            old-val (when (cm/has-value? old-choice) (cm/get-value old-choice))
            old-lp  (if old-val (dist-log-prob dist old-val) (mx/scalar 0.0))]
        (vswap! *state*
          (fn [s] (-> s
                    (update :choices #(cm/set-choice % [addr] new-val))
                    (update :score  #(mx/add % new-lp))
                    (update :weight #(mx/add % (mx/subtract new-lp old-lp))))))
        new-val)
      ;; Not selected: keep old
      (let [val (cm/get-value old-choice)
            lp  (dist-log-prob dist val)]
        (vswap! *state*
          (fn [s] (-> s
                    (update :choices #(cm/set-choice % [addr] val))
                    (update :score #(mx/add % lp)))))
        val))))

;; ---------------------------------------------------------------------------
;; Entry points called by trace/splice inside gen bodies
;; ---------------------------------------------------------------------------

(defn trace-choice!
  "Dispatch to active handler. Outside any handler, sample directly."
  [addr dist]
  (if *handler*
    (*handler* addr dist)
    ;; Direct execution mode — sample and materialize
    (let [v (dist-sample dist nil)]
      (mx/eval! v)
      (mx/item v))))

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
      ;; Merge sub-result into parent state
      (vswap! *state*
        (fn [s]
          (-> s
            (update :choices #(cm/set-choice % [addr] (:choices sub-result)))
            (update :score (fn [sc] (mx/add sc (:score sub-result))))
            (cond->
              (and (contains? s :weight) (:weight sub-result))
              (update :weight (fn [w] (mx/add w (:weight sub-result))))

              (:discard sub-result)
              (update :discard #(cm/set-choice % [addr] (:discard sub-result)))))))
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

;; ---------------------------------------------------------------------------
;; Bridge functions — set by dist.cljs to avoid circular deps
;; ---------------------------------------------------------------------------

(defonce ^:private dist-fns (volatile! nil))

(defn set-dist-fns!
  "Called by genmlx.dist to register sample/log-prob functions."
  [sample-fn log-prob-fn]
  (vreset! dist-fns {:sample sample-fn :log-prob log-prob-fn}))

(defn- dist-sample [dist key]
  (if-let [fns @dist-fns]
    ((:sample fns) dist key)
    (throw (ex-info "Distribution functions not registered. Require genmlx.dist first." {}))))

(defn- dist-log-prob [dist value]
  (if-let [fns @dist-fns]
    ((:log-prob fns) dist value)
    (throw (ex-info "Distribution functions not registered. Require genmlx.dist first." {}))))
