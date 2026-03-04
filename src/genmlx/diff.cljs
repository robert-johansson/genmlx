(ns genmlx.diff
  "Change-tagged values for incremental computation.
   Argdiffs and retdiffs enable `update` to skip unchanged computation,
   which is critical for MCMC performance where each MH step only changes
   one or a few addresses.")

;; ---------------------------------------------------------------------------
;; Diff types
;; ---------------------------------------------------------------------------

(def no-change
  "Indicates a value has not changed."
  {:diff-type :no-change})

;; ---------------------------------------------------------------------------
;; Diff predicates
;; ---------------------------------------------------------------------------

(defn no-change?
  "Returns true if the diff indicates no change."
  [d]
  (= (:diff-type d) :no-change))

(defn changed?
  "Returns true if the diff indicates any change."
  [d]
  (not (no-change? d)))
