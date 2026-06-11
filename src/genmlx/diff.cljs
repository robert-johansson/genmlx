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

(defn vector-diff
  "Argdiff for a vector argument where only the elements whose indices are
   in `changed` (a set of ints) may differ. Map-style combinators dispatch
   on this to update only the changed elements (update-with-diffs)."
  [changed]
  {:diff-type :vector-diff :changed (set changed)})

;; ---------------------------------------------------------------------------
;; Diff predicates
;; ---------------------------------------------------------------------------

(defn no-change?
  "Returns true if the diff indicates no change."
  [d]
  (= (:diff-type d) :no-change))

(defn vector-diff?
  "Returns true if the diff is an element-wise vector diff."
  [d]
  (= (:diff-type d) :vector-diff))
