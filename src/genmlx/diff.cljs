(ns genmlx.diff
  "Change-tagged values for incremental computation.
   Argdiffs enable update/update-with-args to skip unchanged computation,
   which is critical for MCMC performance where each MH step only changes
   one or a few addresses, and for combinator updates over long sequences.

   An argdiff is a caller ASSERTION about which arguments changed (the
   Gen.jl trust model): :unknown (or any unrecognized value) is always
   sound and forces full re-execution; no-change permits the identity
   fast path; vector-diff lets Map/Scan-style combinators skip elements
   whose indices are not in :changed.")

;; ---------------------------------------------------------------------------
;; Diff types
;; ---------------------------------------------------------------------------

(def no-change
  "Indicates a value has not changed."
  {:diff-type :no-change})

(defn vector-diff
  "Argdiff for a vector argument where only the elements whose indices are
   in `changed` (a set of ints) may differ. Map/Scan combinators dispatch
   on this (update-with-diffs / update-with-args) to retain unchanged
   elements verbatim."
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
