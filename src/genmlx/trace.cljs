(ns genmlx.trace
  "Immutable trace records for GenMLX.
   Score convention follows Gen.jl: score = log P(choices | args).
   Scores are MLX scalars (stay on GPU). Choice values are MLX arrays.
   Use mx/item only at the inference boundary."
)

(defprotocol ITrace
  "Marker protocol for trace types. Implemented by Trace here and extended
   by TensorTrace and VectorizedTrace at their definition sites, so
   predicates (e.g. genmlx.schemas) accept every trace representation.")

(defrecord Trace [gen-fn args choices retval score]
  ITrace)

(defn make-trace
  "Create a trace from a map of {:gen-fn :args :choices :retval :score}."
  [m]
  (map->Trace m))

(defn trace?
  "True for any GenMLX trace representation (Trace, TensorTrace,
   VectorizedTrace)."
  [x]
  (satisfies? ITrace x))

;; ---------------------------------------------------------------------------
;; Score-type metadata (genmlx-lbae)
;;
;; Different producing paths assign different meanings to :score
;; (ARCHITECTURE §3.3):
;;   :joint     — log p(tau; x), the joint density of all recorded choices
;;   :marginal  — marginal likelihood: some latents analytically integrated
;;                out (L3 elimination); eliminated latents appear in the
;;                choicemap pinned at their posterior mean
;;   :collapsed — exact marginal likelihood, ALL latents integrated out
;;                (enumerate); the choicemap is empty
;; The tag lives in Clojure metadata under ::score-type, set explicitly by
;; every producing path. Only :marginal traces can cross a joint-scoring
;; boundary (by re-generating from their own choices); :collapsed traces
;; have no choices to re-generate from and must throw.
;; ---------------------------------------------------------------------------

(def ^:private score-type-rank
  "Propagation order for combine-score-types among the path-unstable tags."
  {:joint 0 :marginal 1 :beam-marginal 2})

(defn score-type
  "The score encoding of a trace: its ::score-type metadata, or :joint when
   untagged (hand-rolled traces in tests, deserialized traces)."
  [trace]
  (get (meta trace) ::score-type :joint))

(defn with-score-type
  "Return trace tagged with score-type st, preserving other metadata."
  [trace st]
  (vary-meta trace assoc ::score-type st))

(defn combine-score-types
  "Compose a composite trace's score-type from its parts (spliced
   sub-traces, combinator elements). nil counts as :joint (untagged).

   Only path-UNSTABLE tags propagate (:marginal, :beam-marginal): an
   analytically-marginal part records latent choices its score does not
   cover (posterior-mean pinning; scoring switches decomposition between
   ops), poisoning score deltas for the whole composite. A :collapsed part
   (enumerate) is the opposite — it records NO internal choices, its score
   is the exact reproducible density of its empty block (encapsulated
   exact marginalization), and every op re-derives it consistently, so it
   composes like any deterministic factor: :joint (the certified
   MCMC-around-an-enumerate-splice pattern, exact_test 41). Top-level
   :collapsed tags are assigned directly by enumerate, never derived."
  ([] :joint)
  ([a] (if (contains? score-type-rank a) a :joint))
  ([a b]
   (let [a (if (contains? score-type-rank a) a :joint)
         b (if (contains? score-type-rank b) b :joint)]
     (if (>= (score-type-rank a) (score-type-rank b)) a b))))

(defn assert-joint!
  "Throw unless trace is joint-scored. Consumers whose math requires joint
   scores (trace-MH acceptance ratios) call this so a marginal/collapsed
   trace that slipped past the strip+convert machinery fails loudly
   (genmlx-540f) instead of silently miscalibrating. Returns the trace."
  [trace op]
  (let [st (score-type trace)]
    (if (= :joint st)
      trace
      (throw (ex-info
               (str op " requires a joint-scored trace, got " st
                    (case st
                      :marginal
                      " — this trace came from the L3 analytical path; strip the model with genmlx.dynamic/strip-analytical-path or re-generate the trace from its own choices"
                      :collapsed
                      " — collapsed traces (enumerate/exact) have no recorded choices and cannot be consumed by sampling-based methods"
                      ""))
               {:genmlx/error :score-type-mismatch
                :op op :score-type st :expected :joint})))))
