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
