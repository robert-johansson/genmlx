(ns genmlx.trace
  "Immutable trace records for GenMLX.
   Score convention follows Gen.jl: score = log P(choices | args).
   Scores are MLX scalars (stay on GPU). Choice values are MLX arrays.
   Use mx/item only at the inference boundary."
  (:require [genmlx.choicemap :as cm]))

(defrecord Trace [gen-fn args choices retval score])

(defn make-trace
  "Create a trace from a map of {:gen-fn :args :choices :retval :score}."
  [m]
  (map->Trace m))
