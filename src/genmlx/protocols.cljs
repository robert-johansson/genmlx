(ns genmlx.protocols
  "GFI (Generative Function Interface) protocol definitions.
   Eleven single-operation protocols. There are no protocol-level defaults: each
   generative function implements the operations it supports (simulate is the
   conceptual primitive; the generic run-* dispatch logic lives in the handler).")

(defprotocol IGenerativeFunction
  (simulate [gf args]
    "Forward-sample all choices. Returns Trace."))

(defprotocol IGenerate
  (generate [gf args constraints]
    "Constrained execution. Returns {:trace Trace :weight MLX-scalar}."))

(defprotocol IAssess
  (assess [gf args choices]
    "Score fully-specified choices. Returns {:retval any :weight MLX-scalar}."))

(defprotocol IUpdate
  (update [gf trace constraints]
    "Update a trace with new constraints.
     Returns {:trace Trace :weight MLX-scalar :discard ChoiceMap}."))

(defprotocol IRegenerate
  (regenerate [gf trace selection]
    "Resample selected addresses.
     Returns {:trace Trace :weight MLX-scalar}."))

(defprotocol IPropose
  (propose [gf args]
    "Forward-sample all choices and return choices + their joint log-probability.
     Returns {:choices ChoiceMap :weight MLX-scalar :retval any}."))

(defprotocol IProject
  (project [gf trace selection]
    "Compute log-probability of selected choices in trace.
     Returns MLX-scalar log-weight."))

(defprotocol IUpdateWithDiffs
  (update-with-diffs [gf trace constraints argdiffs]
    "Update a trace with change hints. argdiffs describes which arguments changed.
     Enables combinators to skip unchanged sub-computations.
     Equivalent to update-with-args with new-args = (:args trace).
     Returns {:trace Trace :weight MLX-scalar :discard ChoiceMap}."))

(defprotocol IUpdateWithArgs
  (update-with-args [gf trace new-args argdiffs constraints]
    "Update a trace while the model arguments change (thesis x').
     Retained choices are re-scored under the new arguments; fresh sites
     cancel; removed sites are charged via the old score and discarded.
     Weight: nonfresh-score(t'; x') - score(t; x).
     argdiffs: genmlx.diff/no-change | genmlx.diff/vector-diff | :unknown —
     a caller assertion about which arguments changed (trusted, Gen.jl-style).
     Returns {:trace Trace :weight MLX-scalar :discard ChoiceMap}."))

(defprotocol IHasArgumentGrads
  (has-argument-grads [gf]
    "Vector of booleans: true if that argument position is differentiable.
     Returns nil if unknown."))

(defprotocol IBatchedSplice
  (batched-splice [gf state addr args]
    "Execute this combinator in batched mode within a parent handler.
     state: parent handler state (with :batched? true, :batch-size N, etc.)
     addr: the splice address in the parent
     args: arguments to this combinator
     Returns [state' retval]."))
