(ns genmlx.protocols
  "GFI (Generative Function Interface) protocol definitions.
   Layered design: simulate is required, everything else has defaults.")

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

(defprotocol IUpdateWithDiffs
  (update-with-diffs [gf trace constraints argdiffs]
    "Update a trace with change hints. argdiffs describes which arguments changed.
     Enables combinators to skip unchanged sub-computations.
     Returns {:trace Trace :weight MLX-scalar :discard ChoiceMap}."))
