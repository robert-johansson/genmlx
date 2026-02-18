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
