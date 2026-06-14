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
    "Constrained execution. Returns {:trace Trace :weight MLX-scalar}.

     Default (handler/compiled paths): unconstrained choices are sampled from the
     prior; :weight is the sum of log-probabilities at the constrained addresses
     (log p(constraints | sampled latents)); the trace's score-type is :joint.

     L3 analytical special case: when the model's STATIC keyword addresses form a
     detected conjugate pair and the observation is constrained while its prior
     latent is not, generate marginalizes the latent analytically — the latent in
     the returned trace is the deterministic analytic posterior mean (not a prior
     sample) and :weight is the analytic MARGINAL log p(obs). This is the optimal
     zero-variance importance weight; the trace is tagged :marginal (detect with
     genmlx.trace/score-type). The same joint written with DYNAMIC addresses skips
     conjugacy and uses the default behavior. To force prior-sample + conditional
     (:joint) weight — required for correct multi-particle importance sampling and
     trace-MH — strip the path with genmlx.dynamic/strip-analytical-path (every
     built-in importance/SMC/MCMC entry point does this internally)."))

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
