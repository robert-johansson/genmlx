(ns agentmodels.helpers
  "Ergonomic helpers for the agentmodels examples library.

   Closes the 'partial' PPL-ergonomics gaps with a tiny shared namespace —
   zero core/engine change. Every helper composes existing GenMLX primitives:

   - `factor-dist`   soft conditioning via the handler's :score path
                     (defdist, mirroring dist/delta)
   - `softmax-action` Boltzmann policy = Categorical(softmax(alpha * EU))
                     (the GenMLX realization of agentmodels' factor(alpha * EU))
   - `uniform-draw` / `weighted-draw`  value-carrying categorical draws
                     (generalizes llm/bytes byte-marginals->categorical out of
                     the LLM layer)
   - `boxed-choice`  distribution-as-value via a :kind index + SwitchCombinator

   These belong in an examples library, not the engine: they are conveniences
   over the GFI, not new GFI machinery."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.combinators :as comb]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]
                   [genmlx.dist.macros :refer [defdist]]))

;; ---------------------------------------------------------------------------
;; factor-dist — soft conditioning
;; ---------------------------------------------------------------------------
;;
;; The handler's transitions all do `score += dist-log-prob(value)` (see
;; handler.cljs). A distribution whose log-prob is a constant w — independent
;; of the drawn value — therefore injects exactly w into the trace score. That
;; is precisely WebPPL/agentmodels' `factor(w)`: a soft observation that
;; reweights the trace without binding a random variable.
;;
;; Structurally this mirrors dist/delta (deterministic draw, point support) but
;; replaces delta's 0/-inf indicator log-prob with the free scalar w.

(defdist factor-dist
  "Soft-conditioning pseudo-distribution. Deterministically draws 0 and
   contributes log-weight w to the trace :score for *any* drawn value, so

     (trace :soft (factor-dist (mx/scalar 2.3)))   ; score += 2.3

   realizes agentmodels' factor(2.3). Mirrors dist/delta but injects an
   arbitrary scalar log-factor rather than a 0/-inf point-mass indicator."
  [w]
  (sample [_key] (mx/scalar 0.0))
  (log-prob [_value] w)
  (support [] [(mx/scalar 0.0)]))

;; Batch-sampling support so factor-dist works under vsimulate/vgenerate too:
;; the draw is the constant 0 broadcast to [n] (log-prob already broadcasts w).
(defmethod genmlx.dist.core/dist-sample-n* :factor-dist [_d _key n]
  (mx/broadcast-to (mx/scalar 0.0) [n]))

;; ---------------------------------------------------------------------------
;; softmax-action — Boltzmann / soft-max policy
;; ---------------------------------------------------------------------------
;;
;; agentmodels selects actions with `factor(alpha * EU(action))` inside an
;; enumerated action choice, which is mathematically Categorical(softmax(alpha *
;; EU)). dist/categorical already treats its argument as logits and normalizes
;; via log_softmax, so passing `alpha * EU` directly yields the Boltzmann
;; policy. This is the same construction proven in examples/memo/mdp.cljs
;; (logits = alpha * Q, then dist/categorical).
;;
;; As alpha -> inf the policy concentrates on argmax(EU); we make that limit
;; exact (and uniformly tie-broken) by delegating to exact/categorical-argmax.

(defn softmax-action
  "Boltzmann action policy: returns a Categorical distribution over actions
   with logits `alpha * eu`, i.e. action ~ Categorical(softmax(alpha * eu)).

   `eu` is a vector/array of expected utilities, one per action. `alpha` is the
   (inverse-temperature) rationality parameter. With alpha = ##Inf the policy is
   the deterministic argmax, delegated to exact/categorical-argmax so ties break
   uniformly. This is the primary GenMLX realization of factor(alpha * EU)."
  [alpha eu]
  (if (= alpha ##Inf)
    (exact/categorical-argmax eu)
    (dist/categorical (mx/multiply (mx/scalar alpha) eu))))

;; ---------------------------------------------------------------------------
;; uniform-draw / weighted-draw — value-carrying categorical
;; ---------------------------------------------------------------------------
;;
;; A categorical samples an *index*; the modeller almost always wants the value
;; behind that index. `byte-marginals->categorical` in llm/bytes.cljs solved
;; exactly this for byte characters ({:dist ... :chars ...} + (nth chars i)).
;; These helpers lift that pattern out of the LLM layer into a generic
;; {:dist ... :values ...} box with a gather-by-index lookup.

(defn weighted-draw
  "Value-carrying weighted categorical. Given parallel `values` and `weights`
   vectors, returns {:dist (dist/weighted weights) :values (vec values)}.
   Sampling :dist yields an index i; (draw-value the-box i) recovers the value.
   Generalizes llm/bytes byte-marginals->categorical."
  [values weights]
  {:dist   (dist/weighted (vec weights))
   :values (vec values)})

(defn uniform-draw
  "Value-carrying uniform categorical — weighted-draw with equal weights over
   `values`. Sampling :dist yields an index i; (draw-value the-box i) recovers
   the value."
  [values]
  (weighted-draw values (repeat (count values) 1.0)))

(defn draw-value
  "Gather the value selected by index `idx` from a uniform-/weighted-draw box.
   `idx` may be a plain integer or an MLX integer scalar (as returned by a
   categorical sample)."
  [draw idx]
  (let [i (if (mx/array? idx) (mx/item idx) idx)]
    (nth (:values draw) (int i))))

;; ---------------------------------------------------------------------------
;; Distribution-as-value via a :kind index + SwitchCombinator
;; ---------------------------------------------------------------------------
;;
;; agentmodels often draws *which sub-model* to run (e.g. which agent type,
;; which goal-prior) and then runs it. In GenMLX the idiomatic encoding is:
;;
;;   1. trace a discrete :kind index from a categorical (uniform-/weighted-draw
;;      gives you the value box if the branches carry payloads);
;;   2. dispatch on that index with a SwitchCombinator (combinators.cljs), whose
;;      simulate runs branch `kind` with the supplied branch args.
;;
;; The SwitchCombinator is a first-class generative function, so the chosen
;; sub-model is itself a value living in the trace — distribution-as-value
;; without leaving the GFI. `boxed-choice` packages the two-step convention.

(defn boxed-choice
  "Box a sub-model selection as a single generative function. Returns a
   SwitchCombinator over `branches` (each a generative function). Call its GFI
   ops with args `[kind & branch-args]`: index `kind` selects the branch, and
   the selected sub-model's choices/score live in the resulting trace under the
   :kind path — distribution-as-value without leaving the GFI.

     (def choose (boxed-choice [model-a model-b]))
     (p/simulate choose [kind arg0 arg1])   ; runs branch `kind`"
  [branches]
  (apply comb/switch-combinator branches))
