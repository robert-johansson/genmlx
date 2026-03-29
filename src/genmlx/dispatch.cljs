(ns genmlx.dispatch
  "Data-driven dispatch for GFI operations.

   Replaces the hardcoded cond ladders in DynamicGF with a dispatcher
   stack — a sequence of dispatch functions that each return a dispatch-spec
   or nil. The first non-nil spec wins. Adding a new execution mode means
   adding a dispatcher to the stack, not editing DynamicGF.

   The dispatcher stack is defined in dynamic.cljs (where the dispatchers
   have access to the run-* helpers). This namespace defines the protocol,
   the stack walk, and the with-handler mechanism.

   See ARCHITECTURE.md Part II, Section 2.5."
  (:require [genmlx.schemas :as schemas]))

;; ---------------------------------------------------------------------------
;; Dispatcher protocol
;; ---------------------------------------------------------------------------

(defprotocol IDispatcher
  (resolve-transition [this op schema opts]
    "Return a dispatch-spec or nil.

     op:     :simulate | :generate | :update | :regenerate | :assess | :project | :propose
     schema: the model's schema map (trace-sites, compiled fns, conjugacy, etc.)
     opts:   operation context map, keys depend on op:
             :gf          - the generative function (always present)
             :constraints - choice map (generate, update, assess)
             :trace       - existing trace (update, regenerate, project)
             :selection   - address selection (regenerate, project)

     Returns: {:run (fn [gf args key opts] -> gfi-result)
               :score-type :joint|:marginal|:collapsed|:beam-marginal}
     or nil if this dispatcher cannot handle the operation."))

;; ---------------------------------------------------------------------------
;; Stack resolution
;; ---------------------------------------------------------------------------

(defn resolve
  "Walk the dispatcher stack, return the first non-nil dispatch-spec.
   The stack must have a fallback dispatcher at the bottom that always
   returns non-nil (HandlerDispatcher)."
  [stack op schema opts]
  (loop [dispatchers stack]
    (when (seq dispatchers)
      (or (resolve-transition (first dispatchers) op schema opts)
          (recur (rest dispatchers))))))

;; ---------------------------------------------------------------------------
;; Handler substitution
;; ---------------------------------------------------------------------------

(defn with-handler
  "Attach a custom transition to a generative function via metadata.
   The CustomTransitionDispatcher checks for ::custom-transition in
   the gen-fn's metadata and uses it if present.

   Usage:
     (with-handler model enumerate-transition)
     (with-handler llm (wrap-grammar base-transition grammar))"
  [gf transition]
  (vary-meta gf assoc ::custom-transition transition))
