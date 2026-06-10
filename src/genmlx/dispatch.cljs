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
)

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
  (some #(resolve-transition % op schema opts) stack))

;; ---------------------------------------------------------------------------
;; Handler substitution
;; ---------------------------------------------------------------------------

(defn with-handler
  "Attach a custom handler transition to a generative function via metadata.
   `transition` is either:

   - a single transition fn (fn [state addr dist] -> [value state']) — used
     for EVERY GFI op. Only correct when the transition is genuinely
     op-agnostic; a generate-flavored transition used for update/regenerate
     silently runs generate semantics (genmlx-xwxh).
   - a map {op -> transition} with ops from #{:simulate :generate :update
     :regenerate :assess :project :propose} — the dispatcher picks the
     entry per op and falls back to the STANDARD transition for ops the
     map omits. Prefer this for middleware like grammar constraints.

   The dispatcher wraps the chosen transition into run-handler with the
   correct init-state per GFI operation.

   Usage:
     (with-handler llm {:generate (wrap-grammar generate-transition grammar)})
     (with-handler model (wrap-analytical generate-transition conjugacy-map))"
  [gf transition]
  (vary-meta gf assoc ::custom-transition transition))

(defn with-dispatch
  "Attach a custom dispatch function to a generative function via metadata.
   The dispatch function has signature:

       (fn [op gf args key opts] -> gfi-result)

   where op is :simulate|:generate|:update|:regenerate|:assess|:project|:propose.
   Use this for execution modes that need custom init-state or post-processing
   per operation (e.g., exact enumeration).

   Prefer with-handler when a transition substitution suffices."
  [gf dispatch-fn]
  (vary-meta gf assoc ::custom-dispatch dispatch-fn))
