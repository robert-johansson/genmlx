(ns genmlx.inference.analytical
  "Composable analytical middleware for handler transitions.

   Analytical middleware intercepts trace sites with specific distribution
   types and replaces sampling+scoring with exact computation. The model
   code stays unchanged — the same gen function works under any handler.

   Pattern (Ring-style):
     (-> base-transition
         (wrap-analytical kalman-dispatch)
         (wrap-analytical hmm-dispatch))

   Each dispatch map: {dist-type-keyword -> (fn [state addr dist] -> [value state'])}

   Middleware state lives in namespaced keys (e.g. :kalman-belief, :hmm-belief)
   and does NOT touch :score/:weight. The caller decides how to aggregate
   the analytical log-likelihoods."
  (:require [genmlx.handler :as h]))

(defn wrap-analytical
  "Wrap a handler transition with analytical dispatch.

   dispatch-map: {keyword -> (fn [state addr dist] -> [value state'])}
   base-transition: the transition to delegate to for non-matching sites

   Returns a new transition function that checks (:type dist) against
   the dispatch map before falling through to base-transition."
  [base-transition dispatch-map]
  (fn [state addr dist]
    (if-let [handler-fn (get dispatch-map (:type dist))]
      (handler-fn state addr dist)
      (base-transition state addr dist))))

(defn compose-middleware
  "Compose multiple dispatch maps into a single transition.

   base-transition: the fallback transition (e.g. h/generate-transition)
   dispatch-maps:   sequence of {dist-type -> handler-fn} maps

   Returns a single transition function. Later maps take priority
   (last dispatch-map is checked first)."
  [base-transition & dispatch-maps]
  (reduce wrap-analytical base-transition dispatch-maps))
