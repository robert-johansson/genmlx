(ns genmlx.inference.cost
  "Compute / cost meter (genmlx-i0s4): inference effort as a first-class VALUE.

   A CostMeter is a plain immutable map of monotonic, additive counters:

     {:forced-evals n  ; eval! GPU dispatches    (the honest compute signal)
      :items n         ; mx/item reads
      :clj-reads n     ; mx/->clj reads
      :gfi-ops n       ; caller-supplied GFI-op count (not auto-instrumented)
      :particles n     ; particles processed
      :steps n         ; steppable steps taken
      :structural n    ; static dep-graph blast radius (per model, once)
      :wall-ns n}      ; optional wall-clock (off by default)

   `measure` brackets a thunk and reports the forced-eval/item/clj-read DELTAS it
   incurred, read from the honest mlx.cljs membrane counters (the sole GPU-
   dispatch boundary). `cost+` is additive merge so a scheduler can fold per-step
   costs into a running total. This namespace owns NO mutable state of its own —
   the only mutation is the audited mlx.cljs membrane counters."
  (:require [genmlx.mlx :as mx]
            [genmlx.dep-graph :as dg]))

(def zero
  "The empty CostMeter (all fields present so cost+ is total)."
  {:forced-evals 0 :items 0 :clj-reads 0 :gfi-ops 0
   :particles 0 :steps 0 :structural 0 :wall-ns 0})

(defn cost+
  "Additive merge of CostMeters (associative + commutative on every field).
   Always returns a meter with the full key set."
  [& meters]
  (reduce (fn [acc m] (merge-with + acc m)) zero meters))

(def ^:dynamic *wall-clock?*
  "When true, `measure` brackets the thunk with Bun.nanoseconds and reports
   :wall-ns. Off by default so the hot path pays nothing (and stays sync — no
   promesa). Bun.nanoseconds is a synchronous monotonic-clock read."
  false)

(defn- now-ns []
  (if (exists? js/Bun)
    (js/Number (.nanoseconds (.-Bun js/globalThis)))
    0))

(defn measure
  "Run `thunk`; return {:result r :cost meter} where meter holds the
   forced-eval / item / clj-read DELTAS the thunk incurred (and :wall-ns when
   *wall-clock?* is bound true). Uses read-before / read-after deltas, so it
   composes with nesting and never assumes exclusive counter ownership."
  [thunk]
  (let [before (mx/read-cost-counters)
        t0 (when *wall-clock?* (now-ns))
        result (thunk)
        after (mx/read-cost-counters)]
    {:result result
     :cost (assoc zero
                  :forced-evals (- (:forced-evals after) (:forced-evals before))
                  :items (- (:items after) (:items before))
                  :clj-reads (- (:clj-reads after) (:clj-reads before))
                  :wall-ns (if *wall-clock?* (- (now-ns) t0) 0))}))

(defn measure-step
  "Like `measure`, plus :steps 1 and a caller-supplied :particles count — the
   shape a steppable/scheduler folds per advance."
  [thunk & {:keys [particles] :or {particles 0}}]
  (let [{:keys [result cost]} (measure thunk)]
    {:result result :cost (cost+ cost {:steps 1 :particles particles})}))

(defn structural-cost
  "Static structural cost of a model = sum over latent nodes of
   |descendants(node)| — a blast-radius-weighted measure of how much recompute a
   change to each latent triggers. Computed once per model (no GPU). A 3-site
   chain a->b->c gives 2 + 1 + 0 = 3."
  [model]
  (let [g (dg/build-dep-graph (:schema model))]
    (reduce + 0 (map (fn [n] (count (dg/find-descendants g n))) (:nodes g)))))
