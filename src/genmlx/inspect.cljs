(ns genmlx.inspect
  "Model introspection — reports compilation, conjugacy, and dispatch
   resolution for any generative function. Pure read, no execution.
   Uses the actual dispatcher stack via dyn/resolve-dispatch."
  (:require [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]))

(defn batched-splice-eligibility
  "How gf executes when SPLICED under a batched handler (genmlx-y3ls):
   :dynamic          — DynamicGF, runs the batched sub-handler
   :native           — implements IBatchedSplice (fast path: Map/Unfold/
                       Switch/Scan/Mix, and the Mask/contramap/map-retval
                       wrappers, which delegate to their inner)
   :scalar-fallback  — GFI value with neither: the N-fold scalar host loop
                       (handler/combinator-batched-fallback, ~2-4 ms/particle
                       on Thor sm_110 — Recurse is deliberately here)
   nil               — not a generative function."
  [gf]
  (cond
    (:body-fn gf) :dynamic
    (satisfies? p/IBatchedSplice gf) :native
    (satisfies? p/IGenerativeFunction gf) :scalar-fallback
    :else nil))

(def ^:private ops
  [:simulate :generate :update :regenerate :assess :project :propose])

(def ^:private compiled-schema-keys
  [:compiled-simulate :compiled-generate :compiled-update
   :compiled-regenerate :compiled-assess :compiled-project])

(def ^:private prefix-schema-keys
  [:compiled-prefix :compiled-prefix-generate :compiled-prefix-update
   :compiled-prefix-regenerate :compiled-prefix-assess :compiled-prefix-project])

(defn- compilation-level [schema]
  (let [has-compiled? (some schema compiled-schema-keys)
        has-prefix?   (some schema prefix-schema-keys)]
    (cond
      (and has-compiled? (:static? schema))       :L1-M2
      (and has-compiled? (:has-branches? schema))  :L1-M4
      has-prefix?                                  :L1-M3
      :else                                        :L0)))

(defn- resolve-dispatch [gf]
  (into {}
    (map (fn [op]
           [op (:label (dyn/resolve-dispatch gf op))])
         ops)))

(defn inspect
  "Return a structured report of a generative function's compilation
   state, conjugacy, and dispatch resolution. Pure read — no execution."
  [gf]
  (let [schema (:schema gf)]
    (if-not schema
      ;; Combinators / wrappers carry no schema, but their batched-splice
      ;; eligibility is still the question worth asking (genmlx-y3ls).
      (when-let [e (batched-splice-eligibility gf)]
        {:type (pr-str (type gf)) :batched-splice e})
      (cond->
        {:trace-sites    (mapv #(select-keys % [:addr :dist-type :deps :static?])
                               (:trace-sites schema))
         :classification (select-keys schema [:static? :has-branches? :has-loops?
                                              :dynamic-addresses?])
         :compilation    (compilation-level schema)
         :batched-splice (batched-splice-eligibility gf)
         :dispatch       (resolve-dispatch gf)}

        (:has-conjugate? schema)
        (assoc :conjugacy {:pairs (mapv #(select-keys % [:prior-addr :obs-addr :family
                                                         :dependency-type])
                                        (:conjugate-pairs schema))
                           :analytical-eligible
                           (cond-> #{}
                             (:auto-handlers schema) (into [:generate :assess])
                             (:auto-regenerate-transition schema) (conj :regenerate))})

        (seq (:splice-sites schema))
        (assoc :splice-sites (mapv #(select-keys % [:addr :gf-form :deps])
                                    (:splice-sites schema)))

        (seq (:param-sites schema))
        (assoc :param-sites (:param-sites schema))))))
