(ns genmlx.inspect
  "Model introspection — reports compilation, conjugacy, and dispatch
   resolution for any generative function. Pure read, no execution."
  (:require [genmlx.dispatch :as dispatch]))

(def ^:private ops
  [:simulate :generate :update :regenerate :assess :project :propose])

(def ^:private compiled-schema-keys
  {:simulate :compiled-simulate,   :generate :compiled-generate
   :update :compiled-update,       :regenerate :compiled-regenerate
   :assess :compiled-assess,       :project :compiled-project})

(def ^:private prefix-schema-keys
  {:simulate :compiled-prefix,     :generate :compiled-prefix-generate
   :update :compiled-prefix-update, :regenerate :compiled-prefix-regenerate
   :assess :compiled-prefix-assess, :project :compiled-prefix-project})

(defn- compilation-level [schema]
  (let [has-compiled? (some #(get schema (val %)) compiled-schema-keys)
        has-prefix?   (some #(get schema (val %)) prefix-schema-keys)]
    (cond
      (and has-compiled? (:static? schema))       :L1-M2
      (and has-compiled? (:has-branches? schema))  :L1-M4
      has-prefix?                                  :L1-M3
      :else                                        :L0)))

(defn- resolve-dispatch [schema gf-meta]
  (let [custom? (some? (::dispatch/custom-transition gf-meta))]
    (into {}
      (map (fn [op]
             [op (cond
                   custom?                                :custom
                   (get schema (compiled-schema-keys op)) :compiled
                   (get schema (prefix-schema-keys op))   :prefix
                   :else                                  :handler)])
           ops))))

(defn inspect
  "Return a structured report of a generative function's compilation
   state, conjugacy, and dispatch resolution. Pure read — no execution."
  [gf]
  (let [schema (:schema gf)]
    (when schema
      (cond->
        {:trace-sites    (mapv #(select-keys % [:addr :dist-type :deps :static?])
                               (:trace-sites schema))
         :classification (select-keys schema [:static? :has-branches? :has-loops?
                                              :dynamic-addresses?])
         :compilation    (compilation-level schema)
         :dispatch       (resolve-dispatch schema (meta gf))}

        (:has-conjugate? schema)
        (assoc :conjugacy {:pairs (mapv #(select-keys % [:prior-addr :obs-addr :family
                                                         :dependency-type])
                                        (:conjugate-pairs schema))
                           :analytical-eligible
                           (cond-> #{}
                             (:auto-handlers schema) (into [:generate :assess])
                             (:auto-regenerate-transition schema) (conj :regenerate))})

        (seq (:splice-sites schema))
        (assoc :splice-sites (mapv #(select-keys % [:addr :gf-sym :deps])
                                    (:splice-sites schema)))

        (seq (:param-sites schema))
        (assoc :param-sites (:param-sites schema))))))
