(ns genmlx.schema
  "Schema extraction from gen body source forms.
   Walks the quoted source to extract trace sites, splice sites, param sites,
   dependency structure, loop structure, and classify models as static vs dynamic.

   Used by make-gen-fn to attach structural metadata to DynamicGF records,
   enabling Level 1 compilation (L1-M2+) and loop analysis (VIS-M3+).

   The walker threads a binding environment (env) that maps local symbols to
   the set of trace addresses their values depend on. This enables accurate
   dependency tracking between trace sites."
  (:require [clojure.set :as set]))

;; =========================================================================
;; Helpers
;; =========================================================================

(defn- extract-dist-type
  "Extract distribution type keyword from a dist constructor form.
   (dist/gaussian ...) → :gaussian, (dist/bernoulli ...) → :bernoulli, etc.
   Returns :unknown for non-recognizable forms."
  [dist-form]
  (if (and (seq? dist-form) (seq dist-form) (symbol? (first dist-form)))
    (let [sym (first dist-form)
          ns-part (namespace sym)
          name-part (name sym)]
      (if (or (nil? ns-part)
              (= ns-part "dist")
              (= ns-part "genmlx.dist")
              (and (string? ns-part) (.endsWith ns-part ".dist")))
        (keyword name-part)
        :unknown))
    :unknown))

(defn- find-symbols
  "Find all symbols referenced in a form (recursively)."
  [form]
  (cond
    (symbol? form) #{form}
    (seq? form) (into #{} (mapcat find-symbols) form)
    (vector? form) (into #{} (mapcat find-symbols) form)
    (map? form) (into #{} (mapcat find-symbols) (mapcat identity form))
    (set? form) (into #{} (mapcat find-symbols) form)
    :else #{}))

(defn- deps-of-syms
  "Resolve a set of symbols to the union of trace addresses they depend on,
   given the current binding environment (env maps symbols to #{trace-addrs})."
  [env syms]
  (reduce (fn [deps sym]
            (if-let [trace-addrs (get env sym)]
              (into deps trace-addrs)
              deps))
          #{}
          syms))

(defn- compute-deps
  "Compute the set of trace addresses that a form depends on,
   given the current binding environment.
   env maps symbols to #{trace-addrs} they depend on."
  [env form]
  (deps-of-syms env (find-symbols form)))

(defn- head-sym
  "The head symbol of a non-empty list form, or nil."
  [form]
  (when (and (seq? form) (seq form) (symbol? (first form)))
    (first form)))

(defn- call-named?
  "True when form is a call whose head symbol's name is n
   (e.g. (call-named? form \"trace\"))."
  [form n]
  (when-let [h (head-sym form)]
    (= n (name h))))

(defn- contains-gen-call?
  "Does this form recursively contain any trace, splice, or param calls?"
  [form]
  (cond
    (and (seq? form) (seq form))
    (or (when-let [h (head-sym form)]
          (contains? #{"trace" "splice" "param"} (name h)))
        (some contains-gen-call? form))

    (and (vector? form) (seq form))
    (some contains-gen-call? form)

    (map? form)
    (boolean (some contains-gen-call? (mapcat identity form)))

    (set? form)
    (boolean (some contains-gen-call? form))

    :else false))

;; The gen macro binds these local symbols to runtime closures that PRODUCE
;; trace sites. The walker can only see them in head position — `(trace ...)`,
;; `(splice ...)`. A tracing capability "escapes" — becomes invisible to the
;; walker — in three ways:
;;
;;   1. The bare binding is referenced as a VALUE (anywhere other than the head
;;      of its own call form): `(run-sessions trace ...)`,
;;      `(cl/run-controlled-loop trace {...})`. The callee loops internally and
;;      traces hidden sites.
;;
;;   2. An fn/fn* literal that itself contains a trace/splice call is referenced
;;      as a VALUE — passed to an opaque higher-order function or stored in a
;;      let — rather than invoked immediately: `(run! step xs)`,
;;      `(mapv (fn [i] (trace ...)) is)`. The HOF decides how many times (0/1/N)
;;      it runs, which the walker cannot know, so the single-shot compiled path
;;      diverges from the handler.
;;
;;   3. A `letfn`-bound local function whose body traces: `(letfn [(step [x]
;;      (trace :y ...))] (run! step xs))`. The name `step` is neither the bare
;;      binding nor an fn-literal, so it slips past #1 and #2, but it is the same
;;      indirectly-invoked tracing capability as #2.
;;
;; Any of these makes the body NOT statically analyzable: treating it as static
;; makes the L1-M2 compiled path drop/under-count those sites (silently, e.g.
;; when the opaque call is a non-final statement, or when an HOF invokes the
;; capability N times), so observation constraints are never scored correctly.
;; `param` is excluded throughout — it only READS a parameter and produces no
;; trace site, so handing it off hides nothing.
(def ^:private gen-binding-names #{"trace" "splice"})

(defn- contains-trace-call?
  "Does this form recursively contain a trace or splice call? Unlike
   contains-gen-call?, `param` does NOT count — it produces no trace site, so an
   fn that only reads params is not a tracing capability."
  [form]
  (cond
    (and (seq? form) (seq form))
    (or (when-let [h (head-sym form)]
          (contains? gen-binding-names (name h)))
        (some contains-trace-call? form))

    (and (vector? form) (seq form))
    (some contains-trace-call? form)

    (map? form)
    (boolean (some contains-trace-call? (mapcat identity form)))

    (set? form)
    (boolean (some contains-trace-call? form))

    :else false))

(defn- find-all-calls
  "Recursively find all calls matching pred in forms. Recurses into matched
   forms as well — a match's arguments may contain further matches (e.g. a
   trace call nested in another trace's dist-args)."
  [forms pred]
  (let [results (volatile! [])]
    (letfn [(walk [form]
              (when (pred form) (vswap! results conj form))
              (cond
                (seq? form) (run! walk form)
                (vector? form) (run! walk form)
                (map? form) (run! walk (mapcat identity form))
                (set? form) (run! walk form)))]
      (run! walk forms))
    @results))

(defn- trace-addrs-in
  "Keyword addresses of all literal trace calls anywhere inside form."
  [form]
  (->> (find-all-calls [form] #(call-named? % "trace"))
       (map second)
       (filter keyword?)
       set))

(defn- gen-binding-sym?
  "True for an unqualified symbol that is one of the gen-macro runtime bindings
   (trace/splice) — i.e. a tracing capability that must stay in head position."
  [x]
  (and (symbol? x) (nil? (namespace x)) (contains? gen-binding-names (name x))))

(defn- fn-literal?
  "True for an (fn ...) or (fn* ...) form (reader #(...) expands to fn*)."
  [form]
  (and (seq? form) (seq form) (symbol? (first form))
       (contains? #{"fn" "fn*"} (name (first form)))))

(defn- tracing-fn-literal?
  "True for an fn/fn* literal whose body contains a trace/splice call — a
   first-class tracing capability. As a VALUE this is an escape (see ns note);
   as the operator of its own immediate call it runs exactly once and is fine."
  [form]
  (and (fn-literal? form) (contains-trace-call? form)))

(defn- letfn-binds-tracer?
  "True for a `(letfn [(name [args] body...) ...] ...)` form where any local fn
   body contains a trace/splice call. Such a named tracer can be invoked
   indirectly or repeatedly (handed to a HOF, called in a loop), hiding sites
   from the walker — mechanism #3 in the ns note."
  [form]
  (and (seq? form) (seq form) (symbol? (first form)) (= "letfn" (name (first form)))
       (vector? (second form))
       (some (fn [spec] (and (seq? spec) (contains-trace-call? spec)))
             (second form))))

(defn- escapes-gen-binding?
  "True when a tracing capability escapes into code the walker cannot analyze: a
   bare trace/splice binding used as a value, a tracing fn-literal used as a
   value (not immediately invoked), or a letfn-bound tracing function. Quoted
   forms are data, not bindings, so they are not inspected."
  [form]
  (cond
    (and (seq? form) (seq form))
    (let [h (first form)]
      (if (and (symbol? h) (= "quote" (name h)))
        false
        (boolean
         (or
          ;; A letfn that binds a tracing local function escapes (mechanism #3).
          (letfn-binds-tracer? form)
          ;; Head: a symbol head is a call name (fine). A non-symbol head is a
          ;; sub-form — including `((fn ...) args)` immediate invocation, where
          ;; the fn runs once: scan its body for nested escapes but do NOT treat
          ;; the fn itself as an escaping value.
          (when-not (symbol? h) (escapes-gen-binding? h))
          ;; Arguments: a bare gen-op symbol or a tracing fn-literal escapes here.
          (some (fn [a] (or (tracing-fn-literal? a) (escapes-gen-binding? a)))
                (rest form))))))

    (vector? form) (boolean (some #(or (tracing-fn-literal? %) (escapes-gen-binding? %)) form))
    (set? form)    (boolean (some #(or (tracing-fn-literal? %) (escapes-gen-binding? %)) form))
    (map? form)    (boolean (some #(or (tracing-fn-literal? %) (escapes-gen-binding? %))
                                  (mapcat identity form)))
    (gen-binding-sym? form) true
    :else false))

;; =========================================================================
;; Walker — threads acc (accumulator) and env (binding environment)
;; =========================================================================

(declare walk-form)

(defn- walk-forms
  "Walk a sequence of forms, accumulating into acc with given env."
  [acc env forms]
  (reduce (fn [acc form] (walk-form acc env form)) acc forms))

(defn- bare-trace-alias-addr
  "If `val-form` is a bare trace call to a static (keyword) address —
   (trace :addr <dist> ...) — return that address keyword, else nil. Recorded
   under env's ::arg-aliases so conjugacy classification can tell a DIRECT
   binding (mu = (trace :mu ...)) from an affine REBINDING that reuses the
   symbol name (mu = (mx/add mu 5)) — the latter must not classify :direct
   (genmlx-1thx). Also lets a direct alias with a non-matching name (m =
   (trace :mu ...)) be recognized as direct."
  [val-form]
  (when (and (seq? val-form) (seq val-form)
             (symbol? (first val-form))
             (= "trace" (name (first val-form)))
             (keyword? (second val-form)))
    (second val-form)))

(defn- handle-trace [acc env args]
  (let [addr-form (first args)
        dist-form (second args)
        static? (keyword? addr-form)
        addr (if static? addr-form :dynamic)
        dist-type (extract-dist-type dist-form)
        dist-args (when (and (seq? dist-form) (seq dist-form))
                    (vec (rest dist-form)))
        deps (compute-deps env dist-form)
        ;; Walk sub-forms first (handles nested traces in dist args)
        acc' (walk-forms acc env args)]
    (-> acc'
        (update :trace-sites conj {:addr addr
                                   :addr-form addr-form
                                   :dist-type dist-type
                                   :dist-args (or dist-args [])
                                   :deps deps
                                   ;; direct trace-alias provenance for dist-arg
                                   ;; symbols (genmlx-1thx)
                                   :arg-aliases (or (::arg-aliases env) {})
                                   :static? static?})
        (cond-> (not static?) (assoc :dynamic-addresses? true)))))

(defn- handle-splice [acc env args]
  (let [addr-form (first args)
        gf-form (second args)
        splice-args (vec (drop 2 args))
        static? (keyword? addr-form)
        addr (if static? addr-form :dynamic)
        deps (compute-deps env (cons 'splice args))
        ;; Walk sub-forms
        acc' (walk-forms acc env args)]
    (-> acc'
        (update :splice-sites conj {:addr addr
                                    :addr-form addr-form
                                    :gf-form gf-form
                                    :splice-args splice-args
                                    :deps deps
                                    :static? static?})
        (cond-> (not static?) (assoc :dynamic-addresses? true)))))

(defn- handle-param [acc env args]
  (let [acc' (if-let [default (second args)]
               (walk-form acc env default)
               acc)]
    (update acc' :param-sites conj {:name (first args)
                                    :default-form (second args)})))

(defn- handle-let [acc env args]
  (if (empty? args)
    acc
    (let [bindings-form (first args)
          body (rest args)
          pairs (when (and (vector? bindings-form) (seq bindings-form))
                  (partition 2 bindings-form))
          ;; Process bindings sequentially, updating env
          [acc' env'] (reduce
                       (fn [[acc env] [sym val-form]]
                         (let [acc' (walk-form acc env val-form)
                               ;; Deps flow from referenced bindings AND from
                               ;; any trace call inside the value form — literal
                               ;; (trace :x ...) or wrapped, e.g.
                               ;; (mx/add (trace :x ...) 1).
                               sym-deps (into (compute-deps env val-form)
                                              (trace-addrs-in val-form))]
                           (if (symbol? sym)
                             ;; Maintain direct trace-alias provenance: record
                             ;; sym -> :addr when bound directly to (trace :addr
                             ;; ...), and clear it on any rebinding so an affine
                             ;; reuse of the name is not mistaken for a direct
                             ;; natural parameter (genmlx-1thx).
                             (let [env1 (if (seq sym-deps) (assoc env sym sym-deps) env)
                                   alias-addr (bare-trace-alias-addr val-form)
                                   env2 (cond
                                          alias-addr (assoc-in env1 [::arg-aliases sym] alias-addr)
                                          (get-in env1 [::arg-aliases sym]) (update env1 ::arg-aliases dissoc sym)
                                          :else env1)]
                               [acc' env2])
                             ;; Destructuring binding — conservatively give
                             ;; every bound symbol the full deps of the value
                             ;; form (over-approximation keeps dep edges), and
                             ;; clear any aliases for the bound symbols.
                             (let [env1 (if (seq sym-deps)
                                          (reduce (fn [e s] (assoc e s sym-deps))
                                                  env (find-symbols sym))
                                          env)
                                   env2 (if (::arg-aliases env1)
                                          (update env1 ::arg-aliases
                                                  #(apply dissoc % (find-symbols sym)))
                                          env1)]
                               [acc' env2]))))
                       [acc env]
                       (or pairs []))]
      (walk-forms acc' env' body))))

(defn- handle-branching
  ;; if/when/cond/and/or/case — walk every arg, but a trace anywhere in `scan`
  ;; means execution is conditional, so flag :has-branches?. `scan` is the full
  ;; arg list for everything except case, where the dispatch expr is skipped.
  [acc env args scan]
  (cond-> (walk-forms acc env args)
    (some contains-gen-call? scan) (assoc :has-branches? true)))

;; =========================================================================
;; Loop analysis helpers (VIS-M3)
;; =========================================================================

(defn- detect-addr-pattern
  "Detect the address pattern in a trace address form.
   Returns {:type :keyword-str/:keyword-sym/:static/:unknown ...}"
  [addr-form]
  (cond
    ;; Static keyword: :y, :slope, etc.
    (keyword? addr-form)
    {:type :static :addr addr-form}

    ;; (keyword (str "prefix" sym)) or (keyword (str "prefix" sym "suffix"))
    (and (seq? addr-form)
         (= 2 (count addr-form))
         (symbol? (first addr-form))
         (= "keyword" (name (first addr-form)))
         (seq? (second addr-form))
         (symbol? (first (second addr-form)))
         (= "str" (name (first (second addr-form)))))
    (let [parts (vec (rest (second addr-form)))]
      (cond
        ;; (keyword (str "prefix" sym))
        (and (= 2 (count parts))
             (string? (first parts))
             (symbol? (second parts)))
        {:type :keyword-str
         :prefix (first parts)
         :index-sym (second parts)
         :suffix nil}

        ;; (keyword (str "prefix" sym "suffix"))
        (and (= 3 (count parts))
             (string? (first parts))
             (symbol? (second parts))
             (string? (nth parts 2)))
        {:type :keyword-str
         :prefix (first parts)
         :index-sym (second parts)
         :suffix (nth parts 2)}

        :else
        {:type :unknown :form addr-form}))

    ;; (keyword sym)
    (and (seq? addr-form)
         (= 2 (count addr-form))
         (symbol? (first addr-form))
         (= "keyword" (name (first addr-form)))
         (symbol? (second addr-form)))
    {:type :keyword-sym
     :index-sym (second addr-form)}

    :else
    {:type :unknown :form addr-form}))

(defn- analyze-doseq-bindings
  "Analyze doseq binding form. Returns {:element-sym :index-sym :collection-form}."
  [bindings]
  (when (and (vector? bindings) (>= (count bindings) 2))
    (let [bind-form (first bindings)
          coll-form (second bindings)]
      (cond
        ;; [x coll] — simple element binding
        (symbol? bind-form)
        {:element-sym bind-form :index-sym nil :collection-form coll-form}

        ;; [[j x] coll] — destructured pair (e.g., map-indexed)
        (and (vector? bind-form) (= 2 (count bind-form)))
        (let [[a b] bind-form]
          (if (and (symbol? a) (symbol? b))
            {:index-sym a :element-sym b :collection-form coll-form}
            {:element-sym bind-form :index-sym nil :collection-form coll-form}))

        :else
        {:element-sym bind-form :index-sym nil :collection-form coll-form}))))

(defn- analyze-dotimes-bindings
  "Analyze dotimes binding form. Returns {:index-sym :count-form}."
  [bindings]
  (when (and (vector? bindings) (>= (count bindings) 2))
    {:index-sym (first bindings)
     :count-form (second bindings)}))

(defn- analyze-for-bindings
  "Analyze for binding form. Returns {:element-sym :collection-form} or nil if complex."
  [bindings]
  (when (and (vector? bindings) (>= (count bindings) 2))
    (let [bind-form (first bindings)]
      (when-not (keyword? bind-form)
        {:element-sym bind-form
         :index-sym nil
         :collection-form (second bindings)}))))

(defn- infer-count-arg-idx
  "Infer which gen-fn parameter determines the loop count.
   Returns index into params vector, or -1."
  [count-form params]
  (let [pv (vec params)]
    (cond
      (integer? count-form)
      -1

      ;; (count sym) where sym is a param
      (and (seq? count-form)
           (= 2 (count count-form))
           (symbol? (first count-form))
           (= "count" (name (first count-form)))
           (symbol? (second count-form)))
      (.indexOf pv (second count-form))

      ;; Plain symbol that is a param
      (symbol? count-form)
      (.indexOf pv count-form)

      :else -1)))

(defn- contains-nested-loop?
  "Check if forms contain a nested loop construct."
  [forms]
  (some (fn check [form]
          (cond
            (and (seq? form) (seq form) (symbol? (first form)))
            (let [n (name (first form))]
              (or (#{"doseq" "dotimes" "for" "loop"} n)
                  (some check (rest form))))
            (seq? form) (some check form)
            (vector? form) (some check form)
            (map? form) (some check (mapcat identity form))
            (set? form) (some check form)
            :else false))
        forms))

(defn- contains-branch-with-trace?
  "Check if forms contain a branch (if/when/cond) that has trace calls."
  [forms]
  (let [trace-call? #(call-named? % "trace")]
    (some (fn check [form]
            (cond
              (head-sym form)
              (let [n (name (head-sym form))]
                (if (#{"if" "when" "when-not" "cond" "case" "if-let" "when-let" "if-not"} n)
                  (seq (find-all-calls (rest form) trace-call?))
                  (some check (rest form))))
              (seq? form) (some check form)
              (vector? form) (some check form)
              (map? form) (some check (mapcat identity form))
              (set? form) (some check form)
              :else false))
          forms)))

(defn- extract-loop-trace-sites
  "Extract trace sites from loop body with addr patterns and dep classification.
   loop-syms: set of symbols bound by the loop (element + index).
   env: outer binding environment."
  [body loop-syms env]
  (let [trace-call? #(call-named? % "trace")
        splice-call? #(call-named? % "splice")
        param-call? #(call-named? % "param")
        traces (find-all-calls body trace-call?)
        splices (find-all-calls body splice-call?)
        params (find-all-calls body param-call?)]
    {:trace-sites
     (mapv (fn [form]
             (let [addr-form (second form)
                   dist-form (nth form 2 nil)
                   dist-type (extract-dist-type dist-form)
                   dist-args (when (and (seq? dist-form) (seq dist-form))
                               (vec (rest dist-form)))
                   all-syms (find-symbols dist-form)
                   element-deps (set/intersection all-syms loop-syms)
                   outer-dep-syms (set/difference all-syms loop-syms)
                   outer-deps (deps-of-syms env outer-dep-syms)]
               {:addr :dynamic
                :addr-form addr-form
                :addr-pattern (detect-addr-pattern addr-form)
                :dist-type dist-type
                :dist-args (or dist-args [])
                :element-deps element-deps
                :outer-deps outer-deps
                :static-dist-type? (not= dist-type :unknown)}))
           traces)
     :splice-sites (mapv (fn [form] {:addr-form (second form)}) splices)
     :param-sites (mapv (fn [form] {:name (second form)}) params)}))

(defn- classify-loop
  "Classify whether a loop is homogeneous and rewritable."
  [loop-type trace-sites splice-sites param-sites body]
  (let [blockers (cond-> []
                   (seq splice-sites)
                   (conj "splice in loop body")

                   (seq param-sites)
                   (conj "param in loop body")

                   (= loop-type :loop)
                   (conj "loop/recur not analyzable")

                   (contains-nested-loop? body)
                   (conj "nested loop")

                   (contains-branch-with-trace? body)
                   (conj "branch with trace in loop body")

                   (some #(= :unknown (:type (:addr-pattern %))) trace-sites)
                   (conj "unrecognized address pattern"))
        dist-types (set (map :dist-type trace-sites))
        arg-counts (set (map (comp count :dist-args) trace-sites))
        homogeneous? (and (= 1 (count dist-types))
                          (= 1 (count arg-counts))
                          (not (contains? dist-types :unknown)))
        blockers (cond-> blockers
                   (and (not homogeneous?) (pos? (count trace-sites)))
                   (conj "heterogeneous distribution types"))]
    {:homogeneous? homogeneous?
     :rewritable? (and homogeneous? (empty? blockers))
     :rewrite-blockers blockers}))

(defn- handle-loop-form [acc env args]
  ;; doseq/dotimes/for: first arg is bindings vector, rest is body
  ;; Enhanced for VIS-M3: extracts structured loop metadata into :loop-sites
  (let [bindings-form (first args)
        body (rest args)
        has-trace? (some contains-gen-call? body)
        ;; Walk sub-forms first (backward compat: accumulates trace sites + flags)
        acc' (walk-forms acc env args)]
    (if-not has-trace?
      acc'
      ;; Loop has trace calls — extract structured metadata
      (let [;; Determine loop type from the calling context
            ;; (handle-call passes the form type via the acc's :current-loop-type)
            loop-type (or (:current-loop-type acc) :doseq)
            ;; Analyze bindings based on type
            {:keys [element-sym index-sym collection-form]
             count-form* :count-form}
            (case loop-type
              :doseq (analyze-doseq-bindings bindings-form)
              :dotimes (analyze-dotimes-bindings bindings-form)
              :for (analyze-for-bindings bindings-form)
              nil)
            ;; Collect loop binding symbols for dep classification
            loop-syms (into #{}
                            (filter symbol?)
                            [element-sym index-sym])
            ;; Extract loop-specific trace/splice/param sites
            loop-body-info (extract-loop-trace-sites body loop-syms env)
            loop-traces (:trace-sites loop-body-info)
            loop-splices (:splice-sites loop-body-info)
            loop-params (:param-sites loop-body-info)
            ;; Classify rewritability
            classification (classify-loop loop-type loop-traces loop-splices loop-params body)
            ;; Infer count expression
            count-form (cond
                         (= loop-type :dotimes)
                         count-form*
                         ;; For doseq/for, count is (count collection)
                         collection-form
                         (list 'count collection-form)
                         :else nil)
            ;; Build loop-site entry
            loop-site (merge
                       {:type loop-type
                        :bindings-form bindings-form
                        :element-sym element-sym
                        :index-sym index-sym
                        :collection-form collection-form
                        :count-form count-form
                        :trace-sites loop-traces
                        :splice-sites loop-splices
                        :param-sites loop-params}
                       classification)]
        (-> acc'
            (assoc :has-loops? true)
            (update :loop-sites (fnil conj []) loop-site))))))

(defn- handle-loop-loop [acc env args]
  ;; loop: like let, has bindings + body.
  ;; loop/recur is not analyzable for VIS-M3 (arbitrary state threading),
  ;; but we set :has-loops? if the body contains trace calls.
  (if (empty? args)
    acc
    (let [bindings-form (first args)
          body (rest args)
          has-trace? (some contains-gen-call? body)
          pairs (when (and (vector? bindings-form) (seq bindings-form))
                  (partition 2 bindings-form))
          [acc' env'] (reduce
                       (fn [[acc env] [sym val-form]]
                         (let [acc' (walk-form acc env val-form)]
                           (if (symbol? sym)
                             (let [val-deps (compute-deps env val-form)
                                   env' (if (seq val-deps)
                                          (assoc env sym val-deps)
                                          env)]
                               [acc' env'])
                             [acc' env])))
                       [acc env]
                       (or pairs []))
          acc'' (walk-forms acc' env' body)]
      (cond-> acc''
        has-trace? (assoc :has-loops? true)))))

(defn- handle-fn [acc env args]
  (let [has-name? (symbol? (first args))
        rest-args (if has-name? (rest args) args)]
    (if (vector? (first rest-args))
      ;; Single arity: (fn [params] body...)
      (let [params (first rest-args)
            body (rest rest-args)
            ;; Remove fn params from env (they shadow outer bindings)
            env' (reduce dissoc env params)]
        (walk-forms acc env' body))
      ;; Multi-arity: (fn ([params] body...) ([params] body...))
      (reduce (fn [acc arity]
                (if (and (sequential? arity) (seq arity) (vector? (first arity)))
                  (let [params (first arity)
                        body (rest arity)
                        env' (reduce dissoc env params)]
                    (walk-forms acc env' body))
                  acc))
              acc
              rest-args))))

(defn- handle-letfn [acc env args]
  ;; letfn: (letfn [(f1 [x] body1) (f2 [y] body2)] body...)
  (if (empty? args)
    acc
    (let [fn-defs (first args)
          body (rest args)
          ;; Walk each fn definition body
          acc' (reduce (fn [acc fd]
                         (if (and (sequential? fd) (> (count fd) 2))
                           (let [params (second fd)
                                 fn-body (drop 2 fd)
                                 env' (if (vector? params)
                                        (reduce dissoc env params)
                                        env)]
                             (walk-forms acc env' fn-body))
                           acc))
                       acc
                       (if (vector? fn-defs) fn-defs []))]
      (walk-forms acc' env body))))

(defn- handle-call [acc env head args]
  (let [n (name head)]
    (case n
      "trace" (handle-trace acc env args)
      "splice" (handle-splice acc env args)
      "param" (handle-param acc env args)
      "let" (handle-let acc env args)
      "if" (handle-branching acc env args args)
      "when" (handle-branching acc env args args)
      "when-not" (handle-branching acc env args args)
      "when-let" (handle-branching acc env args args)
      "if-let" (handle-branching acc env args args)
      "if-not" (handle-branching acc env args args)
      "cond" (handle-branching acc env args args)
      "case" (handle-branching acc env args (rest args))
      "and" (handle-branching acc env args args)
      "or" (handle-branching acc env args args)
      "do" (walk-forms acc env args)
      "doseq" (handle-loop-form (assoc acc :current-loop-type :doseq) env args)
      "dotimes" (handle-loop-form (assoc acc :current-loop-type :dotimes) env args)
      "for" (handle-loop-form (assoc acc :current-loop-type :for) env args)
      "loop" (handle-loop-loop acc env args)
      "fn" (handle-fn acc env args)
      "defn" (handle-fn acc env (rest args))
      "letfn" (handle-letfn acc env args)
      "quote" acc
      ;; Default: walk all sub-forms
      (walk-forms acc env args))))

(defn- walk-form
  "Walk a single form, accumulating trace/splice/param sites and flags."
  [acc env form]
  (cond
    ;; List with symbol head → might be a special form or gen call
    (and (seq? form) (seq form) (symbol? (first form)))
    (handle-call acc env (first form) (rest form))

    ;; List without symbol head (e.g., ((fn ...) arg)) → walk all children
    (and (seq? form) (seq form))
    (walk-forms acc env form)

    ;; Vector → walk elements (traces can appear inside vector literals)
    (and (vector? form) (seq form))
    (walk-forms acc env form)

    ;; Map literal → walk keys and values (traces can hide in either)
    (and (map? form) (seq form))
    (walk-forms acc env (mapcat identity form))

    ;; Set literal → walk elements
    (and (set? form) (seq form))
    (walk-forms acc env (seq form))

    ;; Anything else (keyword, symbol, number, string, nil, empty colls)
    :else acc))

;; =========================================================================
;; Topological sort of trace sites by dependency order
;; =========================================================================

(defn- topo-sort
  "Topological sort of static trace addresses by dependency order.
   Returns vector where each trace comes after all its dependencies."
  [trace-sites]
  (let [static-sites (filter :static? trace-sites)
        addr-set (set (map :addr static-sites))
        ;; Only keep deps that are within our static trace set
        dep-map (into {} (map (fn [ts]
                                [(:addr ts)
                                 (set/intersection (:deps ts) addr-set)])
                              static-sites))]
    (loop [remaining (set (keys dep-map))
           result []]
      (if (empty? remaining)
        result
        (let [ready (first (filter (fn [a]
                                     (empty? (set/intersection
                                              (get dep-map a #{})
                                              remaining)))
                                   remaining))]
          (if ready
            (recur (disj remaining ready) (conj result ready))
            ;; Fallback: add remaining in any order (shouldn't happen in valid models)
            (into result remaining)))))))

;; =========================================================================
;; Main API
;; =========================================================================

(defn extract-schema
  "Extract schema from a gen source form.
   Source form is (params-vec body-form1 body-form2 ...) as captured by the gen macro.
   Returns a schema map with trace sites, splice sites, param sites,
   loop sites, dependency structure, and classifications."
  [source]
  (when source
    (let [params (first source)
          body (rest source)
          init-acc {:trace-sites []
                    :splice-sites []
                    :param-sites []
                    :loop-sites []
                    :dynamic-addresses? false
                    :has-branches? false
                    :has-loops? false}
          result (walk-forms init-acc {} body)
          ;; Enrich loop-sites with count-arg-idx
          result (update result :loop-sites
                         (fn [loops]
                           (mapv (fn [ls]
                                   (assoc ls :count-arg-idx
                                          (infer-count-arg-idx
                                           (:count-form ls) params)))
                                 (or loops []))))
          ;; A body that hands `trace`/`splice` to opaque code has hidden trace
          ;; sites the walker cannot see — it is not statically analyzable and
          ;; must take the handler path (no compilation). See escapes-gen-binding?.
          opaque-escape? (boolean (some escapes-gen-binding? body))]
      (-> result
          (dissoc :current-loop-type)
          (assoc :params (vec params)
                 :return-form (last body)
                 :dep-order (topo-sort (:trace-sites result))
                 :opaque-gen-escape? opaque-escape?
                 ;; CLAUDE.md definition: static = all keyword-literal
                 ;; addresses, no branches, no loops, no splices. Splices
                 ;; were previously compensated for only inside the
                 ;; compiled.cljs builders (genmlx-q3x2).
                 :static? (and (not (:dynamic-addresses? result))
                               (not (:has-branches? result))
                               (not (:has-loops? result))
                               (empty? (:splice-sites result))
                               (not opaque-escape?)))))))
