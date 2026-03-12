(ns genmlx.schema
  "Schema extraction from gen body source forms.
   Walks the quoted source to extract trace sites, splice sites, param sites,
   dependency structure, and classify models as static vs dynamic.

   Used by make-gen-fn to attach structural metadata to DynamicGF records,
   enabling Level 1 compilation (L1-M2+).

   The walker threads a binding environment (env) that maps local symbols to
   the set of trace addresses their values depend on. This enables accurate
   dependency tracking between trace sites."
  (:require [clojure.set]))

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
    :else #{}))

(defn- compute-deps
  "Compute the set of trace addresses that a form depends on,
   given the current binding environment.
   env maps symbols to #{trace-addrs} they depend on."
  [env form]
  (let [syms (find-symbols form)]
    (reduce (fn [deps sym]
              (if-let [trace-addrs (get env sym)]
                (into deps trace-addrs)
                deps))
            #{}
            syms)))

(defn- contains-gen-call?
  "Does this form recursively contain any trace, splice, or param calls?"
  [form]
  (cond
    (and (seq? form) (seq form))
    (let [head (first form)]
      (or (and (symbol? head)
               (let [n (name head)]
                 (or (= n "trace") (= n "splice") (= n "param"))))
          (some contains-gen-call? form)))

    (and (vector? form) (seq form))
    (some contains-gen-call? form)

    :else false))

;; =========================================================================
;; Walker — threads acc (accumulator) and env (binding environment)
;; =========================================================================

(declare walk-form)

(defn- walk-forms
  "Walk a sequence of forms, accumulating into acc with given env."
  [acc env forms]
  (reduce (fn [acc form] (walk-form acc env form)) acc forms))

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
  (let [acc' (when (second args)
               (walk-form acc env (second args)))]
    (update (or acc' acc) :param-sites conj {:name (first args)
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
                          (let [acc' (walk-form acc env val-form)]
                            (if (symbol? sym)
                              ;; Simple symbol binding — track deps in env
                              (let [val-deps (compute-deps env val-form)
                                    ;; If val-form is a trace call, include the trace addr
                                    is-trace? (and (seq? val-form)
                                                   (seq val-form)
                                                   (symbol? (first val-form))
                                                   (= "trace" (name (first val-form))))
                                    trace-addr (when (and is-trace?
                                                          (keyword? (second val-form)))
                                                 (second val-form))
                                    sym-deps (if trace-addr
                                               (conj val-deps trace-addr)
                                               val-deps)
                                    env' (if (seq sym-deps)
                                           (assoc env sym sym-deps)
                                           env)]
                                [acc' env'])
                              ;; Destructuring binding — walk but don't track
                              [acc' env])))
                        [acc env]
                        (or pairs []))]
      (walk-forms acc' env' body))))

(defn- handle-branch [acc env args]
  ;; For if/when/when-not/if-let/when-let/if-not
  (let [has-trace? (some contains-gen-call? args)]
    (cond-> (walk-forms acc env args)
      has-trace? (assoc :has-branches? true))))

(defn- handle-cond [acc env args]
  (let [has-trace? (some contains-gen-call? args)]
    (cond-> (walk-forms acc env args)
      has-trace? (assoc :has-branches? true))))

(defn- handle-case [acc env args]
  ;; case: (case expr val1 result1 val2 result2 ... default?)
  (let [has-trace? (some contains-gen-call? (rest args))]
    (cond-> (walk-forms acc env args)
      has-trace? (assoc :has-branches? true))))

(defn- handle-and-or [acc env args]
  ;; and/or with traces → branches (short-circuit = conditional execution)
  (let [has-trace? (some contains-gen-call? args)]
    (cond-> (walk-forms acc env args)
      has-trace? (assoc :has-branches? true))))

(defn- handle-loop-form [acc env args]
  ;; doseq/dotimes/for: first arg is bindings vector, rest is body
  (let [body (rest args)
        has-trace? (some contains-gen-call? body)]
    (cond-> (walk-forms acc env args)
      has-trace? (assoc :has-loops? true))))

(defn- handle-loop-loop [acc env args]
  ;; loop: like let, has bindings + body
  (if (empty? args)
    acc
    (let [bindings-form (first args)
          body (rest args)
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
                        (or pairs []))]
      (walk-forms acc' env' body))))

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
      "trace"    (handle-trace acc env args)
      "splice"   (handle-splice acc env args)
      "param"    (handle-param acc env args)
      "let"      (handle-let acc env args)
      "if"       (handle-branch acc env args)
      "when"     (handle-branch acc env args)
      "when-not" (handle-branch acc env args)
      "when-let" (handle-branch acc env args)
      "if-let"   (handle-branch acc env args)
      "if-not"   (handle-branch acc env args)
      "cond"     (handle-cond acc env args)
      "case"     (handle-case acc env args)
      "and"      (handle-and-or acc env args)
      "or"       (handle-and-or acc env args)
      "do"       (walk-forms acc env args)
      "doseq"    (handle-loop-form acc env args)
      "dotimes"  (handle-loop-form acc env args)
      "for"      (handle-loop-form acc env args)
      "loop"     (handle-loop-loop acc env args)
      "fn"       (handle-fn acc env args)
      "defn"     (handle-fn acc env (rest args))
      "letfn"    (handle-letfn acc env args)
      "quote"    acc
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
                                 (clojure.set/intersection (:deps ts) addr-set)])
                              static-sites))]
    (loop [remaining (set (keys dep-map))
           result []]
      (if (empty? remaining)
        result
        (let [ready (first (filter (fn [a]
                                     (empty? (clojure.set/intersection
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
   dependency structure, and classifications."
  [source]
  (when source
    (let [params (first source)
          body (rest source)
          init-acc {:trace-sites []
                    :splice-sites []
                    :param-sites []
                    :dynamic-addresses? false
                    :has-branches? false
                    :has-loops? false}
          result (walk-forms init-acc {} body)]
      (assoc result
             :params (vec params)
             :return-form (last body)
             :dep-order (topo-sort (:trace-sites result))
             :static? (and (not (:dynamic-addresses? result))
                           (not (:has-branches? result))
                           (not (:has-loops? result)))))))
