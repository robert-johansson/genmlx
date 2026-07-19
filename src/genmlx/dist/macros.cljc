(ns genmlx.dist.macros
  "The defdist macro for defining distributions with minimal boilerplate.

   (defdist gaussian
     \"Gaussian distribution.\"
     [mu sigma]
     (sample [key] ...)
     (log-prob [v] ...)
     (reparam [key] ...))

   Generates: constructor fn, defmethod for dist-sample/dist-log-prob/
   dist-reparam/dist-support.")

#?(:org.babashka/nbb
   (defmacro defdist
     "Define a distribution type with constructor and multimethod implementations.

      name     - symbol, becomes both the constructor fn name and the keyword type
      docstr   - optional docstring
      params   - vector of parameter names (auto-wrapped with ensure-array in constructor)
      clauses  - one or more of:
                   (validate body...)
                   (sample [key] body...)
                   (log-prob [v] body...)
                   (reparam [key] body...)
                   (support [] body...)

      The validate clause body runs inside the emitted public constructor
      BEFORE the ensure-array coercion, with the RAW (uncoerced) params bound —
      the place for check-* parameter guards, which only inspect JS numbers
      and must see the argument before it becomes an MLX array."
     [dist-name & args]
     (let [[docstr args] (if (string? (first args))
                           [(first args) (rest args)]
                           [nil args])
           params (first args)
           clauses (rest args)
           type-kw (keyword (name dist-name))
           clause-map (into {} (map (juxt first rest)) clauses)
           ;; One hygienic binding for the Distribution argument, shared by all
           ;; emitted defmethods. No clause body can see (or shadow) it.
           d-sym (gensym "d")
           ;; Build the destructuring let for params from (:params d)
           params-let (vec (mapcat (fn [p]
                                     [p (list (keyword (name p))
                                              (list :params d-sym))])
                                   params))]
       (letfn [;; Emit a single-arg defmethod (sample/log-prob/reparam): bind the
               ;; params from (:params d), coerce the raw arg, then run the body.
               (emit-arg-method [clause-key method-sym coerce-sym raw-sym]
                 (when-let [clause (get clause-map clause-key)]
                   (let [[clause-args & clause-body] clause
                         arg-sym (first clause-args)]
                     `(defmethod ~method-sym ~type-kw [~d-sym ~raw-sym]
                        (let [~@params-let
                              ~arg-sym (~coerce-sym ~raw-sym)]
                          ~@clause-body)))))]
       `(do
          ;; Constructor function
          ~(let [ctor-body `(genmlx.dist.core/->Distribution
                              ~type-kw
                              ~(into {} (map (juxt (comp keyword name) identity)) params))]
             `(defn ~dist-name ~@(when docstr [docstr]) ~params
                ~@(get clause-map 'validate)
                (let [~@(mapcat (fn [p] [p (list 'genmlx.mlx/ensure-array p)])
                                params)]
                  ~ctor-body)))

          ;; dist-sample method
          ~(emit-arg-method 'sample 'genmlx.dist.core/dist-sample*
                            'genmlx.mlx.random/ensure-key (gensym "raw-key"))

          ;; dist-log-prob method
          ~(emit-arg-method 'log-prob 'genmlx.dist.core/dist-log-prob
                            'genmlx.mlx/ensure-array (gensym "raw-val"))

          ;; dist-reparam method (optional)
          ~(emit-arg-method 'reparam 'genmlx.dist.core/dist-reparam
                            'genmlx.mlx.random/ensure-key (gensym "raw-key"))

          ;; dist-support method (optional)
          ~(when-let [support-clause (get clause-map 'support)]
             (let [[_support-args & support-body] support-clause]
               `(defmethod genmlx.dist.core/dist-support ~type-kw [~d-sym]
                  (let [~@params-let]
                    ~@support-body))))))))

)
