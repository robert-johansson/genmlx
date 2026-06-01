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
                   (sample [key] body...)
                   (log-prob [v] body...)
                   (reparam [key] body...)
                   (support [] body...)"
     [dist-name & args]
     (let [[docstr args] (if (string? (first args))
                           [(first args) (rest args)]
                           [nil args])
           params (first args)
           clauses (rest args)
           type-kw (keyword (name dist-name))
           clause-map (into {} (map (juxt first rest)) clauses)
           ;; Build the destructuring let for params from (:params d)
           params-let (vec (mapcat (fn [p]
                                     [p (list (keyword (name p))
                                              (list :params 'd))])
                                   params))]
       (letfn [;; Emit a single-arg defmethod (sample/log-prob/reparam): bind the
               ;; params from (:params d), coerce the raw arg, then run the body.
               (emit-arg-method [clause-key method-sym coerce-sym raw-sym]
                 (when-let [clause (get clause-map clause-key)]
                   (let [[clause-args & clause-body] clause
                         arg-sym (first clause-args)]
                     `(defmethod ~method-sym ~type-kw [~'d ~raw-sym]
                        (let [~@params-let
                              ~arg-sym (~coerce-sym ~raw-sym)]
                          ~@clause-body)))))]
       `(do
          ;; Constructor function
          ~(let [ctor-body `(genmlx.dist.core/->Distribution
                              ~type-kw
                              ~(into {} (map (juxt (comp keyword name) identity)) params))]
             `(defn ~dist-name ~@(when docstr [docstr]) ~params
                (let [~@(mapcat (fn [p] [p (list 'genmlx.mlx/ensure-array p)])
                                params)]
                  ~ctor-body)))

          ;; dist-sample method
          ~(emit-arg-method 'sample 'genmlx.dist.core/dist-sample*
                            'genmlx.mlx.random/ensure-key 'raw-key#)

          ;; dist-log-prob method
          ~(emit-arg-method 'log-prob 'genmlx.dist.core/dist-log-prob
                            'genmlx.mlx/ensure-array 'raw-val#)

          ;; dist-reparam method (optional)
          ~(emit-arg-method 'reparam 'genmlx.dist.core/dist-reparam
                            'genmlx.mlx.random/ensure-key 'raw-key#)

          ;; dist-support method (optional)
          ~(when-let [support-clause (get clause-map 'support)]
             (let [[_support-args & support-body] support-clause]
               `(defmethod genmlx.dist.core/dist-support ~type-kw [~'d]
                  (let [~@params-let]
                    ~@support-body))))))))

)
