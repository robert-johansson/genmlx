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
           clause-map (into {} (map (fn [clause]
                                      [(first clause) (rest clause)])
                                    clauses))
           ;; Build the destructuring let for params from (:params d)
           params-let (vec (mapcat (fn [p]
                                     [p (list (keyword (name p))
                                              (list :params 'd))])
                                   params))]
       `(do
          ;; Constructor function
          ~(let [ctor-body `(genmlx.dist.core/->Distribution
                              ~type-kw
                              ~(into {} (map (fn [p]
                                               [(keyword (name p)) p])
                                             params)))]
             (if docstr
               `(defn ~dist-name ~docstr ~params
                  (let [~@(mapcat (fn [p] [p (list 'genmlx.mlx/ensure-array p)])
                                  params)]
                    ~ctor-body))
               `(defn ~dist-name ~params
                  (let [~@(mapcat (fn [p] [p (list 'genmlx.mlx/ensure-array p)])
                                  params)]
                    ~ctor-body))))

          ;; dist-sample method
          ~(when-let [sample-clause (get clause-map 'sample)]
             (let [[sample-args & sample-body] sample-clause
                   key-sym (first sample-args)]
               `(defmethod genmlx.dist.core/dist-sample ~type-kw [~'d ~'raw-key#]
                  (let [~@params-let
                        ~key-sym (genmlx.mlx.random/ensure-key ~'raw-key#)]
                    ~@sample-body))))

          ;; dist-log-prob method
          ~(when-let [lp-clause (get clause-map 'log-prob)]
             (let [[lp-args & lp-body] lp-clause
                   val-sym (first lp-args)]
               `(defmethod genmlx.dist.core/dist-log-prob ~type-kw [~'d ~'raw-val#]
                  (let [~@params-let
                        ~val-sym (genmlx.mlx/ensure-array ~'raw-val#)]
                    ~@lp-body))))

          ;; dist-reparam method (optional)
          ~(when-let [reparam-clause (get clause-map 'reparam)]
             (let [[reparam-args & reparam-body] reparam-clause
                   key-sym (first reparam-args)]
               `(defmethod genmlx.dist.core/dist-reparam ~type-kw [~'d ~'raw-key#]
                  (let [~@params-let
                        ~key-sym (genmlx.mlx.random/ensure-key ~'raw-key#)]
                    ~@reparam-body))))

          ;; dist-support method (optional)
          ~(when-let [support-clause (get clause-map 'support)]
             (let [[_support-args & support-body] support-clause]
               `(defmethod genmlx.dist.core/dist-support ~type-kw [~'d]
                  (let [~@params-let]
                    ~@support-body)))))))

)

#?(:org.babashka/nbb
   (defmacro defdist-transform
     "Define a derived distribution via a deterministic transform of a base distribution.

      Example:
        (defdist-transform log-normal-t
          \"Log-normal via exp transform of gaussian.\"
          [mu sigma]
          :base (gaussian mu sigma)
          :forward exp     ;; forward transform: base -> derived
          :inverse log     ;; inverse transform: derived -> base
          :log-det-jac (fn [v] (negative (log v))))  ;; log |d(inverse)/dv|

      Generates: constructor fn, sample method (sample base, apply forward),
      log-prob method (inverse transform + base log-prob + log-det-jacobian)."
     [dist-name & args]
     (let [[docstr args] (if (string? (first args))
                           [(first args) (rest args)]
                           [nil args])
           params (first args)
           kv-args (apply hash-map (rest args))
           type-kw (keyword (name dist-name))
           base-expr (:base kv-args)
           forward-sym (:forward kv-args)
           inverse-sym (:inverse kv-args)
           log-det-jac-expr (:log-det-jac kv-args)]
       `(do
          ~(let [ctor-body `(genmlx.dist.core/->Distribution
                              ~type-kw
                              ~(into {:base-dist-fn `(fn [~@params] ~base-expr)
                                      :forward-fn `~forward-sym
                                      :inverse-fn `~inverse-sym
                                      :log-det-jac-fn `~log-det-jac-expr}
                                     (map (fn [p] [(keyword (name p)) p]) params)))]
             (if docstr
               `(defn ~dist-name ~docstr ~params
                  (let [~@(mapcat (fn [p] [p (list 'genmlx.mlx/ensure-array p)])
                                  params)]
                    ~ctor-body))
               `(defn ~dist-name ~params
                  (let [~@(mapcat (fn [p] [p (list 'genmlx.mlx/ensure-array p)])
                                  params)]
                    ~ctor-body))))

          (defmethod genmlx.dist.core/dist-sample ~type-kw [~'d ~'raw-key#]
            (let [params# (:params ~'d)
                  base-d# (apply (:base-dist-fn params#)
                                  (mapv #(get params# %) ~(mapv keyword (map name params))))
                  base-val# (genmlx.dist.core/dist-sample base-d# ~'raw-key#)]
              ((:forward-fn params#) base-val#)))

          (defmethod genmlx.dist.core/dist-log-prob ~type-kw [~'d ~'raw-val#]
            (let [params# (:params ~'d)
                  v# (genmlx.mlx/ensure-array ~'raw-val#)
                  inv-val# ((:inverse-fn params#) v#)
                  base-d# (apply (:base-dist-fn params#)
                                  (mapv #(get params# %) ~(mapv keyword (map name params))))
                  base-lp# (genmlx.dist.core/dist-log-prob base-d# inv-val#)
                  ldj# ((:log-det-jac-fn params#) v#)]
              (genmlx.mlx/add base-lp# ldj#)))))))
