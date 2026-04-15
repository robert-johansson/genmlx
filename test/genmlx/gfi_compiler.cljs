(ns genmlx.gfi-compiler
  "Compile model specifications (pure Clojure data) into DynamicGF objects.
   Used by the GFI property-based test suite to generate random models."
  (:require [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx :as mx]))

;; ---------------------------------------------------------------------------
;; Distribution resolution
;; ---------------------------------------------------------------------------

(def ^:private dist-constructors
  {:gaussian    dist/gaussian
   :uniform     dist/uniform
   :bernoulli   dist/bernoulli
   :exponential dist/exponential
   :beta        dist/beta-dist
   :gamma       dist/gamma-dist
   :cauchy      dist/cauchy
   :laplace     dist/laplace
   :delta       dist/delta})

(defn resolve-dist
  "Map a distribution keyword to its constructor function."
  [dist-type]
  (or (get dist-constructors dist-type)
      (throw (ex-info (str "Unknown distribution type: " dist-type)
                      {:dist-type dist-type
                       :supported (keys dist-constructors)}))))

;; ---------------------------------------------------------------------------
;; Source form generation
;; ---------------------------------------------------------------------------

(defn- resolve-arg
  "Resolve a dist arg: keywords become symbols, numbers pass through."
  [a]
  (if (keyword? a) (symbol (name a)) a))

(def ^:private dist-names
  "Map spec keywords to actual constructor names in the dist namespace."
  {:gaussian    "gaussian"
   :uniform     "uniform"
   :bernoulli   "bernoulli"
   :exponential "exponential"
   :beta        "beta-dist"
   :gamma       "gamma-dist"
   :cauchy      "cauchy"
   :laplace     "laplace"
   :delta       "delta"})

(defn- dist-sym
  "Build a namespace-qualified dist symbol: :gaussian -> dist/gaussian."
  [dist-type]
  (symbol "dist" (or (get dist-names dist-type) (name dist-type))))

(defn spec->source
  "Transform a spec map into a quoted source form for schema extraction."
  [{:keys [sites args return]}]
  (let [params   (mapv #(symbol (name %)) args)
        bindings (->> sites
                      (mapcat (fn [{:keys [addr dist args]}]
                                [(symbol (name addr))
                                 (list 'trace addr
                                       (apply list (dist-sym dist)
                                              (map resolve-arg args)))]))
                      vec)]
    (list params (list 'let bindings (symbol (name return))))))

;; ---------------------------------------------------------------------------
;; Body function generation
;; ---------------------------------------------------------------------------

(defn spec->body-fn
  "Create an executable body function from a spec map."
  [{:keys [sites args return]}]
  (fn [rt & model-args]
    (let [trace-fn (.-trace rt)
          arg-env  (zipmap args model-args)
          env      (reduce (fn [env {:keys [addr dist args]}]
                             (let [resolved (mapv #(if (keyword? %)
                                                     (or (get env %) (get arg-env %))
                                                     %)
                                                  args)
                                   v (trace-fn addr (apply (resolve-dist dist) resolved))]
                               (assoc env addr v)))
                           {}
                           sites)]
      (get env return))))

;; ---------------------------------------------------------------------------
;; Full pipeline
;; ---------------------------------------------------------------------------

(defn spec->gf
  "Compile a spec map into a DynamicGF with auto-generated PRNG keys."
  [spec]
  (-> (dyn/make-gen-fn (spec->body-fn spec) (spec->source spec))
      dyn/auto-key))

;; ---------------------------------------------------------------------------
;; Branching model compilation (Gap 3)
;; ---------------------------------------------------------------------------

(defn- branch-body-source
  "Generate source form for one branch (true or false)."
  [sites]
  (if (= 1 (count sites))
    (let [{:keys [addr dist args]} (first sites)]
      (list 'trace addr (apply list (dist-sym dist) (map resolve-arg args))))
    (let [bindings (mapcat (fn [{:keys [addr dist args]}]
                             [(symbol (name addr))
                              (list 'trace addr
                                    (apply list (dist-sym dist) (map resolve-arg args)))])
                           sites)]
      (list 'let (vec bindings) (symbol (name (:addr (last sites))))))))

(defn branching-spec->source
  "Transform a branching model spec into a quoted source form.
   Emits (if (> coin 0.5) true-branch false-branch) which schema
   extraction recognizes as has-branches? true."
  [{:keys [pre-sites branch true-sites false-sites args]}]
  (let [params (mapv #(symbol (name %)) args)
        pre-bindings (mapcat (fn [{:keys [addr dist args]}]
                               [(symbol (name addr))
                                (list 'trace addr
                                      (apply list (dist-sym dist) (map resolve-arg args)))])
                             pre-sites)
        coin-sym (symbol (name (:addr branch)))
        coin-trace (list 'trace (:addr branch)
                         (apply list (dist-sym (:dist branch)) (:args branch)))
        true-body (branch-body-source true-sites)
        false-body (branch-body-source false-sites)
        if-form (list 'if (list '> coin-sym 0.5) true-body false-body)
        all-bindings (vec (concat pre-bindings [coin-sym coin-trace]))]
    (list params (list 'let all-bindings if-form))))

(defn branching-spec->body-fn
  "Create an executable body function from a branching model spec.
   Uses mx/item on the coin value to materialize for branching."
  [{:keys [pre-sites branch true-sites false-sites args]}]
  (let [execute-sites
        (fn [trace-fn env arg-env sites]
          (reduce (fn [_ {:keys [addr dist args]}]
                    (let [resolved (mapv #(if (keyword? %)
                                            (or (get env %) (get arg-env %))
                                            %) args)]
                      (trace-fn addr (apply (resolve-dist dist) resolved))))
                  nil
                  sites))]
    (fn [rt & model-args]
      (let [trace-fn (.-trace rt)
            arg-env (zipmap args model-args)
            pre-env (reduce (fn [env {:keys [addr dist args]}]
                              (let [resolved (mapv #(if (keyword? %)
                                                      (or (get env %) (get arg-env %))
                                                      %) args)
                                    v (trace-fn addr (apply (resolve-dist dist) resolved))]
                                (assoc env addr v)))
                            {}
                            pre-sites)
            coin (trace-fn (:addr branch)
                           (apply (resolve-dist (:dist branch)) (:args branch)))]
        (if (> (mx/item coin) 0.5)
          (execute-sites trace-fn pre-env arg-env true-sites)
          (execute-sites trace-fn pre-env arg-env false-sites))))))

(defn branching-spec->gf
  "Compile a branching model spec into a DynamicGF with auto-generated PRNG keys."
  [spec]
  (-> (dyn/make-gen-fn (branching-spec->body-fn spec) (branching-spec->source spec))
      dyn/auto-key))

(comment
  ;; Manual verification:
  ;; bun run --bun nbb -e '(require (quote genmlx.gfi-compiler)) ...'
  (require '[genmlx.protocols :as p] '[genmlx.mlx :as mx])

  (def spec {:sites [{:addr :x :dist :gaussian :args [0 1] :deps []}]
             :args [] :return :x})
  (spec->source spec)  ;; => ([] (let [x (trace :x (dist/gaussian 0 1))] x))
  (spec->gf spec)      ;; => DynamicGF with schema

  (def two-site {:sites [{:addr :x :dist :gaussian :args [0 1] :deps []}
                         {:addr :y :dist :gaussian :args [:x 1] :deps [:x]}]
                 :args [] :return :y})
  (let [gf (spec->gf two-site)
        tr (p/simulate gf [])]
    {:retval (mx/item (:retval tr))
     :score  (mx/item (:score tr))}))
