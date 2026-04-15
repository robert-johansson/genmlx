(ns genmlx.gfi-compiler
  "Compile model specifications (pure Clojure data) into DynamicGF objects.
   Used by the GFI property-based test suite to generate random models."
  (:require [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]))

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
