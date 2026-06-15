(ns genmlx.dist.core
  "Foundation for distributions-as-data: single Distribution record,
   open multimethods, and GFI bridge functions."
  (:require [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.mlx.constants :refer [ZERO]]
            [genmlx.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Open multimethods — dispatch on (:type dist)
;; ---------------------------------------------------------------------------

;; Internal multimethods — distributions implement these
(defmulti dist-sample*   (fn [d _key] (:type d)))
(defmulti dist-log-prob  (fn [d _value] (:type d)))
(defmulti dist-reparam   (fn [d _key] (:type d)))
;; Differentiable log-prob of a REPARAMETERIZED sample. For continuous dists the
;; reparam value is an ordinary sample, so the default delegates to dist-log-prob
;; (unchanged behavior). Discrete relaxations (Gumbel-softmax categorical) whose
;; reparam value is a [K] one-hot — which dist-log-prob would int-truncate and
;; mis-gather — override this with a differentiable score, keeping the relaxation
;; OUT of ordinary categorical scoring/assess/generate (genmlx-0nyj).
(defmulti dist-reparam-log-prob (fn [d _value] (:type d)))
(defmulti dist-support   (fn [d] (:type d)))
(defmulti dist-log-prob-support
  "Log-probabilities for ALL support values at once. Returns tensor of shape
   [K, ...] where K is the support size and remaining dims match the dist
   params shape (from broadcasting with previous enumerate axes)."
  (fn [d] (:type d)))
(defmulti dist-sample-n* (fn [d _key _n] (:type d)))

;; Public API — delegates to multimethod after ensuring key.
(defn dist-sample
  "Sample from distribution d using PRNG key."
  [d key]
  (let [key (rng/ensure-key key)]
    (dist-sample* d key)))

(defn dist-sample-n
  "Batch-sample n values from distribution d using PRNG key."
  [d key n]
  (let [key (rng/ensure-key key)
        result (dist-sample-n* d key n)]
    (mx/auto-cleanup! true) ;; aggressive: leaf op, safe for forced GC
    result))

;; Defaults: helpful errors
(defmethod dist-reparam :default [d _]
  (throw (ex-info (str "Distribution " (:type d) " does not support reparameterized sampling")
                  {:type (:type d)})))

;; Default: the reparam value is an ordinary sample, score it as usual. Only
;; relaxed-discrete dists (categorical) need an override.
(defmethod dist-reparam-log-prob :default [d v] (dist-log-prob d v))

(defmethod dist-support :default [d]
  (throw (ex-info (str "Distribution " (:type d) " is not enumerable")
                  {:type (:type d)})))

;; Default dist-log-prob-support: per-value fallback for distributions without bulk impl.
(defmethod dist-log-prob-support :default [d]
  (let [support (dist-support d)]
    (mx/stack (mapv #(dist-log-prob d %) support))))

;; Default dist-sample-n*: sequential fallback for distributions that can't batch.
(defmethod dist-sample-n* :default [d key n]
  (let [keys (rng/split-n (rng/ensure-key key) n)]
    (mx/stack (mapv #(dist-sample d %) keys))))

;; ---------------------------------------------------------------------------
;; GFI bridge: distribution -> trace
;; ---------------------------------------------------------------------------

(defn- sample-and-score
  "Draw a value from dist (using its threaded PRNG key, if any) and its log-prob.
   Returns [value log-prob]."
  [dist]
  (let [key (or (:genmlx.dynamic/key (meta dist)) (rng/fresh-key))
        v  (dist-sample dist key)
        lp (dist-log-prob dist v)]
    [v lp]))

(defn dist-simulate [dist]
  (let [[v lp] (sample-and-score dist)]
    (tr/make-trace {:gen-fn dist :args [] :choices (cm/->Value v)
                    :retval v :score lp})))

(defn dist-generate [dist constraints]
  (if (cm/has-value? constraints)
    (let [v  (cm/get-value constraints)
          lp (dist-log-prob dist v)]
      {:trace (tr/make-trace {:gen-fn dist :args [] :choices (cm/->Value v)
                              :retval v :score lp})
       :weight lp})
    {:trace (dist-simulate dist) :weight ZERO}))

(defn dist-propose [dist]
  (let [[v lp] (sample-and-score dist)]
    {:choices (cm/->Value v) :weight lp :retval v}))

(defn- dist-assess [dist choices]
  (if (cm/has-value? choices)
    (let [v  (cm/get-value choices)
          lp (dist-log-prob dist v)]
      {:retval v :weight lp})
    (throw (ex-info "assess requires fully-specified choices" {:dist (:type dist)}))))

(defn- root-selected?
  "Does this selection select a Distribution's single root value? The root
   has no address, so only selections that select everything at this level
   can select it: canonical `all`, or a Complement whose inner does not.
   Address-naming selections (SelectAddrs/Hierarchical) name addresses a
   single-value trace does not have."
  [selection]
  (cond
    (identical? sel/all selection) true
    (instance? sel/Complement selection) (not (root-selected? (:inner selection)))
    :else false))

(defn- dist-update
  "Update a distribution trace. A constrained value replaces the old one
   with weight lp(v') - lp(v) (thesis update convention: non-fresh new
   score minus old score; the single site is constrained, so everything
   is non-fresh). Empty constraints leave the trace unchanged."
  [dist trace constraints]
  (if (cm/has-value? constraints)
    (let [v' (cm/get-value constraints)
          lp' (dist-log-prob dist v')]
      {:trace (tr/make-trace {:gen-fn dist :args [] :choices (cm/->Value v')
                              :retval v' :score lp'})
       :weight (mx/subtract lp' (:score trace))
       :discard (:choices trace)})
    {:trace trace :weight ZERO :discard cm/EMPTY}))

(defn- dist-regenerate
  "Regenerate a distribution trace. A selected root value resamples from
   the distribution itself with weight 0 (prior-proposal cancellation:
   the score delta equals the proposal ratio exactly). Unselected leaves
   the trace unchanged, also weight 0."
  [dist trace selection]
  (if (root-selected? selection)
    {:trace (dist-simulate dist) :weight ZERO}
    {:trace trace :weight ZERO}))

;; ---------------------------------------------------------------------------
;; THE single record for all distributions
;; ---------------------------------------------------------------------------

(defrecord Distribution [type params]
  p/IGenerativeFunction
  (simulate [this _] (dist-simulate this))

  p/IGenerate
  (generate [this _ constraints] (dist-generate this constraints))

  p/IAssess
  (assess [this _ choices] (dist-assess this choices))

  p/IPropose
  (propose [this _] (dist-propose this))

  p/IUpdate
  (update [this trace constraints] (dist-update this trace constraints))

  p/IRegenerate
  (regenerate [this trace selection] (dist-regenerate this trace selection))

  p/IProject
  (project [this trace selection]
    ;; A distribution has a single root choice: project returns its score
    ;; iff the selection selects the root. The old check returned the FULL
    ;; score for ANY selection other than canonical `none` — an unrelated
    ;; (select :foo) projected the whole score (genmlx-yeam).
    (if (root-selected? selection)
      (:score trace)
      ZERO)))

;; ---------------------------------------------------------------------------
;; IHasArgumentGrads — distributions do not declare argument differentiability
;; ---------------------------------------------------------------------------

(extend-type Distribution
  p/IHasArgumentGrads
  (has-argument-grads [_] nil))

;; ---------------------------------------------------------------------------
;; map->dist — create a Distribution from a plain map
;; ---------------------------------------------------------------------------

(def ^:private map->dist-types
  "Types registered by map->dist. Re-registering YOUR OWN custom type is
   normal REPL flow; clobbering a type someone else registered (e.g. a
   builtin like :gaussian) silently redefined it process-wide, so that
   now throws (genmlx-yeam). Registry bookkeeping only — never read by
   computation paths."
  (atom #{}))

(defn map->dist
  "Create a Distribution from a plain map with :sample and :log-prob functions.
   Required keys:
     :type      - keyword identifier (auto-generated if omitted); must not
                  collide with an already-registered distribution type
     :sample    - (fn [key] -> MLX-value)
     :log-prob  - (fn [value] -> MLX-scalar)
   Optional keys:
     :reparam   - (fn [key] -> MLX-value) reparameterized sample
     :support   - (fn [] -> seq) enumerable support
     :sample-n  - (fn [key n] -> MLX-array) batch sampling"
  [{:keys [type sample log-prob reparam support sample-n]}]
  (let [type-kw (or type (keyword (gensym "custom-dist")))]
    (when (and (contains? (methods dist-sample*) type-kw)
               (not (contains? @map->dist-types type-kw)))
      (throw (ex-info (str "map->dist: " type-kw " is already a registered "
                           "distribution type — choose a unique :type "
                           "(registering it would silently redefine the "
                           "existing distribution process-wide)")
                      {:type type-kw})))
    (swap! map->dist-types conj type-kw)
    (defmethod dist-sample* type-kw [_ key] (sample key))
    (defmethod dist-log-prob type-kw [_ value] (log-prob value))
    (when reparam
      (defmethod dist-reparam type-kw [_ key] (reparam key)))
    (when support
      (defmethod dist-support type-kw [_] (support)))
    (when sample-n
      (defmethod dist-sample-n* type-kw [_ key n] (sample-n key n)))
    (->Distribution type-kw {})))

;; ---------------------------------------------------------------------------
;; Mixture distribution
;; ---------------------------------------------------------------------------

(defn mixture
  "Create a mixture distribution from component distributions and log-weights.
   components: vector of Distribution records
   log-weights: MLX array of log mixing weights (unnormalized)"
  [components log-weights]
  (let [log-w (if (mx/array? log-weights) log-weights (mx/array log-weights))]
    (->Distribution :mixture {:components components :log-weights log-w})))

(defmethod dist-sample* :mixture [d key]
  (let [{:keys [components log-weights]} (:params d)
        key (rng/ensure-key key)
        [k1 k2] (rng/split key)
        ;; Sample ALL components, then select by categorical index.
        ;; Stays in MLX graph (no mx/item) — vectorizable + differentiable.
        idx (rng/categorical k1 log-weights)
        component-keys (rng/split-n k2 (count components))
        all-samples (mx/stack (mapv (fn [c k] (dist-sample c k))
                                    components component-keys))]
    (mx/index all-samples idx)))

(defmethod dist-log-prob :mixture [d v]
  (let [{:keys [components log-weights]} (:params d)
        v (mx/ensure-array v)
        ;; Normalize log-weights
        log-norm-w (mx/subtract log-weights (mx/logsumexp log-weights))]
    ;; Compute log p(v) = logsumexp_k(log w_k + log p_k(v))
    ;; Stay in MLX graph: compute each log(w_k * p_k(v)) and reduce via a
    ;; logaddexp chain to stay differentiable.
    (->> components
         (map #(dist-log-prob % v))
         (map-indexed (fn [i lp] (mx/add (mx/index log-norm-w i) lp)))
         (reduce mx/logaddexp))))

;; ---------------------------------------------------------------------------
;; Product distribution
;; ---------------------------------------------------------------------------

(defn product
  "Create a product (joint independent) distribution from a vector or map
   of component distributions. Sampling is independent; log-prob is the sum.
   Vector form returns a vector of MLX values; map form returns a map."
  [components]
  (cond
    (vector? components)
    (->Distribution :product {:form :vector :components components})

    (map? components)
    (->Distribution :product {:form :map :components components})

    :else
    (throw (ex-info "product requires a vector or map of distributions"
                    {:got (type components)}))))

(defn- product-map
  "Split key across a product distribution's components and apply f to each.
   f is (fn [component sub-key] -> value). Returns a vector for :vector form,
   a key->value map (preserving component keys) for :map form."
  [d key f]
  (let [{:keys [form components]} (:params d)
        key (rng/ensure-key key)]
    (if (= form :vector)
      (let [keys (rng/split-n key (count components))]
        (mapv f components keys))
      (let [entries (vec components)
            keys (rng/split-n key (count entries))]
        (into {}
          (map (fn [[k comp] sub-key] [k (f comp sub-key)])
               entries keys))))))

(defmethod dist-sample* :product [d key]
  (product-map d key dist-sample))

(defmethod dist-log-prob :product [d value]
  (let [{:keys [form components]} (:params d)]
    (if (= form :vector)
      (reduce mx/add
              (map-indexed (fn [i comp]
                             (dist-log-prob comp (nth value i)))
                           components))
      ;; map form
      (reduce mx/add
              (map (fn [[k comp]]
                     (dist-log-prob comp (get value k)))
                   components)))))

(defmethod dist-reparam :product [d key]
  (product-map d key dist-reparam))

(defmethod dist-support :product [d]
  (let [{:keys [form components]} (:params d)]
    (letfn [(cartesian [supports]
              (reduce (fn [acc s]
                        (for [prefix acc, v s]
                          (conj prefix v)))
                      (mapv vector (first supports))
                      (rest supports)))]
      (if (= form :vector)
        ;; Cartesian product of component supports
        (cartesian (mapv dist-support components))
        ;; Map form: Cartesian product with keys
        (let [entries (vec components)
              ks (mapv first entries)
              supports (mapv (fn [[_ comp]] (dist-support comp)) entries)]
          (mapv (fn [vals] (zipmap ks vals)) (cartesian supports)))))))

(defmethod dist-sample-n* :product [d key n]
  (product-map d key (fn [comp k] (dist-sample-n comp k n))))
