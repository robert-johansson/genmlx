(ns genmlx.dist.core
  "Foundation for distributions-as-data: single Distribution record,
   open multimethods, and GFI bridge functions."
  (:require [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Cached zero constant
;; ---------------------------------------------------------------------------

(def ^:private ZERO (mx/scalar 0.0))

;; ---------------------------------------------------------------------------
;; Open multimethods — dispatch on (:type dist)
;; ---------------------------------------------------------------------------

;; Internal multimethods — distributions implement these
(defmulti dist-sample*   (fn [d _key] (:type d)))
(defmulti dist-log-prob  (fn [d _value] (:type d)))
(defmulti dist-reparam   (fn [d _key] (:type d)))
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

(defn dist-simulate [dist]
  (let [key (or (:genmlx.dynamic/key (meta dist)) (rng/fresh-key))

        v  (dist-sample dist key)
        lp (dist-log-prob dist v)]
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

;; ---------------------------------------------------------------------------
;; THE single record for all distributions
;; ---------------------------------------------------------------------------

(defn dist-propose [dist]
  (let [key (or (:genmlx.dynamic/key (meta dist)) (rng/fresh-key))

        v  (dist-sample dist key)
        lp (dist-log-prob dist v)]
    {:choices (cm/->Value v) :weight lp :retval v}))

(defn dist-assess [dist choices]
  (if (cm/has-value? choices)
    (let [v  (cm/get-value choices)
          lp (dist-log-prob dist v)]
      {:retval v :weight lp})
    (throw (ex-info "assess requires fully-specified choices" {:dist (:type dist)}))))

(defrecord Distribution [type params]
  p/IGenerativeFunction
  (simulate [this _] (dist-simulate this))

  p/IGenerate
  (generate [this _ constraints] (dist-generate this constraints))

  p/IAssess
  (assess [this _ choices] (dist-assess this choices))

  p/IPropose
  (propose [this _] (dist-propose this))

  p/IProject
  (project [this trace selection]
    ;; A distribution has a single choice. If nothing is selected, return 0.
    (if (identical? selection sel/none)
      ZERO
      (:score trace))))

;; ---------------------------------------------------------------------------
;; IHasArgumentGrads — distributions do not declare argument differentiability
;; ---------------------------------------------------------------------------

(extend-type Distribution
  p/IHasArgumentGrads
  (has-argument-grads [_] nil))

;; ---------------------------------------------------------------------------
;; map->dist — create a Distribution from a plain map
;; ---------------------------------------------------------------------------

(defn map->dist
  "Create a Distribution from a plain map with :sample and :log-prob functions.
   Required keys:
     :type      - keyword identifier (auto-generated if omitted)
     :sample    - (fn [key] -> MLX-value)
     :log-prob  - (fn [value] -> MLX-scalar)
   Optional keys:
     :reparam   - (fn [key] -> MLX-value) reparameterized sample
     :support   - (fn [] -> seq) enumerable support
     :sample-n  - (fn [key n] -> MLX-array) batch sampling"
  [{:keys [type sample log-prob reparam support sample-n]}]
  (let [type-kw (or type (keyword (gensym "custom-dist")))]
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
        log-norm-w (mx/subtract log-weights (mx/logsumexp log-weights))
        n (count components)]
    ;; Compute log p(v) = logsumexp_k(log w_k + log p_k(v))
    ;; Stay in MLX graph: compute each log(w_k * p_k(v)) and reduce
    (let [component-lps (mapv #(dist-log-prob % v) components)]
      ;; Build sum via logaddexp chain to stay differentiable
      (reduce (fn [acc i]
                (mx/logaddexp acc
                  (mx/add (mx/index log-norm-w i)
                          (nth component-lps i))))
              (mx/add (mx/index log-norm-w 0) (first component-lps))
              (range 1 n)))))

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

(defmethod dist-sample* :product [d key]
  (let [{:keys [form components]} (:params d)
        key (rng/ensure-key key)]
    (if (= form :vector)
      (let [keys (rng/split-n key (count components))]
        (mapv (fn [comp k] (dist-sample comp k))
              components keys))
      ;; map form
      (let [entries (vec components)
            keys (rng/split-n key (count entries))]
        (into {}
          (map-indexed (fn [i [k comp]]
                         [k (dist-sample comp (nth keys i))])
                       entries))))))

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
  (let [{:keys [form components]} (:params d)
        key (rng/ensure-key key)]
    (if (= form :vector)
      (let [keys (rng/split-n key (count components))]
        (mapv (fn [comp k] (dist-reparam comp k))
              components keys))
      (let [entries (vec components)
            keys (rng/split-n key (count entries))]
        (into {}
          (map-indexed (fn [i [k comp]]
                         [k (dist-reparam comp (nth keys i))])
                       entries))))))

(defmethod dist-support :product [d]
  (let [{:keys [form components]} (:params d)]
    (if (= form :vector)
      ;; Cartesian product of component supports
      (let [supports (mapv dist-support components)]
        (reduce (fn [acc s]
                  (for [prefix acc, v s]
                    (conj prefix v)))
                (mapv vector (first supports))
                (rest supports)))
      ;; Map form: Cartesian product with keys
      (let [entries (vec components)
            ks (mapv first entries)
            supports (mapv (fn [[_ comp]] (dist-support comp)) entries)
            value-seqs (reduce (fn [acc s]
                                 (for [prefix acc, v s]
                                   (conj prefix v)))
                               (mapv vector (first supports))
                               (rest supports))]
        (mapv (fn [vals] (zipmap ks vals)) value-seqs)))))

(defmethod dist-sample-n* :product [d key n]
  (let [{:keys [form components]} (:params d)
        key (rng/ensure-key key)]
    (if (= form :vector)
      (let [keys (rng/split-n key (count components))]
        (mapv (fn [comp k] (dist-sample-n comp k n))
              components keys))
      (let [entries (vec components)
            keys (rng/split-n key (count entries))]
        (into {}
          (map-indexed (fn [i [k comp]]
                         [k (dist-sample-n comp (nth keys i) n)])
                       entries))))))
