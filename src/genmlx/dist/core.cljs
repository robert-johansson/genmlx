(ns genmlx.dist.core
  "Foundation for distributions-as-data: single Distribution record,
   open multimethods, and GFI bridge functions."
  (:require [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

;; ---------------------------------------------------------------------------
;; Open multimethods — dispatch on (:type dist)
;; ---------------------------------------------------------------------------

(defmulti dist-sample   (fn [d _key] (:type d)))
(defmulti dist-log-prob (fn [d _value] (:type d)))
(defmulti dist-reparam  (fn [d _key] (:type d)))
(defmulti dist-support   (fn [d] (:type d)))
(defmulti dist-sample-n  (fn [d _key _n] (:type d)))

;; Defaults: helpful errors
(defmethod dist-reparam :default [d _]
  (throw (ex-info (str "Distribution " (:type d) " does not support reparameterized sampling")
                  {:type (:type d)})))

(defmethod dist-support :default [d]
  (throw (ex-info (str "Distribution " (:type d) " is not enumerable")
                  {:type (:type d)})))

;; Default dist-sample-n: sequential fallback for distributions that can't batch
(defmethod dist-sample-n :default [d key n]
  (let [keys (rng/split-n (rng/ensure-key key) n)]
    (mx/stack (mapv #(dist-sample d %) keys))))

;; ---------------------------------------------------------------------------
;; GFI bridge: distribution -> trace
;; ---------------------------------------------------------------------------

(defn dist-simulate [dist]
  (let [v  (dist-sample dist nil)
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
    {:trace (dist-simulate dist) :weight (mx/scalar 0.0)}))

;; ---------------------------------------------------------------------------
;; THE single record for all distributions
;; ---------------------------------------------------------------------------

(defn dist-propose [dist]
  (let [v  (dist-sample dist nil)
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
  (propose [this _] (dist-propose this)))

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

(defmethod dist-sample :mixture [d key]
  (let [{:keys [components log-weights]} (:params d)
        key (rng/ensure-key key)
        [k1 k2] (rng/split key)
        idx (mx/item (rng/categorical k1 log-weights))
        component (nth components (int idx))]
    (dist-sample component k2)))

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
